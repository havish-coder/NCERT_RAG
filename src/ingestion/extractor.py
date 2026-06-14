"""
EntityRelationExtractor
=======================
Runs on Kaggle/Lightning AI (GPU available).
Uses Gemma 4 or Qwen 3.5 mid-size via HuggingFace transformers to extract
entities and relationships from each L2 text chunk.

Output format per chunk:
{
  "entities": [{"name": str, "entity_type": str, "description": str}, ...],
  "relationships": [{"source": str, "target": str, "relation_type": str, "description": str}, ...]
}

Simple approach: prompt → json.loads() → retry once on failure.
No constrained decoding library needed — modern instruction-tuned models
reliably produce valid JSON with a clear system prompt.
"""
from __future__ import annotations

import logging

from src.ingestion.llm_output import extract_generated_text, parse_json_block

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledge graph builder for Indian school textbooks (NCERT grades 6-12).
Extract entities and relationships from the given passage.

Return ONLY valid JSON in this exact format — no markdown, no explanation:
{
  "entities": [
    {"name": "string", "entity_type": "CONCEPT|PERSON|PLACE|EVENT|ORGANISM|CHEMICAL|LAW|THEOREM|FORMULA|PROCESS", "description": "1-2 sentence description"}
  ],
  "relationships": [
    {"source": "entity name", "target": "entity name", "relation_type": "IS_PART_OF|CAUSES|DEFINES|DISCOVERED_BY|REACTS_WITH|LEADS_TO|IS_TYPE_OF|USED_IN|PRODUCES|PERFORMS", "description": "brief description"}
  ]
}
Only extract entities clearly present or directly implied in the passage.
Relationships must only reference entity names you listed above."""


def load_extractor(model_id: str):
    """Load the HF text-generation pipeline once so it can be reused across many
    batches/shards (avoids re-loading a multi-GB model per call)."""
    from transformers import pipeline  # import here so this module loads on CPU too

    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    # Decoder-only models must pad on the LEFT for correct batched generation,
    # and need a pad token. Without this the pipeline falls back to running one
    # prompt at a time ("using the pipelines sequentially on GPU" warning) — the
    # single biggest extraction-speed killer.
    pipe.tokenizer.padding_side = "left"
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    return pipe


def extract_batch(
    chunks: list[dict],   # each has "chunk_id", "text", "subject", "grade", "chapter"
    model_id: str | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    pipe=None,
) -> list[dict]:
    """
    Run entity/relationship extraction on a list of chunk dicts.
    Returns a list of dicts: {chunk_id, entities, relationships}.
    Runs on GPU via HuggingFace pipeline with device_map="auto".

    Pass a preloaded `pipe` (from load_extractor) to reuse one model across
    shards; otherwise a pipeline is created from `model_id`.
    """
    if pipe is None:
        if model_id is None:
            raise ValueError("extract_batch requires either model_id or a preloaded pipe")
        pipe = load_extractor(model_id)

    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        prompts = [_build_prompt(c) for c in batch]

        try:
            outputs = pipe(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                batch_size=len(prompts),   # actually batch on the GPU, don't run serially
            )
        except Exception as exc:
            logger.warning("extraction_batch_failed batch=%d err=%s", i // batch_size, exc)
            for c in batch:
                results.append({"chunk_id": c["chunk_id"], "entities": [], "relationships": []})
            continue

        for chunk, output in zip(batch, outputs):
            try:
                generated = extract_generated_text(output)
                parsed = _parse(generated, chunk["chunk_id"])
            except Exception as exc:
                logger.warning("extraction_parse_failed chunk=%s err=%s", chunk["chunk_id"], exc)
                parsed = {"chunk_id": chunk["chunk_id"], "entities": [], "relationships": []}
            results.append(parsed)

        if i % (batch_size * 10) == 0:
            logger.info("extraction progress %d/%d", i, len(chunks))

    return results


def _build_prompt(chunk: dict) -> list[dict]:
    context = f"Subject: {chunk.get('subject', 'unknown')}, Grade: {chunk.get('grade', '?')}, Chapter: {chunk.get('chapter', '?')}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[{context}]\n\n{chunk['text']}"},
    ]


def _parse(generated_text: str, chunk_id: str) -> dict:
    """Extract JSON from the model output. Retry prompt is handled by the caller."""
    data = parse_json_block(generated_text)
    if data is None:
        logger.debug("json_parse_failed chunk_id=%s snippet=%s", chunk_id, generated_text[-200:])
        return {"chunk_id": chunk_id, "entities": [], "relationships": []}

    entities = [e for e in data.get("entities", []) if isinstance(e, dict) and "name" in e]
    rels = [r for r in data.get("relationships", []) if isinstance(r, dict) and "source" in r]
    return {"chunk_id": chunk_id, "entities": entities, "relationships": rels}
