"""
CommunitySummarizer
===================
Generates one LLM summary per community.
Runs on Kaggle/Lightning AI with the same model already loaded for extraction.

Input: community_members.json + entities.json
Output: community_summaries.json

Flat (not hierarchical) — one summary per community at each level.
Simple: build entity description text → one LLM call → save title + summary.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.ingestion.llm_output import extract_generated_text, parse_json_block

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are summarizing a cluster of related concepts from NCERT educational content.
Given a list of entities and their relationships, write a concise community summary.

Return ONLY valid JSON:
{
  "title": "5-8 word descriptive title",
  "summary": "2-4 sentences covering the main theme, key concepts, and how they relate"
}"""


def summarize_communities(
    community_members_path: Path,
    entities_path: Path,
    output_path: Path,
    model_id: str | None = None,
    max_entities_per_community: int = 20,
    pipe=None,
) -> list[dict]:
    """
    For each community: build entity context → call LLM → parse JSON → save.
    Returns list of community dicts with title and summary.

    Pass a preloaded `pipe` to reuse the same model already loaded for
    extraction; otherwise a pipeline is created from `model_id`.
    """
    community_members: dict[str, list[str]] = json.loads(community_members_path.read_text())
    entities_raw: list[dict] = json.loads(entities_path.read_text())
    entity_map = {e["entity_id"]: e for e in entities_raw}

    if pipe is None:
        if model_id is None:
            raise ValueError("summarize_communities requires either model_id or a preloaded pipe")
        from transformers import pipeline
        pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype="auto")

    summaries: list[dict] = []

    for comm_id, member_ids in community_members.items():
        members = [entity_map[eid] for eid in member_ids if eid in entity_map]
        if not members:
            continue

        # Sample top N by description length (proxy for informativeness)
        members_sorted = sorted(members, key=lambda e: -len(e.get("description", "")))
        selected = members_sorted[:max_entities_per_community]

        context = _build_context(selected)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        try:
            output = pipe(prompt, max_new_tokens=256, do_sample=False)
            generated = extract_generated_text(output)
            result = parse_json_block(generated) or {}
        except Exception as exc:
            logger.warning("community_summary_failed comm=%s err=%s", comm_id, exc)
            result = {"title": comm_id, "summary": ""}

        summaries.append({
            "community_id": comm_id,
            "level": 0 if comm_id.startswith("c0-") else 1,
            "title": result.get("title", comm_id),
            "summary": result.get("summary", ""),
            "member_entity_ids": member_ids,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2))
    return summaries


def _build_context(members: list[dict]) -> str:
    lines = ["Entities in this community:"]
    for e in members:
        lines.append(f"- {e['name']} ({e.get('entity_type', 'CONCEPT')}): {e.get('description', '')}")
    return "\n".join(lines)


