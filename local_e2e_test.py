"""
Local end-to-end test of the NCERT GraphRAG offline ingestion pipeline.

Runs the EXACT stages the Lightning AI job will run, but on 3 real PDFs and
with a tiny LLM (Qwen2.5-0.5B-Instruct) so it finishes in minutes on a laptop
GPU. The goal is to prove the plumbing is correct before paying for Lightning.

Run from the repo root:
    python local_e2e_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "artifacts_test"
OUT.mkdir(parents=True, exist_ok=True)

TEST_PDFS = [
    ("science", 8, ROOT / "data/raw/science/grade_8/hesc1_ch03.pdf"),
    ("mathematics", 8, ROOT / "data/raw/mathematics/grade_8/hemh1_ch02.pdf"),
    ("history", 11, ROOT / "data/raw/history/grade_11/kehs1_ch03.pdf"),
]
TEST_LLM = "Qwen/Qwen2.5-0.5B-Instruct"   # tiny stand-in for Qwen2.5-7B on Lightning
EXTRACT_N = 12                            # only extract on a subset to keep test fast


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.chunker import HierarchicalSemanticChunker
    from src.ingestion.embedder import EmbeddingEngine

    # ── 1. Parse + chunk ──────────────────────────────────────────────────────
    section("1. PARSE + CHUNK")
    chunker = HierarchicalSemanticChunker()
    all_chunks = []
    for subject, grade, path in TEST_PDFS:
        check(path.exists(), f"PDF exists: {path.name}")
        parser = PDFParser(subject=subject, grade=grade)
        pages = parser.parse(path)
        chunks = chunker.chunk_document(pages, subject=subject, grade=grade, source_pdf=path.name)
        all_chunks.extend(chunks)
        sections_found = sorted({tuple(c.metadata.section_path) for c in chunks if len(c.metadata.section_path) > 1})
        print(f"  {path.name}: {len(pages)} pages -> {len(chunks)} chunks, "
              f"{len(sections_found)} distinct sections detected")

    l1 = [c for c in all_chunks if c.metadata.chunk_level == 1]
    l2 = [c for c in all_chunks if c.metadata.chunk_level == 2]
    l3 = [c for c in all_chunks if c.metadata.chunk_level == 3]
    print(f"  TOTAL: L1={len(l1)}  L2={len(l2)}  L3={len(l3)}")
    check(len(l2) > 0, "produced L2 retrieval chunks")
    check(all(c.metadata.parent_chunk_id for c in l2), "every L2 has a parent L1")
    check(all(c.token_count <= 512 + 80 for c in l2), "L2 chunks respect ~512 token cap")
    # section detection should now fire on real PDFs (the \n parser fix)
    any_section = any(len(c.metadata.section_path) > 1 and c.metadata.section_path[1][0].isdigit() for c in l2)
    print(f"  numbered-section detection fired: {any_section}")

    # ── 2. Embed (per-chunk dense + sparse) ───────────────────────────────────
    section("2. EMBED (per-chunk dense + sparse)")
    t0 = time.time()
    embedder = EmbeddingEngine()
    embedder.load()
    print(f"  bge-m3 loaded in {time.time() - t0:.1f}s")

    embedder.embed_chunks(l2 + l3)

    dims = {len(c.dense_embedding) for c in l2 if c.dense_embedding}
    check(dims == {1024}, f"all L2 dense vectors are 1024-dim (got {dims})")
    check(all(c.dense_embedding is not None for c in l2), "every L2 got a dense embedding")
    check(all(c.sparse_indices for c in l2), "every L2 got non-empty sparse weights")

    # Non-degenerate check: distinct chunks must not collapse to one vector.
    sample = [np.array(c.dense_embedding) for c in l2[:8]]
    if len(sample) >= 2:
        sims = []
        for i in range(len(sample) - 1):
            a, b = sample[i], sample[i + 1]
            sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        print(f"  adjacent-chunk cosine sims: min={min(sims):.3f} max={max(sims):.3f}")
        check(max(sims) < 0.999, "distinct chunks get distinct embeddings")

    # Sanity: a query about the science chapter should rank a science chunk top.
    q_dense, q_idx, q_val = embedder.encode_query("What is friction?")
    qv = np.array(q_dense)
    scored = sorted(
        l2, key=lambda c: float(qv @ np.array(c.dense_embedding) /
                                (np.linalg.norm(qv) * np.linalg.norm(c.dense_embedding))),
        reverse=True,
    )
    top = scored[0]
    print(f"  query 'What is friction?' -> top chunk subject={top.metadata.subject}, "
          f"pdf={top.metadata.source_pdf}")
    print(f"    \"{top.text[:120]}...\"")

    # ── 3. Save + reload Parquet ──────────────────────────────────────────────
    section("3. SAVE + RELOAD PARQUET")
    retrieval = l2 + l3
    records = []
    for c in retrieval:
        records.append({
            "chunk_id": c.chunk_id, "text": c.text, "token_count": c.token_count,
            "dense_embedding": c.dense_embedding,
            "sparse_indices": c.sparse_indices, "sparse_values": c.sparse_values,
            **c.metadata.model_dump(),
        })
    df = pd.DataFrame(records)
    pq = OUT / "chunks.parquet"
    df.to_parquet(pq, index=False)
    df2 = pd.read_parquet(pq)
    check(len(df2) == len(records), f"parquet round-trips {len(records)} rows")
    check(len(df2.iloc[0]["dense_embedding"]) == 1024, "dense embedding survives parquet round-trip")
    print(f"  saved {pq} ({pq.stat().st_size/1024/1024:.1f} MB)")

    # ── 4. Entity extraction (tiny LLM) ───────────────────────────────────────
    section(f"4. ENTITY EXTRACTION  (model={TEST_LLM}, n={EXTRACT_N})")
    from src.ingestion.extractor import extract_batch
    chunk_dicts = [
        {"chunk_id": c.chunk_id, "text": c.text, "subject": c.metadata.subject,
         "grade": c.metadata.grade, "chapter": c.metadata.chapter}
        for c in l2[:EXTRACT_N]
    ]
    t0 = time.time()
    results = extract_batch(chunk_dicts, model_id=TEST_LLM, batch_size=4, max_new_tokens=512)
    n_ent = sum(len(r["entities"]) for r in results)
    n_rel = sum(len(r["relationships"]) for r in results)
    print(f"  extracted {n_ent} entities, {n_rel} relationships in {time.time()-t0:.1f}s")
    check(len(results) == len(chunk_dicts), "one result per chunk")
    check(n_ent > 0, "extractor parsed at least some entities (JSON parsing works)")
    ex = next((e for r in results for e in r["entities"]), None)
    if ex:
        print(f"  sample entity: {ex.get('name')} ({ex.get('entity_type')})")

    # ── 5. Build graph ────────────────────────────────────────────────────────
    section("5. BUILD GRAPH (dedup)")
    from src.ingestion.graph_builder import build_graph
    chunk_meta = {c.chunk_id: {"subject": c.metadata.subject, "grade": c.metadata.grade,
                               "chapter": c.metadata.chapter} for c in l2}
    entities, relationships = build_graph(results, chunk_meta, OUT)
    print(f"  entities={len(entities)}  relationships={len(relationships)}")
    check((OUT / "entities.json").exists(), "entities.json written")
    check((OUT / "relationships.json").exists(), "relationships.json written")
    if entities:
        check(all("entity_id" in e and "name" in e for e in entities), "entities well-formed")

    # ── 6. Community detection ────────────────────────────────────────────────
    section("6. COMMUNITY DETECTION (Leiden)")
    from src.ingestion.community_detector import detect_communities
    memberships = detect_communities(
        OUT / "entities.json", OUT / "relationships.json", OUT / "community_memberships.json")
    print(f"  assigned communities to {len(memberships)} entities")
    check((OUT / "community_members.json").exists(), "community_members.json written")

    # ── 7. Community summaries (tiny LLM) ─────────────────────────────────────
    section("7. COMMUNITY SUMMARIES")
    from src.ingestion.community_summarizer import summarize_communities
    summaries = summarize_communities(
        OUT / "community_members.json", OUT / "entities.json",
        OUT / "community_summaries.json", model_id=TEST_LLM)
    print(f"  generated {len(summaries)} community summaries")
    if summaries:
        s = summaries[0]
        print(f"  sample: [{s['title']}] {s['summary'][:100]}")
        check(any(x.get("summary") for x in summaries), "at least one non-empty summary (chat parsing works)")

    # ── Done ──────────────────────────────────────────────────────────────────
    section("ARTIFACTS")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
    print("\nALL STAGES PASSED")


if __name__ == "__main__":
    main()
