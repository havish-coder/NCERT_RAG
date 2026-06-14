"""
run_pipeline.py — NCERT GraphRAG offline ingestion (Lightning AI / any GPU box)
==============================================================================
One resumable script for the heavy offline work. Every stage writes a
checkpoint to the output dir; re-running SKIPS finished stages and RESUMES
entity extraction from where it stopped. Lightning AI keeps the Studio
filesystem between sessions, so a disconnect or GPU-time limit never loses
completed work — just run the script again and it continues.

Stages
  1. discover PDFs            (data/raw/<subject>/grade_<N>/<file>.pdf)
  2. parse + chunk + embed -> chunks.parquet
  3. extract entities      -> extraction_results.jsonl   (incremental, resumable)
  4. build graph (dedup)   -> entities.json, relationships.json
  5. community detection   -> community_memberships.json, community_members.json
  6. community summaries   -> community_summaries.json

Usage (from repo root, on the Lightning Studio terminal):
    pip install -r requirements-lightning.txt
    python run_pipeline.py                       # full corpus, default paths
    python run_pipeline.py --limit 5             # smoke-test on 5 PDFs first
    python run_pipeline.py --pdf-dir data/raw --out artifacts --model Qwen/Qwen2.5-7B-Instruct

When it prints "PIPELINE COMPLETE", download the whole `artifacts/` folder and
drop it into `data/artifacts/` on your local machine.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXTRACT_SHARD = 50          # results flushed to disk every SHARD chunks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ── 1. Discover PDFs ───────────────────────────────────────────────────────────

def discover_pdfs(pdf_dir: Path) -> list[dict]:
    """Infer subject + grade from the folder layout data/raw/<subject>/grade_<N>/."""
    metas = []
    for path in sorted(pdf_dir.rglob("*.pdf")):
        grade = 0
        subject = "unknown"
        # parent like "grade_8"
        m = re.search(r"grade[_\s]?(\d+)", path.parent.name, re.IGNORECASE)
        if m:
            grade = int(m.group(1))
            subject = path.parent.parent.name.lower()
        else:
            # fallback: try filename, subject from immediate parent
            m2 = re.search(r"grade[_\s]?(\d+)", path.stem, re.IGNORECASE)
            grade = int(m2.group(1)) if m2 else 0
            subject = path.parent.name.lower()
        metas.append({"subject": subject, "grade": grade, "path": path})
    return metas


# ── 2. Parse + chunk + embed -> chunks.parquet ─────────────────────────────────

def stage_chunks(metas: list[dict], out: Path) -> pd.DataFrame:
    pq = out / "chunks.parquet"
    if pq.exists():
        log(f"chunks.parquet exists ({pq.stat().st_size/1e6:.1f} MB) - skipping parse/embed")
        return pd.read_parquet(pq)

    from src.ingestion.pdf_parser import PDFParser
    from src.ingestion.chunker import HierarchicalSemanticChunker
    from src.ingestion.embedder import EmbeddingEngine
    from tqdm import tqdm

    chunker = HierarchicalSemanticChunker()
    all_chunks = []
    for meta in tqdm(metas, desc="parse+chunk"):
        try:
            pages = PDFParser(subject=meta["subject"], grade=meta["grade"]).parse(meta["path"])
            all_chunks.extend(chunker.chunk_document(
                pages, subject=meta["subject"], grade=meta["grade"], source_pdf=meta["path"].name))
        except Exception as exc:  # one bad PDF shouldn't kill the corpus
            log(f"  WARN failed {meta['path'].name}: {exc}")

    l1 = [c for c in all_chunks if c.metadata.chunk_level == 1]
    l2 = [c for c in all_chunks if c.metadata.chunk_level == 2]
    l3 = [c for c in all_chunks if c.metadata.chunk_level == 3]
    log(f"chunks: L1={len(l1)} L2={len(l2)} L3={len(l3)}")

    log("loading bge-m3 + embedding (late chunking)...")
    embedder = EmbeddingEngine()
    embedder.load()
    embedder.embed_chunks(l2 + l3)

    # Free bge-m3 from VRAM before the (much larger) extraction model loads, so a
    # single fresh run doesn't hold both at once on a 16 GB GPU.
    del embedder
    _free_gpu()

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
    df.to_parquet(pq, index=False)
    log(f"wrote {pq} - {len(df)} chunks, {pq.stat().st_size/1e6:.1f} MB")
    return df


# ── 3. Entity extraction -> extraction_results.jsonl (resumable) ───────────────

def stage_extract(df: pd.DataFrame, out: Path, model_id: str,
                  batch_size: int, max_new_tokens: int) -> list[dict]:
    jsonl = out / "extraction_results.jsonl"
    l2 = df[df["chunk_level"] == 2]
    todo = [
        {"chunk_id": r["chunk_id"], "text": r["text"], "subject": r["subject"],
         "grade": int(r["grade"]), "chapter": int(r["chapter"])}
        for _, r in l2.iterrows()
    ]

    done_ids: set[str] = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["chunk_id"])
        log(f"resuming extraction - {len(done_ids)}/{len(todo)} chunks already done")

    remaining = [c for c in todo if c["chunk_id"] not in done_ids]
    if remaining:
        from src.ingestion.extractor import load_extractor, extract_batch
        log(f"loading extraction model {model_id} ...")
        pipe = load_extractor(model_id)

        t0 = time.time()
        with jsonl.open("a", encoding="utf-8") as fh:
            for i in range(0, len(remaining), EXTRACT_SHARD):
                shard = remaining[i: i + EXTRACT_SHARD]
                res = extract_batch(shard, batch_size=batch_size,
                                    max_new_tokens=max_new_tokens, pipe=pipe)
                for r in res:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                n_done = len(done_ids) + i + len(shard)
                rate = (i + len(shard)) / max(time.time() - t0, 1e-6)
                eta = (len(remaining) - i - len(shard)) / max(rate, 1e-6)
                log(f"  extracted {n_done}/{len(todo)}  ({rate:.2f} chunk/s, ETA {eta/60:.0f} min)")
    else:
        log("extraction already complete")

    results = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_ent = sum(len(r["entities"]) for r in results)
    n_rel = sum(len(r["relationships"]) for r in results)
    log(f"extraction totals: {n_ent} entities, {n_rel} relationships across {len(results)} chunks")
    return results


# ── 4-6. Graph, communities, summaries ─────────────────────────────────────────

def stage_graph(results: list[dict], df: pd.DataFrame, out: Path):
    from src.ingestion.graph_builder import build_graph
    if (out / "entities.json").exists() and (out / "relationships.json").exists():
        log("entities/relationships exist - skipping graph build")
        return
    l2 = df[df["chunk_level"] == 2]
    chunk_meta = {r["chunk_id"]: {"subject": r["subject"], "grade": int(r["grade"]),
                                  "chapter": int(r["chapter"])} for _, r in l2.iterrows()}
    entities, rels = build_graph(results, chunk_meta, out)
    log(f"graph: {len(entities)} entities, {len(rels)} relationships")


def stage_communities(out: Path):
    from src.ingestion.community_detector import detect_communities
    if (out / "community_members.json").exists():
        log("communities exist - skipping detection")
        return
    memberships = detect_communities(
        out / "entities.json", out / "relationships.json", out / "community_memberships.json")
    log(f"communities: assigned {len(memberships)} entities")


def stage_summaries(out: Path, model_id: str):
    from src.ingestion.community_summarizer import summarize_communities
    if (out / "community_summaries.json").exists():
        log("summaries exist - skipping")
        return
    from src.ingestion.extractor import load_extractor
    log("loading model for community summaries ...")
    pipe = load_extractor(model_id)
    summaries = summarize_communities(
        out / "community_members.json", out / "entities.json",
        out / "community_summaries.json", pipe=pipe)
    log(f"summaries: {len(summaries)} communities")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="data/raw")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N PDFs (smoke test)")
    ap.add_argument("--batch", type=int, default=4, help="extraction prompts per forward pass (raise on big-VRAM GPUs)")
    ap.add_argument("--max-new-tokens", type=int, default=768, help="max tokens generated per extraction")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metas = discover_pdfs(pdf_dir)
    if args.limit:
        metas = metas[: args.limit]
    if not metas:
        log(f"ERROR: no PDFs found under {pdf_dir.resolve()}")
        sys.exit(1)
    subjects = sorted({m["subject"] for m in metas})
    log(f"found {len(metas)} PDFs across {len(subjects)} subjects: {', '.join(subjects)}")

    df = stage_chunks(metas, out)
    results = stage_extract(df, out, args.model, args.batch, args.max_new_tokens)
    stage_graph(results, df, out)
    stage_communities(out)
    stage_summaries(out, args.model)

    log("=" * 60)
    log("PIPELINE COMPLETE - artifacts:")
    for f in sorted(out.iterdir()):
        log(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
    log("Download the whole artifacts/ folder -> data/artifacts/ locally.")


if __name__ == "__main__":
    main()
