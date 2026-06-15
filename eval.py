"""
eval.py — offline evaluation harness for the NCERT GraphRAG.

A RAG system has two independent failure points, so we measure them separately —
otherwise you can't tell *why* an answer was bad.

  1. RETRIEVAL  — did we fetch the right passages?           (free, no LLM calls)
       • Recall@k   — was a correct source in the top-k?
       • MRR        — how highly was the first correct source ranked?
       • Ablation   — the SAME metrics WITH vs WITHOUT the Neo4j graph-expansion
                      step, which isolates how much the graph layer actually adds.

  2. GENERATION — given the context, is the answer good?     (--judge, costs API calls)
       • Faithfulness — is every claim supported by the retrieved context? (hallucination)
       • Relevance    — does the answer address the question?
       • Correctness  — does it agree with the hand-written gold answer?
     All scored 0-1 by an LLM-as-judge.

Gold set: a hand-written JSON list (see eval/gold_set.json). Each item labels a
question with what the correct passage should contain:

    {"question": "...",
     "keywords": ["force", "acceleration"],  # all must appear in the chunk text
     "subject": "science",                    # optional scope (reliable)
     "grade": 9,                              # optional scope (reliable)
     "source_pdf": "iesc1_ch09",              # optional, precise chapter id
     "gold_answer": "..."}                    # optional; only used by --judge correctness

A retrieved source counts as a HIT when every criterion the gold row provides is
satisfied (see _matches). We match on TEXT CONTENT rather than chapter metadata
because this corpus's chapter / chapter_title fields are unreliable (the parser
labels ~89% of chunks "Introduction"); subject, grade and source_pdf are reliable.

Usage:
    python eval.py                      # retrieval metrics + graph ablation only (free)
    python eval.py --judge              # also LLM-judge the generated answers (API calls)
    python eval.py --limit 10           # only the first 10 questions (save quota)
    python eval.py --gold eval/mine.json --k 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

for _noisy in ("neo4j", "neo4j.notifications", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from rich.console import Console
from rich.table import Table

from src.config import settings
from src.ingestion.embedder import EmbeddingEngine
from src.ingestion.llm_output import parse_json_block
from src.llm.client import LLMClient
from src.retrieval.local_search import LocalSearch
from src.storage.neo4j_client import Neo4jClient
from src.storage.qdrant_client import QdrantClientWrapper

console = Console()


# ── Matching: does a retrieved source satisfy the gold label? ──────────────────

# chunk_id -> full chunk text, loaded from the artifact parquet. The with-graph
# path returns citations whose text is truncated to ~300 chars, while the no-graph
# path keeps full text — matching keywords against the full text by chunk_id keeps
# the two pipelines on identical footing and avoids false misses from truncation.
_CHUNK_TEXT: dict[str, str] = {}


def _norm(v) -> str:
    return str(v).strip().lower()


def _source_text(source: dict) -> str:
    """Full chunk text by id, falling back to whatever text the source carries."""
    return _CHUNK_TEXT.get(source.get("chunk_id", ""), source.get("text", "") or "")


def _matches(source: dict, gold: dict) -> bool:
    """A source is a HIT when every criterion the gold row provides is satisfied.
    All criteria are optional, but a gold row must provide at least one.
      • keywords   — ALL listed terms must appear in the chunk text (the main signal).
      • subject    — substring match either way (tolerates minor naming drift).
      • grade      — exact match.
      • source_pdf — substring of the source filename (precise chapter identifier).
    Chapter / chapter_title are deliberately NOT used — they are unreliable here.
    """
    gs = _norm(gold.get("subject", ""))
    if gs:
        ss = _norm(source.get("subject", ""))
        if gs not in ss and ss not in gs:
            return False

    gg = gold.get("grade")
    if gg not in (None, ""):
        if _norm(gg) != _norm(source.get("grade", "")):
            return False

    gp = _norm(gold.get("source_pdf", ""))
    if gp and gp not in _norm(source.get("source_pdf", "")):
        return False

    kws = [_norm(k) for k in gold.get("keywords", []) if str(k).strip()]
    if kws:
        text = _norm(_source_text(source))
        if not all(k in text for k in kws):
            return False

    return bool(kws or gp or gs or gg not in (None, ""))


def _ranked(sources: list[dict]) -> list[dict]:
    """Sort by relevance score, descending — citations are stored in reading
    order, not rank order, so we re-sort before computing rank-aware metrics."""
    return sorted(sources, key=lambda s: s.get("score", 0.0), reverse=True)


def _first_hit_rank(sources: list[dict], gold: dict) -> int:
    """1-based rank of the first matching source, or 0 if none match."""
    for i, s in enumerate(_ranked(sources), 1):
        if _matches(s, gold):
            return i
    return 0


def _covered_hops(sources: list[dict], hops: list[list[str]]) -> list[bool]:
    """For a multi-hop question, whether each hop (a keyword group) is covered by
    SOME retrieved chunk. A hop is covered when one chunk's full text contains all
    of the hop's keywords. This measures whether retrieval assembled the complete
    multi-concept context — the thing graph expansion is supposed to help with."""
    texts = [_norm(_source_text(s)) for s in sources]
    return [any(all(k in t for k in (_norm(w) for w in hop)) for t in texts) for hop in hops]


def _top_label(sources: list[dict]) -> str:
    """Human-readable summary of the top retrieved source — printed for misses so
    you can pick better keywords or see whether retrieval genuinely failed."""
    if not sources:
        return "(nothing retrieved)"
    s = _ranked(sources)[0]
    snippet = _source_text(s).replace("\n", " ")[:70]
    return f"{s.get('subject', '?')}|g{s.get('grade', '?')}|{s.get('source_pdf', '?')}  «{snippet}…»"


# ── Retrieval pipelines ────────────────────────────────────────────────────────

async def retrieve_with_graph(local: LocalSearch, question: str) -> list[dict]:
    """Full GraphRAG: query → seed entities → Neo4j neighborhood → chunks."""
    prepared = await local.prepare(question)
    return prepared.get("sources", [])  # empty result returns no sources


async def retrieve_vanilla(embedder, qdrant, question: str, k: int) -> list[dict]:
    """Ablation baseline: plain hybrid (dense+sparse) chunk search, NO graph."""
    dense, sparse_idx, sparse_val = embedder.encode_query(question)
    return await qdrant.search_chunks(dense, sparse_idx, sparse_val, top_k=k)


# ── LLM-as-judge ───────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of a tutor's answer. "
    "Score each metric from 0.0 to 1.0 (one decimal is fine). "
    "Return ONLY a JSON object, no prose."
)


def _judge_user(question: str, context: str, answer: str, gold: str | None) -> str:
    parts = [
        "Evaluate the ANSWER using the CONTEXT and QUESTION below.",
        f"\nQUESTION:\n{question}",
        f"\nRETRIEVED CONTEXT:\n{context}",
        f"\nANSWER:\n{answer}",
    ]
    if gold:
        parts.append(f"\nREFERENCE (gold) ANSWER:\n{gold}")
    keys = [
        '  "faithfulness": 0-1  — are ALL claims in the answer supported by the context? '
        "(1 = fully grounded; 0 = hallucinated)",
        '  "relevance": 0-1     — does the answer actually address the question?',
    ]
    if gold:
        keys.append('  "correctness": 0-1   — does the answer agree with the reference answer?')
    keys.append('  "reason": short string')
    parts.append("\nReturn JSON with keys:\n" + "\n".join(keys))
    return "\n".join(parts)


async def judge_answer(llm: LLMClient, question: str, user_prompt: str,
                       answer: str, gold: str | None) -> dict | None:
    """One judge round-trip. `user_prompt` is the prepared 'Question:..\\nContext:..'
    string, from which we lift just the context block."""
    context = user_prompt.split("Context:\n", 1)[-1]
    reply = await llm.complete(JUDGE_SYSTEM, _judge_user(question, context, answer, gold))
    scores = parse_json_block(reply)
    if not scores or "faithfulness" not in scores:
        return None
    return scores


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

async def run(args) -> None:
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.limit:
        gold = gold[: args.limit]
    if not gold:
        console.print("[red]Gold set is empty.[/] Add questions to", args.gold)
        return

    console.print(f"[bold]Evaluating {len(gold)} questions[/] (k={args.k}, judge={'on' if args.judge else 'off'})\n")

    # Full chunk text by id — used for accurate, truncation-free keyword matching.
    if Path(args.chunks).exists():
        import pandas as pd
        df = pd.read_parquet(args.chunks, columns=["chunk_id", "text"])
        _CHUNK_TEXT.update(dict(zip(df["chunk_id"], df["text"])))
    else:
        console.print(f"[yellow]note:[/] {args.chunks} not found — matching on truncated "
                      "citation text, which can cause false misses.")

    neo4j, qdrant = Neo4jClient(), QdrantClientWrapper()
    embedder, llm = EmbeddingEngine(), LLMClient()
    with console.status("connecting to Neo4j + Qdrant and loading bge-m3…"):
        await neo4j.connect()
        await qdrant.connect()
        await asyncio.to_thread(embedder.load)
    local = LocalSearch(neo4j, qdrant, embedder, llm)

    # Per-question accumulators. Keyword items use ranks (1-based rank of the first
    # correct source, 0 = miss); multi-hop items use hop-coverage fractions.
    g_ranks, v_ranks = [], []
    g_cov, v_cov, g_full, v_full = [], [], [], []
    g_lat, v_lat, g_nsrc = [], [], []
    faith, relv, corr = [], [], []
    misses: list[str] = []
    judge_failed = False

    try:
        for i, item in enumerate(gold, 1):
            q = item["question"]

            # --- Retrieval: with graph ---
            t0 = time.perf_counter()
            prepared = await local.prepare(q)
            g_lat.append((time.perf_counter() - t0) * 1000)
            g_src = prepared.get("sources", [])
            g_nsrc.append(len(g_src))

            # --- Retrieval: vanilla (no graph) ablation ---
            t0 = time.perf_counter()
            v_src = await retrieve_vanilla(embedder, qdrant, q, args.k)
            v_lat.append((time.perf_counter() - t0) * 1000)

            if "hops" in item:  # multi-hop coverage metric
                gc, vc = _covered_hops(g_src, item["hops"]), _covered_hops(v_src, item["hops"])
                g_cov.append(_mean([float(x) for x in gc]))
                v_cov.append(_mean([float(x) for x in vc]))
                g_full.append(all(gc))
                v_full.append(all(vc))
                mark = "[green]✓[/]" if all(gc) else ("[yellow]~[/]" if any(gc) else "[red]✗[/]")
                console.print(f"  {mark} {i:>3}. {q[:60]}")
                if not all(gc):
                    missed = ["+".join(h) for h, c in zip(item["hops"], gc) if not c]
                    misses.append(f"{q[:48]}  →  graph missed: {', '.join(missed)}")
            else:  # keyword rank metric
                gr, vr = _first_hit_rank(g_src, item), _first_hit_rank(v_src, item)
                g_ranks.append(gr)
                v_ranks.append(vr)
                mark = "[green]✓[/]" if gr else "[red]✗[/]"
                console.print(f"  {mark} {i:>3}. {q[:60]}")
                if not gr:
                    misses.append(f"{q[:55]}  →  top: {_top_label(g_src)}")

            # --- Generation judging (optional, costs API calls) ---
            if args.judge and "prompt" in prepared and not judge_failed:
                try:
                    system, user = prepared["prompt"]
                    answer = await llm.complete(system, user)
                    scores = await judge_answer(llm, q, user, answer, item.get("gold_answer"))
                    if scores:
                        faith.append(float(scores.get("faithfulness", 0)))
                        relv.append(float(scores.get("relevance", 0)))
                        if "correctness" in scores:
                            corr.append(float(scores["correctness"]))
                except Exception as exc:  # rate limit etc. — stop judging, keep retrieval results
                    judge_failed = True
                    console.print(f"     [yellow]judge stopped:[/] {str(exc)[:90]}")
    finally:
        await neo4j.close()
        await qdrant.close()

    _report(args, g_ranks, v_ranks, g_cov, v_cov, g_full, v_full,
            g_lat, v_lat, g_nsrc, faith, relv, corr, misses, judge_failed)


def _recall_at(ranks: list[int], k: int) -> float:
    return _mean([1.0 if 0 < r <= k else 0.0 for r in ranks])


def _mrr(ranks: list[int]) -> float:
    return _mean([1.0 / r if r else 0.0 for r in ranks])


def _pp(v: float, g: float, scale: float = 100.0, suffix: str = " pp", dec: int = 1):
    """Format a vector/graph pair plus a colored Δ (graph − vector)."""
    d = scale * (g - v)
    color = "green" if d > 0 else ("red" if d < 0 else "dim")
    fmt = f"{{:.{dec}f}}"
    unit = "%" if scale == 100.0 and suffix == " pp" else ""
    return (fmt.format(scale * v) + unit, fmt.format(scale * g) + unit,
            f"[{color}]{d:+.{dec}f}{suffix}[/]")


def _report(args, g_ranks, v_ranks, g_cov, v_cov, g_full, v_full,
            g_lat, v_lat, g_nsrc, faith, relv, corr, misses, judge_failed) -> None:
    def base_table(title: str) -> Table:
        t = Table(title=title, title_justify="left")
        t.add_column("Metric")
        t.add_column("No graph\n(vector only)", justify="right")
        t.add_column("With graph\n(GraphRAG)", justify="right")
        t.add_column("Δ", justify="right")
        return t

    def cost_rows(t: Table) -> None:
        t.add_section()
        t.add_row("Avg latency (ms)", f"{_mean(v_lat):.0f}", f"{_mean(g_lat):.0f}", "")
        t.add_row("Avg sources / q", f"{args.k}", f"{_mean(g_nsrc):.1f}", "")

    if g_ranks:  # keyword (concept-lookup) questions
        console.print()
        t = base_table(f"Retrieval — concept lookups, graph ablation (n={len(g_ranks)})")
        for k in (1, 3, 5, 10):
            t.add_row(f"Recall@{k}", *_pp(_recall_at(v_ranks, k), _recall_at(g_ranks, k)))
        t.add_row("MRR", *_pp(_mrr(v_ranks), _mrr(g_ranks), scale=1.0, suffix="", dec=3))
        cost_rows(t)
        console.print(t)

    if g_cov:  # multi-hop questions
        console.print()
        t = base_table(f"Retrieval — multi-hop coverage, graph ablation (n={len(g_cov)})")
        t.add_row("Hop coverage", *_pp(_mean(v_cov), _mean(g_cov)))
        t.add_row("Full coverage", *_pp(_mean([float(x) for x in v_full]),
                                        _mean([float(x) for x in g_full])))
        cost_rows(t)
        console.print(t)

    if args.judge:
        console.print()
        if not faith:
            console.print("[yellow]No judge scores collected[/] "
                          "(rate-limited, or every retrieval was empty).")
        else:
            jt = Table(title=f"Generation — LLM-as-judge ({settings.online_model}, n={len(faith)})",
                       title_justify="left")
            jt.add_column("Metric")
            jt.add_column("Score (0-1)", justify="right")
            jt.add_row("Faithfulness (grounded)", f"{_mean(faith):.2f}")
            jt.add_row("Answer relevance", f"{_mean(relv):.2f}")
            if corr:
                jt.add_row("Correctness vs gold", f"{_mean(corr):.2f}")
            console.print(jt)
            if judge_failed:
                console.print("[yellow]note:[/] judging stopped early (rate limit); "
                              "scores above are over the questions that completed.")

    if misses:
        console.print(f"\n[bold]Misses ({len(misses)})[/] — gold not found in retrieved sources:")
        for m in misses:
            console.print(f"  [red]·[/] {m}")
        console.print("[dim]If a miss's top source looks correct, the gold label "
                      "(subject/grade/chapter) probably needs adjusting.[/]")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the NCERT GraphRAG.")
    p.add_argument("--gold", default="eval/gold_set.json", help="path to the gold-set JSON")
    p.add_argument("--chunks", default="data/artifacts/chunks.parquet",
                   help="chunks parquet, for full-text keyword matching")
    p.add_argument("--k", type=int, default=settings.top_k_chunks, help="cutoff for Recall@k")
    p.add_argument("--judge", action="store_true", help="also LLM-judge generation (costs API calls)")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N questions")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
