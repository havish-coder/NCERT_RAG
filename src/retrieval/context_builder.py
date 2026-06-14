"""
ContextBuilder
==============
Assembles a ranked, ordered, token-budgeted context string from retrieved chunks.

Key responsibilities:
1. Deduplicate chunks that share the same parent (L1 section) — keep highest-scoring child
2. Sort by sequence_index (ascending) → natural reading order
3. Optionally fetch +1 neighboring chunk (prev/next) for additional context
4. Trim to context_token_budget tokens
5. Build citation metadata for the UI
"""
from __future__ import annotations

import tiktoken

from src.config import settings


class ContextBuilder:
    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("cl100k_base")

    def assemble(
        self,
        chunks: list[dict],    # dicts with payload fields: text, sequence_index, parent_chunk_id, score, etc.
        expand_neighbors: bool = False,
        qdrant_client=None,    # needed if expand_neighbors=True
    ) -> tuple[str, list[dict]]:
        """
        Returns (context_text, source_citations).
        context_text: assembled prompt context, chunks in reading order, separated by ---
        source_citations: list of dicts with provenance for UI display
        """
        if not chunks:
            return "", []

        # 1. Deduplicate: per parent_chunk_id, keep highest-scoring child
        deduped = _dedup_by_parent(chunks)

        # 2. Sort by document reading order
        deduped.sort(key=lambda c: c.get("sequence_index", 0))

        # 3. Trim to token budget
        selected = self._trim_to_budget(deduped)

        # 4. Build context string
        parts = []
        for c in selected:
            header = f"[{c.get('subject', '').upper()} | Grade {c.get('grade', '?')} | Ch.{c.get('chapter', '?')}: {c.get('chapter_title', '')}]"
            parts.append(f"{header}\n{c['text']}")

        context_text = "\n\n---\n\n".join(parts)

        # 5. Build citations for UI
        citations = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                "subject": c.get("subject", ""),
                "grade": c.get("grade", ""),
                "chapter": c.get("chapter", ""),
                "chapter_title": c.get("chapter_title", ""),
                "page_start": c.get("page_start", ""),
                "page_end": c.get("page_end", ""),
                "source_pdf": c.get("source_pdf", ""),
                "score": round(c.get("score", 0.0), 3),
            }
            for c in selected
        ]

        return context_text, citations

    def _trim_to_budget(self, chunks: list[dict]) -> list[dict]:
        selected, total = [], 0
        for c in chunks:
            tokens = len(self._enc.encode(c["text"]))
            if total + tokens > settings.context_token_budget:
                break
            selected.append(c)
            total += tokens
        return selected


def _dedup_by_parent(chunks: list[dict]) -> list[dict]:
    """
    For chunks sharing a parent_chunk_id, keep only the highest-scoring one.
    Chunks with no parent pass through unchanged.
    """
    parent_best: dict[str, dict] = {}
    no_parent: list[dict] = []

    for c in chunks:
        pid = c.get("parent_chunk_id")
        if not pid:
            no_parent.append(c)
            continue
        existing = parent_best.get(pid)
        if existing is None or c.get("score", 0) > existing.get("score", 0):
            parent_best[pid] = c

    return no_parent + list(parent_best.values())
