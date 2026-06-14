"""
HierarchicalSemanticChunker
===========================
Three-tier chunking for NCERT textbooks:

  L1 (Section)   <= 2000 tokens  — late-chunking embedding window, NOT stored in Qdrant
  L2 (Paragraph) <=  512 tokens  — primary retrieval unit, stored in Qdrant
  L3 (Atomic)    <=  128 tokens  — definitions/formulas/laws, kept whole

Semantic boundary detection uses cosine similarity between consecutive sentence
embeddings (rolling window=3) to find natural topic transitions.

Late chunking: each L2 chunk's dense embedding is mean-pooled from the parent L1
section's ColBERT token vectors, giving it full-section context.
"""
from __future__ import annotations

import re
from typing import Optional
from uuid import uuid4

import numpy as np
import tiktoken

from src.config import settings
from src.models.document import ChunkMetadata, ContentType, TextChunk

# ── NCERT special content detectors ───────────────────────────────────────────
_SECTION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?\s+(.{3,80})", re.MULTILINE)

_ATOMIC_PATTERNS: list[tuple[re.Pattern, ContentType]] = [
    (re.compile(r"^(definition|theorem|law|lemma|corollary|axiom)\s*[:—]", re.I), ContentType.DEFINITION),
    (re.compile(r"^(note|remember|recall)\s*[:—]", re.I), ContentType.KEY_POINT),
    (re.compile(r"\$\$.+\$\$|\\begin\{equation\}", re.S), ContentType.FORMULA),
]

_SPECIAL_PATTERNS: list[tuple[re.Pattern, ContentType]] = [
    (re.compile(r"^example\s+\d+", re.I), ContentType.EXAMPLE),
    (re.compile(r"^activity\s+\d+", re.I), ContentType.ACTIVITY),
    (re.compile(r"^(exercises?|problems?)\s*$", re.I | re.M), ContentType.EXERCISE),
    (re.compile(r"^(summary|let us recall|key points|points to remember)", re.I), ContentType.SUMMARY),
    (re.compile(r"^(table|fig\.?\s+)\s*\d+", re.I), ContentType.TABLE),
]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")


class NCERTStructureDetector:
    """
    Converts a flat list of pages (from PDFParser) into structured blocks.
    Each block carries section hierarchy info and a ContentType tag.
    Atomic blocks (definitions, formulas) are flagged is_atomic=True.
    """

    def detect(self, pages: list[dict]) -> list[dict]:
        blocks: list[dict] = []
        state = _fresh_state()
        buffer: list[str] = []
        buf_page_start = 1

        for page in pages:
            text = page["text"].strip()
            if not text:
                continue

            pnum = page["page_num"]

            # Chapter change detected by pdf_parser
            if page["chapter"] != state["chapter"]:
                if buffer:
                    blocks.extend(self._flush(buffer, buf_page_start, pnum - 1, state))
                    buffer, buf_page_start = [], pnum
                state["chapter"] = page["chapter"]
                state["chapter_title"] = page["chapter_title"]
                state["section_num"] = 0
                state["section_path"] = [f"Chapter {page['chapter']}: {page['chapter_title']}"]

            # Detect section headings inline (e.g. "9.2 Newton's First Law")
            for m in _SECTION_RE.finditer(text[:600]):
                if buffer:
                    blocks.extend(self._flush(buffer, buf_page_start, pnum, state))
                    buffer, buf_page_start = [], pnum
                sec = int(m.group(2))
                title = m.group(4).strip()
                state["section_num"] = sec
                state["section_path"] = [
                    f"Chapter {state['chapter']}: {state['chapter_title']}",
                    f"{m.group(1)}.{m.group(2)} {title}",
                ]
                break  # one section heading per page check is enough

            buffer.append(text)

        if buffer:
            last = pages[-1]["page_num"]
            blocks.extend(self._flush(buffer, buf_page_start, last, state))

        return blocks

    def _flush(self, buffer: list[str], p_start: int, p_end: int, state: dict) -> list[dict]:
        text = " ".join(buffer).strip()
        if not text:
            return []
        ctype, is_atomic = _classify(text)
        return [{
            "text": text,
            "page_start": p_start,
            "page_end": p_end,
            "chapter": state["chapter"],
            "chapter_title": state["chapter_title"],
            "section_num": state["section_num"],
            "section_path": list(state["section_path"]),
            "content_type": ctype,
            "is_atomic": is_atomic,
        }]


class SemanticBoundaryDetector:
    """
    Given sentences + their dense embeddings, returns indices where L2 chunk
    boundaries should fall. Uses rolling-window cosine similarity with a
    hard token-count fallback.
    """

    def __init__(
        self,
        threshold: float = settings.semantic_threshold,
        window: int = 3,
    ) -> None:
        self.threshold = threshold
        self.window = window

    def find_boundaries(
        self,
        sentences: list[str],
        embeddings: np.ndarray,   # shape [N, 1024]
        enc: tiktoken.Encoding,
        l2_min: int = settings.chunk_l2_min_tokens,
        l2_max: int = settings.chunk_l2_max_tokens,
    ) -> list[int]:
        n = len(sentences)
        if n <= 1:
            return []

        # Rolling window cosine similarities
        sims = np.ones(n - 1)
        for i in range(n - 1):
            a = embeddings[max(0, i - self.window + 1): i + 1].mean(axis=0)
            b = embeddings[i + 1: min(n, i + 1 + self.window)].mean(axis=0)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                sims[i] = float(np.dot(a, b) / (na * nb))

        boundaries: list[int] = []
        acc = len(enc.encode(sentences[0]))

        for i in range(1, n):
            tok = len(enc.encode(sentences[i]))
            force_split = (acc + tok) > l2_max
            semantic_gap = sims[i - 1] < self.threshold

            if (semantic_gap or force_split) and acc >= l2_min:
                boundaries.append(i - 1)
                acc = tok
            else:
                acc += tok

        return boundaries


class HierarchicalSemanticChunker:
    """
    Main chunking class. Call chunk_document() to get all TextChunks for one PDF.

    Returns chunks at three levels:
      L1 (chunk_level=1): section text — used for late-chunking context, not retrieved directly
      L2 (chunk_level=2): paragraph — primary retrieval unit stored in Qdrant
      L3 (chunk_level=3): atomic definitions/formulas — stored as indivisible units
    """

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("cl100k_base")
        self._structure = NCERTStructureDetector()
        self._boundary = SemanticBoundaryDetector()

    def chunk_document(
        self,
        pages: list[dict],
        subject: str,
        grade: int,
        source_pdf: str,
        sentence_embed_fn=None,   # callable(list[str]) -> np.ndarray, injected at embedding time
    ) -> list[TextChunk]:
        blocks = self._structure.detect(pages)
        all_chunks: list[TextChunk] = []
        counters: dict[tuple, int] = {}   # (chapter, section_num) -> para count

        for block in blocks:
            key = (block["chapter"], block["section_num"])
            para_idx = counters.get(key, 0)
            base_args = dict(
                subject=subject, grade=grade, source_pdf=source_pdf,
                chapter=block["chapter"], chapter_title=block["chapter_title"],
                section_path=block["section_path"],
                page_start=block["page_start"], page_end=block["page_end"],
                content_type=block["content_type"],
            )

            if block["is_atomic"]:
                seq = _seq(block["chapter"], block["section_num"], para_idx)
                all_chunks.append(self._chunk(block["text"], 3, True, seq, **base_args))
                counters[key] = para_idx + 1
                continue

            # L1 section chunk
            l1_seq = _seq(block["chapter"], block["section_num"], para_idx)
            l1 = self._chunk(block["text"], 1, False, l1_seq, **base_args)
            all_chunks.append(l1)

            # Split into L2 chunks
            sentences = [s.strip() for s in _SENTENCE_RE.split(block["text"]) if s.strip()]
            if not sentences:
                counters[key] = para_idx + 1
                continue

            if sentence_embed_fn is not None and len(sentences) > 1:
                emb = sentence_embed_fn(sentences)      # np.ndarray [N, 1024]
                cuts = self._boundary.find_boundaries(sentences, emb, self._enc)
            else:
                cuts = self._token_cuts(sentences)

            groups = _split_at(sentences, cuts)

            # Guarantee the token cap: sentence-boundary cutting cannot split a
            # single oversized segment (long run-on text, tables, heading-only
            # lines with no '.!?'), so hard-split any group that still exceeds it.
            group_texts: list[str] = []
            for grp in groups:
                group_texts.extend(self._enforce_token_cap(" ".join(grp)))

            prev_l2: Optional[TextChunk] = None
            for gi, gtext in enumerate(group_texts):
                seq = _seq(block["chapter"], block["section_num"], para_idx + gi)
                l2 = self._chunk(gtext, 2, False, seq, parent_chunk_id=l1.chunk_id, **base_args)
                if prev_l2 is not None:
                    prev_l2.metadata.next_chunk_id = l2.chunk_id
                    l2.metadata.prev_chunk_id = prev_l2.chunk_id
                all_chunks.append(l2)
                prev_l2 = l2

            counters[key] = para_idx + len(group_texts)

        return sorted(all_chunks, key=lambda c: c.metadata.sequence_index)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _chunk(
        self, text: str, level: int, is_atomic: bool, sequence_index: int,
        *, subject: str, grade: int, source_pdf: str, chapter: int,
        chapter_title: str, section_path: list[str], page_start: int,
        page_end: int, content_type: ContentType,
        parent_chunk_id: Optional[str] = None,
    ) -> TextChunk:
        return TextChunk(
            chunk_id=str(uuid4()),
            text=text,
            token_count=len(self._enc.encode(text)),
            metadata=ChunkMetadata(
                subject=subject, grade=grade, chapter=chapter,
                chapter_title=chapter_title, page_start=page_start,
                page_end=page_end, source_pdf=source_pdf,
                chunk_level=level, content_type=content_type,
                section_path=section_path, sequence_index=sequence_index,
                is_atomic=is_atomic, parent_chunk_id=parent_chunk_id,
            ),
        )

    def _enforce_token_cap(self, text: str) -> list[str]:
        """Return text unchanged if within the L2 cap, else hard-split it into
        consecutive <=l2_max token windows. Last resort for segments that have
        no usable sentence boundary."""
        ids = self._enc.encode(text)
        cap = settings.chunk_l2_max_tokens
        if len(ids) <= cap:
            return [text]
        return [self._enc.decode(ids[i: i + cap]) for i in range(0, len(ids), cap)]

    def _token_cuts(self, sentences: list[str]) -> list[int]:
        cuts: list[int] = []
        acc = 0
        for i, s in enumerate(sentences):
            t = len(self._enc.encode(s))
            if acc + t > settings.chunk_l2_max_tokens and acc >= settings.chunk_l2_min_tokens:
                cuts.append(i - 1)
                acc = t
            else:
                acc += t
        return cuts


# ── module-level helpers ───────────────────────────────────────────────────────

def _seq(chapter: int, section: int, para: int) -> int:
    return chapter * 1_000_000 + section * 10_000 + para


def _split_at(sentences: list[str], cuts: list[int]) -> list[list[str]]:
    if not cuts:
        return [sentences]
    groups, prev = [], 0
    for c in cuts:
        groups.append(sentences[prev: c + 1])
        prev = c + 1
    groups.append(sentences[prev:])
    return [g for g in groups if g]


def _classify(text: str) -> tuple[ContentType, bool]:
    head = text[:150]
    for pat, ctype in _ATOMIC_PATTERNS:
        if pat.search(head):
            return ctype, True
    for pat, ctype in _SPECIAL_PATTERNS:
        if pat.search(head):
            return ctype, False
    return ContentType.TEXT, False


def _fresh_state() -> dict:
    return {"chapter": 0, "chapter_title": "", "section_num": 0, "section_path": []}
