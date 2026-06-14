"""
Unit tests for HierarchicalSemanticChunker.
No external services needed — runs purely on CPU with tiktoken.
"""
import pytest
from src.models.document import ContentType
from src.ingestion.chunker import (
    HierarchicalSemanticChunker,
    NCERTStructureDetector,
    SemanticBoundaryDetector,
    _classify,
    _seq,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _pages(text: str, chapter: int = 1, chapter_title: str = "Test Chapter") -> list[dict]:
    return [{
        "page_num": 1,
        "text": text,
        "chapter": chapter,
        "chapter_title": chapter_title,
    }]


# ── _classify ──────────────────────────────────────────────────────────────────

class TestClassify:
    def test_definition_detected(self):
        ctype, is_atomic = _classify("Definition: Osmosis is the movement of water molecules...")
        assert ctype == ContentType.DEFINITION
        assert is_atomic is True

    def test_law_is_atomic(self):
        ctype, is_atomic = _classify("Law: Newton's first law states that...")
        assert ctype == ContentType.DEFINITION
        assert is_atomic is True

    def test_example_not_atomic(self):
        ctype, is_atomic = _classify("Example 1: A ball is thrown horizontally...")
        assert ctype == ContentType.EXAMPLE
        assert is_atomic is False

    def test_summary_not_atomic(self):
        ctype, is_atomic = _classify("Summary: In this chapter we learned about...")
        assert ctype == ContentType.SUMMARY
        assert is_atomic is False

    def test_plain_text(self):
        ctype, is_atomic = _classify("The cell is the basic unit of life.")
        assert ctype == ContentType.TEXT
        assert is_atomic is False


# ── sequence_index ─────────────────────────────────────────────────────────────

class TestSeqIndex:
    def test_ordering(self):
        assert _seq(1, 0, 0) < _seq(1, 1, 0)
        assert _seq(1, 1, 0) < _seq(1, 1, 1)
        assert _seq(1, 2, 0) < _seq(2, 0, 0)

    def test_unique_within_chapter(self):
        indices = [_seq(3, s, p) for s in range(5) for p in range(10)]
        assert len(indices) == len(set(indices))


# ── NCERTStructureDetector ─────────────────────────────────────────────────────

class TestNCERTStructureDetector:
    def setup_method(self):
        self.detector = NCERTStructureDetector()

    def test_single_block_returned(self):
        pages = _pages("The mitochondria is the powerhouse of the cell.")
        blocks = self.detector.detect(pages)
        assert len(blocks) == 1
        assert "mitochondria" in blocks[0]["text"]

    def test_atomic_definition_flagged(self):
        pages = _pages("Definition: Photosynthesis is the process by which plants make food.")
        blocks = self.detector.detect(pages)
        assert blocks[0]["is_atomic"] is True
        assert blocks[0]["content_type"] == ContentType.DEFINITION

    def test_chapter_change_flushes_buffer(self):
        pages = [
            {"page_num": 1, "text": "Chapter one content here.", "chapter": 1, "chapter_title": "Ch1"},
            {"page_num": 2, "text": "Chapter two content here.", "chapter": 2, "chapter_title": "Ch2"},
        ]
        blocks = self.detector.detect(pages)
        assert len(blocks) == 2

    def test_section_heading_splits_block(self):
        # Real parsed text keeps line breaks; section headings sit at line starts
        # so the MULTILINE ^ anchor in _SECTION_RE can match them.
        text = "Some intro text.\n9.1 Motion and Force\nMore text about force and motion here."
        pages = _pages(text, chapter=9, chapter_title="Force and Laws of Motion")
        blocks = self.detector.detect(pages)
        assert any("9.1" in b.get("section_path", [""])[1] if len(b.get("section_path", [])) > 1 else False for b in blocks)


# ── HierarchicalSemanticChunker ────────────────────────────────────────────────

class TestHierarchicalSemanticChunker:
    def setup_method(self):
        self.chunker = HierarchicalSemanticChunker()

    def test_returns_chunks(self):
        text = "The mitochondria is the powerhouse of the cell. " * 10
        pages = _pages(text)
        chunks = self.chunker.chunk_document(pages, "biology", 9, "test.pdf")
        assert len(chunks) > 0

    def test_l2_chunks_have_parent(self):
        text = "The mitochondria is the powerhouse of the cell. " * 20
        pages = _pages(text)
        chunks = self.chunker.chunk_document(pages, "biology", 9, "test.pdf")
        l2s = [c for c in chunks if c.metadata.chunk_level == 2]
        assert all(c.metadata.parent_chunk_id is not None for c in l2s)

    def test_atomic_chunk_level_3(self):
        text = "Definition: Osmosis is the movement of water across a semi-permeable membrane."
        pages = _pages(text)
        chunks = self.chunker.chunk_document(pages, "biology", 9, "test.pdf")
        atomics = [c for c in chunks if c.metadata.is_atomic]
        assert len(atomics) == 1
        assert atomics[0].metadata.chunk_level == 3

    def test_chunks_sorted_by_sequence_index(self):
        text = ". ".join([f"Sentence {i} about topics in this chapter" for i in range(50)])
        pages = _pages(text)
        chunks = self.chunker.chunk_document(pages, "physics", 10, "test.pdf")
        indices = [c.metadata.sequence_index for c in chunks]
        assert indices == sorted(indices)

    def test_l2_token_limit(self):
        long_text = "This is a sentence about an important concept. " * 100
        pages = _pages(long_text)
        chunks = self.chunker.chunk_document(pages, "physics", 10, "test.pdf")
        l2s = [c for c in chunks if c.metadata.chunk_level == 2]
        for c in l2s:
            assert c.token_count <= 512 + 50  # allow small margin for sentence boundary

    def test_prev_next_links(self):
        text = ". ".join([f"Sentence {i} about the topic at hand" for i in range(30)])
        pages = _pages(text)
        chunks = self.chunker.chunk_document(pages, "chemistry", 8, "test.pdf")
        l2s = [c for c in chunks if c.metadata.chunk_level == 2]
        if len(l2s) >= 2:
            assert l2s[0].metadata.next_chunk_id == l2s[1].chunk_id
            assert l2s[1].metadata.prev_chunk_id == l2s[0].chunk_id

    def test_content_type_propagated(self):
        text = "Example 1: Calculate the force when mass=5kg and acceleration=2m/s2."
        pages = _pages(text)
        chunks = self.chunker.chunk_document(pages, "physics", 9, "test.pdf")
        assert any(c.metadata.content_type == ContentType.EXAMPLE for c in chunks)
