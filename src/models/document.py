from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    DEFINITION = "definition"
    EXAMPLE = "example"
    FORMULA = "formula"
    KEY_POINT = "key_point"
    EXERCISE = "exercise"
    SUMMARY = "summary"
    TABLE = "table"
    FIGURE = "figure"
    ACTIVITY = "activity"


class ChunkMetadata(BaseModel):
    # Core provenance
    subject: str                                # plain string — whatever the folder/filename says
    grade: int
    chapter: int
    chapter_title: str
    page_start: int
    page_end: int
    source_pdf: str

    # Industrial chunking fields
    chunk_level: int = 2                        # 1=section(L1), 2=paragraph(L2), 3=atomic(L3)
    content_type: ContentType = ContentType.TEXT
    section_path: list[str] = []                # ["Chapter 9: Force and Laws of Motion", "9.2 Newton's Laws"]
    sequence_index: int = 0                     # chapter*1_000_000 + section*10_000 + para_idx
    is_atomic: bool = False                     # True → never split (definitions, formulas, laws)
    parent_chunk_id: Optional[str] = None       # L1 section this chunk belongs to
    prev_chunk_id: Optional[str] = None         # preceding L2 in document order
    next_chunk_id: Optional[str] = None         # following L2 in document order


class TextChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    token_count: int
    metadata: ChunkMetadata
    dense_embedding: Optional[list[float]] = None
    sparse_indices: Optional[list[int]] = None
    sparse_values: Optional[list[float]] = None
