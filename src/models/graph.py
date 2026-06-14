from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CONCEPT = "CONCEPT"
    PERSON = "PERSON"
    PLACE = "PLACE"
    EVENT = "EVENT"
    ORGANISM = "ORGANISM"
    CHEMICAL = "CHEMICAL"
    EQUATION = "EQUATION"
    LAW = "LAW"
    THEOREM = "THEOREM"
    INSTITUTION = "INSTITUTION"
    PHENOMENON = "PHENOMENON"
    PROCESS = "PROCESS"
    TERM = "TERM"


class Entity(BaseModel):
    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    entity_type: EntityType
    description: str
    subjects: list[str] = []          # plain strings, same as ChunkMetadata.subject
    grades: list[int] = []
    chunk_ids: list[str] = []
    dense_embedding: Optional[list[float]] = None
    community_ids: dict[str, str] = {}     # {"0": "c-abc", "1": "c-def", "2": "c-ghi"}


class Relationship(BaseModel):
    rel_id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str
    weight: float = 1.0
    chunk_ids: list[str] = []


class Community(BaseModel):
    community_id: str = Field(default_factory=lambda: str(uuid4()))
    level: int
    member_entity_ids: list[str] = []
    parent_community_id: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    subject_coverage: list[str] = []
    grade_coverage: list[int] = []
    dense_embedding: Optional[list[float]] = None
