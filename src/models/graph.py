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


class Relationship(BaseModel):
    rel_id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str
    weight: float = 1.0
    chunk_ids: list[str] = []
