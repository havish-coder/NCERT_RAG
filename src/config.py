from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (OpenAI-compatible: Ollama, Gemini, Groq, OpenAI, …) ──────────────
    # online_base_url + llm_api_key select the provider. Defaults = local Ollama.
    online_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"          # dummy for Ollama; real key for cloud APIs
    online_model: str = "qwen2.5:7b-instruct"
    offline_extraction_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "ncert"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_chunks_collection: str = "ncert_chunks"
    qdrant_entities_collection: str = "ncert_entities"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_l1_max_tokens: int = 2000      # section-level (late-chunking window)
    chunk_l2_max_tokens: int = 512       # paragraph-level (primary retrieval)
    chunk_l2_min_tokens: int = 64        # merge if smaller
    chunk_l3_max_tokens: int = 128       # atomic content (definitions, formulas)
    semantic_threshold: float = 0.45    # similarity drop → L2 boundary

    # ── Graph ─────────────────────────────────────────────────────────────────
    max_entity_neighbors: int = 10
    entity_dedup_threshold: float = 0.92

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_chunks: int = 10
    top_k_entities: int = 5
    context_token_budget: int = 8192

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


settings = Settings()
