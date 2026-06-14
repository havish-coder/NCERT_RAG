"""
import_artifacts.py
==================
Loads artifacts generated offline (Lightning AI GPU notebook) into local Neo4j + Qdrant.

Run with: python -m src.ingestion.import_artifacts

Steps:
1. Read chunks.parquet → upsert to Qdrant (ncert_chunks collection)
2. Read entities.json + entity embeddings → upsert to Neo4j + Qdrant (ncert_entities)
3. Read relationships.json → upsert to Neo4j
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

from src.ingestion.embedder import EmbeddingEngine
from src.storage.neo4j_client import Neo4jClient
from src.storage.qdrant_client import QdrantClientWrapper

logger = logging.getLogger(__name__)

ARTIFACTS = Path("data/artifacts")


async def run_import() -> None:
    neo4j = Neo4jClient()
    qdrant = QdrantClientWrapper()
    embedder = EmbeddingEngine()

    await neo4j.connect()
    await qdrant.connect()
    await qdrant.ensure_collections()
    await neo4j.create_indexes()

    def embed_missing(records: list[dict], text_of) -> None:
        """Embed only records without a dense_embedding (artifacts generated on
        Lightning AI already carry embeddings — skip the ~30s model load + encode
        on re-runs). Loads bge-m3 lazily on first use."""
        todo = [r for r in records if not r.get("dense_embedding")]
        if not todo:
            return
        if not embedder.is_loaded:
            embedder.load()
        for rec, emb in zip(todo, embedder.embed_entities([text_of(r) for r in todo])):
            rec["dense_embedding"] = emb

    logger.info("import_start artifacts_dir=%s", ARTIFACTS)

    # ── 1. Chunks → Qdrant ────────────────────────────────────────────────────
    chunks_path = ARTIFACTS / "chunks.parquet"
    if chunks_path.exists():
        logger.info("importing_chunks")
        df = pd.read_parquet(chunks_path)
        # Only L2/L3 chunks go to Qdrant (L1 are context-only).
        # dense_embedding is stored as list[float] in parquet.
        records = df[df["chunk_level"].isin([2, 3])].to_dict("records")
        for i in range(0, len(records), 500):
            await qdrant.upsert_chunks(records[i : i + 500])
        logger.info("chunks_imported count=%d", len(records))

    # ── 2. Entities → Neo4j + Qdrant ──────────────────────────────────────────
    entities_path = ARTIFACTS / "entities.json"
    if entities_path.exists():
        logger.info("importing_entities")
        entities = json.loads(entities_path.read_text())
        embed_missing(entities, lambda e: e.get("description") or e["name"])

        # Neo4j
        await neo4j.upsert_entities_batch(entities)

        # Qdrant
        await qdrant.upsert_entities(entities)
        logger.info("entities_imported count=%d", len(entities))

    # ── 3. Relationships → Neo4j ──────────────────────────────────────────────
    rels_path = ARTIFACTS / "relationships.json"
    if rels_path.exists():
        logger.info("importing_relationships")
        rels = json.loads(rels_path.read_text())
        await neo4j.upsert_relationships_batch(rels)
        logger.info("relationships_imported count=%d", len(rels))

    await neo4j.close()
    await qdrant.close()
    logger.info("import_complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_import())
