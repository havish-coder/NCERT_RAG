"""
import_artifacts.py
==================
Loads artifacts generated offline (Lightning AI GPU notebook) into local Neo4j + Qdrant.

Run with: python -m src.ingestion.import_artifacts

Steps:
1. Read chunks.parquet → upsert to Qdrant (ncert_chunks collection)
2. Read entities.json + entity embeddings → upsert to Neo4j + Qdrant (ncert_entities)
3. Read relationships.json → upsert to Neo4j
4. Run community_detector locally (CPU) → build community_memberships.json
5. Read community_summaries.json + embed summaries → upsert to Neo4j + Qdrant (ncert_communities)
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.ingestion.community_detector import detect_communities
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

    # ── 4. Community detection (CPU, local) ────────────────────────────────────
    memberships_path = ARTIFACTS / "community_memberships.json"
    if not memberships_path.exists() and entities_path.exists() and rels_path.exists():
        logger.info("running_community_detection")
        detect_communities(entities_path, rels_path, memberships_path)

    # ── 5. Community summaries → Neo4j + Qdrant ───────────────────────────────
    summaries_path = ARTIFACTS / "community_summaries.json"
    if summaries_path.exists() and memberships_path.exists():
        logger.info("importing_communities")
        summaries = json.loads(summaries_path.read_text())
        memberships = json.loads(memberships_path.read_text())

        # Attach member lists to summaries
        comm_members: dict[str, list[str]] = {}
        for eid, comms in memberships.items():
            for _, cid in comms.items():
                comm_members.setdefault(cid, []).append(eid)

        embed_missing(summaries, lambda s: s.get("summary") or s.get("title", ""))
        for s in summaries:
            s["member_entity_ids"] = comm_members.get(s["community_id"], [])

        await neo4j.upsert_communities_batch(summaries)
        await qdrant.upsert_communities(summaries)
        logger.info("communities_imported count=%d", len(summaries))

    await neo4j.close()
    await qdrant.close()
    logger.info("import_complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_import())
