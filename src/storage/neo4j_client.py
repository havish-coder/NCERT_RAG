from __future__ import annotations

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from src.config import settings

logger = structlog.get_logger(__name__)


class Neo4jClient:
    """Thin async wrapper around the Neo4j driver. Singleton via FastAPI lifespan."""

    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._driver.verify_connectivity()
        logger.info("neo4j_connected", uri=settings.neo4j_uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            logger.info("neo4j_closed")

    async def execute_query(
        self,
        cypher: str,
        parameters: dict | None = None,
        database: str = settings.neo4j_database,
    ) -> list[dict]:
        async with self._driver.session(database=database) as session:
            result = await session.run(cypher, parameters or {})
            return [record.data() async for record in result]

    async def execute_write(
        self,
        cypher: str,
        parameters: dict | None = None,
        database: str = settings.neo4j_database,
    ) -> None:
        async with self._driver.session(database=database) as session:
            await session.run(cypher, parameters or {})

    async def execute_write_batch(
        self,
        cypher: str,
        batch: list[dict],
        batch_size: int = 500,
        database: str = settings.neo4j_database,
    ) -> None:
        for i in range(0, len(batch), batch_size):
            chunk = batch[i : i + batch_size]
            await self.execute_write(cypher, {"batch": chunk}, database=database)

    # ── Schema ────────────────────────────────────────────────────────────────

    async def create_indexes(self) -> None:
        # NB: CREATE DATABASE is Enterprise-only. On Community Edition we use the
        # default `neo4j` database (set NEO4J_DATABASE=neo4j), so no DB creation.
        statements = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
        ]
        for stmt in statements:
            try:
                await self.execute_write(stmt, database=settings.neo4j_database)
            except Exception as exc:
                logger.debug("index_skip", stmt=stmt[:60], reason=str(exc))

    # ── Entity queries ────────────────────────────────────────────────────────

    async def get_entity_by_id(self, entity_id: str) -> dict | None:
        rows = await self.get_entities_by_ids([entity_id])
        return rows[0] if rows else None

    async def get_entities_by_ids(self, entity_ids: list[str]) -> list[dict]:
        """One round-trip for any number of entities (avoids N+1 lookups)."""
        rows = await self.execute_query(
            "MATCH (e:Entity) WHERE e.entity_id IN $ids RETURN e",
            {"ids": entity_ids},
        )
        return [r["e"] for r in rows]

    async def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 2,
        max_nodes: int = settings.max_entity_neighbors,
    ) -> dict:
        return await self.get_neighbors_batch([entity_id], max_depth, max_nodes)

    async def get_neighbors_batch(
        self,
        entity_ids: list[str],
        max_depth: int = 2,
        max_nodes: int = settings.max_entity_neighbors,
    ) -> dict:
        """Expand the neighborhood of all seed entities in ONE query.

        Cypher cannot parameterize variable-length bounds (*1..$depth), so the
        validated ints are inlined. The per-seed LIMIT lives inside the CALL
        subquery so dense hubs don't pull in thousands of nodes that would be
        discarded. Edges are matched only among the kept node set, and nodes
        carry chunk_ids so callers don't need a second lookup.
        """
        cypher = f"""
        MATCH (seed:Entity) WHERE seed.entity_id IN $ids
        CALL (seed) {{
            MATCH (seed)-[:RELATES_TO*1..{int(max_depth)}]-(n:Entity)
            WITH DISTINCT n
            LIMIT {int(max_nodes)}
            RETURN collect(n) AS neighbors
        }}
        UNWIND (neighbors + [seed]) AS node
        WITH collect(DISTINCT node) AS nodes
        UNWIND nodes AS a
        OPTIONAL MATCH (a)-[rel:RELATES_TO]->(b:Entity)
        WHERE b IN nodes
        WITH nodes, collect(DISTINCT CASE WHEN rel IS NULL THEN NULL ELSE
               {{source: a.entity_id, target: b.entity_id,
                 relation_type: rel.relation_type, weight: rel.weight}} END) AS edges
        RETURN [n IN nodes | {{entity_id: n.entity_id, name: n.name,
                              entity_type: n.entity_type, description: n.description,
                              chunk_ids: n.chunk_ids}}] AS nodes, edges
        """
        rows = await self.execute_query(cypher, {"ids": entity_ids})
        if not rows:
            return {"nodes": [], "edges": []}
        return {"nodes": rows[0]["nodes"], "edges": rows[0]["edges"]}

    # ── Batch upserts ─────────────────────────────────────────────────────────

    async def upsert_entities_batch(self, entities: list[dict]) -> None:
        cypher = """
        UNWIND $batch AS e
        MERGE (n:Entity {entity_id: e.entity_id})
        SET n += {
          name: e.name,
          entity_type: e.entity_type,
          description: e.description,
          subjects: e.subjects,
          grades: e.grades,
          chunk_ids: e.chunk_ids
        }
        """
        await self.execute_write_batch(cypher, entities)

    async def upsert_relationships_batch(self, rels: list[dict]) -> None:
        cypher = """
        UNWIND $batch AS r
        MATCH (src:Entity {entity_id: r.source_entity_id})
        MATCH (tgt:Entity {entity_id: r.target_entity_id})
        MERGE (src)-[rel:RELATES_TO {rel_id: r.rel_id}]->(tgt)
        SET rel.relation_type = r.relation_type,
            rel.description   = r.description,
            rel.weight        = coalesce(rel.weight, 0) + r.weight
        """
        await self.execute_write_batch(cypher, rels)

    async def upsert_chunks_batch(self, chunks: list[dict]) -> None:
        cypher = """
        UNWIND $batch AS ch
        MERGE (n:Chunk {chunk_id: ch.chunk_id})
        SET n += {
          text: ch.text,
          subject: ch.subject,
          grade: ch.grade,
          chapter: ch.chapter,
          chapter_title: ch.chapter_title,
          page_start: ch.page_start,
          page_end: ch.page_end,
          source_pdf: ch.source_pdf
        }
        """
        await self.execute_write_batch(cypher, chunks)
