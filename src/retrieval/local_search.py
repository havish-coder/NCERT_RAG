"""
LocalSearch
===========
Entity-centric GraphRAG local search:

1. Embed the query with bge-m3 (dense + sparse)
2. Find top-k matching ENTITIES by dense similarity in Qdrant
3. For each seed entity, retrieve its neighborhood (depth 2) from Neo4j
4. Collect all chunk IDs referenced by those entities
5. Fetch those chunks from Qdrant
6. Sort by sequence_index (reading order) and trim to token budget
7. Build prompt context + generate answer via LLM

Best for: specific concept questions ("What is Newton's 3rd law?",
"Explain osmosis", "Who discovered penicillin?")
"""
from __future__ import annotations

import structlog

from src.config import settings
from src.ingestion.embedder import EmbeddingEngine
from src.llm.client import LLMClient
from src.llm.prompts import QUERY_ANSWER_SYSTEM
from src.retrieval.context_builder import ContextBuilder
from src.storage.neo4j_client import Neo4jClient
from src.storage.qdrant_client import QdrantClientWrapper

logger = structlog.get_logger(__name__)


class LocalSearch:
    def __init__(
        self,
        neo4j: Neo4jClient,
        qdrant: QdrantClientWrapper,
        embedder: EmbeddingEngine,
        llm: LLMClient,
    ) -> None:
        self.neo4j = neo4j
        self.qdrant = qdrant
        self.embedder = embedder
        self.llm = llm
        self.ctx_builder = ContextBuilder()

    async def search(
        self,
        query: str,
        filters: dict | None = None,
        top_k_entities: int = settings.top_k_entities,
    ) -> dict:
        """Returns {answer, sources, subgraph, mode_used}."""
        prepared = await self.prepare(query, filters, top_k_entities)
        if "prompt" not in prepared:
            return prepared  # empty / short-circuit case already has an answer
        system, user = prepared.pop("prompt")
        prepared["answer"] = await self.llm.complete(system, user)
        return prepared

    async def prepare(
        self,
        query: str,
        filters: dict | None = None,
        top_k_entities: int = settings.top_k_entities,
    ) -> dict:
        """Do all retrieval and return the prompt + sources WITHOUT calling the
        LLM, so the caller can stream the answer. Returns either
        {prompt:(system,user), sources, subgraph, mode_used} or an empty result
        carrying a ready 'answer'.
        """
        # 1. Embed query
        dense, sparse_idx, sparse_val = self.embedder.encode_query(query)

        # 2. Find seed entities by embedding similarity
        seed_entities = await self.qdrant.search_entities(dense, top_k=top_k_entities, filters=filters)
        if not seed_entities:
            return _empty("local")

        entity_ids = [e["entity_id"] for e in seed_entities]
        entity_scores = {e["entity_id"]: e["score"] for e in seed_entities}

        # 3. Expand all seed neighborhoods in ONE Neo4j round-trip (depth 2).
        # Nodes come back with chunk_ids, so no per-entity follow-up queries.
        subg = await self.neo4j.get_neighbors_batch(entity_ids, max_depth=2)
        subgraph_nodes = {n["entity_id"]: n for n in subg["nodes"]}
        subgraph_edges = subg["edges"]
        if not subgraph_nodes:  # graph empty/unreachable — fall back to the seeds themselves
            subgraph_nodes = {
                e["entity_id"]: e for e in await self.neo4j.get_entities_by_ids(entity_ids)
            }

        all_chunk_ids: set[str] = set()
        for n in subgraph_nodes.values():
            all_chunk_ids.update(n.get("chunk_ids") or [])

        # 4. Retrieve source chunks from Qdrant by chunk ID
        chunks = await self._fetch_chunks(list(all_chunk_ids), dense, sparse_idx, sparse_val, filters)

        # 5. Attach entity relevance scores to chunks (for dedup/ranking)
        _score_chunks(chunks, entity_scores, subgraph_nodes)

        # 6. Assemble context
        context, citations = self.ctx_builder.assemble(chunks)
        if not context:
            return _empty("local")

        logger.info("local_search_prepared", entities=len(seed_entities), chunks=len(chunks))
        return {
            "prompt": (QUERY_ANSWER_SYSTEM, f"Question: {query}\n\nContext:\n{context}"),
            "sources": citations,
            "subgraph": {"nodes": list(subgraph_nodes.values()), "edges": subgraph_edges},
            "mode_used": "local",
        }

    async def _fetch_chunks(
        self,
        chunk_ids: list[str],
        dense: list[float],
        sparse_idx: list[int],
        sparse_val: list[float],
        filters: dict | None,
    ) -> list[dict]:
        if not chunk_ids:
            return await self.qdrant.search_chunks(
                dense, sparse_idx, sparse_val,
                top_k=settings.top_k_chunks, filters=filters
            )
        # Hybrid: search among retrieved chunk IDs
        results = await self.qdrant.search_chunks(
            dense, sparse_idx, sparse_val,
            top_k=settings.top_k_chunks * 2, filters=filters
        )
        id_set = set(chunk_ids)
        matched = [r for r in results if r.get("chunk_id") in id_set]
        return matched or results[:settings.top_k_chunks]


def _score_chunks(chunks: list[dict], entity_scores: dict, subgraph_nodes: dict) -> None:
    """Score chunks by the highest entity score among entities that reference them."""
    # Build a map: chunk_id -> best entity score
    chunk_best: dict[str, float] = {}
    for eid, escore in entity_scores.items():
        node = subgraph_nodes.get(eid, {})
        for cid in node.get("chunk_ids", []):
            if escore > chunk_best.get(cid, 0.0):
                chunk_best[cid] = escore

    for c in chunks:
        cid = c.get("chunk_id", "")
        c["score"] = chunk_best.get(cid, c.get("score", 0.3))


def _empty(mode: str) -> dict:
    return {"answer": "I could not find relevant information for this query.", "sources": [], "subgraph": None, "mode_used": mode}
