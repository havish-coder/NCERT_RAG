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

        # 5. Attach entity relevance scores to chunks, then rank the union and keep
        # the strongest candidates so the token budget is spent on the best chunks.
        _score_chunks(chunks, entity_scores, subgraph_nodes)
        chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        chunks = chunks[: settings.top_k_chunks * 2]

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
        # Query-relevant chunks across the whole corpus (plain vector retrieval).
        global_hits = await self.qdrant.search_chunks(
            dense, sparse_idx, sparse_val, top_k=settings.top_k_chunks, filters=filters
        )
        if not chunk_ids:
            return global_hits
        # UNION in the query-relevant chunks from inside the graph neighborhood.
        # These can include passages outside the global top-k — the multi-hop case a
        # corpus-wide search misses. (The old code used chunk_ids only to FILTER the
        # global hits, so the graph could never introduce a chunk vector search had
        # not already found; recall was capped at the vector baseline by construction.)
        graph_hits = await self.qdrant.search_chunks_in_ids(
            chunk_ids, dense, top_k=settings.top_k_chunks, filters=filters
        )
        by_id: dict[str, dict] = {}
        for c in global_hits + graph_hits:
            cid = c.get("chunk_id")
            if cid not in by_id or c.get("score", 0.0) > by_id[cid].get("score", 0.0):
                by_id[cid] = c
        return list(by_id.values())


def _score_chunks(chunks: list[dict], entity_scores: dict, subgraph_nodes: dict) -> None:
    """Fuse two relevance signals: a chunk's own query similarity (the base, kept so
    the best lexical/semantic match stays on top) plus a bonus for being connected to
    a high-relevance seed entity in the graph. Overwriting the score with the entity
    signal alone reorders the top result by the noisier signal and hurts Recall@1."""
    GRAPH_BONUS = 0.25
    chunk_best: dict[str, float] = {}  # chunk_id -> best seed-entity score referencing it
    for eid, escore in entity_scores.items():
        for cid in subgraph_nodes.get(eid, {}).get("chunk_ids", []):
            chunk_best[cid] = max(escore, chunk_best.get(cid, 0.0))

    for c in chunks:
        base = c.get("score", 0.0)  # query similarity (present on every retrieved chunk)
        c["score"] = base + GRAPH_BONUS * chunk_best.get(c.get("chunk_id", ""), 0.0)


def _empty(mode: str) -> dict:
    return {"answer": "I could not find relevant information for this query.", "sources": [], "subgraph": None, "mode_used": mode}
