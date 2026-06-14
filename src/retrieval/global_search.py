"""
GlobalSearch
============
Simplified GraphRAG global search:

1. Embed the query with bge-m3
2. Find top-3 community summaries by vector similarity (Qdrant)
3. Combine them into one context block
4. Single LLM call → synthesized answer

Best for: broad thematic questions no single chunk can answer
("What are the major topics in class 10 science?",
 "How does NCERT cover the Indian independence movement?")
"""
from __future__ import annotations

import structlog

from src.config import settings
from src.ingestion.embedder import EmbeddingEngine
from src.llm.client import LLMClient
from src.llm.prompts import GLOBAL_REDUCE_SYSTEM
from src.storage.qdrant_client import QdrantClientWrapper

logger = structlog.get_logger(__name__)


class GlobalSearch:
    def __init__(
        self,
        qdrant: QdrantClientWrapper,
        embedder: EmbeddingEngine,
        llm: LLMClient,
    ) -> None:
        self.qdrant = qdrant
        self.embedder = embedder
        self.llm = llm

    async def search(
        self,
        query: str,
        community_level: int = 1,
        top_k: int = 3,
    ) -> dict:
        prepared = await self.prepare(query, community_level, top_k)
        if "prompt" not in prepared:
            return prepared
        system, user = prepared.pop("prompt")
        prepared["answer"] = await self.llm.complete(system, user, max_tokens=1024)
        return prepared

    async def prepare(
        self,
        query: str,
        community_level: int = 1,
        top_k: int = 3,
    ) -> dict:
        """Retrieve community summaries and return the prompt without calling the
        LLM (so the caller can stream). See LocalSearch.prepare."""
        # 1. Embed query
        dense, _, _ = self.embedder.encode_query(query)

        # 2. Find top-k community summaries by vector similarity
        communities = await self.qdrant.search_communities(dense, level=community_level, top_k=top_k)
        if not communities:
            return _empty()

        # 3. Combine summaries into one context block
        context_parts = []
        for i, c in enumerate(communities, 1):
            title = c.get("title", f"Community {i}")
            summary = c.get("summary", "")
            if summary:
                context_parts.append(f"**{title}**\n{summary}")

        if not context_parts:
            return _empty()

        context = "\n\n".join(context_parts)
        logger.info("global_search_prepared", communities_used=len(communities))
        return {
            "prompt": (GLOBAL_REDUCE_SYSTEM, f"Question: {query}\n\nRelevant knowledge areas:\n\n{context}"),
            "sources": [],
            "subgraph": None,
            "mode_used": "global",
        }


def _empty() -> dict:
    return {
        "answer": "I could not find relevant information at the community level for this query. Try Local mode for specific concepts.",
        "sources": [],
        "subgraph": None,
        "mode_used": "global",
    }
