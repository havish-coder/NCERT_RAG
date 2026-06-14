from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from src.config import settings

logger = structlog.get_logger(__name__)

DENSE = "dense"
SPARSE = "sparse"

# Vector fields live in the Qdrant vector, not the payload. They also arrive as
# numpy arrays when loaded from parquet, so they must be normalised explicitly.
_VECTOR_KEYS = {"dense_embedding", "sparse_indices", "sparse_values"}


def _to_list(v):
    """numpy array / pandas value → plain list (or None). Avoids the
    'truth value of an array is ambiguous' error from bare truthiness checks."""
    if v is None:
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


def _clean_payload(d: dict) -> dict:
    """Drop the vector fields and convert any numpy values to JSON-native types."""
    out = {}
    for k, v in d.items():
        if k in _VECTOR_KEYS:
            continue
        out[k] = v.tolist() if hasattr(v, "tolist") else v
    return out


class QdrantClientWrapper:
    """
    Manages three Qdrant collections using hybrid search (dense + sparse via bge-m3).
    Collections: ncert_chunks, ncert_entities, ncert_communities.
    """

    def __init__(self) -> None:
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info("qdrant_connected", host=settings.qdrant_host, port=settings.qdrant_port)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("qdrant_closed")

    async def ensure_collections(self) -> None:
        for name in [
            settings.qdrant_chunks_collection,
            settings.qdrant_entities_collection,
            settings.qdrant_communities_collection,
        ]:
            existing = await self._client.collection_exists(name)
            if not existing:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        DENSE: VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        SPARSE: SparseVectorParams(),
                    },
                )
                logger.info("qdrant_collection_created", name=name)

    # ── Upsert helpers ────────────────────────────────────────────────────────

    def _make_point(self, point_id: str, payload: dict, dense,
                    sparse_indices=None, sparse_values=None) -> PointStruct:
        vectors: dict = {DENSE: _to_list(dense)}
        si, sv = _to_list(sparse_indices), _to_list(sparse_values)
        if si and sv:
            vectors[SPARSE] = SparseVector(indices=si, values=sv)
        return PointStruct(id=point_id, vector=vectors, payload=_clean_payload(payload))

    async def _upsert_batched(self, collection: str, points: list[PointStruct],
                              batch_size: int = 256) -> None:
        """Send points in chunks — one big request blows past Qdrant's 32 MB
        payload limit (30k entity vectors ≈ 600 MB in a single call)."""
        for i in range(0, len(points), batch_size):
            await self._client.upsert(collection, points=points[i : i + batch_size])

    async def upsert_chunks(self, chunks: list[dict]) -> None:
        points = [
            self._make_point(
                c["chunk_id"], c,
                c["dense_embedding"], c.get("sparse_indices"), c.get("sparse_values"),
            )
            for c in chunks if _to_list(c.get("dense_embedding"))
        ]
        await self._upsert_batched(settings.qdrant_chunks_collection, points)

    async def upsert_entities(self, entities: list[dict]) -> None:
        points = [
            self._make_point(e["entity_id"], e, e["dense_embedding"])
            for e in entities if _to_list(e.get("dense_embedding"))
        ]
        await self._upsert_batched(settings.qdrant_entities_collection, points)

    async def upsert_communities(self, communities: list[dict]) -> None:
        points = [
            self._make_point(c["community_id"], c, c["dense_embedding"])
            for c in communities if _to_list(c.get("dense_embedding"))
        ]
        await self._upsert_batched(settings.qdrant_communities_collection, points)

    # ── Search helpers ────────────────────────────────────────────────────────

    def _build_filter(self, filters: dict | None) -> Filter | None:
        if not filters:
            return None
        conditions = []
        if "subjects" in filters and filters["subjects"]:
            conditions.append(FieldCondition(key="subject", match=MatchAny(any=filters["subjects"])))
        if "grades" in filters and filters["grades"]:
            conditions.append(FieldCondition(key="grade", match=MatchAny(any=filters["grades"])))
        if "level" in filters:
            conditions.append(FieldCondition(key="level", match=MatchValue(value=filters["level"])))
        return Filter(must=conditions) if conditions else None

    async def search_chunks(
        self,
        dense_vector: list[float],
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        top_k: int = settings.top_k_chunks,
        filters: dict | None = None,
    ) -> list[dict]:
        qfilter = self._build_filter(filters)
        return await self._query(settings.qdrant_chunks_collection, dense_vector, top_k, qfilter)

    async def search_entities(
        self,
        dense_vector: list[float],
        top_k: int = settings.top_k_entities,
        filters: dict | None = None,
    ) -> list[dict]:
        qfilter = self._build_filter(filters)
        return await self._query(settings.qdrant_entities_collection, dense_vector, top_k, qfilter)

    async def search_communities(
        self,
        dense_vector: list[float],
        level: int,
        top_k: int = settings.top_k_communities,
    ) -> list[dict]:
        qfilter = Filter(must=[FieldCondition(key="level", match=MatchValue(value=level))])
        return await self._query(settings.qdrant_communities_collection, dense_vector, top_k, qfilter)

    async def _query(self, collection: str, dense_vector: list[float], top_k: int,
                     qfilter: Filter | None) -> list[dict]:
        """Dense vector search via query_points (.search() was removed in
        qdrant-client 1.10+). `using` selects the named dense vector."""
        resp = await self._client.query_points(
            collection_name=collection,
            query=dense_vector,
            using=DENSE,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
        )
        return [{"score": r.score, **r.payload} for r in resp.points]
