"""
EmbeddingEngine — wraps BAAI/bge-m3 via FlagEmbedding.

For each text, bge-m3 gives us two things we use:
  - a dense vector (1024-d)        → semantic similarity
  - sparse weights (BM25-style)    → keyword / lexical matching

Used offline to embed chunks/entities/communities, and online to encode a query.
We embed each chunk on its own (simple and easy to debug). No late chunking.
"""
from __future__ import annotations

from src.config import settings
from src.models.document import TextChunk


class EmbeddingEngine:
    """Lazy-loaded bge-m3 encoder. Call .load() once before using it."""

    def __init__(self) -> None:
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        self._model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=(settings.embedding_device == "cuda"),
        )

    # ── Corpus embedding (offline) ────────────────────────────────────────────

    def embed_chunks(self, chunks: list[TextChunk]) -> None:
        """Embed each chunk's text (dense + sparse), writing back in place."""
        if not chunks:
            return
        out = self._encode([c.text for c in chunks], sparse=True)
        for i, c in enumerate(chunks):
            c.dense_embedding = out["dense_vecs"][i].tolist()
            c.sparse_indices, c.sparse_values = _sparse(out["lexical_weights"][i])

    def embed_entities(self, descriptions: list[str]) -> list[list[float]]:
        """Dense vectors for entity descriptions."""
        return self._encode(descriptions, sparse=False)["dense_vecs"].tolist()

    def embed_communities(self, summaries: list[str]) -> list[list[float]]:
        """Dense vectors for community summaries."""
        return self.embed_entities(summaries)

    # ── Query embedding (online) ──────────────────────────────────────────────

    def encode_query(self, query: str) -> tuple[list[float], list[int], list[float]]:
        """Encode one query → (dense_vector, sparse_indices, sparse_values)."""
        out = self._encode([query], sparse=True)
        dense = out["dense_vecs"][0].tolist()
        idx, val = _sparse(out["lexical_weights"][0])
        return dense, idx, val

    # ── internal ──────────────────────────────────────────────────────────────

    def _encode(self, texts: list[str], sparse: bool):
        assert self._model is not None, "call .load() first"
        return self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            return_dense=True,
            return_sparse=sparse,
        )


def _sparse(weights: dict) -> tuple[list[int], list[float]]:
    """bge-m3 sparse weights {token_id: weight} → (indices, values)."""
    return [int(k) for k in weights], [float(v) for v in weights.values()]
