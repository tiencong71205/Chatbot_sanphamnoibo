"""Dense vector retriever using Qdrant."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.database.qdrant_store import QdrantStore
from app.embeddings.ollama_embedding import OllamaEmbedding

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, settings: Settings, qdrant: QdrantStore, embedding: OllamaEmbedding):
        self.settings = settings
        self.qdrant = qdrant
        self.embedding = embedding

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        query_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k documents by dense similarity.

        Pass a precomputed `query_vector` to skip re-embedding the same query
        text (used by HybridRetriever when fanning a multi-product query out
        across several product_id filters — the embedding is identical each
        time, only the filter changes).
        """
        k = top_k or self.settings.dense_top_k
        query_vec = query_vector if query_vector is not None else self.embedding.embed_query(query)
        results = self.qdrant.search(
            vector=query_vec,
            top_k=k,
            filters=filters,
            score_threshold=self.settings.min_dense_score,
        )
        # Add rank
        for i, r in enumerate(results):
            r["dense_rank"] = i + 1
            r["dense_score"] = r.pop("score", 0.0)
        logger.debug("Dense retrieval: %d results for query '%s...'", len(results), query[:50])
        return results
