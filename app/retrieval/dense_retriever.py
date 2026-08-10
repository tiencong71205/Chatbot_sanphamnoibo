"""Dense vector retriever using Qdrant."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.database.qdrant_store import QdrantStore

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
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k documents by dense similarity."""
        k = top_k or self.settings.dense_top_k
        query_vec = self.embedding.embed_query(query)
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
