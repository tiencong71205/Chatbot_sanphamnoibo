"""Qdrant vector store wrapper."""
from __future__ import annotations
import hashlib
import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import Settings
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


def _chunk_id_to_int(chunk_id: str) -> int:
    h = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def _build_filter(filters: Dict[str, Any]) -> qdrant_models.Filter:
    conditions = []
    for field, value in filters.items():
        if isinstance(value, list):
            conditions.append(
                qdrant_models.FieldCondition(
                    key=field,
                    match=qdrant_models.MatchAny(any=value),
                )
            )
        else:
            conditions.append(
                qdrant_models.FieldCondition(
                    key=field,
                    match=qdrant_models.MatchValue(value=value),
                )
            )
    return qdrant_models.Filter(must=conditions)


class QdrantStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.collection = settings.qdrant_collection
        self._client = QdrantClient(
            url=settings.qdrant_url,
            timeout=30.0,
        )

    def collection_exists(self) -> bool:
        try:
            collections = self._client.get_collections().collections
            return any(c.name == self.collection for c in collections)
        except Exception as e:
            logger.error("Error checking collection existence: %s", e)
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self._client.get_collection(self.collection)
            return {
                "status": info.status.value if info.status else "unknown",
                "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", getattr(info, "points_count", 0)),
                "points_count": getattr(info, "points_count", 0),
                "segments_count": getattr(info, "segments_count", 0),
            }
        except Exception as e:
            logger.error("Error getting collection info: %s", e)
            return {"status": "error", "error": str(e)}

    def create_collection(self) -> None:
        logger.info("Creating Qdrant collection: %s (dim=%d)", self.collection, self.settings.embedding_dimension)
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qdrant_models.VectorParams(
                size=self.settings.embedding_dimension,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        self._create_payload_indexes()

    def recreate_collection(self) -> None:
        logger.info("Recreating collection: %s", self.collection)
        if self.collection_exists():
            self._client.delete_collection(self.collection)
        self.create_collection()

    def ensure_collection(self) -> None:
        if not self.collection_exists():
            self.create_collection()
        else:
            logger.info("Qdrant collection exists: %s", self.collection)

    def _create_payload_indexes(self) -> None:
        indexed_fields = ["product_id", "product_group", "content_type", "model", "source_file"]
        for field in indexed_fields:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            except Exception as e:
                logger.warning("Error creating payload index for %s: %s", field, e)

    def upsert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if not chunks or not vectors:
            return
        self.ensure_collection()

        points = []
        for chunk, vec in zip(chunks, vectors):
            payload = chunk.to_dict()
            points.append(
                qdrant_models.PointStruct(
                    id=_chunk_id_to_int(chunk.chunk_id),
                    vector=vec,
                    payload={**payload, "chunk_id": chunk.chunk_id},
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=True,
            )
        logger.info("Upserted %d points to %s", len(points), self.collection)

    def search(
        self,
        query_vector: Optional[List[float]] = None,
        vector: Optional[List[float]] = None,
        top_k: int = 10,
        filters: Optional[Any] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        target_vector = query_vector or vector
        if not target_vector:
            return []
        
        q_filter = None
        if isinstance(filters, dict):
            q_filter = _build_filter(filters)
        elif isinstance(filters, qdrant_models.Filter):
            q_filter = filters

        try:
            results = self._client.query_points(
                collection_name=self.collection,
                query=target_vector,
                limit=top_k,
                query_filter=q_filter,
                score_threshold=score_threshold,
            ).points

            output = []
            for hit in results:
                output.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                })
            return output
        except Exception as e:
            logger.error("Qdrant search error: %s", e)
            return []

    def get_all_products(self) -> List[Dict[str, Any]]:
        if not self.collection_exists():
            return []
        try:
            records, _ = self._client.scroll(
                collection_name=self.collection,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            seen: Dict[str, Dict[str, Any]] = {}
            for r in records:
                p = r.payload or {}
                pid = p.get("product_id")
                if not pid:
                    continue
                if pid not in seen:
                    seen[pid] = {
                        "product_id": pid,
                        "product_name": p.get("product_name", ""),
                        "product_group": p.get("product_group", ""),
                        "models": [],
                        "chunk_count": 0,
                    }
                seen[pid]["chunk_count"] += 1
                model = p.get("model", "")
                if model and model not in seen[pid]["models"]:
                    seen[pid]["models"].append(model)
            return list(seen.values())
        except Exception as e:
            logger.error("Error fetching products from Qdrant: %s", e)
            return []
