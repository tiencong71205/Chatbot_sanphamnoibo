"""Qdrant vector store wrapper with full CRUD, listing, and scroll filtering support."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import Settings
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


def _chunk_id_to_int(chunk_id: str) -> int:
    h = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def _build_filter(filters: Dict[str, Any]) -> qdrant_models.Filter:
    conditions = []
    for field, value in filters.items():
        if value is None or value == "":
            continue
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
        indexed_fields = [
            "product_id", "product_group", "content_type", "chunk_type", "model",
            "source_file", "source_document", "document_version", "source_section",
        ]
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

    def scroll_all(self, limit_per_page: int = 1000) -> List[Dict[str, Any]]:
        """Scroll through all points in collection."""
        if not self.collection_exists():
            return []
        try:
            all_records = []
            offset = None
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self.collection,
                    limit=limit_per_page,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_records.extend(records)
                if next_offset is None:
                    break
                offset = next_offset
            return all_records
        except Exception as e:
            logger.error("Error scrolling collection: %s", e)
            return []

    def scroll_by_filter(
        self,
        filters: Dict[str, Any],
        limit_per_page: int = 500,
    ) -> List[Dict[str, Any]]:
        """Scroll through all points matching specific metadata filters (product_id, content_type, etc.)."""
        if not self.collection_exists():
            return []
        try:
            q_filter = _build_filter(filters) if filters else None
            all_records = []
            offset = None
            while True:
                records, next_offset = self._client.scroll(
                    collection_name=self.collection,
                    scroll_filter=q_filter,
                    limit=limit_per_page,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for r in records:
                    all_records.append({
                        "id": r.id,
                        "score": 1.0,
                        "payload": r.payload or {},
                    })
                if next_offset is None:
                    break
                offset = next_offset
            return all_records
        except Exception as e:
            logger.error("Error scrolling by filter %s: %s", filters, e)
            return []

    def list_products(self) -> List[Dict[str, Any]]:
        """List unique products from collection."""
        if not self.collection_exists():
            return []
        try:
            records = self.scroll_all()
            seen: Dict[str, Dict[str, Any]] = {}
            for r in records:
                p = r.payload if hasattr(r, "payload") else r.get("payload", {})
                pid = p.get("product_id")
                if not pid:
                    continue
                if pid not in seen:
                    seen[pid] = {
                        "product_id": pid,
                        "product_name": p.get("product_name", ""),
                        "product_group": p.get("product_group", ""),
                        "model": p.get("model", ""),
                        "aliases": [],
                        "source_file": p.get("source_file", ""),
                        "chunk_count": 0,
                    }
                seen[pid]["chunk_count"] += 1
            return list(seen.values())
        except Exception as e:
            logger.error("Error fetching products from Qdrant: %s", e)
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """List unique documents from collection."""
        if not self.collection_exists():
            return []
        try:
            records = self.scroll_all()
            seen: Dict[str, Dict[str, Any]] = {}
            for r in records:
                p = r.payload if hasattr(r, "payload") else r.get("payload", {})
                doc_id = p.get("source_document") or p.get("source_file", "")
                if not doc_id:
                    continue
                if doc_id not in seen:
                    seen[doc_id] = {
                        "document_id": doc_id,
                        "product_name": p.get("product_name", ""),
                        "source_file": p.get("source_file", ""),
                        "chunk_count": 0,
                        "status": "indexed",
                    }
                seen[doc_id]["chunk_count"] += 1
            return list(seen.values())
        except Exception as e:
            logger.error("Error listing documents: %s", e)
            return []

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a document."""
        if not self.collection_exists():
            return None
        try:
            records = self.scroll_all()
            chunks = []
            doc_info = None
            for r in records:
                p = r.payload if hasattr(r, "payload") else r.get("payload", {})
                if (p.get("source_document") == document_id or
                        p.get("source_file") == document_id or
                        p.get("product_id") == document_id):
                    chunks.append(p)
                    if doc_info is None:
                        doc_info = {
                            "document_id": document_id,
                            "product_name": p.get("product_name", ""),
                            "product_id": p.get("product_id", ""),
                            "source_file": p.get("source_file", ""),
                        }
            if doc_info:
                doc_info["chunk_count"] = len(chunks)
                doc_info["sections"] = list({c.get("source_section", "") for c in chunks if c.get("source_section")})
                doc_info["content_types"] = list({c.get("content_type", "") for c in chunks if c.get("content_type")})
                return doc_info
            return None
        except Exception as e:
            logger.error("Error getting document info: %s", e)
            return None

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by its chunk_id string."""
        try:
            int_id = _chunk_id_to_int(chunk_id)
            results = self._client.retrieve(
                collection_name=self.collection,
                ids=[int_id],
                with_payload=True,
                with_vectors=False,
            )
            if results:
                return {"id": results[0].id, "payload": results[0].payload}
            # Fallback: scroll and find
            records = self.scroll_all()
            for r in records:
                p = r.payload if hasattr(r, "payload") else r.get("payload", {})
                if p.get("chunk_id") == chunk_id:
                    return {"id": getattr(r, "id", None), "payload": p}
            return None
        except Exception as e:
            logger.error("Error getting chunk: %s", e)
            return None

    def count_by_product(self, product_id: str) -> int:
        """Count chunks for a specific product."""
        try:
            count = self._client.count(
                collection_name=self.collection,
                count_filter=_build_filter({"product_id": product_id}),
            )
            return count.count
        except Exception as e:
            logger.warning("Error counting by product %s: %s", product_id, e)
            return 0

    def get_existing_chunk_metadata(self, source_document: str) -> Dict[str, str]:
        """Return {chunk_id: stored_product_id} currently stored for one source file.

        Used by incremental ingest to diff against freshly generated chunks —
        not just by chunk_id existence, but also by product_id/product_name,
        since chunk_id is a hash of (filename, heading_path, title, content)
        and deliberately does NOT include product_id. A chunk whose text is
        unchanged but whose product_id was corrected (e.g. a catalog
        collision fix) would otherwise be silently skipped forever, leaving
        the old wrong product_id in Qdrant even after the source data and
        catalog are both fixed — confirmed to happen in practice.
        """
        if not self.collection_exists():
            return {}
        records = self.scroll_by_filter({"source_document": source_document})
        result = {}
        for r in records:
            payload = r.get("payload", {})
            cid = payload.get("chunk_id")
            if cid:
                result[cid] = payload.get("product_id", "")
        return result

    def get_existing_chunk_ids(self, source_document: str) -> set:
        """Return the set of chunk_id strings currently stored for one source file.

        Used by incremental ingest to diff against freshly generated chunk_ids:
        unchanged ones are skipped (no re-embedding), removed ones are deleted,
        new ones are embedded and upserted.
        """
        return set(self.get_existing_chunk_metadata(source_document).keys())

    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> None:
        """Delete points by their string chunk_id (converted to Qdrant's int id)."""
        if not chunk_ids or not self.collection_exists():
            return
        int_ids = [_chunk_id_to_int(cid) for cid in chunk_ids]
        self._client.delete(
            collection_name=self.collection,
            points_selector=qdrant_models.PointIdsList(points=int_ids),
            wait=True,
        )
        logger.info("Deleted %d stale points from %s", len(int_ids), self.collection)

    def get_all_products(self) -> List[Dict[str, Any]]:
        """Alias for list_products for backward compatibility."""
        return self.list_products()
