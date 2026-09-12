"""Hybrid Retriever integrating resolver, dense, sparse, RRF fusion, and full-coverage scroll retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.database.bm25_store import BM25Store
from app.database.qdrant_store import QdrantStore
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.product_resolver import ProductResolver
from app.retrieval.rrf_fusion import rrf_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        bm25: BM25Store,
        embedding: OllamaEmbedding,
    ):
        self.settings = settings
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedding = embedding
        self.resolver = ProductResolver()

        self.dense = DenseRetriever(settings, qdrant, embedding)
        self.sparse = BM25Retriever(settings, bm25)

    @staticmethod
    def _round_robin(
        result_groups: List[List[Dict[str, Any]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Merge result groups while preserving product diversity."""
        merged: List[Dict[str, Any]] = []
        seen_chunk_ids = set()
        index = 0

        while len(merged) < limit:
            added = False

            for group in result_groups:
                if index >= len(group):
                    continue

                item = group[index]
                payload = item.get("payload", {})
                chunk_id = payload.get("chunk_id")

                if chunk_id not in seen_chunk_ids:
                    merged.append(item)
                    seen_chunk_ids.add(chunk_id)
                    added = True

                if len(merged) >= limit:
                    break

            if not added:
                break

            index += 1

        return merged

    def retrieve_full(
        self,
        product_id: Optional[str] = None,
        content_type: Optional[str] = None,
        section_name: Optional[str] = None,
        heading_path: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve ALL matching chunks without top_k semantic limits via Qdrant scroll.

        Used for intent: product_full, section_full, compare_full.
        """
        filters: Dict[str, Any] = {}
        if product_id:
            filters["product_id"] = product_id
        if content_type:
            filters["content_type"] = content_type
        if source_document:
            filters["source_document"] = source_document

        records = self.qdrant.scroll_by_filter(filters)

        # Post-filter by section_name / heading_path if requested
        filtered_records = []
        for rec in records:
            p = rec.get("payload", {})
            sec = str(p.get("source_section", "")).lower()
            head = str(p.get("heading_path", "")).lower()

            if section_name and section_name.lower() not in sec and section_name.lower() not in head:
                continue
            if heading_path and heading_path.lower() not in head:
                continue

            filtered_records.append(rec)

        logger.info(
            "retrieve_full (filters=%s, section=%s) returned %d chunks",
            filters, section_name, len(filtered_records)
        )
        return filtered_records

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        product_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        debug = bool(kwargs.get("debug", False))
        product_match_type = str(
            kwargs.get("product_match_type", "explicit_filter")
        )
        k = top_k or self.settings.final_top_k

        resolved_products = self.resolver.resolve_all(query)

        if product_id:
            target_product_ids = [product_id]
            catalog_item = self.resolver.catalog_item(product_id) or {}
            resolved_products = [
                {
                    "product_id": product_id,
                    "product_name": catalog_item.get("product_name", ""),
                    "confidence": 1.0,
                    "match_type": product_match_type,
                }
            ]
        else:
            target_product_ids = [
                item["product_id"]
                for item in resolved_products
                if item.get("confidence", 0.0) >= 0.65
            ]

        all_dense_results: List[Dict[str, Any]] = []
        all_sparse_results: List[Dict[str, Any]] = []
        fused_groups: List[List[Dict[str, Any]]] = []
        applied_filters: List[Dict[str, Any]] = []

        # Embed the query text once. When a query mentions 2-3 products, the
        # loop below issues one Qdrant search per product_id, but the query
        # vector itself never changes — only the filter does. Re-embedding it
        # per product was N redundant Ollama round-trips for identical input.
        precomputed_vector = self.embedding.embed_query(query) if query else None

        # Multi-product retrieval: retrieve and fuse each product separately.
        if target_product_ids:
            for target_pid in target_product_ids:
                req_filters = dict(filters) if filters else {}
                req_filters["product_id"] = target_pid
                applied_filters.append(req_filters)

                logger.info(
                    "Applying hard product filter for '%s'",
                    target_pid,
                )

                dense_results = self.dense.retrieve(
                    query,
                    top_k=self.settings.dense_top_k,
                    filters=req_filters,
                    query_vector=precomputed_vector,
                )
                sparse_results = self.sparse.retrieve(
                    query,
                    top_k=self.settings.sparse_top_k,
                    filters=req_filters,
                )

                fused = rrf_fusion(
                    dense_results,
                    sparse_results,
                    rrf_k=self.settings.rrf_k,
                    target_product_id=target_pid,
                    query=query,
                )

                all_dense_results.extend(dense_results)
                all_sparse_results.extend(sparse_results)
                fused_groups.append(fused)

            if len(fused_groups) > 1:
                final_results = self._round_robin(fused_groups, k)
            else:
                final_results = fused_groups[0][:k]

        # No product resolved: search the entire collection.
        else:
            req_filters = dict(filters) if filters else {}
            applied_filters.append(req_filters)

            dense_results = self.dense.retrieve(
                query,
                top_k=self.settings.dense_top_k,
                filters=req_filters or None,
                query_vector=precomputed_vector,
            )
            sparse_results = self.sparse.retrieve(
                query,
                top_k=self.settings.sparse_top_k,
                filters=req_filters or None,
            )

            fused = rrf_fusion(
                dense_results,
                sparse_results,
                rrf_k=self.settings.rrf_k,
                target_product_id=None,
                query=query,
            )

            all_dense_results = dense_results
            all_sparse_results = sparse_results
            fused_groups = [fused]
            final_results = fused[:k]

        latency_ms = (time.perf_counter() - started_at) * 1000

        resolved_product = (
            resolved_products[0]
            if resolved_products
            else {
                "product_id": "",
                "product_name": "",
                "confidence": 0.0,
                "match_type": "none",
            }
        )

        response: Dict[str, Any] = {
            "query": query,
            "resolved_product": resolved_product,
            "resolved_products": resolved_products,
            "results": final_results,
            "dense_count": len(all_dense_results),
            "sparse_count": len(all_sparse_results),
            "latency_ms": latency_ms,
        }

        if debug:
            response["debug"] = {
                "dense_candidates": all_dense_results,
                "sparse_candidates": all_sparse_results,
                "rrf_results": [
                    item
                    for group in fused_groups
                    for item in group
                ],
                "applied_filters": applied_filters,
                "target_product_ids": target_product_ids,
                "resolved_products": resolved_products,
            }

        return response
