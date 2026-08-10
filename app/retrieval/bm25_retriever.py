"""BM25 sparse retriever."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.database.bm25_store import BM25Store

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, settings: Settings, bm25: BM25Store):
        self.settings = settings
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        filter_product_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.settings.sparse_top_k
        pid = filter_product_id or (filters.get("product_id") if filters else None)

        raw_results = self.bm25.search(query, top_k=k * 2)

        output = []
        for r in raw_results:
            payload = r.get("chunk", {})
            if pid:
                p_pid = str(payload.get("product_id", "")).lower().strip()
                t_pid = str(pid).lower().strip()
                if p_pid and t_pid and p_pid != t_pid and t_pid not in p_pid and p_pid not in t_pid:
                    continue
            output.append({
                "payload": payload,
                "score": r.get("score", 0.0),
                "rank": r.get("rank"),
            })
            if len(output) >= k:
                break

        for i, r in enumerate(output):
            r["sparse_rank"] = i + 1
            r["sparse_score"] = r.pop("score", 0.0)

        logger.debug("BM25 retrieval: %d results for query '%s...'", len(output), query[:50])
        return output
