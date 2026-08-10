"""Reciprocal Rank Fusion (RRF) with Relevance Guard exact match boost."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def rrf_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    target_product_id: Optional[str] = None,
    query: str = "",
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}

    for item in dense_results:
        cid = item.get("payload", {}).get("chunk_id", str(item.get("id")))
        rank = item.get("dense_rank", 999)
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        payloads[cid] = item.get("payload", item)
        dense_ranks[cid] = rank
        if item.get("dense_score") is not None:
            dense_scores[cid] = item["dense_score"]

    for item in sparse_results:
        cid = item.get("payload", {}).get("chunk_id", str(item.get("id")))
        rank = item.get("sparse_rank", 999)
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        if cid not in payloads:
            payloads[cid] = item.get("payload", item)
        sparse_ranks[cid] = rank
        if item.get("sparse_score") is not None:
            sparse_scores[cid] = item["sparse_score"]

    # Intent boost theo loại nội dung
    q_norm = query.lower().strip()

    specification_terms = (
        "thông số",
        "thông số kỹ thuật",
        "nguồn",
        "kích thước",
        "nguồn cấp",
        "điện áp",
        "công suất",
        "giao tiếp",
        "truyền thông",
        "khả năng phát hiện",
        "phát hiện",
        "model",
        "nhiệt độ hoạt động",
        "độ ẩm",
        "khoảng cách phát hiện",
    )

    if any(term in q_norm for term in specification_terms):
        for cid, payload in payloads.items():
            if payload.get("content_type") == "specification":
                scores[cid] *= 2.5

    # Relevance Guard Boost
    if target_product_id:
        t_pid = target_product_id.lower().strip()
        for cid, payload in payloads.items():
            pid = str(payload.get("product_id", "")).lower().strip()
            if pid and (t_pid == pid or t_pid in pid or pid in t_pid):
                scores[cid] *= 1.5

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for cid in sorted_ids:
        fused.append(
            {
                "payload": payloads[cid],
                "rrf_score": scores[cid],
                "dense_rank": dense_ranks.get(cid),
                "dense_score": dense_scores.get(cid),
                "sparse_rank": sparse_ranks.get(cid),
                "sparse_score": sparse_scores.get(cid),
            }
        )

    return fused


def fuse_results(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    target_product_id: Optional[str] = None,
    query: str = "",
) -> List[Dict[str, Any]]:
    return rrf_fusion(
        dense_results,
        sparse_results,
        rrf_k,
        target_product_id,
        query,
    )


def rrf_fuse(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 10,
    target_product_id: Optional[str] = None,
    query: str = "",
) -> List[Dict[str, Any]]:
    """Backward-compatible RRF API with payload fields at the top level."""
    fused = rrf_fusion(
        dense_results,
        sparse_results,
        rrf_k=rrf_k,
        target_product_id=target_product_id,
        query=query,
    )[:top_k]

    flattened: List[Dict[str, Any]] = []
    for item in fused:
        row = dict(item.get("payload", {}))
        row.update({key: value for key, value in item.items() if key != "payload"})
        flattened.append(row)
    return flattened
