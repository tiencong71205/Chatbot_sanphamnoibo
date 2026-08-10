"""Build soft/hard metadata filters for retrieval."""
from __future__ import annotations
from typing import Dict, Optional

from app.retrieval.product_resolver import ResolvedProduct

HARD_CONFIDENCE = 0.7


def build_qdrant_filter(
    resolved: ResolvedProduct,
) -> Optional[Dict]:
    """
    Build Qdrant filter dict.
    Hard filter if confidence >= HARD_CONFIDENCE.
    Soft filter (no filter) if confidence < HARD_CONFIDENCE.
    Returns None if no filter should be applied.
    """
    if not resolved.product_id and not resolved.model:
        return None

    if resolved.confidence < HARD_CONFIDENCE:
        # Soft filter: return None (no filter, RRF will handle ranking)
        return None

    filters: Dict = {}
    if resolved.product_id:
        filters["product_id"] = resolved.product_id
    return filters if filters else None


def should_boost(resolved: ResolvedProduct) -> bool:
    """True if we should boost matching chunks in RRF but not hard-filter."""
    return 0.3 <= resolved.confidence < HARD_CONFIDENCE
