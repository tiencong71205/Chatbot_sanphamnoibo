"""Resolve one or multiple products from the product catalog."""

import difflib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CATALOG_PATHS = [
    Path(os.environ.get("PRODUCT_CATALOG_PATH", "/app/data/product_catalog.json")),
    Path("/app/data/product_catalog.json"),
    Path("./data/product_catalog.json"),
    Path("data/product_catalog.json"),
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.90
    return difflib.SequenceMatcher(None, a, b).ratio()


class ProductResolver:
    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog: List[Dict[str, Any]] = []

        if catalog_path and catalog_path.exists():
            search_paths = [catalog_path]
        else:
            search_paths = _CATALOG_PATHS

        for p in search_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as file:
                    self.catalog = json.load(file)
                logger.info("ProductResolver loaded catalog from: %s (%d products)", p, len(self.catalog))
                break

        if not self.catalog:
            logger.warning("ProductResolver: no catalog found at any path")

    @staticmethod
    def _result(
        item: Dict[str, Any],
        confidence: float,
        match_type: str,
    ) -> Dict[str, Any]:
        return {
            "product_id": item["product_id"],
            "product_name": item["product_name"],
            "confidence": confidence,
            "match_type": match_type,
        }

    def resolve_all(self, query: str) -> List[Dict[str, Any]]:
        """Return every product explicitly mentioned in the query."""
        q_norm = normalize_text(query)

        if not q_norm or not self.catalog:
            return []

        matches: Dict[str, Dict[str, Any]] = {}

        # Model, alias and typo matches may identify multiple products.
        for item in self.catalog:
            product_id = item["product_id"]

            model = normalize_text(item.get("model", ""))
            if (
                model
                and model not in {
                    "chưa xác định - cần xác nhận",
                    "chưa có dữ liệu",
                    "chưa có dữ liệu trong tài liệu",
                }
                and model in q_norm
            ):
                matches[product_id] = self._result(item, 1.0, "model")
                continue

            for alias in item.get("aliases", []):
                alias_norm = normalize_text(alias)
                if alias_norm and alias_norm in q_norm:
                    matches[product_id] = self._result(
                        item,
                        0.95,
                        "exact_alias",
                    )
                    break

            if product_id in matches:
                continue

            for typo in item.get("common_typos", []):
                typo_norm = normalize_text(typo)
                if typo_norm and typo_norm in q_norm:
                    matches[product_id] = self._result(
                        item,
                        0.90,
                        "typo",
                    )
                    break

        if matches:
            return list(matches.values())

        # Fuzzy fallback is only used when no explicit product is found.
        best_score = 0.0
        best_item: Optional[Dict[str, Any]] = None

        for item in self.catalog:
            candidates = (
                [item["product_name"]]
                + item.get("aliases", [])
                + item.get("keywords", [])
            )

            for candidate in candidates:
                score = similarity_ratio(
                    normalize_text(candidate),
                    q_norm,
                )
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item and best_score >= 0.65:
            return [
                self._result(
                    best_item,
                    round(best_score, 2),
                    "fuzzy",
                )
            ]

        return []

    def resolve(self, query: str) -> Dict[str, Any]:
        """Backward-compatible single-product resolution."""
        matches = self.resolve_all(query)

        if matches:
            return matches[0]

        return {
            "product_id": "",
            "confidence": 0.0,
            "match_type": "none",
        }
