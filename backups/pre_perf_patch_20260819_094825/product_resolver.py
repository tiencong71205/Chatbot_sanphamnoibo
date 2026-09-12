"""Resolve one or multiple products from the product catalog."""

import difflib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CATALOG_PATHS = [
    Path(os.environ.get("PRODUCT_CATALOG_PATH", "/app/data/product_catalog.json")),
    Path("/app/data/product_catalog.json"),
    Path("./data/product_catalog.json"),
    Path("data/product_catalog.json"),
]


@dataclass
class ResolvedProduct:
    """Backward-compatible result used by metadata-filter helpers."""

    product_id: Optional[str] = None
    product_name: Optional[str] = None
    model: Optional[str] = None
    confidence: float = 0.0


_MODEL_RE = re.compile(
    r"\b([A-Z]{2,}[-][A-Z0-9]+(?:[-][A-Z0-9]+)*)\b",
    re.IGNORECASE,
)

_CONNECTION_TERMS = (
    "kết nối",
    "ghép nối",
    "thêm thiết bị",
    "đăng ký thiết bị",
)

_RELATION_MARKERS = (
    " thông qua ",
    " với ",
    " qua ",
    " bằng ",
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # Strip common sentence punctuation so it doesn't stick to an adjacent
    # word (e.g. "Dimmer," -> "dimmer") and dilute n-gram window matching in
    # _best_ngram_similarity. Keep "/" since product names sometimes use it
    # (e.g. "Tăng/Giảm" as a button label, not part of a product name).
    text = re.sub(r"[.,;:!?()\[\]\"“”'’]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        # Containment alone isn't enough — a short, generic fragment (e.g.
        # "công tắc", 2 words) is a substring of many different product
        # names ("Công tắc thông minh", "Công tắc thông minh dimmer",
        # "Công tắc chống giật..."), so unconditional high-confidence
        # containment previously caused a short window to false-match
        # whichever unrelated product got iterated first. Scale by how much
        # of the longer string the shorter one actually covers: a near-full
        # match still scores ~0.9, but a trivial fragment scores much lower
        # and won't cross the resolution threshold on its own.
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        coverage = len(shorter) / max(1, len(longer))
        return 0.5 + 0.4 * coverage
    return difflib.SequenceMatcher(None, a, b).ratio()


def _best_ngram_similarity(candidate: str, query_words: List[str]) -> float:
    """Best similarity between `candidate` and any contiguous word-window of
    the query, instead of the whole raw sentence.

    difflib.SequenceMatcher.ratio() over a short candidate name (a few words)
    vs. an entire long question is dominated by whatever characters happen to
    overlap across the whole sentence — mostly noise, and confirmed in
    practice to pick essentially arbitrary wrong products for real questions
    (e.g. "Công tắc Dimmer" fuzzy-matched onto "Cảm biến chuyển động ánh
    sáng"). Comparing against sliding windows sized close to the candidate's
    own word count is far closer to how a person spots a product name inside
    a sentence, and scores the correct product much more reliably (e.g. the
    window "công tắc dimmer" vs. candidate "công tắc thông minh dimmer"
    scores ~0.73, well above threshold, where whole-sentence comparison
    scored too low to ever match).
    """
    if not candidate or not query_words:
        return similarity_ratio(candidate, " ".join(query_words))

    candidate_len = max(1, len(candidate.split()))
    best = 0.0
    # Try windows both shorter and longer than the candidate's own word
    # count: the query may omit a word the official name has (e.g. "Công
    # tắc Dimmer" vs. catalog's "Công tắc thông minh dimmer" — 3 words vs
    # 4), or add one. A window range fixed at exactly candidate_len-and-up
    # misses the shorter, better-matching phrase entirely.
    min_window = max(1, candidate_len - 2)
    max_window = candidate_len + 2
    for window_size in range(min_window, max_window + 1):
        for start in range(0, max(1, len(query_words) - window_size + 1)):
            window = " ".join(query_words[start:start + window_size])
            if not window:
                continue
            score = similarity_ratio(candidate, window)
            if score > best:
                best = score
    return best


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

            product_name = normalize_text(item.get("product_name", ""))
            if product_name and product_name in q_norm:
                matches[product_id] = self._result(item, 1.0, "exact_name")
                continue

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
        query_words = q_norm.split()

        for item in self.catalog:
            candidates = (
                [item["product_name"]]
                + item.get("aliases", [])
                + item.get("keywords", [])
            )

            for candidate in candidates:
                score = _best_ngram_similarity(
                    normalize_text(candidate),
                    query_words,
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

    def catalog_item(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return one catalog row without exposing catalog traversal to callers."""
        return next(
            (
                item
                for item in self.catalog
                if str(item.get("product_id", "")) == product_id
            ),
            None,
        )

    def resolve_primary(self, query: str) -> Dict[str, Any]:
        """Resolve the product being operated on in a relational question.

        For example, in ``kết nối Cảm biến cửa Mesh với Gateway`` the sensor
        is the primary product while Gateway is the connection target.  A
        normal multi-product fact or comparison still remains multi-product
        through :meth:`resolve_all`.
        """
        matches = self.resolve_all(query)
        if len(matches) <= 1:
            return matches[0] if matches else {
                "product_id": "",
                "product_name": "",
                "confidence": 0.0,
                "match_type": "none",
            }

        normalized = normalize_text(query)
        if any(term in normalized for term in _CONNECTION_TERMS):
            marker_positions = [
                normalized.find(marker)
                for marker in _RELATION_MARKERS
                if normalized.find(marker) >= 0
            ]
            if marker_positions:
                prefix = normalized[: min(marker_positions)]
                prefix_matches = [
                    item
                    for item in self.resolve_all(prefix)
                    if item.get("product_id")
                    and item.get("confidence", 0.0) >= 0.65
                ]
                if len(prefix_matches) == 1:
                    return prefix_matches[0]

        return {
            "product_id": "",
            "product_name": "",
            "confidence": 0.0,
            "match_type": "multiple",
        }

    def resolve(self, query: str) -> Dict[str, Any]:
        """Backward-compatible single-product resolution."""
        return self.resolve_primary(query)


def resolve_product(
    query: str,
    product_id_hint: Optional[str] = None,
) -> ResolvedProduct:
    """Compatibility API retained for older tests and metadata filters."""
    if product_id_hint:
        return ResolvedProduct(product_id=product_id_hint, confidence=1.0)

    match = ProductResolver().resolve_primary(query)
    model_match = _MODEL_RE.search(query)
    model = model_match.group(1).upper() if model_match else None

    if match.get("product_id"):
        item = ProductResolver().catalog_item(str(match["product_id"]))
        return ResolvedProduct(
            product_id=str(match["product_id"]),
            product_name=str(match.get("product_name") or ""),
            model=str(item.get("model") or "") if item else model,
            confidence=float(match.get("confidence", 0.0)),
        )

    return ResolvedProduct(model=model, confidence=0.6 if model else 0.0)
