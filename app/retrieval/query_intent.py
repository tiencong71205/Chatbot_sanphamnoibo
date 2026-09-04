"""Query intent classifier for Hybrid RAG.

Categorizes user queries into distinct execution intents:
- fact: specific question about one product or general query.
- section_full: request full content of a specific section (e.g. installation, specs, troubleshooting).
- product_full: request all information of a single product.
- multi_product_fact: query asking for a shared field/criteria across multiple products without explicit comparison.
- compare_field: comparison of specific criteria across multiple products.
- compare_full: comprehensive comparison across products.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class QueryIntent(str, Enum):
    FACT = "fact"
    SECTION_FULL = "section_full"
    PRODUCT_FULL = "product_full"
    MULTI_PRODUCT_FACT = "multi_product_fact"
    COMPARE_FIELD = "compare_field"
    COMPARE_FULL = "compare_full"


@dataclass
class IntentResult:
    intent: QueryIntent
    requested_products: List[str] = field(default_factory=list)
    requested_sections: List[str] = field(default_factory=list)
    requested_fields: List[str] = field(default_factory=list)
    confidence: float = 1.0


# Keyword patterns for full section intent
SECTION_FULL_PATTERNS = [
    r"(?:toàn bộ|tất cả|chi tiết|đầy đủ)\s+(?:hướng dẫn|quy trình|các bước|mục)?\s*(lắp đặt|thông số|kỹ thuật|sự cố|lỗi|kết nối|cấu hình|tính năng|khẩu lệnh|automation|kịch bản)",
    r"hướng dẫn lắp đặt(?:\s+chi tiết|\s+đầy đủ|\s+toàn bộ)?",
    r"cho tôi (?:toàn bộ|tất cả|đầy đủ) (?:hướng dẫn|thông tin|nội dung|mục)?\s*(lắp đặt|thông số|kỹ thuật|sự cố|lỗi|kết nối|cấu hình)",
]

# Keyword patterns for full product intent
PRODUCT_FULL_PATTERNS = [
    r"trình bày (?:toàn bộ|tất cả|đầy đủ) thông tin",
    r"toàn bộ thông tin",
    r"tất cả thông tin",
    r"tổng hợp (?:toàn bộ|tất cả|đầy đủ) (?:về|thông tin)?",
    r"giới thiệu (?:toàn bộ|chi tiết|đầy đủ) (?:về)?",
    r"cho tôi (?:biết )?toàn bộ (?:thông tin|dữ liệu|chi tiết) (?:về|của)?",
]

# Keyword patterns for compare full intent
COMPARE_FULL_PATTERNS = [
    r"so sánh tổng thể",
    r"so sánh toàn bộ",
    r"so sánh đầy đủ",
    r"so sánh tất cả",
    r"so sánh tổng quan",
    r"đối chiếu tổng thể",
    r"đối chiếu toàn bộ",
]

# Keyword patterns for compare field intent
COMPARE_FIELD_PATTERNS = [
    r"so sánh",
    r"khác nhau",
    r"điểm khác",
    r"sự khác biệt",
    r"đối chiếu",
]

# Mapping Vietnamese query terms to canonical sections
SECTION_MAP = {
    "lắp đặt": ["lắp đặt", "cài đặt", "gắn", "treo", "thi công"],
    "thông số": ["thông số", "kỹ thuật", "specification", "kích thước", "điện áp", "công suất", "nguồn"],
    "sự cố": ["sự cố", "lỗi", "khắc phục", "faq", "troubleshoot", "không hoạt động", "hỏng"],
    "kết nối": ["kết nối", "cấu hình", "pairing", "thêm thiết bị", "ghép nối", "bluetooth", "mesh", "wifi"],
    "tính năng": ["tính năng", "chức năng", "feature", "công dụng"],
    "automation": ["automation", "kịch bản", "tự động", "nhà thông minh", "scene"],
    "khẩu lệnh": ["khẩu lệnh", "giọng nói", "google assistant", "alexa"],
}


def classify_intent(
    question: str,
    resolved_products: Optional[List[Dict[str, Any]]] = None,
) -> IntentResult:
    """Classify user query into QueryIntent with extracted products & sections."""
    q_norm = (question or "").lower().strip()
    products = [
        p.get("product_id", "")
        for p in (resolved_products or [])
        if p.get("product_id") and p.get("confidence", 0.0) >= 0.65
    ]
    # Remove duplicates preserving order
    unique_products = list(dict.fromkeys([p for p in products if p]))
    num_products = len(unique_products)

    # Detect sections mentioned in query
    detected_sections = []
    for canonical_sec, keywords in SECTION_MAP.items():
        if any(kw in q_norm for kw in keywords):
            detected_sections.append(canonical_sec)

    # 1. Check compare_full: "so sánh tổng thể" / "so sánh toàn bộ" OR "so sánh" with 2+ products & no specific field
    for pat in COMPARE_FULL_PATTERNS:
        if re.search(pat, q_norm, re.IGNORECASE):
            return IntentResult(
                intent=QueryIntent.COMPARE_FULL,
                requested_products=unique_products,
                requested_sections=detected_sections,
                confidence=1.0,
            )

    # Handle "so sánh" explicit query
    is_compare_query = any(re.search(pat, q_norm, re.IGNORECASE) for pat in COMPARE_FIELD_PATTERNS)

    if is_compare_query:
        # If asks for "so sánh tổng thể" or no specific section detected + 2+ products
        if num_products >= 2 and not detected_sections and any(w in q_norm for w in ["tất cả", "các sản phẩm", "ba sản phẩm", "hai sản phẩm"]):
            return IntentResult(
                intent=QueryIntent.COMPARE_FULL,
                requested_products=unique_products,
                requested_sections=[],
                confidence=0.95,
            )

        return IntentResult(
            intent=QueryIntent.COMPARE_FIELD,
            requested_products=unique_products,
            requested_sections=detected_sections,
            confidence=0.95,
        )

    # 2. Check product_full: "Trình bày toàn bộ thông tin...", "Toàn bộ thông tin Cảm biến..."
    for pat in PRODUCT_FULL_PATTERNS:
        if re.search(pat, q_norm, re.IGNORECASE):
            return IntentResult(
                intent=QueryIntent.PRODUCT_FULL,
                requested_products=unique_products,
                requested_sections=[],
                confidence=1.0,
            )

    # 3. Check section_full: "Cho tôi toàn bộ hướng dẫn lắp đặt...", "Toàn bộ thông số..."
    for pat in SECTION_FULL_PATTERNS:
        if re.search(pat, q_norm, re.IGNORECASE):
            return IntentResult(
                intent=QueryIntent.SECTION_FULL,
                requested_products=unique_products,
                requested_sections=detected_sections,
                confidence=1.0,
            )

    # Also detect section_full if "toàn bộ" / "tất cả" + specific section
    if ("toàn bộ" in q_norm or "tất cả" in q_norm or "chi tiết" in q_norm) and detected_sections:
        return IntentResult(
            intent=QueryIntent.SECTION_FULL,
            requested_products=unique_products,
            requested_sections=detected_sections,
            confidence=0.9,
        )

    # 4. Check multi_product_fact: 2+ products mentioned without "so sánh"
    # Example: "Hai cảm biến dùng chuẩn kết nối gì?"
    if num_products >= 2 or any(term in q_norm for term in ["hai cảm biến", "ba sản phẩm", "cả hai", "cả ba"]):
        return IntentResult(
            intent=QueryIntent.MULTI_PRODUCT_FACT,
            requested_products=unique_products,
            requested_sections=detected_sections,
            confidence=0.9,
        )

    # 5. Default: fact query
    return IntentResult(
        intent=QueryIntent.FACT,
        requested_products=unique_products,
        requested_sections=detected_sections,
        confidence=0.8,
    )
