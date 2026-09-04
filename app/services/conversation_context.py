"""Resolve the active product for contextual follow-up questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from app.retrieval.product_resolver import ProductResolver
from app.schemas import ConversationTurn, SourceReference

FOLLOW_UP_MARKERS = (
    "còn ",
    "thế còn",
    "vậy còn",
    "nó ",
    "thiết bị này",
    "sản phẩm này",
    "cái này",
    "tính năng này",
    "trường này",
    "trạng thái xác nhận",
)

BROAD_QUERY_MARKERS = (
    "các loại",
    "danh sách",
    "liệt kê",
    "có bao nhiêu loại",
    "có những sản phẩm",
    "tất cả sản phẩm",
)


@dataclass(frozen=True)
class ConversationScope:
    """Resolved retrieval scope for one chat request."""

    query: str
    product_id: Optional[str]
    product_name: str = ""
    source: str = "none"
    is_follow_up: bool = False


class ConversationContextResolver:
    """Carry the last unambiguous product into a short follow-up question."""

    def __init__(self, product_resolver: ProductResolver):
        self.product_resolver = product_resolver

    def _catalog_name(self, product_id: str) -> str:
        for item in self.product_resolver.catalog:
            if item.get("product_id") == product_id:
                return str(item.get("product_name", ""))
        return ""

    @staticmethod
    def _unique_product(matches: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
        confident = [
            item
            for item in matches
            if item.get("product_id") and item.get("confidence", 0.0) >= 0.65
        ]
        product_ids = {item["product_id"] for item in confident}
        if len(product_ids) != 1:
            return None
        return max(confident, key=lambda item: item.get("confidence", 0.0))

    @staticmethod
    def _history_turns(
        history: Optional[Sequence[ConversationTurn]],
    ) -> list[ConversationTurn]:
        return list(history or [])[-8:]

    def _product_from_history(
        self,
        history: Optional[Sequence[ConversationTurn]],
    ) -> Optional[dict[str, Any]]:
        turns = self._history_turns(history)

        # A user's explicit product mention is the strongest conversational anchor.
        for preferred_role in ("user", "assistant"):
            for turn in reversed(turns):
                if turn.role != preferred_role:
                    continue
                match = self._unique_product(
                    self.product_resolver.resolve_all(turn.content)
                )
                if match:
                    return match
        return None

    @staticmethod
    def referenced_product_count(question: str) -> Optional[int]:
        normalized = " ".join(question.lower().strip().split())
        patterns = (
            (2, ("2 thiết bị", "hai thiết bị", "2 sản phẩm",
                 "hai sản phẩm", "hai cảm biến", "cả hai")),
            (3, ("3 thiết bị", "ba thiết bị", "3 sản phẩm",
                 "ba sản phẩm", "ba cảm biến", "cả ba")),
        )
        for count, markers in patterns:
            if any(marker in normalized for marker in markers):
                return count
        return None

    @classmethod
    def is_multi_product_reference(cls, question: str) -> bool:
        normalized = " ".join(question.lower().strip().split())
        if cls.referenced_product_count(question):
            return True
        return any(marker in normalized for marker in (
            "các thiết bị",
            "những thiết bị",
            "các sản phẩm",
            "những sản phẩm",
            "các cảm biến",
            "những cảm biến",
        ))

    def products_from_history(
        self,
        history: Optional[Sequence[ConversationTurn]],
        expected_count: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        turns = self._history_turns(history)

        for preferred_role in ("user", "assistant"):
            for turn in reversed(turns):
                if turn.role != preferred_role:
                    continue

                unique: dict[str, dict[str, Any]] = {}
                for item in self.product_resolver.resolve_all(turn.content):
                    product_id = item.get("product_id")
                    if product_id and item.get("confidence", 0.0) >= 0.65:
                        unique.setdefault(product_id, item)

                matches = list(unique.values())
                if expected_count is not None:
                    if len(matches) == expected_count:
                        return matches
                elif len(matches) >= 2:
                    return matches

        return []

    def _looks_like_follow_up(self, question: str) -> bool:
        normalized = " ".join(question.lower().strip().split())
        if not normalized:
            return False
        if any(marker in normalized for marker in BROAD_QUERY_MARKERS):
            return False
        if any(marker in normalized for marker in FOLLOW_UP_MARKERS):
            return True
        # Short questions without an explicit product/referential marker
        # default to "probably a continuation" — e.g. "Công suất tối đa
        # bao nhiêu?" (6 words) right after a product-specific turn.
        #
        # Confirmed real bug at the previous threshold of 14: a standalone
        # troubleshooting question with no product name and no referential
        # marker easily reaches 10-13 words in Vietnamese just by
        # describing the symptom (e.g. "Rèm không di chuyển dù động cơ
        # hoạt động, cần kiểm tra gì?" — 13 words) — this is a genuinely
        # NEW topic, not a continuation, yet the old threshold silently
        # inherited whatever product the previous turn was about. Directly
        # confirmed in testing: this exact question inherited "Cảm biến
        # chuyển động ánh sáng" from an earlier, unrelated turn and
        # answered as if the curtain-motor question were about the light
        # sensor, until the product name was typed explicitly. Genuine
        # one-line attribute continuations ("Nguồn cấp là gì?", "Điện áp
        # hoạt động là bao nhiêu?") stay comfortably under 8 words, so
        # lowering the bar here trades away essentially none of the
        # intended UX while closing off the demonstrated failure mode for
        # standalone questions that happen to lack an explicit product name.
        return len(normalized.split()) <= 8

    def resolve(
        self,
        question: str,
        explicit_product_id: Optional[str] = None,
        history: Optional[Sequence[ConversationTurn]] = None,
        compare_mode: bool = False,
    ) -> ConversationScope:
        """Resolve an effective query and product filter for retrieval."""
        if explicit_product_id:
            product_name = self._catalog_name(explicit_product_id)
            return ConversationScope(
                query=question,
                product_id=explicit_product_id,
                product_name=product_name,
                source="explicit_filter",
            )

        current_match = self.product_resolver.resolve_primary(question)
        if (
            current_match.get("product_id")
            and current_match.get("confidence", 0.0) >= 0.65
        ):
            return ConversationScope(
                query=question,
                product_id=current_match["product_id"],
                product_name=current_match.get("product_name", ""),
                source="current_question",
            )

        if compare_mode or not self._looks_like_follow_up(question):
            return ConversationScope(query=question, product_id=None)

        previous_match = self._product_from_history(history)
        if not previous_match:
            return ConversationScope(query=question, product_id=None)

        product_name = previous_match.get("product_name", "")
        rewritten_query = (
            f"{question}\n"
            f"Sản phẩm đang được hỏi: {product_name}."
        )
        return ConversationScope(
            query=rewritten_query,
            product_id=previous_match["product_id"],
            product_name=product_name,
            source="conversation_history",
            is_follow_up=True,
        )


_NO_CONTEXT_MARKERS = (
    "chưa cung cấp thông tin này",
    "chưa có dữ liệu trong tài liệu",
)


def filter_cited_sources(
    answer: str,
    sources: list[SourceReference],
) -> list[SourceReference]:
    """Return only sources the generated answer actually cites.

    Falls back to returning every source that was fed to the model when the
    answer is substantive but contains no SOURCE_N marker at all — that's a
    citation-format slip (the model forgot the "📚 Nguồn tham khảo:" footer
    required by the system prompt, or generation got cut off by max_tokens
    before reaching it), not a sign that no source was actually used to
    write the answer. Distinguishing this from a genuine "no data" answer
    (which correctly has zero citations, because nothing in context was
    relevant) matters: silently hiding real sources behind a formatting
    miss is worse for answer trustworthiness than showing the sources that
    were actually in context, even in the rare case where one of them
    wasn't strictly needed for this particular sentence.
    """
    import re

    cited_ids = {
        f"SOURCE_{number}"
        for number in re.findall(
            r"\bSOURCE[_\s-]?(\d+)\b",
            answer,
            flags=re.IGNORECASE,
        )
    }
    if cited_ids:
        return [source for source in sources if source.source_id in cited_ids]

    if not sources:
        return []

    normalized = (answer or "").lower()
    is_explicit_no_data = any(marker in normalized for marker in _NO_CONTEXT_MARKERS)
    if is_explicit_no_data:
        return []

    return list(sources)
