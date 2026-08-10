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
    "vậy ",
    "thế ",
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

    def _looks_like_follow_up(self, question: str) -> bool:
        normalized = " ".join(question.lower().strip().split())
        if not normalized:
            return False
        if any(marker in normalized for marker in BROAD_QUERY_MARKERS):
            return False
        if any(marker in normalized for marker in FOLLOW_UP_MARKERS):
            return True
        # Short questions without an explicit product are normally continuations,
        # e.g. "Công suất tối đa bao nhiêu?" after a product-specific turn.
        return len(normalized.split()) <= 14

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

        current_match = self._unique_product(
            self.product_resolver.resolve_all(question)
        )
        if current_match:
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


def filter_cited_sources(
    answer: str,
    sources: list[SourceReference],
) -> list[SourceReference]:
    """Return only sources the generated answer actually cites."""
    import re

    cited_ids = {
        f"SOURCE_{number}"
        for number in re.findall(
            r"\bSOURCE[_\s-]?(\d+)\b",
            answer,
            flags=re.IGNORECASE,
        )
    }
    if not cited_ids:
        return sources
    return [source for source in sources if source.source_id in cited_ids]
