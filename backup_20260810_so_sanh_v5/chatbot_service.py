"""Chatbot service — orchestrates retrieve + generate with history and compare mode."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.generation.context_builder import ContextBuilder
from app.generation.comparison_formatter import repair_comparison_table
from app.generation.ollama_generator import OllamaGenerator
from app.generation.prompts import COMPARE_SYSTEM_PROMPT
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas import ChatResponse, ConversationTurn, RetrieveResponse, SourceReference
from app.services.conversation_context import (
    ConversationContextResolver,
    filter_cited_sources,
)

logger = logging.getLogger(__name__)


class ChatbotService:
    _COMPARE_MIN_CHUNKS_PER_PRODUCT = 2
    _COMPARE_TARGET_MIN_CHUNKS_PER_PRODUCT = 3
    _COMPARE_TARGET_MAX_CHUNKS_PER_PRODUCT = 5
    _COMPARE_MAX_CHUNKS_PER_PRODUCT = 6
    _COMPARE_MAX_TOTAL_CHUNKS = 20

    _COMPARE_FOCUS = (
        "thông số kỹ thuật nguồn cấp điện áp truyền thông giao tiếp "
        "công suất đầu ra nút bấm điều khiển tại chỗ điều khiển từ xa "
        "điều khiển giọng nói hẹn giờ tính năng chính"
    )
    _COMPARE_FIELD_TERMS = {
        "identity": ("model", "mã sản phẩm", "mục đích", "loại sản phẩm"),
        "power": ("nguồn", "điện áp", "vac", "vdc", "pin"),
        "communication": (
            "truyền thông", "giao tiếp", "wi-fi", "wifi", "bluetooth",
            "mesh", "zigbee", "rf433", "ethernet",
        ),
        "output": (
            "công suất", "đầu ra", "relay", "tiếp điểm", "dòng tải",
            "tải led", "tải thuần trở", "kênh",
        ),
        "local_control": (
            "tại chỗ", "nút bấm", "nút cảm ứng", "cảm ứng điện dung",
        ),
        "remote_control": ("từ xa", "ứng dụng", "vhomenex"),
        "voice": ("giọng nói", "google assistant", "amazon alexa"),
        "schedule": ("hẹn giờ", "lịch", "timer"),
        "feature": (
            "tính năng", "cảnh báo", "tự động", "automation", "kịch bản",
            "khóa trẻ em", "dự phòng", "ban đêm",
        ),
    }

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: OllamaGenerator,
    ):
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.context_builder = ContextBuilder(settings)
        self.conversation_context = ConversationContextResolver(retriever.resolver)

    def _build_source_references(
        self, selected: List[Dict[str, Any]], start: int = 1
    ) -> List[SourceReference]:
        sources: List[SourceReference] = []
        for i, item in enumerate(selected, start=start):
            payload = item.get("payload", {})
            sources.append(
                SourceReference(
                    source_id=f"SOURCE_{i}",
                    product_name=payload.get("product_name", ""),
                    product_id=payload.get("product_id", ""),
                    product_group=payload.get("product_group", ""),
                    source_document=payload.get("source_document", "") or payload.get("source_file", ""),
                    source_section=payload.get("source_section", "") or payload.get("title", ""),
                    content_type=payload.get("content_type", ""),
                    content=payload.get("content", "")[:500],
                    chunk_id=payload.get("chunk_id", ""),
                    dense_rank=item.get("dense_rank"),
                    dense_score=item.get("dense_score"),
                    sparse_rank=item.get("sparse_rank"),
                    sparse_score=item.get("sparse_score"),
                    rrf_score=item.get("rrf_score"),
                )
            )
        return sources

    def chat(
        self,
        question: str,
        product_id: Optional[str] = None,
        history: Optional[List[ConversationTurn]] = None,
        compare_mode: bool = False,
        debug: bool = False,
    ) -> ChatResponse:
        """Full RAG pipeline: retrieve → build context → generate."""
        t0 = time.perf_counter()

        # Convert history to dict format for generator.
        history_dicts = None
        if history:
            history_dicts = [{"role": h.role, "content": h.content} for h in history]

        # Câu hỏi liệt kê sản phẩm theo nhóm
        normalized_question = question.lower().strip()
        explicit_products = [
            item
            for item in self.retriever.resolver.resolve_all(question)
            if item.get("product_id") and item.get("confidence", 0.0) >= 0.65
        ]
        compare_mode = compare_mode or any(
            term in normalized_question
            for term in ("so sánh", "khác nhau", "đối chiếu")
        )
        product_list_terms = (
            "các loại",
            "danh sách",
            "có những",
            "gồm những",
            "có bao nhiêu loại",
            "liệt kê",
        )

        group_aliases = {
            "cảm biến": "Cảm biến",
            "công tắc": "Công tắc",
            "bộ điều khiển": "Bộ điều khiển",
            "khóa": "Khóa thông minh",
            "động cơ": "Động cơ thông minh",
            "thiết bị đo": "Thiết bị đo lường",
        }

        requested_group = next(
            (
                catalog_group
                for phrase, catalog_group in group_aliases.items()
                if phrase in normalized_question
            ),
            None,
        )

        if (
            requested_group
            and any(term in normalized_question for term in product_list_terms)
            # "Công tắc cửa cuốn ... có những cách điều khiển nào?" is a
            # product question, not a request to list the whole Công tắc group.
            and not explicit_products
        ):
            catalog_path = Path(self.settings.product_catalog_path)
            if not catalog_path.exists():
                for alt in ["/app/data/product_catalog.json", "./data/product_catalog.json"]:
                    if Path(alt).exists():
                        catalog_path = Path(alt)
                        break

            if catalog_path.exists():
                catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                products = [
                    item
                    for item in catalog
                    if item.get("product_group") == requested_group
                ]

                if products:
                    lines = [
                        f"Hiện có {len(products)} sản phẩm thuộc nhóm "
                        f"**{requested_group}**:"
                    ]
                    for index, item in enumerate(products, start=1):
                        name = item.get("product_name", "Chưa xác định")
                        model = item.get("model", "Chưa xác định")
                        keywords = item.get("keywords", [])
                        keyword_text = ", ".join(keywords[:3])
                        detail = f" — Model: **{model}**"
                        if keyword_text:
                            detail += f". Đặc điểm: {keyword_text}."
                        lines.append(f"{index}. **{name}**{detail}")

                    lines.append(
                        "\nBạn có thể chọn một sản phẩm để xem thông số, "
                        "cách kết nối, lắp đặt hoặc tính năng chi tiết."
                    )

                    total_latency = (time.perf_counter() - t0) * 1000
                    return ChatResponse(
                        answer="\n\n".join(lines),
                        sources=[],
                        debug_info={
                            "intent": "product_list",
                            "product_group": requested_group,
                            "product_count": len(products),
                        } if debug else None,
                        latency_ms=total_latency,
                    )

        # COMPARE MODE: retrieve per product, then generate comparison table.
        if compare_mode:
            return self._compare_chat(
                question,
                product_id,
                history_dicts,
                debug,
                t0,
                explicit_products=explicit_products,
            )

        # A question can ask for one fact across several named products without
        # containing the word "so sánh", e.g. the Internet-loss LED state of
        # the IR controller and curtain motor. Retrieve each named product with
        # a hard filter so an unrelated high-scoring product cannot take over.
        if len(explicit_products) > 1:
            return self._multi_product_chat(
                question,
                explicit_products,
                history_dicts,
                debug,
                t0,
            )

        # Carry the active product from conversation history into a contextual
        # follow-up. This scope is used by retrieval as a hard product filter.
        scope = self.conversation_context.resolve(
            question=question,
            explicit_product_id=product_id,
            history=history,
            compare_mode=False,
        )
        effective_query = scope.query
        effective_product_id = scope.product_id

        # Standard retrieve
        t_retrieval_start = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=effective_query,
            product_id=effective_product_id,
            top_k=self.settings.final_top_k,
            debug=debug,
        )
        retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000
        results = retrieval["results"]

        # Intent-based result filtering
        if any(term in normalized_question for term in ("thông số", "thông tin kỹ thuật", "specification")):
            spec_results = [r for r in results if r.get("payload", {}).get("content_type") == "specification"]
            if spec_results:
                results = spec_results

        if any(term in normalized_question for term in ("tổng quan", "giới thiệu", "sản phẩm dùng để")):
            overview_results = [r for r in results if r.get("payload", {}).get("content_type") == "overview"]
            if not overview_results:
                expanded = self.retriever.retrieve(query=question, product_id=product_id, top_k=20, debug=False)
                overview_results = [r for r in expanded["results"] if r.get("payload", {}).get("content_type") == "overview"]
            if overview_results:
                results = overview_results[:3]

        if any(term in normalized_question for term in ("reset", "cài đặt lại", "khôi phục", "đặt lại")):
            reset_results = [
                r for r in results
                if r.get("payload", {}).get("content_type") == "configuration"
                and any(t in (r.get("payload", {}).get("content", "") + " " + r.get("payload", {}).get("title", "")).lower()
                        for t in ("reset", "cài đặt lại", "khôi phục"))
            ]
            if reset_results:
                results = reset_results

        # Build context and generate
        generation_question = question
        if scope.is_follow_up and scope.product_name:
            generation_question = (
                f"{question}\n\n"
                f"Ngữ cảnh hội thoại cần giữ: sản phẩm đang hỏi là "
                f"{scope.product_name}. Chỉ trả lời cho sản phẩm này."
            )
        user_message, selected = self.context_builder.build(
            results,
            generation_question,
        )

        t_gen_start = time.perf_counter()
        gen_result = self.generator.generate(user_message, history=history_dicts)
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        answer = gen_result["answer"]
        sources = filter_cited_sources(
            answer,
            self._build_source_references(selected),
        )
        total_latency = (time.perf_counter() - t0) * 1000

        debug_info: Optional[Dict[str, Any]] = None
        if debug:
            debug_info = {
                "resolved_product": retrieval.get("resolved_product"),
                "resolved_products": retrieval.get("resolved_products", []),
                "dense_candidates": retrieval.get("debug", {}).get("dense_candidates", []),
                "sparse_candidates": retrieval.get("debug", {}).get("sparse_candidates", []),
                "rrf_results": retrieval.get("debug", {}).get("rrf_results", []),
                "context_chunks": len(selected),
                "effective_query": effective_query,
                "conversation_scope": {
                    "product_id": effective_product_id,
                    "product_name": scope.product_name,
                    "source": scope.source,
                    "is_follow_up": scope.is_follow_up,
                },
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": generation_ms,
            }

        return ChatResponse(
            answer=answer,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def _compare_chat(
        self,
        question: str,
        product_id: Optional[str],
        history_dicts: Optional[List[Dict[str, Any]]],
        debug: bool,
        t0: float,
        explicit_products: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Compare mode: retrieve broadly then generate comparison table."""
        # Retrieve each explicitly named product independently. This guarantees
        # that every compared product contributes context, including its
        # specification chunk for source/power/communication questions.
        targets = explicit_products or []
        if product_id:
            targets = [{"product_id": product_id, "product_name": ""}]

        if len(targets) > 1:
            results, retrieval_debug = self._retrieve_named_products(
                question,
                targets,
                per_product_k=max(self.settings.final_top_k, 10),
                debug=debug,
                comparison=True,
            )
        else:
            retrieval = self.retriever.retrieve(
                query=question,
                product_id=product_id,
                top_k=min(self.settings.final_top_k * 2, 20),
                debug=debug,
            )
            results = retrieval["results"]
            retrieval_debug = retrieval.get("debug", {})
        coverage_lines = []
        requested_fields = self._comparison_fields(question)
        for target in targets:
            target_pid = str(target.get("product_id", ""))
            target_results = [
                item for item in results
                if item.get("payload", {}).get("product_id") == target_pid
            ]
            has_specification = any(
                item.get("payload", {}).get("content_type") == "specification"
                for item in target_results
            )
            covered_fields = set().union(
                *(self._covered_comparison_fields(item) for item in target_results)
            ) if target_results else set()
            missing_fields = requested_fields - covered_fields
            coverage_lines.append(
                f"- {target.get('product_name') or target_pid}: "
                f"{len(target_results)} chunk; thông số kỹ thuật = "
                f"{'CÓ' if has_specification else 'KHÔNG CÓ'}; "
                f"trường chưa có bằng chứng = "
                f"{', '.join(sorted(missing_fields)) if missing_fields else 'không'}"
            )

        comparison_question = (
            f"{question}\n\n"
            "KIỂM TRA ĐỘ PHỦ TRƯỚC KHI TRẢ LỜI:\n"
            + "\n".join(coverage_lines)
            + "\nChỉ ghi 'Chưa có dữ liệu' sau khi đã kiểm tra toàn bộ "
              "chunk của đúng sản phẩm trong ngữ cảnh."
        )
        user_message, selected = self.context_builder.build(
            results,
            comparison_question,
        )

        gen_result = self.generator.generate(
            user_message,
            system_prompt=COMPARE_SYSTEM_PROMPT,
            history=history_dicts,
        )
        answer = repair_comparison_table(
            gen_result["answer"],
            [str(item.get("product_name", "")) for item in targets],
        )
        sources = filter_cited_sources(
            answer,
            self._build_source_references(selected),
        )
        total_latency = (time.perf_counter() - t0) * 1000

        return ChatResponse(
            answer=answer,
            sources=sources,
            debug_info={
                "mode": "compare",
                "context_chunks": len(selected),
                "named_products": [item.get("product_id") for item in targets],
                "per_product_retrieval": retrieval_debug,
            } if debug else None,
            latency_ms=total_latency,
        )

    def _retrieve_named_products(
        self,
        question: str,
        products: List[Dict[str, Any]],
        per_product_k: int,
        debug: bool,
        comparison: bool = False,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve context for every explicitly named product."""
        groups: List[List[Dict[str, Any]]] = []
        debug_rows: List[Dict[str, Any]] = []

        for product in products:
            product_id = str(product.get("product_id", ""))
            product_name = str(product.get("product_name", ""))
            product_query = f"{question}\nSản phẩm cần lấy dữ liệu: {product_name}."
            if comparison:
                product_query += f"\nCác trường bắt buộc ưu tiên: {self._COMPARE_FOCUS}."
            retrieval = self.retriever.retrieve(
                query=product_query,
                product_id=product_id,
                top_k=per_product_k,
                debug=debug,
            )
            product_results = retrieval.get("results", [])

            # A comparison must inspect the specification chunk before it may
            # conclude that power, communication or output data is missing.
            if comparison and not any(
                item.get("payload", {}).get("content_type") == "specification"
                for item in product_results
            ):
                spec_retrieval = self.retriever.retrieve(
                    query=(
                        f"{product_name}: thông số kỹ thuật, nguồn cấp, điện áp, "
                        "truyền thông, công suất, đầu ra, nút bấm"
                    ),
                    product_id=product_id,
                    top_k=5,
                    debug=debug,
                )
                product_results = self._deduplicate_results(
                    product_results + spec_retrieval.get("results", [])
                )

            if comparison:
                product_results = self._select_comparison_evidence(
                    product_results,
                    question,
                )

            groups.append(product_results)
            if debug:
                debug_rows.append(
                    {
                        "product_id": product_id,
                        "result_count": len(product_results),
                        "has_specification": any(
                            item.get("payload", {}).get("content_type") == "specification"
                            for item in product_results
                        ),
                        "retrieval_debug": retrieval.get("debug", {}),
                    }
                )

        # Interleave groups so all products remain visible even if a later
        # context limiter is introduced.
        max_group_size = max((len(group) for group in groups), default=0)
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for index in range(max_group_size):
            for group in groups:
                if index >= len(group):
                    continue
                item = group[index]
                payload = item.get("payload", {})
                key = str(payload.get("chunk_id") or id(item))
                if key not in seen:
                    merged.append(item)
                    seen.add(key)
                    if comparison and len(merged) >= self._COMPARE_MAX_TOTAL_CHUNKS:
                        return merged, debug_rows

        return merged, debug_rows

    @staticmethod
    def _deduplicate_results(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            payload = item.get("payload", {})
            key = str(payload.get("chunk_id") or id(item))
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return unique

    @classmethod
    def _comparison_fields(cls, question: str) -> set[str]:
        """Return fields explicitly requested, or the safe default comparison set."""
        normalized = question.lower()
        requested = {
            field
            for field, terms in cls._COMPARE_FIELD_TERMS.items()
            if any(term in normalized for term in terms)
        }
        if requested:
            return requested
        return set(cls._COMPARE_FIELD_TERMS)

    @classmethod
    def _covered_comparison_fields(
        cls,
        item: Dict[str, Any],
    ) -> set[str]:
        payload = item.get("payload", {})
        text = " ".join(
            str(payload.get(field, ""))
            for field in ("source_section", "title", "content_type", "content")
        ).lower()
        return {
            field
            for field, terms in cls._COMPARE_FIELD_TERMS.items()
            if any(term in text for term in terms)
        }

    @classmethod
    def _select_comparison_evidence(
        cls,
        results: List[Dict[str, Any]],
        question: str,
    ) -> List[Dict[str, Any]]:
        """Select 2-6 adaptive chunks for one compared product.

        The selector keeps technical specifications and overview evidence,
        reaches a normal target of 3 chunks, then adds chunks only when they
        cover comparison fields that are still missing. Five chunks is the
        normal ceiling; a sixth is allowed only for unresolved coverage.
        """
        results = cls._deduplicate_results(results)
        if len(results) <= cls._COMPARE_MIN_CHUNKS_PER_PRODUCT:
            return results

        selected: List[Dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(item: Dict[str, Any]) -> None:
            payload = item.get("payload", {})
            key = str(payload.get("chunk_id") or id(item))
            if (
                key not in selected_ids
                and len(selected) < cls._COMPARE_MAX_CHUNKS_PER_PRODUCT
            ):
                selected.append(item)
                selected_ids.add(key)

        # Technical tables are the authoritative source for core comparison
        # fields; the overview usually summarizes purpose and standout features.
        for required_type in ("specification", "overview"):
            candidates = [
                item for item in results
                if item.get("payload", {}).get("content_type") == required_type
            ]
            if candidates:
                add(max(candidates, key=lambda item: item.get("rrf_score") or 0.0))

        requested_fields = cls._comparison_fields(question)
        focus_terms = {
            token
            for token in (question.lower() + " " + cls._COMPARE_FOCUS).split()
            if len(token) >= 4
        }

        def relevance(item: Dict[str, Any]) -> tuple[int, float]:
            payload = item.get("payload", {})
            text = " ".join(
                str(payload.get(field, ""))
                for field in ("source_section", "title", "content_type", "content")
            ).lower()
            overlap = sum(1 for term in focus_terms if term in text)
            return overlap, float(item.get("rrf_score") or 0.0)

        ranked = sorted(results, key=relevance, reverse=True)

        # Never return fewer than two chunks when the retriever has enough,
        # and normally start generation with at least three chunks.
        for target_size in (
            cls._COMPARE_MIN_CHUNKS_PER_PRODUCT,
            cls._COMPARE_TARGET_MIN_CHUNKS_PER_PRODUCT,
        ):
            for item in ranked:
                add(item)
                if len(selected) >= target_size:
                    break

        def covered() -> set[str]:
            return set().union(
                *(cls._covered_comparison_fields(item) for item in selected)
            ) if selected else set()

        # Grow through the normal 3-5 target only when a new chunk supplies a
        # requested field that is not already represented in the context.
        while len(selected) < cls._COMPARE_TARGET_MAX_CHUNKS_PER_PRODUCT:
            missing = requested_fields - covered()
            candidates = [
                item for item in ranked
                if str(item.get("payload", {}).get("chunk_id") or id(item))
                not in selected_ids
            ]
            if not missing or not candidates:
                break
            best = max(
                candidates,
                key=lambda item: (
                    len(cls._covered_comparison_fields(item) & missing),
                    relevance(item),
                ),
            )
            if not (cls._covered_comparison_fields(best) & missing):
                break
            add(best)

        # The sixth chunk is exceptional: use it only if coverage is still
        # incomplete and it contributes at least one missing requested field.
        if len(selected) < cls._COMPARE_MAX_CHUNKS_PER_PRODUCT:
            missing = requested_fields - covered()
            candidates = [
                item for item in ranked
                if str(item.get("payload", {}).get("chunk_id") or id(item))
                not in selected_ids
                and cls._covered_comparison_fields(item) & missing
            ]
            if candidates:
                add(max(
                    candidates,
                    key=lambda item: (
                        len(cls._covered_comparison_fields(item) & missing),
                        relevance(item),
                    ),
                ))

        return selected

    def _multi_product_chat(
        self,
        question: str,
        products: List[Dict[str, Any]],
        history_dicts: Optional[List[Dict[str, Any]]],
        debug: bool,
        t0: float,
    ) -> ChatResponse:
        """Answer one focused question about multiple explicitly named products."""
        results, retrieval_debug = self._retrieve_named_products(
            question,
            products,
            per_product_k=min(self.settings.final_top_k, 10),
            debug=debug,
        )
        user_message, selected = self.context_builder.build(results, question)
        gen_result = self.generator.generate(user_message, history=history_dicts)
        answer = gen_result["answer"]
        sources = filter_cited_sources(
            answer,
            self._build_source_references(selected),
        )
        total_latency = (time.perf_counter() - t0) * 1000
        return ChatResponse(
            answer=answer,
            sources=sources,
            debug_info={
                "mode": "multi_product",
                "context_chunks": len(selected),
                "named_products": [item.get("product_id") for item in products],
                "per_product_retrieval": retrieval_debug,
            } if debug else None,
            latency_ms=total_latency,
        )

    def retrieve_only(
        self,
        query: str,
        product_id: Optional[str] = None,
        top_k: int = 8,
        debug: bool = False,
    ) -> RetrieveResponse:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=query,
            product_id=product_id,
            top_k=top_k,
            debug=debug,
        )
        results = [
            SourceReference(
                source_id=f"RESULT_{i}",
                product_name=item.get("payload", {}).get("product_name", ""),
                product_id=item.get("payload", {}).get("product_id", ""),
                product_group=item.get("payload", {}).get("product_group", ""),
                source_document=item.get("payload", {}).get("source_document", "") or item.get("payload", {}).get("source_file", ""),
                source_section=item.get("payload", {}).get("source_section", "") or item.get("payload", {}).get("title", ""),
                content_type=item.get("payload", {}).get("content_type", ""),
                content=item.get("payload", {}).get("content", "")[:400],
                chunk_id=item.get("payload", {}).get("chunk_id", ""),
                dense_rank=item.get("dense_rank"),
                dense_score=item.get("dense_score"),
                sparse_rank=item.get("sparse_rank"),
                sparse_score=item.get("sparse_score"),
                rrf_score=item.get("rrf_score"),
            )
            for i, item in enumerate(retrieval["results"], start=1)
        ]
        latency = (time.perf_counter() - t0) * 1000
        debug_info = retrieval.get("debug") if debug else None
        return RetrieveResponse(results=results, debug_info=debug_info, latency_ms=latency)
