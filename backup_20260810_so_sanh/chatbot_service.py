"""Chatbot service — orchestrates retrieve + generate with history and compare mode."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.generation.context_builder import ContextBuilder
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
                per_product_k=min(self.settings.final_top_k, 10),
                debug=debug,
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
        user_message, selected = self.context_builder.build(results, question)

        gen_result = self.generator.generate(
            user_message,
            system_prompt=COMPARE_SYSTEM_PROMPT,
            history=history_dicts,
        )
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
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve context for every explicitly named product."""
        groups: List[List[Dict[str, Any]]] = []
        debug_rows: List[Dict[str, Any]] = []

        for product in products:
            product_id = str(product.get("product_id", ""))
            product_name = str(product.get("product_name", ""))
            product_query = f"{question}\nSản phẩm cần lấy dữ liệu: {product_name}."
            retrieval = self.retriever.retrieve(
                query=product_query,
                product_id=product_id,
                top_k=per_product_k,
                debug=debug,
            )
            groups.append(retrieval.get("results", []))
            if debug:
                debug_rows.append(
                    {
                        "product_id": product_id,
                        "result_count": len(retrieval.get("results", [])),
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

        return merged, debug_rows

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
