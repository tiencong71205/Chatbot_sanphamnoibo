"""Chatbot service — orchestrates retrieve + generate."""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.ollama_generator import OllamaGenerator
from app.generation.context_builder import ContextBuilder
from app.schemas import ChatResponse, SourceReference, RetrieveResponse

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

    def chat(
        self,
        question: str,
        product_id: Optional[str] = None,
        debug: bool = False,
    ) -> ChatResponse:
        """Full RAG pipeline: retrieve → build context → generate."""
        t0 = time.perf_counter()

        # Retrieve
        retrieval = self.retriever.retrieve(
            query=question,
            product_id=product_id,
            top_k=self.settings.final_top_k,
            debug=debug,
        )
        results = retrieval["results"]

        # Giới hạn context theo ý định hỏi thông số kỹ thuật
        normalized_question = question.lower()
        specification_terms = (
            "thông số",
            "thông tin kỹ thuật",
            "specification",
            "technical specification",
        )

        if any(term in normalized_question for term in specification_terms):
            specification_results = [
                item
                for item in results
                if item.get("payload", {}).get("content_type") == "specification"
            ]
            if specification_results:
                results = specification_results

        reset_terms = (
            "reset",
            "cài đặt lại",
            "khôi phục",
            "đặt lại",
        )

        if any(term in normalized_question for term in reset_terms):
            reset_results = [
                item
                for item in results
                if (
                    item.get("payload", {}).get("content_type") == "configuration"
                    and any(
                        term in (
                            item.get("payload", {}).get("title", "")
                            + " "
                            + item.get("payload", {}).get("source_section", "")
                            + " "
                            + item.get("payload", {}).get("content", "")
                        ).lower()
                        for term in reset_terms
                    )
                )
            ]
            if reset_results:
                results = reset_results

        # Build context
        user_message, selected = self.context_builder.build(results, question)

        # Generate
        gen_result = self.generator.generate(user_message)
        answer = gen_result["answer"]

        # Build source references
        sources: List[SourceReference] = []
        for i, item in enumerate(selected, start=1):
            payload = item.get("payload", {})
            sources.append(
                SourceReference(
                    source_id=f"SOURCE_{i}",
                    product_name=payload.get("product_name", ""),
                    source_document=payload.get("source_document", ""),
                    source_section=payload.get("source_section", ""),
                    content_type=payload.get("content_type", ""),
                    content=payload.get("content", "")[:300],
                    dense_rank=item.get("dense_rank"),
                    dense_score=item.get("dense_score"),
                    sparse_rank=item.get("sparse_rank"),
                    sparse_score=item.get("sparse_score"),
                    rrf_score=item.get("rrf_score"),
                )
            )

        total_latency = (time.perf_counter() - t0) * 1000

        debug_info: Optional[Dict[str, Any]] = None
        if debug:
            debug_info = {
                "resolved_product": retrieval.get("resolved_product"),
                "dense_candidates": retrieval.get("debug", {}).get("dense_candidates", []),
                "sparse_candidates": retrieval.get("debug", {}).get("sparse_candidates", []),
                "rrf_results": retrieval.get("debug", {}).get("rrf_results", []),
                "context_chunks": len(selected),
                "retrieval_latency_ms": retrieval.get("latency_ms", 0),
                "generation_latency_ms": gen_result.get("latency_ms", 0),
            }

        return ChatResponse(
            answer=answer,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
        )

    def retrieve_only(
        self,
        query: str,
        product_id: Optional[str] = None,
        top_k: int = 6,
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
                source_document=item.get("payload", {}).get("source_document", ""),
                source_section=item.get("payload", {}).get("source_section", ""),
                content_type=item.get("payload", {}).get("content_type", ""),
                content=item.get("payload", {}).get("content", "")[:300],
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
