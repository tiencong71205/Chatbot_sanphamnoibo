"""Chatbot service — orchestrates multi-intent retrieve & multi-stage synthesis with complete coverage & citations."""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.generation.context_builder import ContextBuilder, estimate_tokens
from app.generation.vllm_generator import VLLMGenerator
from app.generation.prompts import AMBIGUITY_CHECK_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_intent import QueryIntent, classify_intent
from app.schemas import ChatResponse, ConversationTurn, RetrieveResponse, SourceReference
from app.services.conversation_context import (
    ConversationContextResolver,
    filter_cited_sources,
)

logger = logging.getLogger(__name__)

# 11 Canonical Coverage Groups as required by specification
COVERAGE_GROUPS = [
    ("overview", "1. Nhận diện/tổng quan", ["overview", "identity", "nhận diện", "tổng quan", "mô tả sản phẩm"]),
    ("purpose", "2. Mô tả và mục đích", ["purpose", "mục đích", "mô tả", "giải pháp", "ứng dụng thực tế"]),
    ("specification", "3. Thông số kỹ thuật", ["specification", "thông số", "kỹ thuật", "điện áp", "công suất", "kích thước"]),
    ("device_feature", "4. Tính năng trên thiết bị", ["device_feature", "tính năng trên thiết bị", "nút bấm", "đèn báo", "còi"]),
    ("app_feature", "5. Chức năng trên ứng dụng", ["app_feature", "chức năng trên ứng dụng", "ứng dụng", "vhomenex", "app"]),
    ("configuration", "6. Kết nối/cấu hình", ["configuration", "kết nối", "cấu hình", "pairing", "thêm thiết bị", "reset"]),
    ("automation", "7. Automation/kịch bản", ["automation", "kịch bản", "tự động", "nhà thông minh", "scene"]),
    ("installation", "8. Lắp đặt", ["installation", "lắp đặt", "hướng dẫn lắp đặt", "thi công", "gắn"]),
    ("post_installation", "9. Kiểm tra sau lắp đặt", ["post_installation", "kiểm tra sau lắp đặt", "kiểm tra", "vận hành thử"]),
    ("troubleshooting", "10. Xử lý sự cố/FAQ", ["troubleshooting", "faq", "sự cố", "lỗi", "khắc phục", "hỏi đáp"]),
    ("voice_command", "11. Khẩu lệnh hoặc nội dung đặc thù", ["voice_command", "khẩu lệnh", "giọng nói", "google assistant", "alexa"]),
]

# Direct content_type -> coverage-group-key map, checked BEFORE the
# keyword/substring pass in _group_chunks_by_coverage. app/ingestion/
# content_classifier.py only ever produces these exact values:
# identification, overview, specification, usage_tip, installation,
# connection, configuration, reset, troubleshooting, safety,
# compatibility, automation, feature (its own default fallback).
#
# The keyword lists in COVERAGE_GROUPS above were written against a
# *different*, more granular vocabulary (device_feature, app_feature,
# voice_command, post_installation, purpose) that the classifier never
# actually emits, and critically "connection" (English) never matches
# "kết nối" (Vietnamese) as a substring either way. Left unmapped, every
# "connection" and "feature" chunk (the bulk of most product docs) fell
# through to the "overview" catch-all default, producing a bloated,
# misclassified, slow-to-summarize overview group instead of populating
# groups 4/5/6/11 where they actually belong.
_CONTENT_TYPE_TO_GROUP = {
    "identification": "overview",
    "overview": "overview",
    "specification": "specification",
    "usage_tip": "device_feature",
    "installation": "installation",
    "connection": "configuration",
    "configuration": "configuration",
    "reset": "configuration",
    "troubleshooting": "troubleshooting",
    "safety": "installation",
    "compatibility": "specification",
    "automation": "automation",
    # "feature" is the classifier's own catch-all default, used for
    # on-device features, app features, and voice-command tables alike, so
    # it needs its own text-based disambiguation rather than one fixed
    # group — see _group_chunks_by_coverage below.
}


class ChatbotService:
    # Generation length ceiling per query intent. This is the primary
    # "dynamic max_tokens" lever: it cuts decode time (the dominant
    # per-request GPU cost on a single-GPU Ollama setup, since decode is
    # autoregressive while prefill of a few thousand context tokens is
    # comparatively cheap) for query types that don't need long answers.
    # Deliberately does NOT touch retrieval breadth (dense_top_k /
    # sparse_top_k / final_top_k / context token budget) — the set of
    # evidence chunks selected for the prompt is unchanged, so recall/
    # accuracy is unaffected; only the ceiling on how much the model is
    # allowed to write back is reduced for the common, short-answer case.
    # PRODUCT_FULL goes through _multi_stage_synthesize's own per-stage
    # budgets (group summaries + final merge) below instead of this table.
    _MAX_TOKENS_BY_INTENT = {
        QueryIntent.FACT: 1000,
        QueryIntent.MULTI_PRODUCT_FACT: 1100,
        QueryIntent.COMPARE_FIELD: 1300,
        QueryIntent.SECTION_FULL: 1500,  # may hold a full multi-step procedure
        QueryIntent.COMPARE_FULL: 1500,  # full comparison table, needs headroom
    }
    _GROUP_SUMMARY_MAX_TOKENS = 900

    # Canonical, ordered comparison criteria used by the deterministic
    # compare-table pipeline (see _extract_product_criteria /
    # _assemble_comparison_table). A fixed vocabulary — rather than
    # letting each per-product extraction invent its own field names —
    # is what makes the rows line up correctly across products when the
    # table is assembled in code afterward.
    _COMPARISON_CRITERIA = [
        ("business_id", "Mã sản phẩm"),
        ("model", "Model"),
        ("power_supply", "Nguồn cấp"),
        ("communication", "Truyền thông"),
        ("output_power", "Đầu ra / Công suất"),
        ("dimensions", "Kích thước"),
        ("environment", "Môi trường hoạt động"),
        ("buttons", "Nút bấm"),
        ("indicator", "Đèn báo"),
        ("control", "Điều khiển"),
        ("timer", "Hẹn giờ / Lịch"),
        ("warranty", "Bảo hành"),
        ("features", "Tính năng nổi bật"),
    ]
    # Headroom for a full 13-criteria JSON response — some values (e.g.
    # "Điều khiển", "Tính năng nổi bật") can be several sentences long, and
    # a truncated response before the closing '}' fails to parse entirely
    # (see _parse_criteria_json), silently zeroing out that whole product's
    # row instead of just one field.
    _CRITERIA_EXTRACTION_MAX_TOKENS = 1500

    _PROCEDURE_INTENT_TERMS = (
        "hướng dẫn",
        "cách kết nối",
        "các bước",
        "quy trình",
        "làm thế nào để kết nối",
    )
    _STEP_RE = re.compile(
        r"(?ms)^\s*(B\d+)\s*[:.)-]\s*(.+?)"
        r"(?=^\s*B\d+\s*[:.)-]|\Z)"
    )

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: VLLMGenerator,
    ):
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.context_builder = ContextBuilder(settings)
        self.conversation_context = ConversationContextResolver(retriever.resolver)
        # Same directory SemanticChunker._save_image() writes to during
        # ingest (see ingest_service.py) — must match exactly, since a
        # chunk's image_paths are relative filenames resolved against this.
        self.images_dir = Path(settings.data_directory).parent / "images"

    def _load_image_base64(self, filename: str) -> Optional[str]:
        """Read one saved image file and return it as a base64 data URI.

        Never raises — a missing/unreadable file (e.g. the images
        directory was pruned separately from Qdrant, or a chunk predates
        this feature) just means that one image doesn't render; it
        shouldn't ever break the answer it's attached to.
        """
        if not filename:
            return None
        try:
            path = self.images_dir / filename
            if not path.exists() or not path.is_file():
                return None
            ext = path.suffix.lstrip(".").lower() or "png"
            mime = "jpeg" if ext == "jpg" else ext
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/{mime};base64,{data}"
        except Exception as e:
            logger.warning("Failed to load image '%s': %s", filename, e)
            return None

    def _build_source_references(
        self, selected: List[Dict[str, Any]], start: int = 1
    ) -> List[SourceReference]:
        sources: List[SourceReference] = []
        for i, item in enumerate(selected, start=start):
            payload = item.get("payload", item)
            images = [
                b64 for b64 in (
                    self._load_image_base64(fname)
                    for fname in payload.get("image_paths", []) or []
                )
                if b64
            ]
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
                    heading_path=payload.get("heading_path", ""),
                    source_locator=payload.get("source_locator", ""),
                    chunk_type=payload.get("chunk_type", ""),
                    dense_rank=item.get("dense_rank"),
                    dense_score=item.get("dense_score"),
                    sparse_rank=item.get("sparse_rank"),
                    sparse_score=item.get("sparse_score"),
                    rrf_score=item.get("rrf_score"),
                    images=images,
                )
            )
        return sources

    @classmethod
    def _procedure_steps(cls, content: str) -> List[tuple[str, str]]:
        return [
            (label.upper(), text.strip())
            for label, text in cls._STEP_RE.findall(content or "")
        ]

    @classmethod
    def _preserve_procedure_steps(
        cls,
        question: str,
        answer: str,
        selected: List[Dict[str, Any]],
    ) -> str:
        question_norm = question.lower()
        if not any(term in question_norm for term in cls._PROCEDURE_INTENT_TERMS):
            return answer

        procedure_scope_terms = (
            "lắp đặt",
            "kết nối",
            "giám sát",
            "automation",
            "kịch bản",
            "kiểm tra",
            "thêm thiết bị",
            "điều khiển",
            "reset",
        )
        requested_scopes = [
            term
            for term in procedure_scope_terms
            if term in question_norm
        ]

        candidates = []
        for index, item in enumerate(selected, start=1):
            payload = item.get("payload", item)

            # Include the chunk's own `title` and a content snippet, not
            # just feature_name/source_section/heading_path/content_type.
            #
            # Confirmed real bug this fixes: for a section's own leading
            # body content (e.g. a Reset preamble under "4. Tính năng trên
            # thiết bị", before its "4.1 Chế độ kết nối..." child section
            # starts), feature_name is empty (only ever set for level>=3
            # nodes — see semantic_chunker.py's _new_chunk), content_type
            # is an English classifier label ("connection") that can never
            # textually contain a Vietnamese scope term like "kết nối", and
            # source_section/heading_path just repeat the generic parent
            # heading. That chunk was being filtered OUT of `candidates`
            # here even when it was the retrieval-ranked #1 result and its
            # own content directly answered the question — while a sibling
            # chunk whose *section title* happened to literally contain the
            # scope keyword passed the filter and got selected instead,
            # silently overwriting a correct model answer with the wrong
            # procedure. `title` carries semantic_chunker's disambiguated
            # preamble title (e.g. "...— Lưu ý: ...sẵn sàng kết nối..."),
            # which does contain the scope keyword; the content snippet is
            # an extra safety net for documents where the title alone
            # isn't distinguishing.
            candidate_scope = " ".join([
                str(payload.get("feature_name", "")),
                str(payload.get("title", "")),
                str(payload.get("source_section", "")),
                str(payload.get("heading_path", "")),
                str(payload.get("source_locator", "")),
                str(payload.get("content_type", "")),
                str(payload.get("content", ""))[:300],
            ]).lower()

            if requested_scopes and not any(
                scope in candidate_scope
                for scope in requested_scopes
            ):
                continue

            steps = cls._procedure_steps(str(payload.get("content", "")))
            if payload.get("chunk_type") == "procedure" and len(steps) >= 2:
                candidates.append((index, payload, steps))

        if not candidates:
            return answer

        source_index, payload, steps = candidates[0]
        expected_labels = [label for label, _ in steps]
        answer_labels = [
            label.upper()
            for label in re.findall(
                r"(?m)^\s*(B\d+)\s*[:.)-]",
                answer or "",
                flags=re.IGNORECASE,
            )
        ]
        if answer_labels == expected_labels:
            return answer

        feature_name = str(
            payload.get("feature_name")
            or payload.get("source_section")
            or "quy trình trong tài liệu"
        ).strip()
        lines = [f"Hướng dẫn {feature_name.lower()}:"]
        for label, text in steps:
            lines.append(f"- **{label}:** {text}")
        lines.append("")
        lines.append("📚 Nguồn tham khảo:")
        lines.append(f"- SOURCE_{source_index}")
        return "\n".join(lines)

    @staticmethod
    def _remove_dangling_markdown_bullet(answer: str) -> str:
        return re.sub(r"[\r\n]+\s*[-*]\s*$", "", (answer or "").rstrip())

    @classmethod
    def _finalize_answer(
        cls,
        question: str,
        answer: str,
        selected: List[Dict[str, Any]],
    ) -> str:
        answer = cls._preserve_procedure_steps(question, answer, selected)
        return cls._remove_dangling_markdown_bullet(answer)

    def _group_chunks_by_coverage(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Map retrieved chunks into the 11 canonical coverage groups."""
        result: Dict[str, Dict[str, Any]] = {}
        for key, title, keywords in COVERAGE_GROUPS:
            result[key] = {
                "title": title,
                "chunks": [],
                "has_data": False,
            }

        for item in chunks:
            payload = item.get("payload", item)
            ct = str(payload.get("content_type", "")).lower()
            sec = str(payload.get("source_section", "")).lower()

            # 1. Direct content_type -> group lookup (see _CONTENT_TYPE_TO_GROUP
            # for why this has to run first: most content_type values never
            # textually match the group keyword lists below).
            target_key = _CONTENT_TYPE_TO_GROUP.get(ct)

            # 2. "feature" is an intentionally ambiguous catch-all emitted by
            # the classifier for on-device features, app features, and voice
            # command tables alike — disambiguate using the section heading.
            if target_key is None and ct == "feature":
                if any(kw in sec for kw in ("khẩu lệnh", "giọng nói")):
                    target_key = "voice_command"
                elif any(kw in sec for kw in ("ứng dụng", "app", "vhomenex")):
                    target_key = "app_feature"
                elif any(kw in sec for kw in ("kiểm tra sau lắp đặt", "vận hành thử")):
                    target_key = "post_installation"
                else:
                    target_key = "device_feature"

            if target_key:
                result[target_key]["chunks"].append(item)
                result[target_key]["has_data"] = True
                continue

            # 3. Fallback: keyword/substring pass against content_type or
            # source_section, for any content_type not covered above.
            matched = False
            for key, title, keywords in COVERAGE_GROUPS:
                if any(kw in ct or kw in sec for kw in keywords):
                    result[key]["chunks"].append(item)
                    result[key]["has_data"] = True
                    matched = True
                    break

            if not matched:
                result["overview"]["chunks"].append(item)
                result["overview"]["has_data"] = True

        return result

    def _multi_stage_synthesize(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        product_name: str = "",
        debug: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Multi-stage synthesis: Group chunks by section, summarize large groups, merge deterministically."""
        orchestration_steps = ["1_group_coverage"]
        coverage_map = self._group_chunks_by_coverage(chunks)

        coverage_by_product = {product_name or "product": {}}
        missing_sections = []

        for key, group in coverage_map.items():
            cnt = len(group["chunks"])
            coverage_by_product[product_name or "product"][group["title"]] = cnt
            if cnt == 0:
                missing_sections.append(group["title"])

        user_msg, selected = self.context_builder.build(chunks, query=question)
        token_est = estimate_tokens(user_msg)

        if token_est <= getattr(self.settings, "max_context_tokens", 6000):
            orchestration_steps.append("2_single_pass_generation")
            gen_res = self.generator.generate(user_msg)
            answer = gen_res["answer"]
            last_generation_stats = {
                "prompt_eval_count": gen_res.get("prompt_eval_count"),
                "eval_count": gen_res.get("eval_count"),
                "generate_latency_ms": gen_res.get("latency_ms"),
            }
        else:
            orchestration_steps.append("2_stage1_summarize_groups")
            # Groups are independent of each other, so their summaries are
            # generated concurrently instead of one-by-one. This is the
            # single biggest latency cost for "toàn bộ thông tin sản phẩm"
            # style questions: up to 11 sequential Ollama calls otherwise.
            # Actual wall-clock speedup depends on how many requests the
            # Ollama server accepts concurrently (OLLAMA_NUM_PARALLEL env var
            # on the Ollama side, and available VRAM) — with a single RTX
            # 3060 running qwen at Q8, 3 is a safe default that won't starve
            # a single generation of VRAM/compute; raise it if the server is
            # configured for more parallel slots.
            groups_needing_generation = [
                (key, title) for key, title, _ in COVERAGE_GROUPS
                if coverage_map[key]["has_data"]
            ]
            summaries_by_key: Dict[str, str] = {}
            max_workers = min(
                getattr(self.settings, "synthesis_max_parallel_groups", 3),
                max(1, len(groups_needing_generation)),
            )
            if groups_needing_generation:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_key = {
                        pool.submit(self._summarize_group, coverage_map[key], title, question): key
                        for key, title in groups_needing_generation
                    }
                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            summaries_by_key[key] = future.result()
                        except Exception as e:
                            logger.error("Group summary failed for '%s': %s", key, e, exc_info=True)
                            summaries_by_key[key] = "Không thể tạo tóm tắt cho mục này do lỗi hệ thống."

            group_summaries = []
            for key, title, _ in COVERAGE_GROUPS:
                grp = coverage_map[key]
                if not grp["has_data"]:
                    group_summaries.append(f"### {title}\nChưa có dữ liệu trong tài liệu.")
                else:
                    group_summaries.append(f"### {title}\n{summaries_by_key.get(key, '')}")

            orchestration_steps.append("3_stage2_merge_summaries")
            merged_context = "\n\n".join(group_summaries)
            final_prompt = (
                f"Dưới đây là bản tóm tắt đầy đủ theo từng mục của sản phẩm {product_name}:\n\n"
                f"{merged_context}\n\n"
                f"=== CÂU HỎI NGƯỜI DÙNG ===\n{question}\n\n"
                "Hãy tổng hợp thành câu trả lời đầy đủ, chi tiết, chuyên nghiệp theo thứ tự 11 mục trên."
            )
            gen_res = self.generator.generate(final_prompt)
            answer = gen_res["answer"]
            last_generation_stats = {
                "prompt_eval_count": gen_res.get("prompt_eval_count"),
                "eval_count": gen_res.get("eval_count"),
                "generate_latency_ms": gen_res.get("latency_ms"),
                "note": "stats are for the final merge call only; group "
                        "summaries ran in parallel before this and aren't "
                        "individually broken out here",
            }

        debug_data = {
            "orchestration_steps": orchestration_steps,
            "coverage_by_product": coverage_by_product,
            "missing_sections": missing_sections,
            "context_token_estimate": token_est,
            "total_candidate_chunks": len(chunks),
            "selected_chunks": len(selected),
            # Diagnostic: prompt_eval_count is prefill size (context tokens
            # processed), eval_count is decode size (tokens actually
            # generated). If eval_count is near the max_tokens ceiling for
            # this branch, generation time is decode-bound and lowering
            # max_tokens further will help; if it's well under, the model
            # stopped naturally (hit EOS) and the bottleneck is prefill
            # (large context) instead — lowering max_tokens would do nothing.
            "last_generation_stats": last_generation_stats,
        }
        return answer, selected, debug_data

    def _summarize_group(self, group: Dict[str, Any], title: str, question: str) -> str:
        """Summarize a single coverage group. Runs inside a worker thread —
        no shared mutable state, only reads from context_builder/generator
        (both stateless per-call aside from the shared thread-safe httpx
        client), so this is safe to call concurrently.

        Capped at _GROUP_SUMMARY_MAX_TOKENS: a summary of ONE coverage
        group (e.g. just "Thông số kỹ thuật") rarely needs the full 1500-token
        ceiling meant for a final, complete answer — up to 11 of these run
        per PRODUCT_FULL question, so shaving each one's decode ceiling is
        the single biggest lever for that query type's wall-clock time.
        900 leaves real headroom for verbose groups (full spec tables,
        multi-item troubleshooting) so completeness isn't sacrificed for
        speed on the harder cases.
        """
        grp_msg, _ = self.context_builder.build(
            group["chunks"],
            query=f"Tóm tắt ngắn gọn và chính xác toàn bộ nội dung của mục {title}.",
        )
        grp_res = self.generator.generate(
            grp_msg, max_tokens=self._GROUP_SUMMARY_MAX_TOKENS
        )
        return grp_res["answer"]

    def _route(
        self,
        question: str,
        product_id: Optional[str],
        history: Optional[List[ConversationTurn]],
        compare_mode: bool,
        t0: float,
    ):
        """Everything chat() does before generation: product resolution,
        conversation-history disambiguation, intent classification, and
        scope resolution. Pulled out of chat() so chat_stream() can reuse
        the exact same routing decisions without duplicating this logic —
        the two must never diverge on *which* products/intent a question
        resolves to, only on whether the final generation call streams.

        Returns either ("early_return", ChatResponse) when the question is
        genuinely ambiguous (mirrors chat()'s early-return branch exactly),
        or ("dispatch", question, intent_res, explicit_products,
        effective_pid, scope, history_dicts).
        """
        history_dicts = None
        if history:
            history_dicts = [{"role": h.role, "content": h.content} for h in history]

        explicit_products = [
            item
            for item in self.retriever.resolver.resolve_all(question)
            if item.get("product_id") and item.get("confidence", 0.0) >= 0.65
        ]
        if product_id and not any(p["product_id"] == product_id for p in explicit_products):
            catalog_item = self.retriever.resolver.catalog_item(product_id) or {}
            explicit_products.append({
                "product_id": product_id,
                "product_name": catalog_item.get("product_name", product_id),
                "confidence": 1.0,
            })

        multi_reference = self.conversation_context.is_multi_product_reference(
            question
        )
        expected_count = self.conversation_context.referenced_product_count(
            question
        )

        # Only raid conversation history for extra products when the current
        # question is genuinely under-specified. A generic phrase like "các
        # thiết bị" ("the devices") is a weak signal on its own — it appears
        # in plenty of self-contained, single-product questions (e.g. "...xóa
        # Gateway khỏi tài khoản, các thiết bị Zigbee sẽ ra sao?" is clearly
        # about Gateway, not an ambiguous multi-product reference). Confirmed
        # in practice: this generic-phrase path was pulling in 2 unrelated
        # products from an earlier, unrelated turn even though the current
        # question already named its one real product explicitly. Only treat
        # the question as needing history when either (a) it has an explicit
        # numeric count ("2 thiết bị", "cả ba"...) not yet satisfied by what
        # the current turn already resolved, or (b) the current turn resolved
        # zero products at all, so there is genuinely nothing to go on.
        needs_history_lookup = multi_reference and (
            (expected_count is not None and len(explicit_products) < expected_count)
            or (expected_count is None and len(explicit_products) == 0)
        )

        if needs_history_lookup:
            historical_products = self.conversation_context.products_from_history(
                history,
                expected_count=expected_count,
            )

            known_ids = {item["product_id"] for item in explicit_products}
            for item in historical_products:
                if item["product_id"] not in known_ids:
                    explicit_products.append(item)
                    known_ids.add(item["product_id"])

            insufficient = (
                len(explicit_products) < 2
                if expected_count is None
                else len(explicit_products) != expected_count
            )
            if insufficient:
                total_latency = (time.perf_counter() - t0) * 1000
                expected_text = (
                    f"{expected_count} thiết bị"
                    if expected_count
                    else "các thiết bị"
                )
                return (
                    "early_return",
                    ChatResponse(
                        answer=(
                            f"Bạn vui lòng cho biết rõ tên {expected_text} cần hỏi. "
                            "Lịch sử hội thoại hiện chưa xác định đủ danh sách sản phẩm "
                            "nên tôi chưa thể trả lời chính xác."
                        ),
                        latency_ms=total_latency,
                    ),
                )

        intent_question = question
        if multi_reference and len(explicit_products) == 2:
            intent_question += "\nCả hai sản phẩm."
        elif multi_reference and len(explicit_products) == 3:
            intent_question += "\nCả ba sản phẩm."

        intent_res = classify_intent(intent_question, explicit_products)
        logger.info("Classified Intent: %s (products=%s, sections=%s)", intent_res.intent.value, intent_res.requested_products, intent_res.requested_sections)

        scope = self.conversation_context.resolve(
            question=question,
            explicit_product_id=product_id or (explicit_products[0]["product_id"] if explicit_products else None),
            history=history,
            compare_mode=compare_mode or intent_res.intent in (QueryIntent.COMPARE_FIELD, QueryIntent.COMPARE_FULL),
        )
        effective_pid = scope.product_id

        return (
            "dispatch",
            intent_res,
            explicit_products,
            effective_pid,
            scope,
            history_dicts,
        )

    def retrieve_only(
        self,
        query: str,
        product_id: Optional[str] = None,
        top_k: int = 8,
        debug: bool = False,
    ) -> RetrieveResponse:
        """Raw hybrid retrieval results with no generation step.

        Backs /api/search — the diagnostic entry point for inspecting what
        the retriever alone surfaces (and how it ranked things: dense_rank,
        sparse_rank, rrf_score are all populated on each SourceReference)
        for a query, independent of whether/how the LLM ends up using those
        chunks. This lets a ranking problem be told apart from a generation
        problem: if the right chunk isn't even in these results, the fix
        belongs in retrieval; if it's here but the model still answers from
        the wrong one, the fix belongs in generation/prompting instead.

        NOTE: this method didn't exist until now even though /api/search
        already called it — the endpoint was wired up but never finished,
        so every call to /api/search failed with an AttributeError.
        """
        t0 = time.perf_counter()
        result = self.retriever.retrieve(query, product_id=product_id, top_k=top_k, debug=debug)
        sources = self._build_source_references(result.get("results", []))
        latency_ms = (time.perf_counter() - t0) * 1000

        debug_info = None
        if debug:
            debug_info = {
                "resolved_product": result.get("resolved_product"),
                "resolved_products": result.get("resolved_products", []),
                "dense_count": result.get("dense_count"),
                "sparse_count": result.get("sparse_count"),
                "applied_filters": result.get("debug", {}).get("applied_filters"),
                "target_product_ids": result.get("debug", {}).get("target_product_ids"),
            }

        return RetrieveResponse(results=sources, debug_info=debug_info, latency_ms=latency_ms)

    def chat(
        self,
        question: str,
        product_id: Optional[str] = None,
        history: Optional[List[ConversationTurn]] = None,
        compare_mode: bool = False,
        debug: bool = False,
    ) -> ChatResponse:
        """Full RAG pipeline with intent classification & full-coverage multi-stage synthesis."""
        t0 = time.perf_counter()

        routed = self._route(question, product_id, history, compare_mode, t0)
        if routed[0] == "early_return":
            return routed[1]
        _, intent_res, explicit_products, effective_pid, scope, history_dicts = routed

        if intent_res.intent == QueryIntent.PRODUCT_FULL:
            return self._handle_product_full(question, intent_res, effective_pid, history_dicts, debug, t0)

        elif intent_res.intent == QueryIntent.SECTION_FULL:
            return self._handle_section_full(question, intent_res, effective_pid, history_dicts, debug, t0)

        elif intent_res.intent in (QueryIntent.COMPARE_FIELD, QueryIntent.COMPARE_FULL):
            return self._handle_compare(question, intent_res, explicit_products, history_dicts, debug, t0)

        elif intent_res.intent == QueryIntent.MULTI_PRODUCT_FACT:
            return self._handle_multi_product_fact(question, intent_res, explicit_products, history_dicts, debug, t0)

        return self._handle_fact(question, intent_res, effective_pid, scope, history_dicts, debug, t0)

    def chat_stream(
        self,
        question: str,
        product_id: Optional[str] = None,
        history: Optional[List[ConversationTurn]] = None,
        compare_mode: bool = False,
        debug: bool = False,
    ):
        """Generator yielding {"type": "delta", "text": ...} events followed
        by one final {"type": "done", "response": ChatResponse} event.

        Only the FACT intent (a single, direct question — the dominant
        traffic pattern for a product-QA chatbot, e.g. "điện áp hoạt động
        bao nhiêu?") streams tokens as they're generated, for a fast
        time-to-first-token. The other intents (compare, multi-product-fact,
        product_full, section_full) are rarer, heavier queries where the
        existing parallel multi-stage synthesis already dominates latency
        far more than perceived streaming smoothness would — for those,
        this reuses the exact same, already-tested non-streaming handler
        and yields its full answer as one delta, so behavior for those
        intents is byte-for-byte identical to chat(); only FACT gets new
        code, minimizing risk to everything else.
        """
        t0 = time.perf_counter()

        routed = self._route(question, product_id, history, compare_mode, t0)
        if routed[0] == "early_return":
            response = routed[1]
            yield {"type": "delta", "text": response.answer}
            yield {"type": "done", "response": response}
            return
        _, intent_res, explicit_products, effective_pid, scope, history_dicts = routed

        if intent_res.intent == QueryIntent.FACT:
            yield from self._stream_fact(
                question, intent_res, effective_pid, scope, history_dicts, debug, t0
            )
            return

        if intent_res.intent == QueryIntent.PRODUCT_FULL:
            response = self._handle_product_full(question, intent_res, effective_pid, history_dicts, debug, t0)
        elif intent_res.intent == QueryIntent.SECTION_FULL:
            response = self._handle_section_full(question, intent_res, effective_pid, history_dicts, debug, t0)
        elif intent_res.intent in (QueryIntent.COMPARE_FIELD, QueryIntent.COMPARE_FULL):
            response = self._handle_compare(question, intent_res, explicit_products, history_dicts, debug, t0)
        else:
            response = self._handle_multi_product_fact(question, intent_res, explicit_products, history_dicts, debug, t0)

        yield {"type": "delta", "text": response.answer}
        yield {"type": "done", "response": response}

    def _stream_fact(
        self,
        question: str,
        intent_res: Any,
        effective_product_id: Optional[str],
        scope: Any,
        history_dicts: Any,
        debug: bool,
        t0: float,
    ):
        """Streaming twin of _handle_fact — same retrieval, same context
        building, same post-processing; only the final generation call
        streams instead of blocking for the whole answer. Kept as a
        separate method (rather than a `stream: bool` flag threaded through
        _handle_fact) so the well-tested non-streaming path can't be
        accidentally altered by streaming-specific control flow.
        """
        t_ret_start = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=scope.query if scope else question,
            product_id=effective_product_id,
            top_k=self.settings.final_top_k,
            debug=debug,
        )
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000
        results = retrieval["results"]

        # No product was resolved from the question text -- before
        # streaming a normal answer, check whether the retrieved evidence
        # actually conflicts across products (see
        # _check_multi_product_conflict's docstring). Only runs when 2+
        # distinct products are GENUINELY competing for relevance — not
        # just "appeared anywhere in the raw top-K" — a no-op for the
        # common single-product case.
        if not effective_product_id and results:
            grouped_for_check = self._relevant_products_for_ambiguity_check(results)
            if len(grouped_for_check) >= 2:
                conflict = self._check_multi_product_conflict(question, grouped_for_check)
                if conflict and not conflict.get("same_for_all", True):
                    # "Differing" is meaningless with fewer than 2 named
                    # products -- fall back to the full retrieved product
                    # list whenever the model's own enumeration is missing
                    # or incomplete (confirmed real case: the model
                    # correctly determined same_for_all=false but only
                    # named ONE product, producing a self-contradictory
                    # "differs across multiple products" message naming
                    # just one).
                    differing = conflict.get("differing_products") or []
                    if len(differing) < 2:
                        differing = list(grouped_for_check.keys())
                    total_latency = (time.perf_counter() - t0) * 1000
                    clarify_text = (
                        "Câu hỏi này có thể áp dụng cho nhiều sản phẩm khác nhau với "
                        "thông tin không giống nhau: " + ", ".join(differing) + ". "
                        "Bạn đang hỏi về sản phẩm nào để mình trả lời chính xác?"
                    )
                    clarify_debug_info = None
                    if debug:
                        clarify_debug_info = {
                            "query_intent": intent_res.intent.value,
                            "requested_products": [],
                            "orchestration_steps": ["hybrid_retrieval", "multi_product_ambiguity_check"],
                            "retrieved_products": list(grouped_for_check.keys()),
                            "ambiguity_check_result": conflict,
                            "retrieval_ms": retrieval_ms,
                            "total_ms": total_latency,
                        }
                    clarify_response = ChatResponse(
                        answer=clarify_text,
                        latency_ms=total_latency,
                        retrieval_ms=retrieval_ms,
                        debug_info=clarify_debug_info,
                    )
                    yield {"type": "delta", "text": clarify_text}
                    yield {"type": "done", "response": clarify_response}
                    return

        generation_question = question
        if scope and scope.is_follow_up and scope.product_name:
            generation_question = (
                f"{question}\n\n"
                f"Ngữ cảnh hội thoại cần giữ: sản phẩm đang hỏi là "
                f"{scope.product_name}. Chỉ trả lời cho sản phẩm này."
            )

        user_message, selected = self.context_builder.build(results, generation_question)
        token_est = estimate_tokens(user_message)

        t_gen_start = time.perf_counter()
        generation_history = (
            None
            if scope and scope.is_follow_up and effective_product_id
            else history_dicts
        )

        full_text_parts: List[str] = []
        try:
            for delta in self.generator.generate_stream(
                user_message,
                history=generation_history,
                max_tokens=self._MAX_TOKENS_BY_INTENT.get(intent_res.intent),
            ):
                full_text_parts.append(delta)
                yield {"type": "delta", "text": delta}
        except Exception as e:
            logger.error("Streaming generation failed: %s", e, exc_info=True)
            yield {
                "type": "error",
                "message": "Đã xảy ra lỗi khi tạo câu trả lời. Vui lòng thử lại.",
            }
            return
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        raw_answer = "".join(full_text_parts).strip()
        answer = self._finalize_answer(question, raw_answer, selected)
        sources = filter_cited_sources(answer, self._build_source_references(selected))
        total_latency = (time.perf_counter() - t0) * 1000

        # _finalize_answer can rewrite the streamed text entirely (e.g.
        # _preserve_procedure_steps reformatting a procedure answer) — when
        # that happens the client already rendered the raw streamed text, so
        # send a correction delta with the final version rather than silently
        # returning a "done" event whose `response.answer` disagrees with
        # what's on screen.
        if answer != raw_answer:
            yield {"type": "replace", "text": answer}

        debug_info = None
        if debug:
            debug_info = {
                "query_intent": intent_res.intent.value,
                "requested_products": intent_res.requested_products or ([effective_product_id] if effective_product_id else []),
                "requested_sections": intent_res.requested_sections,
                "total_candidate_chunks": len(results),
                "selected_chunks": len(selected),
                "coverage_by_product": {effective_product_id or "all": len(results)},
                "missing_sections": [],
                "context_token_estimate": token_est,
                "orchestration_steps": ["hybrid_retrieval", "generation_stream"],
                "resolved_product": retrieval.get("resolved_product"),
                "resolved_products": retrieval.get("resolved_products", []),
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_latency,
            }

        yield {
            "type": "done",
            "response": ChatResponse(
                answer=answer,
                sources=sources,
                debug_info=debug_info,
                latency_ms=total_latency,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            ),
        }

    def _handle_product_full(
        self,
        question: str,
        intent_res: Any,
        product_id: Optional[str],
        history_dicts: Any,
        debug: bool,
        t0: float,
    ) -> ChatResponse:
        t_ret_start = time.perf_counter()
        pid = product_id or (intent_res.requested_products[0] if intent_res.requested_products else "")
        catalog_item = self.retriever.resolver.catalog_item(pid) or {}
        pname = catalog_item.get("product_name", pid)

        chunks = self.retriever.retrieve_full(product_id=pid)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        t_gen_start = time.perf_counter()
        answer, selected, orch_debug = self._multi_stage_synthesize(
            question, chunks, product_name=pname, debug=debug
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        final_ans = self._finalize_answer(question, answer, selected)
        sources = filter_cited_sources(final_ans, self._build_source_references(selected))
        total_latency = (time.perf_counter() - t0) * 1000

        debug_info = None
        if debug:
            debug_info = {
                "query_intent": intent_res.intent.value,
                "requested_products": intent_res.requested_products or [pid],
                "requested_sections": intent_res.requested_sections,
                "total_candidate_chunks": len(chunks),
                "selected_chunks": len(selected),
                "coverage_by_product": orch_debug["coverage_by_product"],
                "missing_sections": orch_debug["missing_sections"],
                "context_token_estimate": orch_debug["context_token_estimate"],
                "orchestration_steps": orch_debug["orchestration_steps"],
                "last_generation_stats": orch_debug.get("last_generation_stats"),
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_latency,
            }

        return ChatResponse(
            answer=final_ans,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def _handle_section_full(
        self,
        question: str,
        intent_res: Any,
        product_id: Optional[str],
        history_dicts: Any,
        debug: bool,
        t0: float,
    ) -> ChatResponse:
        t_ret_start = time.perf_counter()
        pid = product_id or (intent_res.requested_products[0] if intent_res.requested_products else "")
        sec_name = intent_res.requested_sections[0] if intent_res.requested_sections else ""

        chunks = self.retriever.retrieve_full(product_id=pid, section_name=sec_name)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        user_msg, selected = self.context_builder.build(chunks, query=question)
        token_est = estimate_tokens(user_msg)

        t_gen_start = time.perf_counter()
        gen_res = self.generator.generate(
            user_msg,
            history=history_dicts,
            max_tokens=self._MAX_TOKENS_BY_INTENT.get(intent_res.intent),
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        final_ans = self._finalize_answer(question, gen_res["answer"], selected)
        sources = filter_cited_sources(final_ans, self._build_source_references(selected))
        total_latency = (time.perf_counter() - t0) * 1000

        debug_info = None
        if debug:
            debug_info = {
                "query_intent": intent_res.intent.value,
                "requested_products": intent_res.requested_products or [pid],
                "requested_sections": intent_res.requested_sections,
                "total_candidate_chunks": len(chunks),
                "selected_chunks": len(selected),
                "coverage_by_product": {pid: {sec_name: len(chunks)}},
                "missing_sections": [] if chunks else [sec_name],
                "context_token_estimate": token_est,
                "orchestration_steps": ["section_scroll", "generation"],
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_latency,
            }

        return ChatResponse(
            answer=final_ans,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    @staticmethod
    def _repair_json_text(text: str) -> str:
        """Best-effort repair for common small-model JSON mistakes, applied
        before json.loads(). Every repair here is a narrow, well-known LLM
        JSON-generation failure mode — none of these rewrite anything a
        genuinely well-formed response would contain, so this can only help
        parse a malformed response, never break a valid one.
        """
        # Smart/curly quotes instead of straight ASCII quotes — some models
        # drift into typographic quotes for natural-language text, which
        # json.loads() does not accept as string delimiters.
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # A literal (unescaped) newline inside a JSON string is invalid —
        # none of the extracted values are meant to be multi-line, so
        # collapsing raw newlines to spaces is safe here.
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        # Trailing comma before a closing brace/bracket, e.g. `..., }` —
        # one of the most common structured-output mistakes small models
        # make.
        text = re.sub(r",\s*([\}\]])", r"\1", text)
        return text

    def _extract_product_criteria(
        self,
        product_name: str,
        indexed_chunks: List[Tuple[int, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, str]]:
        """Extract {criterion_key: {"value":..., "source_id":...}} for ONE
        product from ONLY that product's own chunks.

        Runs as an independent LLM call per product (see _handle_compare)
        so the model literally cannot see a second product's data while
        filling in this one's values. This is the structural fix for a
        confirmed bug: even after 2 rounds of explicit prompt reinforcement
        telling the model "do not mix data between products, double-check
        the SOURCE's own product label before filling a cell", a small (4B)
        model still cross-attributed a spec between two similarly-named
        products (e.g. a dimmer's button spec ending up in the non-dimmer
        switch's column) in live testing — confirmed via /api/search that
        retrieval itself was correct, so the mistake was purely in how the
        model wrote the free-form table. Splitting extraction by product
        removes the *opportunity* for that mistake entirely instead of
        continuing to hope better wording fixes it.
        """
        if not indexed_chunks:
            return {}

        blocks = [
            ContextBuilder.format_source_block(idx, payload)
            for idx, payload in indexed_chunks
        ]
        context_str = "\n\n---\n\n".join(blocks)
        criteria_lines = []
        for key, label in self._COMPARISON_CRITERIA:
            if key == "business_id":
                criteria_lines.append(
                    f'- "{key}": {label} — CHỈ nhận mã nghiệp vụ dạng số/ký hiệu ngắn '
                    'được ghi rõ ràng là mã sản phẩm trong nguồn (ví dụ "3106"). '
                    'KHÔNG dùng tên sản phẩm, tên file hay product_id nội bộ thay thế.'
                )
            elif key == "model":
                criteria_lines.append(
                    f'- "{key}": {label} — CHỈ nhận model thương mại được ghi rõ trong '
                    'nguồn (ví dụ "VCN-WSRGC2"). KHÔNG dùng tên sản phẩm thay thế.'
                )
            else:
                criteria_lines.append(f'- "{key}": {label}')
        criteria_desc = "\n".join(criteria_lines)
        prompt = (
            f"Sản phẩm: {product_name}\n\n"
            f"{context_str}\n\n"
            "Trích xuất dữ liệu cho các tiêu chí sau, CHỈ dùng context ở trên "
            f"(context này chỉ thuộc về sản phẩm {product_name}, không có sản phẩm nào khác):\n"
            f"{criteria_desc}\n\n"
            'Trả lời CHỈ bằng JSON đúng định dạng sau, không thêm chữ nào khác:\n'
            '{"business_id": {"value": "...", "source_id": "SOURCE_x"} hoặc null, '
            '"model": ..., "power_supply": ..., ...}'
        )
        try:
            gen_res = self.generator.generate(
                prompt,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                max_tokens=self._CRITERIA_EXTRACTION_MAX_TOKENS,
            )
            raw_answer = gen_res.get("answer", "")
            parsed = self._parse_criteria_json(raw_answer)
            if not parsed:
                # Parsing silently returning {} is indistinguishable from
                # "model genuinely had zero data to report" without seeing
                # the raw text. Log the FULL text (not truncated) — a
                # previous version of this log line cut off at 500 chars,
                # which for a full 13-criteria response (commonly 800-1000+
                # chars) hid exactly the part of the text most likely to
                # contain the actual formatting break.
                logger.warning(
                    "Criteria extraction for '%s' returned no usable JSON "
                    "(%d chars). Raw model output: %r",
                    product_name, len(raw_answer), raw_answer,
                )
            return parsed
        except Exception as e:
            logger.error(
                "Criteria extraction failed for '%s': %s", product_name, e, exc_info=True
            )
            return {}

    @staticmethod
    def _parse_criteria_json(raw_text: str) -> Dict[str, Dict[str, str]]:
        """Robustly parse the extraction model's JSON response.

        Never raises — a malformed response just yields an empty dict
        (that product's row shows "Chưa có dữ liệu" for every criterion
        instead of crashing the whole comparison), since a small model
        occasionally wraps JSON in a code fence or adds a stray word
        despite being told not to.
        """
        text = (raw_text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        text = ChatbotService._repair_json_text(text)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(parsed, dict):
            return {}

        result: Dict[str, Dict[str, str]] = {}
        for key, entry in parsed.items():
            # Prefer the intended schema {"value": ..., "source_id": ...}.
            # Fallback: a small model reliably ignores the nested-object
            # instruction and just writes "key": "plain string value"
            # directly despite the schema/example given — confirmed via
            # real log output (valid, parseable JSON, but every value was
            # a flat string, not a {"value":...} object), which meant every
            # single field was silently dropped by the strict isinstance
            # check below, even though the underlying data was completely
            # correct. Accepting a flat string here means citation
            # precision is lost for those specific fields (no source_id
            # attached, so that cell won't get a [SOURCE_x] tag or count
            # toward the citation footer) — but showing the real value
            # without a citation tag is far better than silently discarding
            # genuinely correct data outright.
            if isinstance(entry, dict) and entry.get("value"):
                result[key] = {
                    "value": str(entry.get("value", "")).strip(),
                    "source_id": str(entry.get("source_id", "")).strip(),
                }
            elif isinstance(entry, str) and entry.strip():
                result[key] = {"value": entry.strip(), "source_id": ""}
        return result

    def _assemble_comparison_table(
        self,
        pids: List[str],
        product_names_by_pid: Dict[str, str],
        extracted_by_pid: Dict[str, Dict[str, Dict[str, str]]],
        indexed_by_pid: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    ) -> Tuple[str, set]:
        """Deterministically build the comparison markdown table from
        already-extracted per-product data.

        No LLM call in this step — every cell's value comes directly from
        extracted_by_pid[pid][key], indexed by the correct pid, so there is
        no code path by which a value could land in the wrong column.
        """
        used_criteria = [
            (key, label) for key, label in self._COMPARISON_CRITERIA
            if any(extracted_by_pid.get(pid, {}).get(key) for pid in pids)
        ]
        cited_source_ids: set = set()
        if not used_criteria:
            return "", cited_source_ids

        names = [product_names_by_pid.get(pid, pid) for pid in pids]
        header = "| Tiêu chí | " + " | ".join(names) + " |"
        sep = "|---|" + "|".join("---" for _ in pids) + "|"
        lines = [header, sep]

        # A product whose extraction used the flat-string fallback (no
        # per-field source_id — see _parse_criteria_json) has no precise
        # per-cell citation to attach. Rather than let that product's whole
        # column go uncited, fall back to citing its FULL evidence set (all
        # chunks that were actually in its own extraction context) — every
        # one of those genuinely was provided as this product's own
        # evidence, so this stays fully honest, just less granular than a
        # per-field citation would be.
        pids_with_data = {
            pid for pid in pids if extracted_by_pid.get(pid)
        }
        pids_needing_fallback_citation = {
            pid for pid in pids_with_data
            if not any(
                entry.get("source_id")
                for entry in extracted_by_pid.get(pid, {}).values()
            )
        }

        for key, label in used_criteria:
            cells = []
            for pid in pids:
                entry = extracted_by_pid.get(pid, {}).get(key)
                if entry and entry.get("value"):
                    sid = entry.get("source_id", "")
                    cell = entry["value"].replace("|", "/")
                    if sid:
                        cited_source_ids.add(sid)
                        cell = f"{cell} [{sid}]"
                    cells.append(cell)
                else:
                    cells.append("Chưa có dữ liệu trong tài liệu")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        for pid in pids_needing_fallback_citation:
            for idx, _payload in indexed_by_pid.get(pid, []):
                cited_source_ids.add(f"SOURCE_{idx}")

        return "\n".join(lines), cited_source_ids

    def _handle_compare(
        self,
        question: str,
        intent_res: Any,
        explicit_products: List[Dict[str, Any]],
        history_dicts: Any,
        debug: bool,
        t0: float,
    ) -> ChatResponse:
        t_ret_start = time.perf_counter()
        pids = [p["product_id"] for p in explicit_products]

        all_chunks = []
        coverage_by_product = {}
        missing_sections = []

        if intent_res.intent == QueryIntent.COMPARE_FULL:
            for pid in pids:
                p_chunks = self.retriever.retrieve_full(product_id=pid)
                all_chunks.extend(p_chunks)
                grp_map = self._group_chunks_by_coverage(p_chunks)
                coverage_by_product[pid] = {v["title"]: len(v["chunks"]) for v in grp_map.values()}
        else:
            # Same identical-query, different-filter situation as
            # _handle_multi_product_fact — embed once, reuse across pids.
            shared_query_vector = (
                self.retriever.embedding.embed_query(question) if question else None
            )
            for pid in pids:
                ret = self.retriever.retrieve(
                    question,
                    product_id=pid,
                    top_k=8,
                    precomputed_vector=shared_query_vector,
                )
                all_chunks.extend(ret["results"])
                coverage_by_product[pid] = {"candidates": len(ret["results"])}

        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        product_names_by_pid = {
            p["product_id"]: p.get("product_name", p["product_id"])
            for p in explicit_products
        }

        # Assign GLOBAL SOURCE_N indices up front (flat over all_chunks, in
        # retrieval order), then group by product while preserving those
        # indices — citation numbering stays consistent even though
        # extraction now runs as N independent per-product LLM calls
        # instead of one combined call asking the model to write the whole
        # table freehand.
        indexed_by_pid: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {pid: [] for pid in pids}
        all_indexed_payloads: List[Dict[str, Any]] = []
        for idx, item in enumerate(all_chunks, start=1):
            payload = item.get("payload", item)
            all_indexed_payloads.append(payload)
            item_pid = payload.get("product_id", "")
            if item_pid in indexed_by_pid:
                indexed_by_pid[item_pid].append((idx, payload))

        token_est = sum(
            estimate_tokens(str(p.get("content", ""))) for p in all_indexed_payloads
        )

        # Extract each product's data from an independent LLM call that
        # only ever sees that product's own chunks — see
        # _extract_product_criteria for why this structurally prevents
        # cross-product data mixing instead of just asking the model
        # nicely not to do it (confirmed insufficient after 2 rounds of
        # explicit prompt reinforcement in live testing).
        t_gen_start = time.perf_counter()
        extracted_by_pid: Dict[str, Dict[str, Dict[str, str]]] = {}
        with ThreadPoolExecutor(max_workers=min(max(len(pids), 1), 4)) as pool:
            future_to_pid = {
                pool.submit(
                    self._extract_product_criteria,
                    product_names_by_pid.get(pid, pid),
                    indexed_by_pid[pid],
                ): pid
                for pid in pids
            }
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    extracted_by_pid[pid] = future.result()
                except Exception as e:
                    logger.error(
                        "Criteria extraction future failed for '%s': %s", pid, e, exc_info=True
                    )
                    extracted_by_pid[pid] = {}

        table_md, cited_source_ids = self._assemble_comparison_table(
            pids, product_names_by_pid, extracted_by_pid, indexed_by_pid
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        if not table_md:
            final_ans = (
                "Tài liệu hiện có chưa cung cấp đủ dữ liệu để so sánh các sản phẩm "
                "theo yêu cầu này; cần xác nhận với bộ phận kỹ thuật Vhomenex."
            )
            sources: List[SourceReference] = []
        else:
            footer_lines = ["", "📚 Nguồn tham khảo:"]
            for sid in sorted(
                cited_source_ids, key=lambda s: int(re.sub(r"\D", "", s) or 0)
            ):
                footer_lines.append(f"- {sid}")
            final_ans = table_md + "\n" + "\n".join(footer_lines)

            # Sources are built precisely from which SOURCE_N ids the
            # extraction step actually attached to a real value — no
            # text-scanning/guessing needed, since the table was assembled
            # by code and we know exactly what was used.
            all_sources = self._build_source_references(
                [{"payload": p} for p in all_indexed_payloads]
            )
            sources = [s for s in all_sources if s.source_id in cited_source_ids]

        total_latency = (time.perf_counter() - t0) * 1000

        debug_info = None
        if debug:
            debug_info = {
                "query_intent": intent_res.intent.value,
                "requested_products": pids,
                "requested_sections": intent_res.requested_sections,
                "total_candidate_chunks": len(all_chunks),
                "selected_chunks": len(all_indexed_payloads),
                "coverage_by_product": coverage_by_product,
                "missing_sections": missing_sections,
                "context_token_estimate": token_est,
                "orchestration_steps": [
                    "compare_retrieval",
                    "per_product_criteria_extraction",
                    "deterministic_table_assembly",
                ],
                "extracted_criteria_by_product": {
                    pid: sorted(extracted_by_pid.get(pid, {}).keys()) for pid in pids
                },
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_latency,
            }

        return ChatResponse(
            answer=final_ans,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def _handle_multi_product_fact(
        self,
        question: str,
        intent_res: Any,
        explicit_products: List[Dict[str, Any]],
        history_dicts: Any,
        debug: bool,
        t0: float,
    ) -> ChatResponse:
        t_ret_start = time.perf_counter()
        all_chunks = []
        pids = [p["product_id"] for p in explicit_products]
        coverage_by_product = {}

        # Embed the question once and reuse it for every product filter
        # below — the query text is identical across the loop, only the
        # product_id filter changes, so re-embedding per product was pure
        # redundant Ollama round trips (N-1 wasted calls for N products).
        shared_query_vector = (
            self.retriever.embedding.embed_query(question) if question else None
        )

        for pid in pids:
            ret = self.retriever.retrieve(
                question,
                product_id=pid,
                top_k=6,
                precomputed_vector=shared_query_vector,
            )
            product_chunks = ret["results"]
            coverage_by_product[pid] = len(product_chunks)
            all_chunks.extend(product_chunks)

        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        user_msg, selected = self.context_builder.build(all_chunks, query=question)
        token_est = estimate_tokens(user_msg)

        resolved_product_names = [
            p.get("product_name", p["product_id"])
            for p in explicit_products
        ]

        resolved_context = (
            "Các sản phẩm đã được hệ thống xác định cho câu hỏi hiện tại: "
            + ", ".join(resolved_product_names)
        )

        multi_product_prompt = chr(10).join([
            resolved_context,
            "",
            user_msg,
            "",
            "=== YÊU CẦU TỔNG HỢP NHIỀU SẢN PHẨM ===",
            "1. Phải kiểm tra và sử dụng thông tin của TẤT CẢ sản phẩm đã xác định.",
            "2. Đối chiếu bằng chứng riêng của từng sản phẩm trước khi kết luận.",
            "3. Nếu hỏi điểm chung, chỉ nêu nội dung được tài liệu của tất cả sản phẩm cùng xác nhận.",
            "4. Nếu không có điểm chung cho tất cả sản phẩm, nêu rõ điều kiện của từng sản phẩm.",
            "5. Không bỏ qua sản phẩm và không chép riêng một SOURCE không trả lời câu hỏi.",
            "6. Mỗi nhận định phải gắn SOURCE tương ứng; cuối câu trả lời chỉ liệt kê SOURCE thực sự dùng.",
        ])

        t_gen_start = time.perf_counter()
        gen_res = self.generator.generate(
            multi_product_prompt,
            history=None,
            max_tokens=self._MAX_TOKENS_BY_INTENT.get(QueryIntent.MULTI_PRODUCT_FACT),
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        # Không dùng _preserve_procedure_steps cho câu hỏi nhiều sản phẩm.
        # Hàm đó chỉ chọn một procedure chunk và có thể thay mất câu trả lời tổng hợp.
        final_ans = self._remove_dangling_markdown_bullet(gen_res["answer"])
        sources = filter_cited_sources(final_ans, self._build_source_references(selected))
        total_latency = (time.perf_counter() - t0) * 1000

        debug_info = None
        if debug:
            debug_info = {
                "query_intent": intent_res.intent.value,
                "requested_products": pids,
                "requested_sections": intent_res.requested_sections,
                "total_candidate_chunks": len(all_chunks),
                "selected_chunks": len(selected),
                "coverage_by_product": coverage_by_product,
                "missing_sections": [
                    f"{pid}:retrieval"
                    for pid, count in coverage_by_product.items()
                    if count == 0
                ],
                "context_token_estimate": token_est,
                "orchestration_steps": ["multi_product_retrieval", "generation"],
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_latency,
            }

        return ChatResponse(
            answer=final_ans,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    @staticmethod
    def _relevant_products_for_ambiguity_check(
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group retrieval results by product, keeping only products whose
        best-ranked chunk is genuinely competitive with the top result —
        not just "appeared somewhere in the raw top-K".

        Confirmed real bug this fixes: a totally unrelated Gateway chunk
        ("Xoá thiết bị trong Gateway") scraped into rank #10 of a search
        for a roller-shutter switch's touch-lock feature, with an
        rrf_score only ~48% of the top result's — yet the old grouping
        (any 2+ distinct products anywhere in the top-K) treated Gateway
        as if it were a second genuine interpretation of the question,
        producing a nonsensical "could mean either of these two totally
        different products" clarification. A product this far behind the
        top score is retrieval noise that happened to clear the top-K
        cutoff, not a competing interpretation — only products within a
        reasonable fraction of the leading score are kept as candidates
        for the multi-product ambiguity check.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        best_score_by_product: Dict[str, float] = {}
        for item in results:
            payload = item.get("payload", item)
            pname = payload.get("product_name", "")
            if not pname:
                continue
            grouped.setdefault(pname, []).append(payload)
            score = item.get("rrf_score") or 0.0
            if score > best_score_by_product.get(pname, 0.0):
                best_score_by_product[pname] = score

        if not best_score_by_product:
            return grouped

        top_score = max(best_score_by_product.values())
        if top_score <= 0:
            return grouped

        _RELEVANCE_RATIO = 0.6
        return {
            pname: chunks
            for pname, chunks in grouped.items()
            if best_score_by_product.get(pname, 0.0) >= top_score * _RELEVANCE_RATIO
        }

    def _check_multi_product_conflict(
        self,
        question: str,
        grouped_by_product: Dict[str, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """When a FACT question resolves to no specific product and
        unrestricted retrieval (searched across the whole catalog) surfaces
        2+ distinct products, check whether they actually give the SAME
        answer (safe to answer directly, current behavior) or DIFFERENT
        answers (must tell the user which products differ, rather than
        silently picking one and presenting it as universal).

        Confirmed real bug this addresses: a question naming no product
        ("Muốn vào SmartConfig Auto và Manual thì giữ nút bao lâu, màu đèn
        nào?") retrieved evidence from 3 products whose real values
        genuinely differ (5s/10s + red/blue for one product, entirely
        different 3s/5s/7s + blue/yellow for another) — the model picked
        one product's numbers and presented them as if universally
        applicable. A purely mechanical heuristic (e.g. "multiple products
        in context → ask") would misfire constantly for the common,
        harmless case where multiple products happen to share the same
        answer (e.g. "what app does this support?" — Vhomenex for every
        product) — determining "do these values actually differ" requires
        real comprehension of the retrieved content, which is why this
        uses one extra, cheap LLM call rather than a keyword heuristic.

        Only runs for this narrow trigger case (FACT question, no product
        resolved, multi-product retrieval spread) — not for the vast
        majority of well-scoped single-product questions, so it adds no
        cost to the common path. Fails open (returns None, meaning
        "proceed with the normal answer") on any error, since this is a
        safety-net check, not a required step — an LLM hiccup here should
        never block a real answer from being given.
        """
        if len(grouped_by_product) < 2:
            return None

        blocks = []
        idx = 1
        for _pname, items in grouped_by_product.items():
            for payload in items:
                blocks.append(ContextBuilder.format_source_block(idx, payload))
                idx += 1
        context_str = "\n\n---\n\n".join(blocks)

        prompt = (
            f"Câu hỏi: {question}\n\n"
            f"{context_str}\n\n"
            "Câu trả lời cho câu hỏi trên có giống nhau ở mọi sản phẩm xuất hiện trong context, hay khác nhau tùy sản phẩm?"
        )
        try:
            gen_res = self.generator.generate(
                prompt,
                system_prompt=AMBIGUITY_CHECK_SYSTEM_PROMPT,
                max_tokens=300,
            )
            return self._parse_ambiguity_json(gen_res.get("answer", ""))
        except Exception as e:
            logger.error("Ambiguity check failed: %s", e, exc_info=True)
            return None

    @staticmethod
    def _parse_ambiguity_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Reuses the same tolerant-parsing approach as
        _parse_criteria_json — a malformed response just means "couldn't
        determine", so the caller falls back to answering normally rather
        than blocking on a parse failure."""
        text = (raw_text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parsed, dict) or "same_for_all" not in parsed:
            return None
        return {
            "same_for_all": bool(parsed.get("same_for_all", True)),
            "differing_products": [
                str(p) for p in (parsed.get("differing_products") or []) if str(p).strip()
            ],
        }

    def _handle_fact(
        self,
        question: str,
        intent_res: Any,
        effective_product_id: Optional[str],
        scope: Any,
        history_dicts: Any,
        debug: bool,
        t0: float,
    ) -> ChatResponse:
        t_ret_start = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query=scope.query if scope else question,
            product_id=effective_product_id,
            top_k=self.settings.final_top_k,
            debug=debug,
        )
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000
        results = retrieval["results"]

        # No product was resolved from the question text -- before
        # generating a normal answer, check whether the retrieved evidence
        # actually conflicts across products (see
        # _check_multi_product_conflict's docstring). Only runs when 2+
        # distinct products are GENUINELY competing for relevance — not
        # just "appeared anywhere in the raw top-K" — a no-op for the
        # common single-product case.
        if not effective_product_id and results:
            grouped_for_check = self._relevant_products_for_ambiguity_check(results)
            if len(grouped_for_check) >= 2:
                conflict = self._check_multi_product_conflict(question, grouped_for_check)
                if conflict and not conflict.get("same_for_all", True):
                    # "Differing" is meaningless with fewer than 2 named
                    # products -- fall back to the full retrieved product
                    # list whenever the model's own enumeration is missing
                    # or incomplete (confirmed real case: the model
                    # correctly determined same_for_all=false but only
                    # named ONE product, producing a self-contradictory
                    # "differs across multiple products" message naming
                    # just one).
                    differing = conflict.get("differing_products") or []
                    if len(differing) < 2:
                        differing = list(grouped_for_check.keys())
                    total_latency = (time.perf_counter() - t0) * 1000
                    clarify_text = (
                        "Câu hỏi này có thể áp dụng cho nhiều sản phẩm khác nhau với "
                        "thông tin không giống nhau: " + ", ".join(differing) + ". "
                        "Bạn đang hỏi về sản phẩm nào để mình trả lời chính xác?"
                    )
                    clarify_debug_info = None
                    if debug:
                        clarify_debug_info = {
                            "query_intent": intent_res.intent.value,
                            "requested_products": [],
                            "orchestration_steps": ["hybrid_retrieval", "multi_product_ambiguity_check"],
                            "retrieved_products": list(grouped_for_check.keys()),
                            "ambiguity_check_result": conflict,
                            "retrieval_ms": retrieval_ms,
                            "total_ms": total_latency,
                        }
                    return ChatResponse(
                        answer=clarify_text,
                        latency_ms=total_latency,
                        retrieval_ms=retrieval_ms,
                        debug_info=clarify_debug_info,
                    )

        generation_question = question
        if scope and scope.is_follow_up and scope.product_name:
            generation_question = (
                f"{question}\n\n"
                f"Ngữ cảnh hội thoại cần giữ: sản phẩm đang hỏi là "
                f"{scope.product_name}. Chỉ trả lời cho sản phẩm này."
            )

        user_message, selected = self.context_builder.build(results, generation_question)
        token_est = estimate_tokens(user_message)

        t_gen_start = time.perf_counter()
        # History is used by the context resolver to resolve follow-up references.
        # Once the product is resolved, generation must rely only on current
        # product-filtered RAG context to prevent facts leaking from older answers.
        generation_history = (
            None
            if scope and scope.is_follow_up and effective_product_id
            else history_dicts
        )
        gen_result = self.generator.generate(
            user_message,
            history=generation_history,
            max_tokens=self._MAX_TOKENS_BY_INTENT.get(intent_res.intent),
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        answer = self._finalize_answer(question, gen_result["answer"], selected)
        sources = filter_cited_sources(answer, self._build_source_references(selected))
        total_latency = (time.perf_counter() - t0) * 1000

        debug_info = None
        if debug:
            debug_info = {
                "query_intent": intent_res.intent.value,
                "requested_products": intent_res.requested_products or ([effective_product_id] if effective_product_id else []),
                "requested_sections": intent_res.requested_sections,
                "total_candidate_chunks": len(results),
                "selected_chunks": len(selected),
                "coverage_by_product": {effective_product_id or "all": len(results)},
                "missing_sections": [],
                "context_token_estimate": token_est,
                "orchestration_steps": ["hybrid_retrieval", "generation"],
                "resolved_product": retrieval.get("resolved_product"),
                "resolved_products": retrieval.get("resolved_products", []),
                "last_generation_stats": {
                    "prompt_eval_count": gen_result.get("prompt_eval_count"),
                    "eval_count": gen_result.get("eval_count"),
                    "generate_latency_ms": gen_result.get("latency_ms"),
                },
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_latency,
            }

        return ChatResponse(
            answer=answer,
            sources=sources,
            debug_info=debug_info,
            latency_ms=total_latency,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
