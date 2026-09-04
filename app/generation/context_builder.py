"""Context builder grouping retrieved chunks by product for LLM generation with strict token budgeting."""
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count for Vietnamese + English technical text.

    Approximates ~1.35 tokens per word for Vietnamese, or tiktoken if installed.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        words = len(text.split())
        return int(words * 1.35)


class ContextBuilder:
    def __init__(self, settings: Any = None):
        self.settings = settings
        self.max_tokens = getattr(settings, "max_context_tokens", 6000)

    def _is_unbreakable_chunk(self, content: str) -> bool:
        """Check if chunk contains procedure steps (B1..Bn) or FAQ/spec record that must not be split."""
        if re.search(r"(?ms)^\s*B\d+\s*[:.)-]", content):
            return True
        if "thông số kỹ thuật" in content.lower() or "bảng thông số" in content.lower():
            return True
        if "hướng dẫn" in content.lower() or "quy trình" in content.lower():
            return True
        return False

    @staticmethod
    def format_source_block(idx: int, p: Dict[str, Any]) -> str:
        """Format one chunk as a `[SOURCE_n] (...)` block for LLM context.

        The single, canonical implementation of this — build_context()
        below calls it for the normal chat/compare flow, and
        ChatbotService's deterministic per-product comparison extraction
        (see _extract_product_criteria) calls it directly to build a
        single-product context using pre-assigned *global* SOURCE_n
        indices (so citation numbers stay consistent when a compare
        question's evidence is split into several per-product LLM calls).

        Do not duplicate this logic elsewhere — a near-identical but
        subtly stale copy of exactly this block (missing the disambiguated
        `title` fallback below) sat unused in app/generation/prompts.py for
        a long time and could easily have been mistaken for live code by a
        future edit.
        """
        sid = f"SOURCE_{idx}"
        doc = p.get("source_document", p.get("source_file", ""))
        # semantic_chunker.py can derive a disambiguating `title` for a
        # chunk (e.g. "Tính năng trên thiết bị — Lưu ý: ...sẵn sàng kết
        # nối...") that's more specific than the raw section name baked
        # into `source_locator`/`heading_path`/`source_section` (those are
        # all built from the plain document heading hierarchy). Append it
        # whenever it differs, so this signal reaches the model regardless
        # of which locator field would otherwise win.
        base_locator = (
            p.get("source_locator", "") or p.get("heading_path", "") or p.get("source_section", "")
        )
        chunk_title = p.get("title", "")
        raw_section = p.get("source_section", "")
        if chunk_title and chunk_title != raw_section and chunk_title not in base_locator:
            locator = f"{base_locator} ({chunk_title})" if base_locator else chunk_title
        else:
            locator = base_locator or chunk_title
        ct = p.get("content_type", "")
        chunk_type = p.get("chunk_type", "")
        content = p.get("content", "")
        image_paths = p.get("image_paths") or []
        # Give the model textual awareness that this chunk has an attached
        # picture — it never sees the actual image bytes (those are only
        # attached at the API/schema layer for the UI to render, entirely
        # outside this text prompt), so a chunk with real image_paths but
        # empty/near-empty `content` (the common case: a section titled
        # e.g. "Hình ảnh sản phẩm" that IS just a picture, no caption text)
        # otherwise reads to the model as "nothing here", producing a
        # self-contradictory answer like "tài liệu không cung cấp hình ảnh"
        # even while the real photo is rendering right below it in the UI.
        # Confirmed in live testing against real product docs.
        if image_paths and not content.strip():
            content = (
                f"[Mục này có {len(image_paths)} hình ảnh minh họa đính kèm, "
                "sẽ được hiển thị trực tiếp cho người dùng — không cần mô tả lại bằng lời.]"
            )
        return (
            f"[{sid}] (Tài liệu: {doc} | Vị trí: {locator} | "
            f"Loại nội dung: {ct} | Loại chunk: {chunk_type})\n{content}"
        )

    def build_context(
        self,
        sources: List[Dict[str, Any]],
        query: str = "",
        max_tokens_override: int = 0,
    ) -> Tuple[str, List[Dict[str, Any]], int, bool]:
        """Build formatted context string respecting token budget."""
        if not sources:
            return "Không tìm thấy dữ liệu liên quan.", [], 0, False

        token_budget = max_tokens_override or self.max_tokens

        grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        for idx, src in enumerate(sources, 1):
            payload = src.get("payload", src)
            pname = payload.get("product_name", "Thiết bị Vhomenex")
            if pname not in grouped:
                grouped[pname] = []
            grouped[pname].append((idx, payload))

        num_products = max(len(grouped), 1)

        selected_items = []
        total_tokens = 0
        is_overflow = False

        blocks = []
        for pname, items in grouped.items():
            header = f"=== SẢN PHẨM: {pname.upper()} ==="
            header_tokens = estimate_tokens(header)
            product_tokens = header_tokens
            item_blocks = []

            voice_budget = int(token_budget * 0.20)
            voice_tokens_used = 0

            for idx, p in items:
                ct = p.get("content_type", "")
                chunk_type = p.get("chunk_type", "")
                content = p.get("content", "")

                if ct == "voice_command" or chunk_type == "voice_command":
                    chunk_tok = estimate_tokens(content)
                    if voice_tokens_used + chunk_tok > voice_budget:
                        lines = content.splitlines()
                        kept_lines = []
                        cur_tok = 0
                        for line in lines:
                            l_tok = estimate_tokens(line)
                            if cur_tok + l_tok > voice_budget:
                                break
                            kept_lines.append(line)
                            cur_tok += l_tok
                        content = "\n".join(kept_lines) + "\n... (đã rút gọn bảng khẩu lệnh để đảm bảo context)"
                        p = {**p, "content": content}

                block = self.format_source_block(idx, p)
                block_tok = estimate_tokens(block)

                if total_tokens + block_tok > token_budget:
                    is_overflow = True
                    if self._is_unbreakable_chunk(content) and (total_tokens + block_tok <= token_budget + 500):
                        item_blocks.append(block)
                        selected_items.append(p)
                        total_tokens += block_tok
                        product_tokens += block_tok
                    else:
                        logger.warning(
                            "Context overflow: dropping chunk SOURCE_%d (%d tokens). Total so far: %d / %d",
                            idx, block_tok, total_tokens, token_budget
                        )
                    continue

                item_blocks.append(block)
                selected_items.append({"payload": p} if "payload" not in p else p)
                total_tokens += block_tok
                product_tokens += block_tok

            if item_blocks:
                blocks.append(header + "\n" + "\n\n".join(item_blocks))

        ctx_str = "\n\n" + ("=" * 40) + "\n\n" + "\n\n".join(blocks) if blocks else "Không tìm thấy dữ liệu liên quan."
        return ctx_str, selected_items, total_tokens, is_overflow

    def build(
        self,
        sources: List[Dict[str, Any]],
        query: str = "",
        max_tokens_override: int = 0,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        ctx_str, selected, token_est, _ = self.build_context(sources, query, max_tokens_override)
        user_message = (
            f"Ngữ cảnh tài liệu:\n{ctx_str}\n\n"
            f"=== CÂU HỎI NGƯỜI DÙNG ===\n{query}\n\n"
            "Hãy trả lời đúng trọng tâm câu hỏi trên và chỉ sử dụng dữ liệu trong ngữ cảnh."
        )
        return user_message, selected
