"""Context builder grouping retrieved chunks by product for LLM generation."""
from typing import List, Dict, Any, Tuple

class ContextBuilder:
    def __init__(self, settings: Any = None):
        self.settings = settings

    def build_context(self, sources: List[Dict[str, Any]], query: str = "") -> str:
        if not sources:
            return "Không tìm thấy dữ liệu liên quan."

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for idx, src in enumerate(sources, 1):
            payload = src.get("payload", src)
            pname = payload.get("product_name", "Thiết bị Vhomenex")
            if pname not in grouped:
                grouped[pname] = []
            grouped[pname].append((idx, payload))

        blocks = []
        for pname, items in grouped.items():
            header = f"=== SẢN PHẨM: {pname.upper()} ==="
            item_blocks = []
            for idx, p in items:
                sid = f"SOURCE_{idx}"
                doc = p.get("source_document", p.get("source_file", ""))
                sec = p.get("source_section", p.get("title", ""))
                ct = p.get("content_type", "")
                content = p.get("content", "")
                
                block = f"[{sid}] (Tài liệu: {doc} | Section: {sec} | Type: {ct})\n{content}"
                item_blocks.append(block)
            
            blocks.append(header + "\n" + "\n\n".join(item_blocks))

        return "\n\n" + ("="*40) + "\n\n".join(blocks)

    def build(self, sources: List[Dict[str, Any]], query: str = "") -> Tuple[str, List[Dict[str, Any]]]:
        ctx_str = self.build_context(sources, query)
        user_message = (
            f"Ngữ cảnh tài liệu:\n{ctx_str}\n\n"
            f"=== CÂU HỎI NGƯỜI DÙNG ===\n{query}\n\n"
            "Hãy trả lời đúng trọng tâm câu hỏi trên và chỉ sử dụng dữ liệu trong ngữ cảnh."
        )
        selected = sources
        return user_message, selected
