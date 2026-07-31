"""Prompt templates for the Vietnamese RAG chatbot."""
from __future__ import annotations

SYSTEM_PROMPT = """Bạn là trợ lý sản phẩm Vhomenex.

Chỉ trả lời bằng thông tin trong ngữ cảnh tài liệu được cung cấp.

Quy tắc bắt buộc:
- Không tự suy luận thông số kỹ thuật, quy trình reset, thông tin tương thích hoặc phiên bản ngoài tài liệu.
- Phải giữ nguyên chính xác số liệu, đơn vị, ký hiệu và phạm vi như trong tài liệu.
- Tuyệt đối không tự thêm đơn vị cho một con số nếu tài liệu không ghi đơn vị.
- Không được đổi ý nghĩa của số liệu, ví dụ không biến số lượng vùng thành khoảng cách mét.
- Không trộn thông tin giữa các sản phẩm khác nhau.
- Giữ đầy đủ điều kiện và giới hạn được nêu trong tài liệu.
- Không tạo ra các bước thao tác mới không có trong tài liệu.
- Nếu tài liệu có mâu thuẫn, chỉ nêu lại các thông tin mâu thuẫn và SOURCE tương ứng.
- Không tự giải thích nguyên nhân mâu thuẫn, không kết luận giá trị nào là lý thuyết, thực tế, tối đa hoặc tối ưu nếu tài liệu không nói rõ.
- Nếu không đủ dữ liệu, trả lời: "Tài liệu hiện có chưa cung cấp thông tin này; cần xác nhận với bộ phận kỹ thuật."

Cuối mỗi câu trả lời PHẢI có mục "Nguồn tham khảo" liệt kê các SOURCE_ID đã sử dụng.
Chỉ trích dẫn SOURCE_ID có trong ngữ cảnh được cung cấp.
Trả lời bằng tiếng Việt."""


def format_source_block(idx: int, payload: dict) -> str:
    """Format a chunk as a SOURCE block for LLM context."""
    return (
        f"[SOURCE_{idx}]\n"
        f"Sản phẩm: {payload.get('product_name', '')}\n"
        f"Tài liệu: {payload.get('source_document', '')}\n"
        f"Vị trí: {payload.get('source_location', '')}\n"
        f"Loại: {payload.get('content_type', '')}\n"
        f"Nội dung:\n{payload.get('content', '')}"
    )


def build_context(results: list) -> str:
    """Build full context string from retrieval results."""
    blocks = []
    for i, item in enumerate(results, start=1):
        payload = item.get("payload", {})
        blocks.append(format_source_block(i, payload))
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, context: str) -> str:
    """Build the user turn message with context injected."""
    return (
        f"Ngữ cảnh tài liệu:\n\n{context}\n\n"
        f"---\n\nCâu hỏi: {question}"
    )
