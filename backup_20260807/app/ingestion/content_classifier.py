"""General Content Type Classifier for RAG chunks."""
import re

class ContentTypeClassifier:
    """Classifies RAG chunks into standardized content types."""

    SECTION_MAP = [
        (re.compile(r"metadata", re.IGNORECASE), "identification"),
        (re.compile(r"tóm tắt|tổng quan|giới thiệu", re.IGNORECASE), "overview"),
        (re.compile(r"thông số|đặc tính|thông số kỹ thuật|đặc tả", re.IGNORECASE), "specification"),
        (re.compile(r"nguyên lý|thao tác|sử dụng|tính năng tại thiết bị", re.IGNORECASE), "usage_tip"),
        (re.compile(r"lắp đặt|đấu nối|sơ đồ", re.IGNORECASE), "installation"),
        (re.compile(r"kết nối|đăng ký|thêm thiết bị", re.IGNORECASE), "connection"),
        (re.compile(r"cấu hình|cài đặt|tính năng trên ứng dụng", re.IGNORECASE), "configuration"),
        (re.compile(r"reset|khôi phục", re.IGNORECASE), "reset"),
        (re.compile(r"lỗi|xử lý lỗi|sự cố|troubleshoot", re.IGNORECASE), "troubleshooting"),
        (re.compile(r"cảnh báo|an toàn|khuyên cáo", re.IGNORECASE), "safety"),
        (re.compile(r"tương thích|hỗ trợ", re.IGNORECASE), "compatibility"),
        (re.compile(r"kịch bản|automation|ngữ cảnh", re.IGNORECASE), "automation"),
    ]

    @classmethod
    def classify(cls, section_title: str, content: str, parent_title: str = "") -> str:
        sec_clean = section_title.lower()
        par_clean = parent_title.lower()
        comb_sec = f"{par_clean} {sec_clean}"

        # 1. Match Section Title & Parent Title
        for pattern, ct in cls.SECTION_MAP:
            if pattern.search(comb_sec):
                return ct

        # 2. Match Content Keywords
        cnt_lower = content.lower()
        if any(w in cnt_lower for w in ["nguồn cấp", "điện áp", "công suất", "kích thước", "tiêu chuẩn"]):
            return "specification"
        if any(w in cnt_lower for w in ["bước 1", "nhấn giữ", "đèn nháy", "tự động"]):
            return "connection"
        if any(w in cnt_lower for w in ["khôi phục", "reset", "mặc định"]):
            return "reset"
        if any(w in cnt_lower for w in ["cảnh báo", "không sử dụng", "nguy hiểm"]):
            return "safety"
        if any(w in cnt_lower for w in ["sự cố", "khắc phục", "lỗi"]):
            return "troubleshooting"

        return "feature"
