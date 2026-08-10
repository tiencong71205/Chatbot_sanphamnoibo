"""Extract product metadata from DOCX content and structure."""
from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Content type keyword mapping — ORDER MATTERS (more specific first)
CONTENT_TYPE_PATTERNS = {
    "reset": [
        r"\breset\b", r"khôi phục", r"factory reset", r"đặt lại",
        r"giữ nút.*\d+.*giây", r"nhấn.*nút.*\d+",
    ],
    "troubleshooting": [
        r"xử lý lỗi", r"troubleshoot", r"sự cố", r"\blỗi\b",
        r"không hoạt động", r"không phản hồi", r"không thể",
    ],
    "safety": [
        r"an toàn", r"cảnh báo", r"\bwarning\b", r"\bsafety\b", r"nguy hiểm",
        r"\bcaution\b",
    ],
    "installation": [
        r"lắp đặt", r"cài đặt ban đầu", r"\binstallation\b", r"\bmounting\b",
        r"kết nối vật lý",
    ],
    "connection": [
        r"kết nối", r"\bpairing\b", r"ghép nối", r"\bconnection\b",
        r"\bbluetooth\b", r"\bwifi\b", r"\bzigbee\b", r"\bz-wave\b",
    ],
    "specification": [
        r"thông số kỹ thuật", r"thông số", r"\bspecification\b", r"\bspecs\b",
        r"công suất", r"điện áp", r"tần số", r"kích thước", r"\d+[-]\d+V",
        r"\d+[-]\d+Hz",
    ],
    "configuration": [
        r"cấu hình", r"thiết lập", r"\bconfiguration\b", r"\bsetting\b",
        r"cài đặt hệ thống",
    ],
    "feature": [
        r"tính năng", r"chức năng", r"\bfeature\b", r"\bfunction\b",
    ],
    "automation": [
        r"tự động hóa", r"\bautomation\b", r"\blịch\b", r"\bschedule\b",
        r"\bscene\b", r"kịch bản",
    ],
    "maintenance": [
        r"bảo trì", r"bảo dưỡng", r"\bmaintenance\b", r"vệ sinh",
    ],
    "compatibility": [
        r"tương thích", r"\bcompatibility\b", r"hỗ trợ.*phiên bản",
        r"\bcompatible\b",
    ],
    "identification": [
        r"nhận diện", r"mã sản phẩm", r"mã model", r"\bidentification\b",
        r"\bserial\b", r"\bbarcode\b",
    ],
    "usage_tip": [
        r"mẹo sử dụng", r"lưu ý", r"ghi chú", r"\btip\b", r"\bnote\b",
        r"khuyến nghị",
    ],
    "overview": [
        r"tổng quan", r"giới thiệu", r"mô tả chung", r"\boverview\b",
    ],
}

_COMPILED_PATTERNS = {
    ct: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ct, patterns in CONTENT_TYPE_PATTERNS.items()
}

VERSION_RE = re.compile(
    r"(?:FW|firmware|phần cứng|HW|hardware|app|ứng dụng)\s*:?\s*"
    r"(v?\d+\.\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
MODEL_RE = re.compile(
    r"(?:model|mã model|sản phẩm)\s*:?\s*([A-Z]{2,}[-]?[A-Z0-9]+(?:[-][A-Z0-9]+)*)",
    re.IGNORECASE,
)


def detect_content_type(text: str, heading: str = "") -> str:
    """Detect content type from text and section heading — ordered by priority."""
    combined = f"{heading} {text}".lower()
    for ct, patterns in _COMPILED_PATTERNS.items():
        for pat in patterns:
            if pat.search(combined):
                return ct
    return "overview"


def extract_versions(text: str) -> dict:
    versions: dict[str, str] = {}
    for m in VERSION_RE.finditer(text):
        full = m.group(0).lower()
        ver = m.group(1)
        if "fw" in full or "firmware" in full:
            versions["firmware_version"] = ver
        elif "hw" in full or "hardware" in full or "phần cứng" in full:
            versions["hardware_version"] = ver
        elif "app" in full:
            versions["app_version"] = ver
    return versions


def extract_model(text: str) -> Optional[str]:
    m = MODEL_RE.search(text)
    return m.group(1) if m else None


def extract_conditions_limitations(text: str) -> tuple[list[str], list[str]]:
    conditions: list[str] = []
    limitations: list[str] = []
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if any(kw in lower for kw in ["điều kiện:", "yêu cầu:", "khi:"]):
            conditions.append(stripped)
        elif any(kw in lower for kw in ["giới hạn:", "tối đa:", "tối thiểu:", "không được:", "lưu ý:"]):
            limitations.append(stripped)
    return conditions, limitations


def extract_keywords(text: str, product_name: str = "", model: str = "") -> list[str]:
    keywords: list[str] = []
    tech_re = re.compile(
        r"(?:[A-Z]{2,}[-][A-Z0-9]+|\d{1,3}[-]\d{1,3}V|\d{2,3}[-]\d{2}Hz"
        r"|[A-Z]{1,3}\d{3,5}|FW\d+\.\d+\.\d+|v\d+\.\d+\.\d+)",
        re.IGNORECASE,
    )
    for m in tech_re.finditer(text):
        kw = m.group(0)
        if kw not in keywords:
            keywords.append(kw)
    if product_name and product_name not in keywords:
        keywords.append(product_name)
    if model and model not in keywords:
        keywords.append(model)
    return keywords[:20]
