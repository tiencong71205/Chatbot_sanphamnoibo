"""Vietnamese text normalization utilities."""
from __future__ import annotations
import re
import unicodedata

# Preserved states — never replace these
PRESERVE_STATES = {
    "không có",
    "không áp dụng",
    "chưa xác định - cần xác nhận",
    "mâu thuẫn - chờ xác minh",
    "n/a",
}

# Technical tokens to keep intact
_TECH_TOKEN_RE = re.compile(
    r"(?:"
    r"[A-Z]{2,}[-][A-Z0-9]+[A-Z0-9]*"   # VCN-WSRGC2
    r"|\d{1,3}[-]\d{1,3}V"              # 110-240V
    r"|\d{2,3}[-]\d{2}Hz"               # 50-60Hz
    r"|[A-Z]{1,3}\d{3,5}"               # CR2450
    r"|FW\d+\.\d+\.\d+"             # FW2.3.0
    r"|v\d+\.\d+"                      # v2.3
    r"|\d+\.\d+\.\d+"               # 2.3.0
    r")",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs to single space; strip lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_control_chars(text: str) -> str:
    """Remove control characters except newline and tab."""
    return "".join(c for c in text if c == "\n" or c == "\t" or not unicodedata.category(c).startswith("C"))


def is_preserved_state(text: str) -> bool:
    return text.strip().lower() in PRESERVE_STATES


def normalize_text(text: str) -> str:
    """Full normalization pipeline."""
    if is_preserved_state(text):
        return text
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)
    return text


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 3.5 chars for Vietnamese."""
    return max(1, len(text) // 3)
