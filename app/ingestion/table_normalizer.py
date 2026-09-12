"""Table normalization for retrieval-friendly, row-level chunks."""
from __future__ import annotations

import re
from typing import Dict, List

# Truly-empty placeholders: dropping these rows loses nothing, since they
# carry no information at all (a blank field, a "to-do" marker, a bare dash
# used as a filler).
_EMPTY_VALUE_RE = re.compile(r"^(?:n/?a|tbd|todo|-|\.)$", re.IGNORECASE)

# "Không có" / "Không áp dụng" are, per the project's own documented
# template convention, meaningful NEGATIVE facts, not empty placeholders —
# "no indicator light" or "not applicable to this product" is real, citable
# information a user might directly ask about (e.g. "does this device have
# an indicator light?"). Confirmed real bug: these were being treated
# identically to genuinely-empty values and silently dropped from every
# table row, permanently losing the chatbot's ability to answer a direct
# "does X exist / apply" question with a confident, sourced "no" instead of
# a vague "no data in the document" (which reads as "maybe it exists but
# wasn't captured", a materially different and less trustworthy answer).
# These rows are now KEPT — only the truly-empty patterns above get
# dropped. See semantic_chunker.py's _clean_line for the matching
# prose-level (non-table) fix.


class TableNormalizer:
    """Convert Word tables into independent semantic records.

    Specification and FAQ rows should be retrievable on their own. Navigation
    tables (``Xem hướng dẫn``) are skipped because the detailed headings that
    follow are the authoritative content.
    """

    @staticmethod
    def _clean_rows(table_data: List[List[str]]) -> List[List[str]]:
        rows: List[List[str]] = []
        for row in table_data or []:
            cells = [str(cell or "").strip() for cell in row]
            while cells and not cells[-1]:
                cells.pop()
            if any(cells):
                rows.append(cells)
        return rows

    @classmethod
    def records(cls, table_data: List[List[str]], section_title: str = "") -> List[Dict[str, str]]:
        rows = cls._clean_rows(table_data)
        if not rows:
            return []

        first_text = " | ".join(rows[0]).lower()
        if "xem hướng dẫn" in first_text:
            return []

        # Three-column FAQ/troubleshooting table.
        if "câu hỏi" in first_text and "cách khắc phục" in first_text:
            headers = rows[0]
            output = []
            for row in rows[1:]:
                values = row + [""] * (len(headers) - len(row))
                question = values[0].strip()
                if not question or _EMPTY_VALUE_RE.match(question):
                    continue
                parts = [f"{headers[i]}: {values[i]}" for i in range(len(headers)) if values[i].strip()]
                output.append({"chunk_type": "faq", "title": question, "content": "\n".join(parts)})
            return output

        # Key/value specification table: emit one chunk per populated row.
        if all(len(row) == 2 for row in rows):
            output = []
            for key, value in rows:
                if not key or not value or _EMPTY_VALUE_RE.match(value):
                    continue
                # Skip generic table headers rather than embedding them as data.
                if key.lower() in {"thuộc tính", "tên thông số", "tên tài liệu", "stt"} and value.lower() in {
                    "giá trị", "thông số", "file tài liệu"
                }:
                    continue
                output.append({
                    "chunk_type": "spec",
                    "title": key,
                    "content": f"{key}: {value}",
                })
            return output

        # Generic multi-column table: preserve column alignment row by row.
        headers = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        output = []
        for row_index, row in enumerate(data_rows, start=1):
            values = row + [""] * (len(headers) - len(row))
            meaningful = [v for v in values if v and not _EMPTY_VALUE_RE.match(v)]
            if not meaningful or all(v.isdigit() for v in meaningful):
                continue
            parts = [f"{headers[i]}: {values[i]}" for i in range(min(len(headers), len(values))) if values[i]]
            if parts:
                output.append({
                    "chunk_type": "table_row",
                    "title": f"{section_title or 'Bảng'} - dòng {row_index}",
                    "content": " | ".join(parts),
                })
        return output

    @classmethod
    def normalize(cls, table_data: List[List[str]], max_tokens: int = 500) -> List[str]:
        """Backward-compatible text-only API used by older callers/tests."""
        return [record["content"] for record in cls.records(table_data)]
