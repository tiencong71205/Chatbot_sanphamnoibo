"""Shared document/chunk validation used by both the CLI validator and ingest_service.

Kept intentionally dependency-free (no Qdrant/Ollama) so it can run as a fast
pre-check before any embedding call is made.

Two tiers, on purpose:
- errors  (validate_chunks): block ingestion of that file entirely. Reserved
  for problems that would make the chatbot answer *wrong* (broken step
  order, missing locator/hash, cover-page leakage).
- warnings (warn_chunks): do NOT block ingestion. Flag data-quality issues
  that degrade answer completeness/precision but aren't outright wrong
  (missing model, spec table mostly "chưa có dữ liệu", a procedure with a
  single step, a suspiciously short chunk). These are meant to be reviewed
  by whoever is doing data entry for the next batch of product docs.
"""
from __future__ import annotations

import re
from typing import List

from app.models.chunk import Chunk

_STEP_NUM_RE = re.compile(r"(?m)^B(\d+)\s*:")
_PLACEHOLDER_VALUE_RE = re.compile(
    r"^(?:chưa có dữ liệu|không áp dụng|n/?a|tbd|todo)$", re.IGNORECASE
)
MIN_CHUNK_WORDS = 4


def _split_into_subprocedures(steps: List[int]) -> List[List[int]]:
    """Split a flat step-number sequence into independent sub-procedures.

    A document section legitimately contains more than one procedure — e.g.
    "5.5. Kết nối với ứng dụng" has an "Đăng ký tự động:" flow (B1, B2) and a
    separate "Đăng ký thủ công:" flow (B1, B2) right after it. Each restarts
    its own numbering at B1 by design, not by mistake. Whenever B1 shows up
    again after at least one prior step, treat it as the start of a new
    sub-procedure instead of a broken sequence — continuity is then checked
    independently within each sub-procedure.
    """
    runs: List[List[int]] = []
    current: List[int] = []
    for value in steps:
        if value == 1 and current:
            runs.append(current)
            current = [value]
        else:
            current.append(value)
    if current:
        runs.append(current)
    return runs


def validate_chunks(chunks: List[Chunk], filename: str = "") -> List[str]:
    """Return a list of human-readable error strings. Empty list = valid.

    Checks (same rules as scripts/validate_template_doc.py):
    - at least one chunk was produced
    - every chunk has heading_path / source_locator / content_hash
    - cover-page boilerplate did not leak into a chunk
    - within each `procedure` chunk, every B1..Bn sub-procedure (see
      _split_into_subprocedures) is contiguous and starts at 1
    """
    errors: List[str] = []
    prefix = f"[{filename}] " if filename else ""

    if not chunks:
        errors.append(f"{prefix}Tài liệu không tạo được chunk nào.")
        return errors

    if any(not chunk.heading_path for chunk in chunks):
        errors.append(f"{prefix}Có chunk thiếu heading_path.")
    if any(not chunk.source_locator for chunk in chunks):
        errors.append(f"{prefix}Có chunk thiếu source_locator.")
    if any(not chunk.content_hash for chunk in chunks):
        errors.append(f"{prefix}Có chunk thiếu content_hash.")
    if any("CÔNG TY CỔ PHẦN" in chunk.content for chunk in chunks):
        errors.append(f"{prefix}Nội dung bìa bị đưa vào chunk.")

    for chunk in chunks:
        if chunk.chunk_type != "procedure":
            continue
        steps = [int(value) for value in _STEP_NUM_RE.findall(chunk.content)]
        for sub_steps in _split_into_subprocedures(steps):
            if sub_steps != list(range(1, max(sub_steps) + 1)):
                errors.append(
                    f"{prefix}{chunk.title}: thứ tự bước không liên tục: {sub_steps} "
                    f"(toàn bộ các bước trong mục: {steps})"
                )

    return errors


def warn_chunks(chunks: List[Chunk], filename: str = "") -> List[str]:
    """Return non-blocking data-quality warnings for a successfully-validated file.

    Only called on files that already passed validate_chunks (0 errors), so
    these are purely advisory — ingestion proceeds either way.
    """
    warnings: List[str] = []
    prefix = f"[{filename}] " if filename else ""
    if not chunks:
        return warnings

    product_name = chunks[0].product_name
    model = chunks[0].model
    document_version = chunks[0].document_version

    if not model or _PLACEHOLDER_VALUE_RE.match(model or ""):
        warnings.append(f"{prefix}Thiếu model cho sản phẩm '{product_name}'.")
    if not document_version:
        warnings.append(f"{prefix}Thiếu phiên bản tài liệu (PHIÊN BẢN).")

    spec_chunks = [c for c in chunks if c.chunk_type == "spec"]
    if not spec_chunks:
        warnings.append(f"{prefix}Không có chunk 'spec' nào — thiếu bảng thông số kỹ thuật?")
    else:
        placeholder_specs = [
            c for c in spec_chunks
            if _PLACEHOLDER_VALUE_RE.search(c.content.split(":", 1)[-1].strip())
        ]
        if len(placeholder_specs) >= max(1, len(spec_chunks) // 2):
            warnings.append(
                f"{prefix}{len(placeholder_specs)}/{len(spec_chunks)} thông số kỹ thuật "
                "đang để 'Chưa có dữ liệu'/'Không áp dụng'."
            )

    if not any(c.chunk_type == "faq" for c in chunks):
        warnings.append(f"{prefix}Không có mục Câu hỏi thường gặp/FAQ.")

    for chunk in chunks:
        if chunk.chunk_type == "procedure":
            steps = _STEP_NUM_RE.findall(chunk.content)
            if len(steps) == 1:
                warnings.append(
                    f"{prefix}{chunk.title}: quy trình chỉ có 1 bước — có thể thiếu bước."
                )
        word_count = len(chunk.content.split())
        if word_count < MIN_CHUNK_WORDS and chunk.chunk_type not in {"spec", "table_row"}:
            warnings.append(
                f"{prefix}{chunk.title}: nội dung rất ngắn ({word_count} từ) — "
                "kiểm tra xem có bị cắt/parse lỗi không."
            )

    return warnings
