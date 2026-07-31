"""
DOCX loader that preserves XML order between paragraphs and tables.
Uses python-docx body element iteration to maintain document flow.
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Union

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

# Styles considered headings
HEADING_STYLES = {
    "Heading 1", "Heading 2", "Heading 3", "Heading 4", "Heading 5",
    "Tiêu đề 1", "Tiêu đề 2", "Tiêu đề 3",
}

# Regex for page number lines
_PAGE_RE = re.compile(r"^\s*(Trang|Page)\s*\d+\s*$", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"^(\[.*?\]|<.*?>|TODO|TBD|N\/A|Không áp dụng)$", re.IGNORECASE
)
DO_NOT_INGEST_MARKER = "DO_NOT_INGEST"


@dataclass
class DocElement:
    """Represents a loaded element from DOCX preserving source order."""
    kind: str          # "heading" | "paragraph" | "bullet" | "numbered" | "table" | "caption"
    text: str          # plain text content
    level: int = 0     # heading level (1-6) or list level
    raw_rows: List[List[str]] = field(default_factory=list)  # for tables


def _iter_block_items(document: Document) -> Generator[Union[Paragraph, Table], None, None]:
    """Iterate document body elements in XML order (paragraphs AND tables)."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _classify_paragraph(para: Paragraph) -> str:
    style = para.style.name if para.style else ""
    if style in HEADING_STYLES or style.startswith("Heading") or style.startswith("Tiêu đề"):
        return "heading"
    pPr = para._p.find(qn("w:pPr"))
    if pPr is not None:
        numPr = pPr.find(qn("w:numPr"))
        if numPr is not None:
            ilvl = numPr.find(qn("w:ilvl"))
            lvl = int(ilvl.get(qn("w:val"), 0)) if ilvl is not None else 0
            return "numbered" if lvl == 0 else "bullet"
    text_lower = para.text.strip().lower()
    if style == "Caption" or text_lower.startswith(("hình ", "bảng ", "figure ", "table ")):
        return "caption"
    return "paragraph"


def _heading_level(para: Paragraph) -> int:
    style = para.style.name if para.style else ""
    for i in range(1, 7):
        if style == f"Heading {i}" or style == f"Tiêu đề {i}":
            return i
    return 1


def _table_to_rows(table: Table) -> List[List[str]]:
    rows = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        # Dedupe merged cells
        deduped = []
        prev = None
        for c in cells:
            if c != prev:
                deduped.append(c)
            prev = c
        rows.append(deduped)
    return rows


def _rows_to_text(rows: List[List[str]]) -> str:
    lines = []
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _is_skip(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if _PAGE_RE.match(t):
        return True
    if DO_NOT_INGEST_MARKER in t:
        return True
    return False


def load_docx(path: Union[str, Path]) -> List[DocElement]:
    """Load a DOCX file and return ordered list of DocElements."""
    path = Path(path)
    logger.info("Loading DOCX: %s", path.name)
    doc = Document(str(path))

    seen_texts: set[str] = set()  # for deduplication of repeated headers/footers
    elements: List[DocElement] = []

    for item in _iter_block_items(doc):
        if isinstance(item, Paragraph):
            raw_text = item.text.strip()
            if _is_skip(raw_text):
                continue
            # Deduplicate repeated header/footer text
            if len(raw_text) < 120 and raw_text in seen_texts:
                continue
            if len(raw_text) < 120:
                seen_texts.add(raw_text)

            kind = _classify_paragraph(item)
            level = _heading_level(item) if kind == "heading" else 0
            elements.append(DocElement(kind=kind, text=raw_text, level=level))

        elif isinstance(item, Table):
            rows = _table_to_rows(item)
            text = _rows_to_text(rows)
            if text.strip():
                elements.append(DocElement(kind="table", text=text, raw_rows=rows))

    logger.info("Loaded %d elements from %s", len(elements), path.name)
    return elements
