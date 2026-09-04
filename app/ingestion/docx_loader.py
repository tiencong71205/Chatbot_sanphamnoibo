"""
DOCX loader that preserves XML order between paragraphs and tables.
Uses python-docx body element iteration to maintain document flow.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Union

from docx import Document
from docx.oxml.ns import nsmap, qn
from docx.table import Table
from docx.text.paragraph import Paragraph

# python-docx registers namespace prefixes for reading/writing modern
# DrawingML pictures (<w:drawing>) out of the box, but has no built-in
# support for VML (<w:pict>/<v:imagedata>) — an older embedding mechanism
# some real documents still contain (confirmed: a genuine product photo
# in one of this project's real docs turned out to be VML-only). Without
# registering "v" here, qn("v:imagedata") below would raise a KeyError.
if "v" not in nsmap:
    nsmap["v"] = "urn:schemas-microsoft-com:vml"

logger = logging.getLogger(__name__)

# Styles considered headings
HEADING_STYLES = {
    "Heading 1", "Heading 2", "Heading 3", "Heading 4", "Heading 5",
    "Tiêu đề 1", "Tiêu đề 2", "Tiêu đề 3",
}

# Regex for page number lines
_PAGE_RE = re.compile(r"^\s*(Trang|Page)\s*\d+\s*$", re.IGNORECASE)
_TOP_LEVEL_RE = re.compile(r"^\s*[A-ZÀ-Ỹ]\s*[.\)]\s+\S+", re.IGNORECASE)
_PUNCTUATION_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
DO_NOT_INGEST_MARKER = "DO_NOT_INGEST"


@dataclass
class DocElement:
    """Represents a loaded element from DOCX preserving source order."""
    kind: str          # "heading" | "paragraph" | "bullet" | "numbered" | "table" | "caption" | "image"
    text: str          # plain text content
    level: int = 0     # heading level (1-6) or list level
    raw_rows: List[List[str]] = field(default_factory=list)  # for tables
    style_name: str = ""
    source_index: int = 0
    is_front_matter: bool = False
    # For kind="image": raw image bytes and its MIME content type (e.g.
    # "image/png"). A paragraph made up of ONLY an image has empty `text`
    # (see _paragraph_image_blobs's docstring for why that used to make the
    # whole paragraph — image included — silently vanish before any image
    # check ran) — `text` on an image element instead holds a short caption
    # derived from an adjacent "Hình ..." caption paragraph when one exists,
    # or stays empty otherwise.
    image_bytes: Optional[bytes] = None
    image_content_type: str = ""


def _iter_block_items(document: Document) -> Generator[Union[Paragraph, Table], None, None]:
    """Iterate document body elements in XML order (paragraphs AND tables)."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _paragraph_image_blobs(para: Paragraph) -> List[tuple]:
    """Extract (raw_bytes, content_type) for every image embedded in
    this paragraph's runs — modern DrawingML pictures (<w:drawing>,
    covering both inline and floating/anchored placement) and legacy VML
    pictures, wherever a <v:imagedata> element appears within the run.

    VML pictures aren't always wrapped in a plain <w:pict> — an image
    inserted via Word's "Insert Object" (rather than "Insert Picture"),
    e.g. a pasted Paintbrush/bitmap object, wraps it in <w:object> instead,
    with the <v:imagedata> holding the *preview* bitmap Word displays for
    that embedded object (a separate <o:OLEObject> element alongside it
    references the "live" OLE package itself, a different part entirely —
    not what a chunk should show a user, since it's not a directly
    viewable image format). Searching the run's whole subtree for
    <v:imagedata> rather than only inside a specific wrapper tag catches
    both cases without needing to separately special-case every wrapper
    Word might use. Confirmed necessary against a real product doc during
    this feature's rollout: <w:pict>-only detection missed a genuine OLE-
    wrapped product photo entirely, with zero warning or trace anywhere.

    The two mechanisms (DrawingML vs. VML) use different XML structures
    and even a different attribute name for the same underlying
    relationship reference (`r:embed` for DrawingML's blip vs. `r:id` for
    VML's imagedata), so both need to be checked explicitly — neither
    subsumes the other.
    """
    blobs = []
    document_part = para.part
    for run in para.runs:
        for drawing in run._element.findall(qn("w:drawing")):
            for blip in drawing.findall(".//" + qn("a:blip")):
                r_id = blip.get(qn("r:embed"))
                if r_id and r_id in document_part.rels:
                    image_part = document_part.rels[r_id].target_part
                    blobs.append((image_part.blob, image_part.content_type))
        for imagedata in run._element.findall(".//" + qn("v:imagedata")):
            r_id = imagedata.get(qn("r:id"))
            if r_id and r_id in document_part.rels:
                image_part = document_part.rels[r_id].target_part
                blobs.append((image_part.blob, image_part.content_type))
    return blobs


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
    if _PUNCTUATION_ONLY_RE.fullmatch(t):
        return True
    return False


def _looks_like_content_start(para: Paragraph) -> bool:
    """Return True for the first real top-level document heading.

    The template normally uses Heading 1, but some files contain a visually
    identical lettered heading (for example ``B. THÔNG TIN SẢN PHẨM``) with
    ``List Paragraph`` style. Supporting both keeps ingestion tolerant without
    changing the author's Heading 1/2 structure.
    """
    if _classify_paragraph(para) == "heading" and _heading_level(para) == 1:
        return True
    return bool(_TOP_LEVEL_RE.match(para.text.strip()))


def _table_image_blobs(table: Table) -> List[tuple]:
    """Extract (raw_bytes, content_type) for every image embedded inside
    any cell of this table — e.g. a "Mô tả | Hình minh họa" style table
    with a product photo placed directly in the second column, a layout
    _table_to_rows() alone can never surface since it only reads
    `cell.text`.

    Merged cells repeat the SAME underlying cell object across every grid
    position they span (already handled for text via the dedup in
    _table_to_rows()); tracked here by identity so a merged image cell
    isn't extracted once per spanned column/row.
    """
    blobs = []
    seen_cell_ids = set()
    for row in table.rows:
        for cell in row.cells:
            cell_id = id(cell._tc)
            if cell_id in seen_cell_ids:
                continue
            seen_cell_ids.add(cell_id)
            for para in cell.paragraphs:
                blobs.extend(_paragraph_image_blobs(para))
    return blobs


def load_docx(path: Union[str, Path]) -> List[DocElement]:
    """Load a DOCX file and return ordered list of DocElements."""
    path = Path(path)
    logger.info("Loading DOCX: %s", path.name)
    doc = Document(str(path))

    elements: List[DocElement] = []
    in_content = False

    for source_index, item in enumerate(_iter_block_items(doc)):
        if isinstance(item, Paragraph):
            raw_text = item.text.strip()

            # Check for embedded image(s) BEFORE the empty-text skip below —
            # a paragraph holding only a picture (the common case: an image
            # on its own line) has empty `raw_text` and would otherwise be
            # dropped by _is_skip() here, taking the image down with it,
            # before any image-specific handling ever got a chance to run.
            image_blobs = _paragraph_image_blobs(item) if in_content else []
            if image_blobs:
                for blob, content_type in image_blobs:
                    elements.append(DocElement(
                        kind="image",
                        text=raw_text,  # usually empty; kept in case of an inline caption
                        style_name=item.style.name if item.style else "",
                        source_index=source_index,
                        is_front_matter=not in_content,
                        image_bytes=blob,
                        image_content_type=content_type,
                    ))
                if not raw_text:
                    continue  # nothing else useful in an image-only paragraph

            if _is_skip(raw_text):
                continue

            if not in_content and _looks_like_content_start(item):
                in_content = True

            kind = _classify_paragraph(item)
            level = _heading_level(item) if kind == "heading" else 0
            elements.append(DocElement(
                kind=kind if in_content else "metadata",
                text=raw_text,
                level=level,
                style_name=item.style.name if item.style else "",
                source_index=source_index,
                is_front_matter=not in_content,
            ))

        elif isinstance(item, Table):
            rows = _table_to_rows(item)
            text = _rows_to_text(rows)
            if text.strip():
                elements.append(DocElement(
                    kind="table",
                    text=text,
                    raw_rows=rows,
                    source_index=source_index,
                    is_front_matter=not in_content,
                ))
            # Confirmed real gap in real product docs: a table cell can
            # hold a product photo directly (e.g. a "Mô tả | Hình minh
            # họa" layout table) — _table_to_rows() above only reads cell
            # TEXT, so such an image was previously invisible to the
            # pipeline entirely, with no warning or trace of it anywhere.
            if in_content:
                for blob, content_type in _table_image_blobs(item):
                    elements.append(DocElement(
                        kind="image",
                        text="",
                        source_index=source_index,
                        is_front_matter=False,
                        image_bytes=blob,
                        image_content_type=content_type,
                    ))

    logger.info("Loaded %d elements from %s", len(elements), path.name)
    return elements
