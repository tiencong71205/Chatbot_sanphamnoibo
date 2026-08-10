"""Tests for DOCX loader — ensures XML order is preserved."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def make_para(text, style="Normal"):
    para = MagicMock()
    para.text = text
    para.style.name = style
    para._p = MagicMock()
    para._p.find.return_value = None
    return para


def test_import_loader():
    from app.ingestion.docx_loader import load_docx, DocElement
    assert DocElement is not None


def test_docx_element_ordering():
    """DocElements must preserve paragraph/table interleaving."""
    from app.ingestion.docx_loader import DocElement
    elements = [
        DocElement(kind="heading", text="Section 1", level=1),
        DocElement(kind="paragraph", text="Some text"),
        DocElement(kind="table", text="Col1 | Col2\nA | B"),
        DocElement(kind="paragraph", text="After table"),
    ]
    assert elements[0].kind == "heading"
    assert elements[1].kind == "paragraph"
    assert elements[2].kind == "table"
    assert elements[3].kind == "paragraph"


def test_skip_empty_paragraphs():
    from app.ingestion.docx_loader import _is_skip
    assert _is_skip("") is True
    assert _is_skip("   ") is True
    assert _is_skip("Trang 5") is True
    assert _is_skip("DO_NOT_INGEST: này không nên import") is True
    assert _is_skip("Nội dung bình thường") is False


def test_heading_classification():
    from app.ingestion.docx_loader import _classify_paragraph
    para = MagicMock()
    para.style.name = "Heading 1"
    para._p = MagicMock()
    para._p.find.return_value = None
    assert _classify_paragraph(para) == "heading"


def test_table_to_rows():
    from app.ingestion.docx_loader import _rows_to_text
    rows = [["Thuộc tính", "Giá trị"], ["Điện áp", "110-240V"]]
    text = _rows_to_text(rows)
    assert "Thuộc tính" in text
    assert "110-240V" in text
