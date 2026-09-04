"""Tests for semantic chunker."""
from app.ingestion.docx_loader import DocElement
from app.ingestion.semantic_chunker import Chunk, SemanticChunker


def make_elements(specs):
    """specs: list of (kind, text) or (kind, text, level)"""
    result = []
    for item in specs:
        if len(item) == 2:
            kind, text = item
            result.append(DocElement(kind=kind, text=text, level=0))
        else:
            kind, text, level = item
            result.append(DocElement(kind=kind, text=text, level=level))
    return result


def test_basic_chunking():
    chunker = SemanticChunker(target_tokens=100, max_tokens=200, overlap_tokens=20)
    elements = make_elements([
        ("heading", "Thong so ky thuat", 1),
        ("paragraph", "Dien ap: 110-240V. Tan so: 50-60Hz. Cong suat: 300W."),
        ("paragraph", "Nhiet do hoat dong: -10C den 50C."),
    ])
    chunks = chunker.chunk(elements, "test.docx", product_name="Switch A", product_id="switch_a")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_no_product_mixing():
    """Two separate heading sections must each produce chunks with their respective headings."""
    chunker = SemanticChunker(target_tokens=200, max_tokens=400, overlap_tokens=30)
    elements = make_elements([
        ("heading", "Cua cuon VCN-A", 1),
        ("paragraph", "Reset: giu nut 10 giay de khoi phuc cai dat goc cho thiet bi."),
        ("paragraph", "Tiep tuc giu nut cho den khi den nhap 3 lan."),
        ("heading", "Cua cong VCN-B", 1),
        ("paragraph", "Reset: giu nut 5 giay de khoi phuc cai dat goc cho thiet bi VCN-B."),
        ("paragraph", "Den se nhap 2 lan khi reset thanh cong."),
    ])
    chunks = chunker.chunk(elements, "mixed.docx", product_name="Vhomenex")
    assert len(chunks) >= 2
    full_text = " ".join(f"{c.title} {c.section_heading} {c.content}" for c in chunks)
    assert "VCN-A" in full_text
    assert "VCN-B" in full_text


def test_chunk_id_deterministic():
    """Same content must produce same chunk_id."""
    from app.ingestion.semantic_chunker import _make_chunk_id
    id1 = _make_chunk_id("file.docx", "sec1", "title", "content text")
    id2 = _make_chunk_id("file.docx", "sec1", "title", "content text")
    assert id1 == id2


def test_no_fake_source_page():
    """source_page must be None (not auto-generated)."""
    chunker = SemanticChunker()
    elements = make_elements([
        ("paragraph", "Some content here about the product specifications."),
    ])
    chunks = chunker.chunk(elements, "test.docx")
    for c in chunks:
        assert c.source_page is None


def test_token_limit_respected():
    """Single paragraph chunk should not be split if under max_tokens."""
    chunker = SemanticChunker(target_tokens=50, max_tokens=500, overlap_tokens=10)
    big_text = "Day la mot doan van ban rat dai hon. " * 20
    elements = make_elements([("paragraph", big_text)])
    chunks = chunker.chunk(elements, "big.docx")
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.content) > 0


def test_mesh_template_keeps_heading_path_and_all_steps():
    chunker = SemanticChunker(target_tokens=100, max_tokens=500)
    elements = make_elements([
        ("heading", "B. THÔNG TIN SẢN PHẨM", 1),
        ("heading", "4. Tính năng trên thiết bị", 2),
        ("heading", "4.1. Chế độ kết nối với bộ điều khiển trung tâm", 3),
        ("paragraph", "a) Mô tả: Kết nối qua Bluetooth Mesh."),
        ("paragraph", "b) Hướng dẫn:"),
        ("paragraph", "B1: Mở chi tiết Gateway"),
        ("paragraph", "B2: Nhấn Thêm thiết bị"),
        ("paragraph", "B3: Chọn cảm biến cửa"),
    ])
    chunks = chunker.chunk(
        elements,
        "mesh.docx",
        product_id="cam_bien_cua_mesh",
        product_name="Cảm biến cửa Mesh",
    )
    procedure = next(c for c in chunks if c.chunk_type == "procedure")
    assert all(step in procedure.content for step in ("B1:", "B2:", "B3:"))
    assert procedure.heading_path == (
        "THÔNG TIN SẢN PHẨM > Tính năng trên thiết bị > "
        "Chế độ kết nối với bộ điều khiển trung tâm"
    )
    assert procedure.source_locator.startswith("Cảm biến cửa Mesh →")


def test_spec_table_is_split_into_one_chunk_per_row():
    chunker = SemanticChunker()
    elements = make_elements([
        ("heading", "B. THÔNG TIN SẢN PHẨM", 1),
        ("heading", "2. Thông số kỹ thuật", 2),
    ])
    elements.append(DocElement(
        kind="table",
        text="Truyền thông | Bluetooth Mesh\nChất liệu vỏ | Nhựa",
        raw_rows=[["Truyền thông", "Bluetooth Mesh"], ["Chất liệu vỏ", "Nhựa"]],
    ))
    chunks = chunker.chunk(elements, "mesh.docx", product_name="Cảm biến cửa Mesh")
    specs = [c for c in chunks if c.chunk_type == "spec"]
    assert len(specs) == 2
    assert {c.title for c in specs} == {"Truyền thông", "Chất liệu vỏ"}


def test_empty_image_labels_and_navigation_table_are_not_indexed():
    chunker = SemanticChunker()
    elements = make_elements([
        ("heading", "B. THÔNG TIN SẢN PHẨM", 1),
        ("heading", "3. Hình ảnh sản phẩm", 2),
        ("paragraph", "Mặt trước:"),
        ("paragraph", "Mặt sau:"),
        ("heading", "4. Tính năng trên thiết bị", 2),
        ("paragraph", "- Bảng danh sách tính năng:"),
    ])
    elements.append(DocElement(
        kind="table",
        text="STT | Tên tính năng | Xem hướng dẫn\n1 | Kết nối | 4.1",
        raw_rows=[["STT", "Tên tính năng", "Xem hướng dẫn"], ["1", "Kết nối", "4.1"]],
    ))
    chunks = chunker.chunk(elements, "mesh.docx", product_name="Cảm biến cửa Mesh")
    assert chunks == []
