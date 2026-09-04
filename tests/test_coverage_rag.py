"""Comprehensive tests for full coverage, token budget, catalog, and procedure steps."""
import json
import pytest
from pathlib import Path
from app.config import Settings
from app.generation.context_builder import ContextBuilder, estimate_tokens
from app.services.chatbot_service import ChatbotService


def test_catalog_has_three_products():
    catalog_path = Path("data/product_catalog.json")
    if not catalog_path.exists():
        catalog_path = Path("/app/data/product_catalog.json")

    assert catalog_path.exists(), "product_catalog.json must exist"
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 3, f"Catalog must contain exactly 3 products, got {len(data)}"
    pids = [item["product_id"] for item in data]
    assert "bo_dieu_khien_hong_ngoai_thong_minh" in pids
    assert "cam_bien_cua_mesh" in pids
    assert "cam_bien_chuyen_dong_anh_sang" in pids


def test_max_context_tokens_respected():
    settings = Settings()
    builder = ContextBuilder(settings)

    # Generate 50 dummy chunks
    chunks = [
        {
            "payload": {
                "product_name": "Cảm biến cửa Mesh",
                "source_document": "1.Vhomenex_Mô tả sản phẩm_Cảm biến cửa Mesh_v1.0_final.docx",
                "source_section": f"Mục thứ {i}",
                "content_type": "specification",
                "content": f"Đây là nội dung thử nghiệm cho mục thứ {i} với dữ liệu dài để kiểm tra giới hạn token budget trong context builder.",
            }
        }
        for i in range(50)
    ]

    # Test with strict override of 1000 tokens
    user_msg, selected = builder.build(chunks, query="Test query", max_tokens_override=1000)
    est = estimate_tokens(user_msg)
    assert est <= 1600, f"Token count {est} exceeded reasonable budget envelope"
    assert len(selected) < 50, "Context builder should drop chunks to fit budget"


def test_preserve_procedure_steps():
    selected = [
        {
            "payload": {
                "chunk_type": "procedure",
                "source_section": "Kết nối Gateway",
                "content": "B1: Mở app Vhomenex\nB2: Chọn thêm thiết bị\nB3: Nhấn giữ nút reset 5s\nB4: Nhập mật khẩu Wi-Fi",
            }
        }
    ]
    incomplete_answer = "Bạn làm theo hướng dẫn: Nhấn nút reset và nhập wifi."
    final_ans = ChatbotService._preserve_procedure_steps(
        question="Hướng dẫn các bước kết nối",
        answer=incomplete_answer,
        selected=selected,
    )
    assert "B1:" in final_ans
    assert "B2:" in final_ans
    assert "B3:" in final_ans
    assert "B4:" in final_ans
