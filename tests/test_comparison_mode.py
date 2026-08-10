from pathlib import Path
from types import SimpleNamespace

from app.generation.comparison_formatter import repair_comparison_table
from app.generation.prompts import COMPARE_SYSTEM_PROMPT
from app.retrieval.product_resolver import ProductResolver
from app.services.chatbot_service import ChatbotService


CATALOG_PATH = Path(__file__).parents[1] / "data" / "product_catalog.json"


def test_removed_mesh_sensor_is_not_in_catalog():
    resolver = ProductResolver(CATALOG_PATH)
    assert all(
        item.get("product_id") != "cam_bien_cua_mesh"
        for item in resolver.catalog
    )


def test_resolver_finds_all_four_switches():
    matches = ProductResolver(CATALOG_PATH).resolve_all(
        "so sánh công tắc cửa cuốn, công tắc chống giật bnn, "
        "công tắc dimmer và công tắc thông minh v2"
    )
    assert {item["product_id"] for item in matches} == {
        "cong_tac_cua_cuon_cua_cong",
        "cong_tac_chong_giat_bnn",
        "cong_tac_dimmer",
        "cong_tac_thong_minh_v2",
    }


def _result(index, product_id, content_type, content, score=0.1):
    return {
        "payload": {
            "chunk_id": f"{product_id}-chunk-{index}",
            "product_id": product_id,
            "content_type": content_type,
            "source_section": content_type,
            "content": content,
        },
        "rrf_score": score,
    }


def test_comparison_evidence_adapts_between_three_and_five_chunks():
    results = [
        _result(
            0,
            "switch",
            "identity",
            "Mã sản phẩm / product_id: 3106 Model: Chưa có dữ liệu",
            1.0,
        ),
        _result(1, "switch", "specification", "nguồn Wi-Fi relay công suất", 0.9),
        _result(2, "switch", "overview", "model loại sản phẩm tính năng", 0.8),
        _result(3, "switch", "configuration", "nút cảm ứng tại chỗ ứng dụng từ xa", 0.7),
        _result(4, "switch", "feature", "hẹn giờ theo lịch", 0.6),
        _result(5, "switch", "feature", "điều khiển giọng nói", 0.5),
        _result(6, "switch", "usage_tip", "vệ sinh bằng khăn mềm", 0.4),
    ]

    selected = ChatbotService._select_comparison_evidence(
        results,
        "so sánh hai công tắc",
    )

    assert 3 <= len(selected) <= 5
    assert {item["payload"]["content_type"] for item in selected} >= {
        "identity",
        "specification",
        "overview",
    }
    combined = " ".join(item["payload"]["content"] for item in selected)
    assert "hẹn giờ" in combined or "giọng nói" in combined


def test_comparison_evidence_allows_sixth_chunk_only_for_missing_coverage():
    results = [
        _result(1, "switch", "specification", "nguồn điện áp", 0.9),
        _result(2, "switch", "overview", "model loại sản phẩm", 0.8),
        _result(3, "switch", "connection", "Wi-Fi Bluetooth", 0.7),
        _result(4, "switch", "output", "relay công suất đầu ra", 0.6),
        _result(5, "switch", "control", "nút cảm ứng tại chỗ ứng dụng từ xa", 0.5),
        _result(6, "switch", "feature", "giọng nói hẹn giờ cảnh báo", 0.4),
        _result(7, "switch", "usage_tip", "vệ sinh bằng khăn mềm", 0.3),
    ]

    selected = ChatbotService._select_comparison_evidence(
        results,
        "so sánh toàn bộ hai công tắc",
    )

    assert len(selected) == 6
    assert all("vệ sinh" not in item["payload"]["content"] for item in selected)


class _FakeRetriever:
    def retrieve(self, query, product_id, top_k, debug):
        rows = [
            _result(1, product_id, "specification", "nguồn điện áp", 0.9),
            _result(2, product_id, "overview", "model loại sản phẩm", 0.8),
            _result(3, product_id, "connection", "Wi-Fi Bluetooth", 0.7),
            _result(4, product_id, "output", "relay công suất đầu ra", 0.6),
            _result(5, product_id, "control", "nút cảm ứng tại chỗ ứng dụng từ xa", 0.5),
            _result(6, product_id, "feature", "giọng nói hẹn giờ cảnh báo", 0.4),
        ]
        return {"results": rows, "debug": {}}


def test_four_product_comparison_keeps_twelve_to_twenty_total_chunks():
    service = ChatbotService.__new__(ChatbotService)
    service.retriever = _FakeRetriever()
    service.settings = SimpleNamespace(product_catalog_path=str(CATALOG_PATH))
    products = [
        {"product_id": f"product-{index}", "product_name": f"Product {index}"}
        for index in range(4)
    ]

    merged, _ = service._retrieve_named_products(
        "so sánh toàn bộ bốn sản phẩm",
        products,
        per_product_k=10,
        debug=False,
        comparison=True,
    )

    assert 12 <= len(merged) <= 20
    counts = {
        product["product_id"]: sum(
            item["payload"]["product_id"] == product["product_id"]
            for item in merged
        )
        for product in products
    }
    assert all(3 <= count <= 6 for count in counts.values())


def test_business_product_id_and_model_are_separate_fields():
    fields = ChatbotService._COMPARE_FIELD_TERMS

    assert "business_product_id" in fields
    assert "model" in fields
    assert "model" not in fields["business_product_id"]
    assert "mã sản phẩm" not in fields["model"]


def test_trusted_identity_evidence_uses_verified_catalog_values():
    service = ChatbotService.__new__(ChatbotService)
    service.settings = SimpleNamespace(product_catalog_path=str(CATALOG_PATH))

    expected = {
        "cong_tac_chong_giat_bnn": ("3043", "Model: Chưa có dữ liệu"),
        "cong_tac_cua_cuon_cua_cong": ("3036", "Model: VCN-WSRGC2"),
        "cong_tac_dimmer": ("3106", "Model: Chưa có dữ liệu"),
        "cong_tac_thong_minh_v2": ("3082", "Model: Chưa có dữ liệu"),
    }

    for retrieval_key, (business_id, model_line) in expected.items():
        result = service._comparison_identity_result(retrieval_key, "")
        assert result is not None
        payload = result["payload"]
        assert payload["product_id"] == retrieval_key
        assert f"Mã sản phẩm / product_id: {business_id}" in payload["content"]
        assert model_line in payload["content"]
        assert payload["content_type"] == "identity"


def test_compare_prompt_forbids_slug_or_model_as_business_product_id():
    assert "retrieval_product_key" in COMPARE_SYSTEM_PROMPT
    assert "không phải mã sản phẩm" in COMPARE_SYSTEM_PROMPT
    assert "Mã sản phẩm / product_id" in COMPARE_SYSTEM_PROMPT
    assert "Model" in COMPARE_SYSTEM_PROMPT


def test_repair_comparison_table_rebuilds_only_header():
    malformed = (
        "Kết quả:\n\n"
        "| Tiêu chíCông tắc A Công tắc B | | |\n"
        "| --- | --- | --- |\n"
        "| Nguồn | 100-240VAC | 110-240VAC |\n\n"
        "📚 Nguồn tham khảo: SOURCE_1, SOURCE_2"
    )

    repaired = repair_comparison_table(malformed, ["Công tắc A", "Công tắc B"])

    assert "| Tiêu chí | Công tắc A | Công tắc B |" in repaired
    assert "|---|---|---|" in repaired
    assert "| Nguồn | 100-240VAC | 110-240VAC |" in repaired
