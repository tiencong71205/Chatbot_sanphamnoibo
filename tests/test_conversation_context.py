"""Regression tests for contextual product scoping and source counts."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from app.retrieval.product_resolver import ProductResolver
from app.schemas import ConversationTurn, SourceReference
from app.services.conversation_context import (
    ConversationContextResolver,
    filter_cited_sources,
)

# Avoid importing native BM25/NumPy dependencies in this focused service test.
hybrid_module = ModuleType("app.retrieval.hybrid_retriever")
hybrid_module.HybridRetriever = object
sys.modules.setdefault("app.retrieval.hybrid_retriever", hybrid_module)

from app.services.chatbot_service import ChatbotService  # noqa: E402

CATALOG_PATH = Path(__file__).parents[1] / "data" / "product_catalog.json"


def _resolver() -> ConversationContextResolver:
    return ConversationContextResolver(ProductResolver(CATALOG_PATH))


def test_follow_up_keeps_cong_tac_v2_scope():
    history = [
        ConversationTurn(
            role="user",
            content="Trạng thái đèn nền của công tắc thông minh V2",
        ),
        ConversationTurn(
            role="assistant",
            content="Công tắc thông minh V2 bật màu cam và tắt màu xanh.",
        ),
    ]

    scope = _resolver().resolve(
        "còn trạng thái xác nhận của trạng thái đèn nền",
        history=history,
    )

    assert scope.product_id == "cong_tac_thong_minh_v2"
    assert scope.product_name == "Công tắc thông minh V2"
    assert scope.is_follow_up is True
    assert "Công tắc thông minh V2" in scope.query


def test_current_product_overrides_previous_product():
    history = [
        ConversationTurn(
            role="user",
            content="Trạng thái đèn nền của công tắc thông minh V2",
        )
    ]

    scope = _resolver().resolve(
        "Công tắc dimmer có công suất tối đa bao nhiêu?",
        history=history,
    )

    assert scope.product_id == "cong_tac_dimmer"
    assert scope.source == "current_question"
    assert scope.is_follow_up is False


def test_broad_product_list_does_not_inherit_scope():
    history = [
        ConversationTurn(
            role="user",
            content="Trạng thái đèn nền của công tắc thông minh V2",
        )
    ]

    scope = _resolver().resolve(
        "Liệt kê các loại cảm biến",
        history=history,
    )

    assert scope.product_id is None
    assert scope.is_follow_up is False


def test_only_cited_sources_are_returned_to_ui():
    sources = [
        SourceReference(source_id="SOURCE_1"),
        SourceReference(source_id="SOURCE_2"),
        SourceReference(source_id="SOURCE_8"),
    ]
    answer = "Kết quả đúng. Nguồn: [SOURCE_2]."

    filtered = filter_cited_sources(answer, sources)

    assert [source.source_id for source in filtered] == ["SOURCE_2"]


def test_chat_service_applies_history_product_filter():
    class FakeRetriever:
        def __init__(self):
            self.resolver = ProductResolver(CATALOG_PATH)
            self.last_call = None

        def retrieve(self, **kwargs):
            self.last_call = kwargs
            return {
                "results": [
                    {
                        "payload": {
                            "product_id": "cong_tac_thong_minh_v2",
                            "product_name": "Công tắc thông minh V2",
                            "source_document": "12_Cong_tac_thong_minh_V2_RAG_v3.docx",
                            "source_section": "Trạng thái đèn nền khi điều khiển",
                            "content_type": "feature",
                            "content": "Trạng thái xác nhận: Đã xác nhận",
                            "chunk_id": "chunk-v2-led-status",
                        }
                    }
                ],
                "resolved_product": {},
                "resolved_products": [],
            }

    class FakeGenerator:
        def generate(self, user_message, **kwargs):
            assert "Công tắc thông minh V2" in user_message
            return {
                "answer": (
                    "Trạng thái xác nhận của tính năng trạng thái đèn nền khi "
                    "điều khiển trên Công tắc thông minh V2 là Đã xác nhận. "
                    "[SOURCE_1]"
                )
            }

    retriever = FakeRetriever()
    service = ChatbotService(
        settings=SimpleNamespace(
            final_top_k=8,
            product_catalog_path=str(CATALOG_PATH),
        ),
        retriever=retriever,
        generator=FakeGenerator(),
    )
    history = [
        ConversationTurn(
            role="user",
            content="Trạng thái đèn nền của công tắc thông minh V2",
        ),
        ConversationTurn(
            role="assistant",
            content="Công tắc thông minh V2 bật màu cam và tắt màu xanh.",
        ),
    ]

    response = service.chat(
        "còn trạng thái xác nhận của trạng thái đèn nền",
        history=history,
        debug=True,
    )

    assert retriever.last_call["product_id"] == "cong_tac_thong_minh_v2"
    assert response.debug_info["conversation_scope"]["source"] == "conversation_history"
    assert response.debug_info["conversation_scope"]["is_follow_up"] is True
    assert [source.source_id for source in response.sources] == ["SOURCE_1"]


def test_resolver_finds_both_products_in_comparison_question():
    matches = ProductResolver(CATALOG_PATH).resolve_all(
        "So sánh Cảm biến hiện diện và Cảm biến chuyển động ánh sáng "
        "về nguồn, giao tiếp và khả năng phát hiện."
    )

    assert {item["product_id"] for item in matches} == {
        "cam_bien_hien_dien",
        "cam_bien_chuyen_dong_anh_sang",
    }


def test_resolver_finds_ir_controller_and_curtain_motor():
    matches = ProductResolver(CATALOG_PATH).resolve_all(
        "Khi có Wi-Fi nhưng mất Internet, đèn của Bộ điều khiển hồng ngoại "
        "và Động cơ rèm hiển thị thế nào?"
    )

    assert {item["product_id"] for item in matches} == {
        "bo_dieu_khien_hong_ngoai",
        "dong_co_rem",
    }


class _NamedProductRetriever:
    def __init__(self):
        self.resolver = ProductResolver(CATALOG_PATH)
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        product_id = kwargs.get("product_id")
        contents = {
            "cam_bien_hien_dien": (
                "Nguồn cấp: 5VDC_2A. Truyền thông: Wi-Fi, Bluetooth Mesh. "
                "Phát hiện người tĩnh/động, tối đa 7 người."
            ),
            "cam_bien_chuyen_dong_anh_sang": (
                "Điện áp hoạt động: Pin AAAx3. Truyền thông: Bluetooth Mesh. "
                "Phát hiện chuyển động, góc phát hiện 120°."
            ),
            "bo_dieu_khien_hong_ngoai": (
                "Có Wi-Fi nhưng mất Internet: đèn nháy chu kỳ 2S "
                "(1S On, 1S Off)."
            ),
            "dong_co_rem": (
                "Có Wi-Fi nhưng mất Internet: đèn đỏ nháy chu kỳ 2S "
                "(1S On, 1S Off)."
            ),
            "cong_tac_cua_cuon_cua_cong": (
                "Điều khiển cửa cuốn: Mở, Mở thoáng, Dừng, Đóng. "
                "Điều khiển cửa cổng: 1 nút, 2 nút, 3 nút."
            ),
        }
        return {
            "results": [
                {
                    "payload": {
                        "product_id": product_id,
                        "product_name": product_id,
                        "source_document": f"{product_id}.docx",
                        "source_section": "Thông tin liên quan",
                        "content_type": "specification",
                        "content": contents[product_id],
                        "chunk_id": f"{product_id}_spec",
                    }
                }
            ],
            "resolved_product": {},
            "resolved_products": [],
        }


class _EchoContextGenerator:
    def generate(self, user_message, **kwargs):
        return {"answer": user_message + "\n📚 Nguồn tham khảo: SOURCE_1"}


def _named_product_service(retriever):
    return ChatbotService(
        settings=SimpleNamespace(
            final_top_k=8,
            product_catalog_path=str(CATALOG_PATH),
        ),
        retriever=retriever,
        generator=_EchoContextGenerator(),
    )


def test_compare_retrieves_each_named_product_with_hard_filter():
    retriever = _NamedProductRetriever()
    response = _named_product_service(retriever).chat(
        "So sánh Cảm biến hiện diện và Cảm biến chuyển động ánh sáng "
        "về nguồn, giao tiếp và khả năng phát hiện.",
        debug=True,
    )

    assert response.debug_info["mode"] == "compare"
    assert {call["product_id"] for call in retriever.calls} == {
        "cam_bien_hien_dien",
        "cam_bien_chuyen_dong_anh_sang",
    }
    assert "5VDC_2A" in response.answer
    assert "Pin AAAx3" in response.answer


def test_multi_product_fact_question_uses_both_hard_filters():
    retriever = _NamedProductRetriever()
    response = _named_product_service(retriever).chat(
        "Khi có Wi-Fi nhưng mất Internet, đèn của Bộ điều khiển hồng ngoại "
        "và Động cơ rèm hiển thị thế nào?",
        debug=True,
    )

    assert response.debug_info["mode"] == "multi_product"
    assert {call["product_id"] for call in retriever.calls} == {
        "bo_dieu_khien_hong_ngoai",
        "dong_co_rem",
    }
    assert "đèn nháy chu kỳ 2S" in response.answer
    assert "đèn đỏ nháy chu kỳ 2S" in response.answer


def test_specific_product_with_co_nhung_is_not_group_listing():
    retriever = _NamedProductRetriever()
    response = _named_product_service(retriever).chat(
        "Công tắc cửa cuốn, cửa cổng V2 có những cách điều khiển nào?",
        debug=True,
    )

    assert retriever.calls[0]["product_id"] == "cong_tac_cua_cuon_cua_cong"
    assert response.debug_info["conversation_scope"]["source"] == "current_question"
    assert "Hiện có 4 sản phẩm thuộc nhóm" not in response.answer
    assert "Mở thoáng" in response.answer
