"""Test query intent classification for required sample queries."""
import pytest
from app.retrieval.query_intent import classify_intent, QueryIntent


def test_intent_fact():
    products = [{"product_id": "cam_bien_cua_mesh", "confidence": 0.9}]
    res = classify_intent("Pin cảm biến cửa dùng được bao lâu?", products)
    assert res.intent == QueryIntent.FACT


def test_intent_section_full():
    products = [{"product_id": "cam_bien_cua_mesh", "confidence": 0.9}]
    res = classify_intent("Cho tôi toàn bộ hướng dẫn lắp đặt Cảm biến cửa Mesh", products)
    assert res.intent == QueryIntent.SECTION_FULL
    assert "lắp đặt" in res.requested_sections


def test_intent_product_full():
    products = [{"product_id": "cam_bien_cua_mesh", "confidence": 0.9}]
    res = classify_intent("Trình bày toàn bộ thông tin Cảm biến cửa Mesh", products)
    assert res.intent == QueryIntent.PRODUCT_FULL


def test_intent_multi_product_fact():
    products = [
        {"product_id": "cam_bien_cua_mesh", "confidence": 0.9},
        {"product_id": "cam_bien_chuyen_dong_anh_sang", "confidence": 0.9},
    ]
    res = classify_intent("Hai cảm biến dùng chuẩn kết nối gì?", products)
    assert res.intent == QueryIntent.MULTI_PRODUCT_FACT
    assert len(res.requested_products) == 2


def test_intent_compare_field():
    products = [
        {"product_id": "cam_bien_cua_mesh", "confidence": 0.9},
        {"product_id": "cam_bien_chuyen_dong_anh_sang", "confidence": 0.9},
    ]
    res = classify_intent("So sánh kết nối của hai cảm biến", products)
    assert res.intent == QueryIntent.COMPARE_FIELD
    assert len(res.requested_products) == 2


def test_intent_compare_full():
    products = [
        {"product_id": "bo_dieu_khien_hong_ngoai_thong_minh", "confidence": 0.9},
        {"product_id": "cam_bien_cua_mesh", "confidence": 0.9},
        {"product_id": "cam_bien_chuyen_dong_anh_sang", "confidence": 0.9},
    ]
    res = classify_intent("So sánh tổng thể ba sản phẩm", products)
    assert res.intent == QueryIntent.COMPARE_FULL
    assert len(res.requested_products) == 3
