"""Test Vietnamese slug generation with đ/Đ and NFKD decomposition."""
import pytest
from app.ingestion.document_metadata import _slug, _key


def test_slug_vietnamese_d():
    assert _slug("Bộ điều khiển hồng ngoại thông minh") == "bo_dieu_khien_hong_ngoai_thong_minh"
    assert _slug("Cảm biến cửa Mesh") == "cam_bien_cua_mesh"
    assert _slug("Cảm biến chuyển động ánh sáng") == "cam_bien_chuyen_dong_anh_sang"


def test_slug_no_dropped_d():
    slug = _slug("điều khiển")
    assert "dieu" in slug
    assert "ieu" not in slug

    slug2 = _slug("chuyển động")
    assert "dong" in slug2
    assert "ong" not in slug2.split("_")[-1]


def test_slug_uppercase_d():
    assert _slug("Đặt lại thiết bị") == "dat_lai_thiet_bi"
    assert _slug("Đèn báo LED") == "den_bao_led"
