"""Tests for metadata extraction."""
import pytest
from app.ingestion.metadata_extractor import detect_content_type, extract_keywords, extract_versions


def test_content_type_specification():
    text = "Điện áp hoạt động: 110-240V, tần số 50-60Hz, công suất 300W"
    ct = detect_content_type(text, "Thông số kỹ thuật")
    assert ct == "specification"


def test_content_type_reset():
    text = "Để reset thiết bị, giữ nút trong 10 giây"
    ct = detect_content_type(text, "Hướng dẫn reset")
    assert ct == "reset"


def test_content_type_troubleshooting():
    text = "Xử lý lỗi: thiết bị không hoạt động, không phản hồi lệnh điều khiển"
    ct = detect_content_type(text, "Sự cố thường gặp")
    assert ct == "troubleshooting"


def test_keywords_technical_tokens():
    """Technical tokens like model codes must be extracted as keywords."""
    text = "Model VCN-WSRGC2 sử dụng pin CR2450, firmware FW2.3.0"
    keywords = extract_keywords(text, "VCN Switch", "VCN-WSRGC2")
    all_kw = " ".join(keywords)
    assert "VCN-WSRGC2" in all_kw or "CR2450" in all_kw or "FW2.3.0" in all_kw


def test_versions_extraction():
    versions = extract_versions("Yêu cầu firmware FW2.3.0 trở lên")
    assert "firmware_version" in versions
    assert "2.3.0" in versions["firmware_version"]


def test_no_fake_product_mix():
    """Content types from distinct products should be detected independently."""
    text_a = "Cửa cuốn VCN-A: reset bằng cách giữ nút 10 giây"
    text_b = "Cửa cổng VCN-B: reset bằng cách giữ nút 5 giây"
    ct_a = detect_content_type(text_a, "Reset cửa cuốn")
    ct_b = detect_content_type(text_b, "Reset cửa cổng")
    assert ct_a == "reset"
    assert ct_b == "reset"
