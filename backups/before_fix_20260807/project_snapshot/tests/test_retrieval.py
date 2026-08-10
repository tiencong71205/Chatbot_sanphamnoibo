"""Tests for retrieval components."""
import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.product_resolver import resolve_product
from app.retrieval.metadata_filter import build_qdrant_filter, HARD_CONFIDENCE
from app.retrieval.rrf_fusion import rrf_fuse


def test_product_resolver_no_query():
    r = resolve_product("Điện áp hoạt động bao nhiêu?")
    # Without product keywords, confidence should be low
    assert r.confidence < HARD_CONFIDENCE or r.product_id is None


def test_product_resolver_with_hint():
    r = resolve_product("Câu hỏi gì đó", product_id_hint="switch_a")
    assert r.product_id == "switch_a"
    assert r.confidence == 1.0


def test_metadata_filter_high_confidence():
    """High-confidence resolution should produce a filter."""
    from app.retrieval.product_resolver import ResolvedProduct
    resolved = ResolvedProduct(product_id="switch", product_name="Switch", confidence=0.9)
    f = build_qdrant_filter(resolved)
    assert f is not None
    assert "product_id" in f


def test_metadata_filter_low_confidence():
    """Low-confidence resolution should produce no filter (soft)."""
    from app.retrieval.product_resolver import ResolvedProduct
    resolved = ResolvedProduct(product_id="switch", confidence=0.4)
    f = build_qdrant_filter(resolved)
    assert f is None


def test_bm25_technical_tokens():
    """BM25 tokenizer must correctly handle technical tokens."""
    from app.database.bm25_store import tokenize_vi
    tokens = tokenize_vi("Model VCN-WSRGC2 sử dụng 110-240V và CR2450")
    token_str = " ".join(tokens)
    assert "VCN-WSRGC2" in token_str
    assert "110-240V" in token_str
    assert "CR2450" in token_str


def test_bm25_bluetooth_mesh():
    from app.database.bm25_store import tokenize_vi
    tokens = tokenize_vi("Hỗ trợ Bluetooth Mesh và FW2.3.0")
    token_str = " ".join(tokens)
    assert "FW2.3.0" in token_str
