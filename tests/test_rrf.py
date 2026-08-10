"""Tests for Reciprocal Rank Fusion."""
import pytest
from app.retrieval.rrf_fusion import rrf_fuse


def make_dense(chunk_ids, start_rank=1):
    return [
        {
            "payload": {"chunk_id": cid, "content": f"content_{cid}", "product_name": "Test"},
            "dense_rank": i + start_rank,
            "dense_score": 1.0 / (i + 1),
        }
        for i, cid in enumerate(chunk_ids)
    ]


def make_sparse(chunk_ids, start_rank=1):
    return [
        {
            "payload": {"chunk_id": cid, "content": f"content_{cid}", "product_name": "Test"},
            "sparse_rank": i + start_rank,
            "sparse_score": 10.0 / (i + 1),
        }
        for i, cid in enumerate(chunk_ids)
    ]


def test_rrf_formula():
    """Score must be 1/(k+rank). Default k=60."""
    dense = make_dense(["a"])
    sparse = []
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=5)
    assert len(results) == 1
    expected = 1.0 / (60 + 1)
    assert abs(results[0]["rrf_score"] - expected) < 1e-9


def test_rrf_deduplication():
    """Same chunk from dense and sparse must appear only once."""
    dense = make_dense(["a", "b", "c"])
    sparse = make_sparse(["a", "d", "e"])
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=10)
    chunk_ids = [r["chunk_id"] for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_rrf_both_sources_boost():
    """Chunk appearing in both dense and sparse gets higher score."""
    dense = make_dense(["shared", "dense_only"])
    sparse = make_sparse(["shared", "sparse_only"])
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=10)
    shared = next(r for r in results if r["chunk_id"] == "shared")
    dense_only = next(r for r in results if r["chunk_id"] == "dense_only")
    assert shared["rrf_score"] > dense_only["rrf_score"]


def test_rrf_top_k():
    dense = make_dense([f"d{i}" for i in range(10)])
    sparse = make_sparse([f"s{i}" for i in range(10)])
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=5)
    assert len(results) <= 5


def test_rrf_no_direct_score_addition():
    """RRF must NOT directly add cosine + BM25 scores."""
    dense = [{"payload": {"chunk_id": "x", "content": "test"}, "dense_rank": 1, "dense_score": 0.95}]
    sparse = [{"payload": {"chunk_id": "x", "content": "test"}, "sparse_rank": 1, "sparse_score": 100.0}]
    results = rrf_fuse(dense, sparse, rrf_k=60, top_k=5)
    # RRF score should be ~2/61 ≈ 0.0328, NOT 0.95 + 100.0
    assert results[0]["rrf_score"] < 1.0
