"""Tests for FastAPI endpoints."""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.api import app
    from app.dependencies import (
        get_bm25_store,
        get_chatbot_service,
        get_embedding,
        get_qdrant_store,
    )
    from app.schemas import ChatResponse, RetrieveResponse

    qdrant_mock = MagicMock()
    qdrant_mock.list_products.return_value = []
    qdrant_mock.count_by_product.return_value = 0

    chatbot_svc = MagicMock()
    chatbot_svc.chat.return_value = ChatResponse(
        answer="Test answer. Nguon tham khao: SOURCE_1",
        sources=[],
        debug_info=None,
        latency_ms=100.0,
    )
    chatbot_svc.retrieve_only.return_value = RetrieveResponse(
        results=[], latency_ms=50.0
    )

    app.dependency_overrides[get_chatbot_service] = lambda: chatbot_svc
    app.dependency_overrides[get_qdrant_store] = lambda: qdrant_mock
    app.dependency_overrides[get_bm25_store] = MagicMock
    app.dependency_overrides[get_embedding] = MagicMock

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Vhomenex" in r.json()["name"]


def test_chat_requires_question(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_question_too_long(client):
    r = client.post("/chat", json={"question": "x" * 2001})
    assert r.status_code == 422


def test_chat_valid(client):
    r = client.post("/chat", json={"question": "Thiet bi co reset khong?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data


def test_retrieve_requires_query(client):
    r = client.post("/retrieve", json={})
    assert r.status_code == 422


def test_retrieve_valid(client):
    r = client.post("/retrieve", json={"query": "thong so ky thuat"})
    assert r.status_code == 200


def test_products_endpoint(client):
    r = client.get("/products")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
