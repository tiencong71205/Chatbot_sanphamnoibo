"""Tests for FastAPI endpoints."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.api.get_chatbot_service") as mock_chatbot_dep, \
         patch("app.api.get_qdrant_store") as mock_qdrant_dep, \
         patch("app.api.get_bm25_store") as mock_bm25_dep, \
         patch("app.api.get_embedding") as mock_emb_dep:

        qdrant_mock = MagicMock()
        qdrant_mock.list_products.return_value = []
        qdrant_mock.count.return_value = 0
        mock_qdrant_dep.return_value = qdrant_mock
        mock_bm25_dep.return_value = MagicMock()
        mock_emb_dep.return_value = MagicMock()

        chatbot_svc = MagicMock()
        from app.schemas import ChatResponse, RetrieveResponse
        chatbot_svc.chat.return_value = ChatResponse(
            answer="Test answer. Nguon tham khao: SOURCE_1",
            sources=[],
            debug_info=None,
            latency_ms=100.0,
        )
        chatbot_svc.retrieve_only.return_value = RetrieveResponse(
            results=[], latency_ms=50.0
        )
        mock_chatbot_dep.return_value = chatbot_svc

        from app.api import app
        with TestClient(app) as c:
            yield c


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
