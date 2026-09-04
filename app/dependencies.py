"""FastAPI dependency injection providers.

IMPORTANT: These no longer construct new objects per request. QdrantStore,
BM25Store, OllamaEmbedding, HybridRetriever, the generator, and
ChatbotService are all expensive to build (disk I/O for the BM25 joblib
index and product catalog, new QdrantClient/httpx.Client connections) and
are safe to share across requests -- none of them hold per-request mutable
state, and the objects they wrap (QdrantClient, httpx.Client, BM25Okapi
reads) are safe for concurrent use from multiple threads/coroutines.

The actual instances are built once at startup in the `lifespan` handler in
app/api.py and stored on `app.state`. These functions just look them up.
"""
from __future__ import annotations

from fastapi import Request

from app.database.bm25_store import BM25Store
from app.database.qdrant_store import QdrantStore
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.chatbot_service import ChatbotService


def get_qdrant_store(request: Request) -> QdrantStore:
    return request.app.state.qdrant_store


def get_bm25_store(request: Request) -> BM25Store:
    return request.app.state.bm25_store


def get_embedding(request: Request) -> OllamaEmbedding:
    return request.app.state.embedding


def get_hybrid_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_generator(request: Request):
    return request.app.state.generator


def get_chatbot_service(request: Request) -> ChatbotService:
    return request.app.state.chatbot_service
