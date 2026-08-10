"""FastAPI dependency injection providers."""
from __future__ import annotations
from functools import lru_cache
from fastapi import Depends
from app.config import Settings, get_settings
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.ollama_generator import OllamaGenerator
from app.services.chatbot_service import ChatbotService


def get_qdrant_store(settings: Settings = Depends(get_settings)) -> QdrantStore:
    return QdrantStore(settings)


def get_bm25_store(settings: Settings = Depends(get_settings)) -> BM25Store:
    return BM25Store(settings)


def get_embedding(settings: Settings = Depends(get_settings)) -> OllamaEmbedding:
    return OllamaEmbedding(settings)


def get_hybrid_retriever(
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    bm25: BM25Store = Depends(get_bm25_store),
    embedding: OllamaEmbedding = Depends(get_embedding),
) -> HybridRetriever:
    return HybridRetriever(settings, qdrant, bm25, embedding)


def get_generator(settings: Settings = Depends(get_settings)) -> OllamaGenerator:
    return OllamaGenerator(settings)


def get_chatbot_service(
    settings: Settings = Depends(get_settings),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    generator: OllamaGenerator = Depends(get_generator),
) -> ChatbotService:
    return ChatbotService(settings, retriever, generator)
