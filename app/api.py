"""FastAPI application — all endpoints."""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.logging_config import setup_logging
from app.schemas import (
    ChatRequest, ChatResponse,
    RetrieveRequest, RetrieveResponse,
    IngestRequest, IngestResponse,
    HealthStatus, FeedbackRequest, ProductInfo,
)
from app.dependencies import (
    get_chatbot_service, get_hybrid_retriever,
    get_qdrant_store, get_bm25_store, get_embedding,
)
from app.services.chatbot_service import ChatbotService
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.ingestion.ingest_service import IngestService

# Setup logging at startup
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vhomenex Hybrid RAG API",
    description="Chatbot Hybrid RAG cho tài liệu sản phẩm Vhomenex",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check(settings: Settings = Depends(get_settings)) -> HealthStatus:
    """Check health of all system components."""
    details: Dict[str, Any] = {}
    ollama_ok = False
    llm_model_ok = False
    embed_model_ok = False
    qdrant_ok = False
    bm25_ok = False

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            ollama_ok = True
            llm_model_ok = settings.ollama_llm_model in models
            embed_model_ok = settings.ollama_embedding_model in models
            details["ollama_models"] = models
    except Exception as e:
        details["ollama_error"] = str(e)

    # Check Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
            qdrant_ok = resp.status_code == 200
            if qdrant_ok:
                col_info = resp.json().get("result", {})
                details["qdrant_points"] = col_info.get("points_count", 0)
    except Exception as e:
        details["qdrant_error"] = str(e)

    # Check BM25
    bm25_path = Path(settings.bm25_index_path)
    bm25_ok = bm25_path.exists()
    details["bm25_index_path"] = str(bm25_path)
    details["bm25_exists"] = bm25_ok

    overall = "ok" if (ollama_ok and llm_model_ok and embed_model_ok and qdrant_ok) else "degraded"

    return HealthStatus(
        status=overall,
        backend=True,
        ollama=ollama_ok,
        ollama_llm_model=llm_model_ok,
        ollama_embedding_model=embed_model_ok,
        qdrant=qdrant_ok,
        bm25_index=bm25_ok,
        details=details,
    )


@app.get("/products", response_model=List[ProductInfo], tags=["Products"])
async def list_products(
    qdrant: QdrantStore = Depends(get_qdrant_store),
) -> List[ProductInfo]:
    """List all products in the knowledge base."""
    try:
        products = qdrant.list_products()
        return [ProductInfo(**p) for p in products]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(
    request: IngestRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    bm25: BM25Store = Depends(get_bm25_store),
    embedding: OllamaEmbedding = Depends(get_embedding),
) -> IngestResponse:
    """Ingest DOCX documents from data directory."""
    svc = IngestService(settings, qdrant, bm25, embedding)
    try:
        result = svc.ingest_all(
            recreate_collection=request.recreate_collection,
            dry_run=request.dry_run,
        )
        return IngestResponse(
            status="ok",
            files_processed=result["files"],
            chunks_created=result["chunks"],
            chunks_upserted=0 if request.dry_run else result["chunks"],
            errors=result["errors"],
            dry_run=request.dry_run,
        )
    except Exception as e:
        logger.error("Ingest failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve(
    request: RetrieveRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> RetrieveResponse:
    """Retrieve relevant chunks for a query."""
    try:
        return chatbot.retrieve_only(
            query=request.query,
            product_id=request.product_id,
            top_k=request.top_k,
            debug=request.debug,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponse:
    """Chat with the RAG system."""
    try:
        return chatbot.chat(
            question=request.question,
            product_id=request.product_id,
            debug=request.debug,
        )
    except Exception as e:
        logger.error("Chat failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest) -> Dict[str, str]:
    """Submit feedback for a Q&A pair."""
    logger.info(
        "Feedback received: rating=%d, question=%s...",
        request.rating, request.question[:50],
    )
    return {"status": "ok", "message": "Feedback recorded"}


@app.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    return {
        "name": "Vhomenex Hybrid RAG API",
        "version": "2.0.0",
        "docs": "/docs",
    }
