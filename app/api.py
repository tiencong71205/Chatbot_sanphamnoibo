"""FastAPI application — all endpoints for Vhomenex Hybrid RAG v2."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.database.bm25_store import BM25Store
from app.database.qdrant_store import QdrantStore
from app.dependencies import (
    get_bm25_store,
    get_chatbot_service,
    get_embedding,
    get_hybrid_retriever,
    get_qdrant_store,
)
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.generation.ollama_generator import OllamaGenerator
from app.generation.vllm_generator import VLLMGenerator
from app.ingestion.ingest_service import IngestService
from app.logging_config import setup_logging
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    FeedbackRequest,
    HealthStatus,
    IngestRequest,
    IngestResponse,
    ProductInfo,
    ReindexRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.chatbot_service import ChatbotService

# Setup logging at startup
setup_logging()
logger = logging.getLogger(__name__)


def _build_generator(settings: Settings):
    backend = settings.llm_backend.strip().lower()
    if backend == "ollama":
        return OllamaGenerator(settings)
    if backend == "vllm":
        return VLLMGenerator(settings)
    raise ValueError(
        f"Unsupported LLM_BACKEND={settings.llm_backend!r}. Use 'ollama' or 'vllm'."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build every expensive, stateless-per-request object exactly once.

    Before this, every single request re-created QdrantClient, BM25Store
    (which meant re-reading the .joblib index from disk on every /api/chat
    and /api/search call), OllamaEmbedding/generator httpx.Clients, and
    ProductResolver (re-reading + re-parsing product_catalog.json). None of
    these hold per-request mutable state, so they're safe to share across
    concurrent requests — building them once removes that repeated disk I/O
    and connection setup from the hot path entirely.
    """
    settings = get_settings()

    qdrant_store = QdrantStore(settings)
    bm25_store = BM25Store(settings)
    bm25_store.load_index()  # ok if it doesn't exist yet; search() still
    # lazily retries load_index() on a cache miss, so a later /api/ingest
    # populates this same shared instance in-memory via build_index().
    embedding = OllamaEmbedding(settings)
    retriever = HybridRetriever(settings, qdrant_store, bm25_store, embedding)
    generator = _build_generator(settings)
    chatbot_service = ChatbotService(settings, retriever, generator)

    app.state.settings = settings
    app.state.qdrant_store = qdrant_store
    app.state.bm25_store = bm25_store
    app.state.embedding = embedding
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.chatbot_service = chatbot_service

    logger.info(
        "Startup complete: bm25_loaded=%s catalog_products=%d backend=%s",
        bm25_store.is_loaded(),
        len(retriever.resolver.catalog),
        settings.llm_backend,
    )

    yield

    generator.close()
    embedding.close()


app = FastAPI(
    title="Vhomenex Hybrid RAG API",
    description="Chatbot Hybrid RAG cho tài liệu sản phẩm Vhomenex - v2",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────── Auth helpers ───────────────

def _check_admin(admin_token: str, settings: Settings) -> None:
    if not settings.admin_token:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_TOKEN is not configured",
        )
    if admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")


# ─────────────── System endpoints ───────────────

@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check(settings: Settings = Depends(get_settings)) -> HealthStatus:
    """Check health of all system components."""
    details: Dict[str, Any] = {}

    backend = settings.llm_backend.strip().lower()
    details["generation_backend"] = backend

    ollama_ok = False
    ollama_llm_model_ok = False
    embed_model_ok = False

    vllm_ok = False
    vllm_model_ok = False

    qdrant_ok = False
    bm25_ok = False

    # Ollama is always required for embeddings.
    # It is also used for generation when LLM_BACKEND=ollama.
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()

            models = [m["name"] for m in resp.json().get("models", [])]

            ollama_ok = True
            ollama_llm_model_ok = settings.ollama_llm_model in models
            embed_model_ok = settings.ollama_embedding_model in models

            details["ollama_models"] = models
            details["ollama_url"] = settings.ollama_base_url

            try:
                ps_resp = await client.get(
                    f"{settings.ollama_base_url}/api/ps"
                )
                if ps_resp.status_code == 200:
                    running = ps_resp.json().get("models", [])
                    details["ollama_running"] = [
                        {
                            "name": m.get("name"),
                            "size_vram": m.get("size_vram", 0),
                        }
                        for m in running
                    ]
            except Exception:
                pass

    except Exception as e:
        details["ollama_error"] = str(e)

    # vLLM is required only when selected as generation backend.
    if backend == "vllm":
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    f"{settings.vllm_base_url}/v1/models"
                )
                resp.raise_for_status()

                models = [
                    m.get("id", "")
                    for m in resp.json().get("data", [])
                ]

                vllm_ok = True
                vllm_model_ok = settings.vllm_llm_model in models

                details["vllm_models"] = models
                details["vllm_url"] = settings.vllm_base_url

        except Exception as e:
            details["vllm_error"] = str(e)
    else:
        details["vllm_status"] = "not_required"

    # Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.qdrant_url}/collections/"
                f"{settings.qdrant_collection}"
            )

            qdrant_ok = resp.status_code == 200

            if qdrant_ok:
                col_info = resp.json().get("result", {})
                details["qdrant_points"] = col_info.get(
                    "points_count", 0
                )
                details["qdrant_collection"] = (
                    settings.qdrant_collection
                )

    except Exception as e:
        details["qdrant_error"] = str(e)

    # BM25
    bm25_path = Path(settings.bm25_index_path)
    bm25_ok = bm25_path.exists()

    details["bm25_index_path"] = str(bm25_path)
    details["bm25_exists"] = bm25_ok

    if backend == "ollama":
        generation_ok = (
            ollama_ok and ollama_llm_model_ok
        )
    elif backend == "vllm":
        generation_ok = (
            vllm_ok and vllm_model_ok
        )
    else:
        generation_ok = False
        details["generation_error"] = (
            f"Unsupported LLM_BACKEND={backend!r}"
        )

    overall = (
        "ok"
        if (
            ollama_ok
            and embed_model_ok
            and generation_ok
            and qdrant_ok
            and bm25_ok
        )
        else "degraded"
    )

    return HealthStatus(
        status=overall,
        backend=True,
        ollama=ollama_ok,
        ollama_llm_model=ollama_llm_model_ok,
        ollama_embedding_model=embed_model_ok,
        vllm=vllm_ok,
        vllm_llm_model=vllm_model_ok,
        qdrant=qdrant_ok,
        bm25_index=bm25_ok,
        details=details,
    )


@app.get("/ready", tags=["System"])
async def readiness_check(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Simple readiness check - just verify backend is alive."""
    return {"status": "ready", "timestamp": time.time()}


@app.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    return {
        "name": "Vhomenex Hybrid RAG API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# ─────────────── Chat endpoints ───────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponse:
    """Chat with the RAG system. Supports product filter and history."""
    try:
        return await run_in_threadpool(
            chatbot.chat,
            question=request.question,
            product_id=request.product_id,
            history=request.history,
            compare_mode=request.compare_mode,
            debug=request.debug,
        )
    except Exception as e:
        logger.error("Chat failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Also expose legacy endpoint without /api prefix
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_legacy(
    request: ChatRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponse:
    return await chat(request, chatbot)


# ─────────────── Streaming chat endpoint (SSE) ───────────────

@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(
    request: ChatRequest,
    raw_request: Request,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> StreamingResponse:
    """Server-Sent Events stream of the answer.

    Only FACT-intent questions (the common single-product/general question
    case — see ChatbotService.chat_stream) get real token-by-token
    streaming for a fast time-to-first-token; other intents (compare,
    multi-product-fact, product_full, section_full) still arrive as one
    'delta' event with the complete answer followed by 'done', so the SSE
    event contract is uniform for the frontend regardless of intent.

    Client disconnect: polled via `raw_request.is_disconnected()` between
    events, and the underlying generator is always explicitly `.close()`d
    in `finally` — this propagates GeneratorExit through any open `with
    self._client.stream(...)` block in the generator client, closing the
    Ollama/vLLM HTTP connection promptly instead of leaving it for the
    garbage collector (and, on the Ollama/vLLM side, a dropped connection is
    what stops it from continuing to burn GPU time on an abandoned request).

    Event types sent: delta (text chunk), replace (final post-processing
    changed the streamed text — see _stream_fact's docstring), error, done
    (data is the full ChatResponse JSON, matching the non-streaming
    endpoint's response shape).
    """
    gen = chatbot.chat_stream(
        question=request.question,
        product_id=request.product_id,
        history=request.history,
        compare_mode=request.compare_mode,
        debug=request.debug,
    )

    async def event_source():
        try:
            while True:
                if await raw_request.is_disconnected():
                    logger.info("Client disconnected mid-stream; stopping generation.")
                    break
                try:
                    item = await run_in_threadpool(next, gen)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error("chat_stream failed: %s", e, exc_info=True)
                    payload = json.dumps({"message": str(e)}, ensure_ascii=False)
                    yield f"event: error\ndata: {payload}\n\n"
                    break

                event_type = item.get("type", "delta")
                if event_type == "done":
                    payload = item["response"].model_dump_json()
                    yield f"event: done\ndata: {payload}\n\n"
                    break
                else:
                    payload = json.dumps(
                        {"text": item.get("text"), "message": item.get("message")},
                        ensure_ascii=False,
                    )
                    yield f"event: {event_type}\ndata: {payload}\n\n"
        finally:
            gen.close()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx proxy buffering, if any
        },
    )


# ─────────────── Search endpoint ───────────────

@app.post("/api/search", response_model=RetrieveResponse, tags=["Retrieval"])
async def search(
    request: RetrieveRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> RetrieveResponse:
    """Search/retrieve relevant chunks for a query without generation."""
    try:
        return await run_in_threadpool(
            chatbot.retrieve_only,
            query=request.query,
            product_id=request.product_id,
            top_k=request.top_k,
            debug=request.debug,
        )
    except Exception as e:
        logger.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Legacy path
@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve_legacy(
    request: RetrieveRequest,
    chatbot: ChatbotService = Depends(get_chatbot_service),
) -> RetrieveResponse:
    return await search(request, chatbot)


# ─────────────── Ingest endpoints ───────────────

@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(
    request: IngestRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    bm25: BM25Store = Depends(get_bm25_store),
    embedding: OllamaEmbedding = Depends(get_embedding),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> IngestResponse:
    """Ingest DOCX documents from data/source_docs directory.

    qdrant/bm25/embedding here are the shared singletons (see lifespan in
    this module) — ingest mutates bm25's in-memory index directly via
    build_index(), so chat/search requests see the new index immediately
    with no restart or reload step. product_catalog.json can still change
    on disk (auto-registration of new products), so the singleton
    ProductResolver is explicitly refreshed below.
    """
    _check_admin(request.admin_token, settings)
    svc = IngestService(settings, qdrant, bm25, embedding)
    try:
        result = await run_in_threadpool(
            svc.ingest_all,
            recreate_collection=request.recreate_collection,
            dry_run=request.dry_run,
        )
        if not request.dry_run:
            retriever.resolver.reload()
            retriever.invalidate_cache()
        return IngestResponse(
            status="ok",
            files_processed=result.get("files", 0),
            chunks_created=result.get("chunks", 0),
            chunks_upserted=0 if request.dry_run else result.get("embedded", 0),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            dry_run=request.dry_run,
            doc_details=result.get("doc_details", []),
        )
    except Exception as e:
        logger.error("Ingest failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Legacy path
@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_legacy(
    request: IngestRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    bm25: BM25Store = Depends(get_bm25_store),
    embedding: OllamaEmbedding = Depends(get_embedding),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> IngestResponse:
    return await ingest_documents(request, settings, qdrant, bm25, embedding, retriever)


@app.get("/api/ingest/report", tags=["Ingestion"])
async def get_last_ingest_report() -> Dict[str, Any]:
    """Return the persisted report (errors + warnings + per-file stats) from
    the most recent ingest run — so data-quality issues stay visible without
    needing to re-run ingest or dig through console logs.
    """
    report_path = Path("./data/processed/last_ingest_report.json")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có lần ingest nào được ghi lại.")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/reindex", response_model=IngestResponse, tags=["Ingestion"])
async def reindex_documents(
    request: ReindexRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    bm25: BM25Store = Depends(get_bm25_store),
    embedding: OllamaEmbedding = Depends(get_embedding),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> IngestResponse:
    """Reindex all documents (idempotent - skips unchanged files)."""
    _check_admin(request.admin_token, settings)
    svc = IngestService(settings, qdrant, bm25, embedding)
    try:
        result = await run_in_threadpool(
            svc.ingest_all,
            recreate_collection=True,
            dry_run=request.dry_run,
        )
        if not request.dry_run:
            retriever.resolver.reload()
            retriever.invalidate_cache()
        return IngestResponse(
            status="ok",
            files_processed=result.get("files", 0),
            chunks_created=result.get("chunks", 0),
            chunks_upserted=0 if request.dry_run else result.get("embedded", 0),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            dry_run=request.dry_run,
            doc_details=result.get("doc_details", []),
        )
    except Exception as e:
        logger.error("Reindex failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────── Document/Product catalog endpoints ───────────────

@app.get("/api/products", response_model=List[ProductInfo], tags=["Products"])
async def list_products(
    qdrant: QdrantStore = Depends(get_qdrant_store),
    settings: Settings = Depends(get_settings),
) -> List[ProductInfo]:
    """List all products in the knowledge base from catalog and Qdrant."""
    try:
        # Load from catalog file
        catalog_path = Path(settings.product_catalog_path)
        if not catalog_path.exists():
            # Try alternate paths
            for alt in ["/app/data/product_catalog.json", "./data/product_catalog.json"]:
                if Path(alt).exists():
                    catalog_path = Path(alt)
                    break

        products = []
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            for item in catalog:
                # Get chunk count from Qdrant
                chunk_count = 0
                try:
                    chunk_count = await run_in_threadpool(
                        qdrant.count_by_product, item.get("product_id", "")
                    )
                except Exception:
                    pass
                products.append(ProductInfo(
                    product_id=item.get("product_id", ""),
                    product_name=item.get("product_name", ""),
                    product_group=item.get("product_group", ""),
                    model=item.get("model", ""),
                    aliases=item.get("aliases", []),
                    source_file=item.get("source_file", ""),
                    chunk_count=chunk_count,
                ))
        else:
            # Fallback: get from Qdrant
            raw_products = await run_in_threadpool(qdrant.list_products)
            products = [ProductInfo(**p) for p in raw_products]
        return products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products", response_model=List[ProductInfo], tags=["Products"])
async def list_products_legacy(
    qdrant: QdrantStore = Depends(get_qdrant_store),
    settings: Settings = Depends(get_settings),
) -> List[ProductInfo]:
    """Legacy path retained for the current Streamlit frontend."""
    return await list_products(qdrant=qdrant, settings=settings)


@app.get("/api/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(
    qdrant: QdrantStore = Depends(get_qdrant_store),
) -> List[DocumentInfo]:
    """List all ingested documents with stats."""
    try:
        docs = await run_in_threadpool(qdrant.list_documents)
        return [DocumentInfo(**d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}", tags=["Documents"])
async def get_document(
    document_id: str,
    qdrant: QdrantStore = Depends(get_qdrant_store),
) -> Dict[str, Any]:
    """Get document details by ID."""
    try:
        doc = await run_in_threadpool(qdrant.get_document_info, document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks/{chunk_id}", tags=["Chunks"])
async def get_chunk(
    chunk_id: str,
    qdrant: QdrantStore = Depends(get_qdrant_store),
) -> Dict[str, Any]:
    """Get a specific chunk by ID."""
    try:
        chunk = await run_in_threadpool(qdrant.get_chunk_by_id, chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk '{chunk_id}' not found")
        return chunk
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────── Feedback endpoint ───────────────

@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest) -> Dict[str, str]:
    """Submit feedback for a Q&A pair."""
    logger.info(
        "Feedback received: rating=%d, question=%s...",
        request.rating, request.question[:50],
    )
    return {"status": "ok", "message": "Feedback recorded"}
