"""Pydantic schemas for API request/response models."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_id: str = ""
    product_id: str = ""
    product_name: str = ""
    product_group: str = ""
    model: str = ""
    hardware_version: str = ""
    firmware_version: str = ""
    app_version: str = ""
    content_type: str = ""
    section_number: str = ""
    section_heading: str = ""
    title: str = ""
    content: str = ""
    conditions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    source_file: str = ""
    source_document: str = ""
    source_page: Optional[int] = None
    source_section: str = ""
    source_heading: str = ""
    source_location: str = ""
    extraction_method: str = "verbatim"
    verification_status: str = "pending_review"


class SourceReference(BaseModel):
    source_id: str = ""
    product_name: str = ""
    source_document: str = ""
    source_section: str = ""
    content_type: str = ""
    content: str = ""
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    product_id: Optional[str] = None
    debug: bool = False


class ChatResponse(BaseModel):
    answer: str = ""
    sources: List[SourceReference] = Field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    product_id: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=20)
    debug: bool = False


class RetrieveResponse(BaseModel):
    results: List[SourceReference] = Field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


class IngestRequest(BaseModel):
    recreate_collection: bool = False
    dry_run: bool = False


class IngestResponse(BaseModel):
    status: str = "ok"
    files_processed: int = 0
    chunks_created: int = 0
    chunks_upserted: int = 0
    errors: List[str] = Field(default_factory=list)
    dry_run: bool = False


class HealthStatus(BaseModel):
    status: str = "ok"
    backend: bool = True
    ollama: bool = False
    ollama_llm_model: bool = False
    ollama_embedding_model: bool = False
    qdrant: bool = False
    bm25_index: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class ProductInfo(BaseModel):
    product_id: str = ""
    product_name: str = ""
    product_group: str = ""
    models: List[str] = Field(default_factory=list)
    chunk_count: int = 0
