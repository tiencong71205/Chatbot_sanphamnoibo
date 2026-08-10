"""Application settings loaded from environment / .env file."""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_llm_model: str = "qwen3.5:4b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    # Qdrant
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "vhomenex_products_v2"
    # Embedding
    embedding_dimension: int = 1024
    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    final_top_k: int = 8
    rrf_k: int = 60
    # Chunking
    chunk_target_tokens: int = 450
    chunk_max_tokens: int = 750
    chunk_overlap_tokens: int = 60
    # Generation
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 1500
    ollama_temperature: float = 0.1
    # Scoring
    min_dense_score: float = 0.20
    max_context_tokens: int = 6000
    # Paths
    data_directory: str = "./data/source_docs"
    processed_directory: str = "./data/processed"
    bm25_index_path: str = "./storage/bm25/index.joblib"
    product_catalog_path: str = "./data/product_catalog.json"
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    backend_url: str = "http://127.0.0.1:8000"
    # Logging
    log_level: str = "INFO"
    # Admin
    admin_token: str = "vhomenex_admin_2024"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
