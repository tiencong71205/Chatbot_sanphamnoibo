# PLAN — vhomenex_hybrid_rag_v2

Xem chi tiết trong implementation_plan.md (tạo bởi Antigravity agent).

## Tóm tắt

- **LLM:** qwen3.5:4b (Ollama Docker container, port 11434)
- **Embedding:** qwen3-embedding:0.6b (1024 dims)
- **Vector DB:** Qdrant existing container (port 6333)
- **BM25:** rank-bm25 + underthesea tokenizer
- **Fusion:** Reciprocal Rank Fusion (k=60)
- **Backend:** FastAPI (port 8000)
- **Frontend:** Streamlit (port 8501)
- **Docker Compose project:** vhomenex-rag-v2
- **Collection:** vhomenex_products_v2
