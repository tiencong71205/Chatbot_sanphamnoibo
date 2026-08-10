# Vhomenex Hybrid RAG v2

Hệ thống chatbot Hybrid RAG cho tài liệu sản phẩm Vhomenex, chạy hoàn toàn local.

## Kiến trúc

```
Frontend (Streamlit :8501)
    ↓
Backend (FastAPI :8000)
    ↓ Dense retrieval    ↓ Sparse retrieval
Qdrant (:6333)         BM25 Index (file)
    ↓
RRF Fusion → Context Builder
    ↓
Ollama (:11434) [qwen3.5:4b]
    ↓
Câu trả lời + Nguồn tham khảo
```

## Cài đặt nhanh

```bash
# 1. Copy config
cp .env.docker.example .env.docker

# 2. Build và chạy
docker compose -p vhomenex-rag-v2 build --no-cache
docker compose -p vhomenex-rag-v2 up -d

# 3. Ingest tài liệu
python scripts/ingest_documents.py

# 4. Kiểm tra health
curl http://localhost:8000/health
```

## Truy cập

- **Frontend:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

## Cấu hình

- `.env.example` — chạy trực tiếp trên Ubuntu
- `.env.docker.example` → `.env.docker` — chạy trong Docker

## Models

| Model | Mục đích | Kích thước |
|-------|----------|------------|
| `qwen3.5:4b` | LLM sinh câu trả lời | 3.4 GB |
| `qwen3-embedding:0.6b` | Dense embedding | 639 MB |

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/health` | Kiểm tra hệ thống |
| GET | `/products` | Danh sách sản phẩm |
| POST | `/ingest` | Ingest tài liệu |
| POST | `/retrieve` | Truy vấn retrieval |
| POST | `/chat` | Chat với RAG |
| POST | `/feedback` | Gửi feedback |

## Quy tắc an toàn

- Không xóa project cũ
- Không động tới Docker volume ngoài project này
- Collection Qdrant: `vhomenex_products_v2` (không liên quan `vconnex_products`)
- Compose project name: `vhomenex-rag-v2`
