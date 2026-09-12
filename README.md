# Vhomenex Hybrid RAG - Cảm biến cửa Mesh

Dự án hợp nhất chatbot Hybrid RAG, chỉ sử dụng một tài liệu nguồn:
`data/source_docs/01_Cam_bien_cua_Mesh_v1.0.docx`.

## Thành phần chính

- Frontend: Streamlit, cổng `8501`.
- Backend: FastAPI, cổng `8000`.
- Retrieval: Qdrant + BM25 + Reciprocal Rank Fusion.
- Generation: Ollama với `qwen3.5:4b`.
- Embedding: `qwen3-embedding:0.6b`, vector 1024 chiều.
- Collection mặc định: `vhomenex_products_unified_v1`.

Collection này là collection staging riêng, không ghi đè
`vhomenex_products_v2`.

## Khởi động trên máy GPU

```bash
cp .env.docker.example .env.docker
```

Thay `ADMIN_TOKEN=replace-with-a-strong-random-token` trong `.env.docker` bằng
một token ngẫu nhiên riêng, sau đó chạy:

```bash
docker compose -p vhomenex-rag-v2 build --no-cache
docker compose -p vhomenex-rag-v2 up -d
docker compose -p vhomenex-rag-v2 ps
```

## Kiểm tra tài liệu trước khi ingest

```bash
docker exec vhomenex-backend-v2 \
  python scripts/validate_template_doc.py \
  /app/data/source_docs/01_Cam_bien_cua_Mesh_v1.0.docx
```

```bash
docker exec vhomenex-backend-v2 \
  python scripts/ingest_documents.py \
  --data-dir /app/data/source_docs \
  --dry-run
```

Kết quả dry-run phải chỉ liệt kê đúng một file Cảm biến cửa Mesh.

## Ingest vào collection mới

Chỉ chạy `--recreate` sau khi xác nhận `.env.docker` vẫn dùng
`QDRANT_COLLECTION=vhomenex_products_unified_v1`:

```bash
docker exec vhomenex-backend-v2 \
  python scripts/ingest_documents.py \
  --data-dir /app/data/source_docs \
  --recreate
```

## Kiểm tra hệ thống

```bash
curl http://localhost:8000/health
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Truy cập:

- Web: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Dữ liệu được phép giữ

```text
data/
├── product_catalog.json
└── source_docs/
    └── 01_Cam_bien_cua_Mesh_v1.0.docx
```

`product_catalog.json` chỉ chứa metadata của Cảm biến cửa Mesh. Gói bàn giao
không chứa tài liệu sản phẩm khác, evaluation cũ, preview chunk cũ, BM25 index,
Qdrant storage, `.venv`, cache hoặc file backup.
