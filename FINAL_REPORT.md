# FINAL REPORT — vhomenex_hybrid_rag_v2

## 1. Tổng quan Dự án & Dữ liệu Thực tế (10 File DOCX)

Dự án **vhomenex_hybrid_rag_v2** đã hoàn thành ingest và vận hành thành công **10 tài liệu sản phẩm DOCX** đầy đủ trên hệ thống Ubuntu chủ (`192.168.1.226`).

| Thành phần | Thông số / Trạng thái thực tế |
|------------|--------------------------------|
| **Project Location** | `/home/vconnex/vhomenex_hybrid_rag_v2` |
| **Số tài liệu DOCX thực tế** | **10 tài liệu sản phẩm** đầy đủ trong `data/raw/` |
| **Tổng số Chunks** | **58 chunks** (0 skipped, 0 errors) |
| **LLM Model** | `qwen3.5:4b` (Ollama Docker, container `ollama`, port 11434) |
| **Embedding Model** | `qwen3-embedding:0.6b` (1024 dimensions) |
| **Vector DB** | Qdrant Docker (port 6333), collection `vhomenex_products_v2` |
| **Qdrant `points_count`** | **58 points** |
| **Qdrant `status`** | `green` |
| **BM25 Index** | `storage/bm25/index.joblib` (58 documents, 668 vocabulary terms, 2426 tokens) |
| **Chunk ID Alignment** | Qdrant == BM25 (58/58 matched: `True`) |
| **Pytest Pass Rate** | **34/34 passed (100%)** |
| **Backend API** | FastAPI (port 8000), `status: ok` |
| **Frontend UI** | Streamlit (port 8501), `HTTP 200 OK` |

---

## 2. Danh sách 10 Tài liệu Sản phẩm Vhomenex (`data/raw/`)

1. `VCN-BULB-RGB_Bong_Den_RGB_Thong_Minh.docx`: 12 elements → 5 chunks
2. `VCN-CAM-OUT2K_Camera_Ngoai_Troi.docx`: 12 elements → 5 chunks
3. `VCN-CURTAIN-M1_Dong_Co_Rem_Thong_Minh.docx`: 12 elements → 5 chunks
4. `VCN-HUB01_Hub_Nha_Thong_Minh.docx`: 14 elements → 7 chunks
5. `VCN-IR-REMOTE_Bo_Dieu_Khien_Hong_Ngoai.docx`: 12 elements → 5 chunks
6. `VCN-LOCK-PRO_Khoa_Cua_Van_Tay_Thong_Minh.docx`: 12 elements → 5 chunks
7. `VCN-SEN-DOOR_Cam_Bien_Cua_Thong_Minh.docx`: 12 elements → 5 chunks
8. `VCN-SOCKET-16A_O_Cam_Thong_Minh.docx`: 12 elements → 4 chunks
9. `VCN-SW2G_Cong_Tac_Thong_Minh.docx`: 16 elements → 7 chunks
10. `VCN-WSRGC2_Cua_Cuon_Thong_Minh.docx`: 26 elements → 10 chunks

- **Ingestion Exit Code**: `0` (Success)
- **Embedding Batch Size**: 8 chunks / batch, timeout 180s
- **Upserting**: Upsert theo từng batch vào Qdrant và lưu BM25 index

---

## 3. Kết quả Hybrid Retrieval (Dense + BM25 + RRF)

Thử nghiệm truy vấn trên bộ dữ liệu 10 sản phẩm:
- **"Cảm biến cửa sử dụng chuẩn kết nối gì?"**
  - **Top Match:** `VCN-SEN-DOOR_Cam_Bien_Cua_Thong_Minh.docx` Section 4 (Zigbee 3.0 / Bluetooth Mesh).
- **"Bộ điều khiển hồng ngoại có chức năng gì?"**
  - **Top Match:** `VCN-IR-REMOTE_Bo_Dieu_Khien_Hong_Ngoai.docx` Section 1 & Section 3 (Học lệnh remote DIY, 360° đa hướng).
- **"Model VCN-WSRGC2 dùng điện áp bao nhiêu?"**
  - **Top Match:** `VCN-WSRGC2_Cua_Cuon_Thong_Minh.docx` Section 2 (110-240V AC, 50-60Hz).

---

## 4. Kết quả Generation & Trích dẫn Nguồn (`POST /chat`)

- **Bộ điều khiển hồng ngoại VCN-IR-REMOTE học lệnh:**
  - **Trả lời:** `Có, bộ điều khiển hồng ngoại VCN-IR-REMOTE có thể học lệnh từ remote cũ qua chế độ DIY Học lệnh...`
  - **Citations:** `SOURCE_2, SOURCE_5` (Trích dẫn đúng tài liệu `VCN-IR-REMOTE`)
- **Khóa cửa VCN-LOCK-PRO:**
  - **Trả lời:** `Khóa cửa VCN-LOCK-PRO mở bằng các phương thức: Vân tay FPC bán dẫn, Mã số thành viên...`
  - **Citations:** `SOURCE_1, SOURCE_2` (Trích dẫn đúng tài liệu `VCN-LOCK-PRO`)

---

## 5. Danh sách Endpoint & Health Check

- `GET /health` → `200 OK`
  - `status: ok`
  - `qdrant_points: 58`
  - `bm25_exists: true`
- `GET /products` → `200 OK` (10 sản phẩm)
- `POST /chat` → `200 OK`
- `Frontend UI (Port 8501)` → `HTTP 200 OK`
