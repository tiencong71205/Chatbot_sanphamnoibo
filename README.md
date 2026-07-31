# Vhomenex Smart Assistant

<p align="center">

![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-DC244C?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)
![RAG](https://img.shields.io/badge/RAG-Hybrid_Retrieval-purple?style=for-the-badge)

</p>

# 🏠🤖 Vhomenex Hybrid RAG Assistant

Hệ thống trợ lý AI tra cứu tài liệu sản phẩm **Vconnex Smart Home**, sử dụng kiến trúc **Hybrid RAG** kết hợp:

- Dense Vector Search
- BM25 Keyword Search
- Reciprocal Rank Fusion
- Product Metadata Filtering
- Local LLM qua Ollama
- Product Resolver
- Multi-product Retrieval



---

## 📋 Tổng quan

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Frontend | Streamlit | Giao diện chatbot và bộ lọc sản phẩm |
| Backend | FastAPI | Xử lý API, retrieval và generation |
| Generation model | Qwen3.5 4B | Sinh câu trả lời tiếng Việt |
| Embedding model | Qwen3 Embedding 0.6B | Chuyển câu hỏi và chunk thành vector |
| Vector Database | Qdrant | Lưu vector, nội dung và metadata |
| Keyword retrieval | BM25 | Tìm kiếm chính xác theo từ khóa |
| Fusion | RRF | Hợp nhất kết quả Dense và BM25 |
| Runtime | Ollama | Chạy model hoàn toàn local |
| Deployment | Docker Compose | Quản lý toàn bộ dịch vụ |

### Dữ liệu hiện tại

| Thuộc tính | Giá trị |
|---|---:|
| Số tài liệu sản phẩm | 10 |
| Số retrieval chunks | 242 |
| Qdrant collection | `vhomenex_products_v2` |
| Ngôn ngữ chính | Tiếng Việt |
| Phạm vi dữ liệu | Sản phẩm Smart Home Vconnex |

---

## 🧠 Kiến trúc hệ thống

```text
DOCX Product Template
        │
        ▼
Parser / Extractor
        │
        ├──────────────► Product Metadata
        │                       │
        │                       ▼
        │                Product Catalog
        │
        ├──────────────► Ecosystem Scenarios
        │
        ▼
Retrieval Chunks
        │
        ├──► Embedding Model ──► Qdrant Vector DB
        │
        └──► BM25 Index
                              
User Question
        │
        ▼
Intent Detection
        │
        ▼
Product Resolver
        │
        ├──► Product Metadata Filter
        │
        ├──► Dense Vector Search
        │
        └──► BM25 Search
                │
                ▼
       Reciprocal Rank Fusion
                │
                ▼
          Context Builder
                │
                ▼
          Qwen3.5 4B LLM
                │
                ▼
       Answer + Sources + Debug
```

---

## 🔍 Luồng xử lý câu hỏi

Ví dụ câu hỏi:

```text
So sánh thông số kỹ thuật của công tơ và bộ điều khiển hồng ngoại.
```

Hệ thống thực hiện:

```text
1. Nhận diện intent: comparison + specifications
2. Product Resolver tìm ra 2 sản phẩm
3. Truy xuất riêng cho từng sản phẩm
4. Dense Search tìm nội dung gần nghĩa
5. BM25 tìm các từ khóa chính xác
6. RRF hợp nhất và xếp hạng kết quả
7. Round-robin đảm bảo cả hai sản phẩm đều có context
8. LLM tạo bảng so sánh
9. Trả về câu trả lời cùng nguồn tham khảo
```

---

## 📥 Input

Nguồn dữ liệu chính là tài liệu DOCX theo template chuẩn hóa.

Mỗi tài liệu tương ứng với một sản phẩm và bao gồm:

- Thông tin nhận diện
- Mô tả sản phẩm
- Thông số kỹ thuật
- Hướng dẫn lắp đặt
- Điều khiển trực tiếp trên thiết bị
- Ý nghĩa đèn báo và âm thanh
- Các chế độ kết nối
- Tính năng đơn lẻ
- Tính năng trên ứng dụng Vhomenex
- Tính năng kết hợp trong hệ sinh thái
- Cảnh báo và lưu ý
- Reset, bảo trì và OTA
- Phạm vi áp dụng theo phiên bản
- Hình ảnh và tài liệu liên quan

### Nguyên tắc dữ liệu

- Một thông số được lưu trong một trường riêng.
- Giá trị và đơn vị được tách rõ ràng.
- Không gộp nhiều thông số vào một đoạn văn.
- Trường chưa có dữ liệu được đánh dấu `pending`.
- Trường không áp dụng được đánh dấu `not_applicable`.
- Nội dung phải gắn đúng model, hardware, firmware và app version.

---

## 📤 Output dữ liệu

Sau khi parse tài liệu, hệ thống tạo ra ba lớp dữ liệu chính:

```text
1. Product Metadata
2. Ecosystem Scenarios
3. Retrieval Chunks
```

### Product Metadata

Mô tả toàn bộ thông tin cố hữu của sản phẩm.

```json
{
  "product_id": "vconnex_presence_sensor_vcn_hps",
  "name": "Cảm biến hiện diện",
  "model": "VCN-HPS",
  "category": "sensor",
  "brand": "Vconnex",
  "warranty_months": 24,
  "supported_app": [
    "Vhomenex iOS",
    "Vhomenex Android"
  ]
}
```

### Ecosystem Scenario

Mô tả quan hệ và kịch bản kết hợp giữa nhiều thiết bị.

```json
{
  "scenario_id": "eco_presence_light_01",
  "name": "Tự động bật đèn khi phát hiện người",
  "participating_product_ids": [
    "vconnex_presence_sensor_vcn_hps",
    "vconnex_dimmer_switch"
  ],
  "trigger": "Phát hiện có người",
  "action": "Bật đèn ở mức sáng 70%",
  "verification_status": "approved"
}
```

### Retrieval Chunk

Đoạn nội dung nhỏ dùng trực tiếp cho tìm kiếm RAG.

```json
{
  "chunk_id": "vcn_hps_pairing_auto_001",
  "content": "Nhấn giữ nút trong 3 giây đến khi đèn xanh nhấp nháy để vào chế độ kết nối tự động. Thời gian chờ là 90 giây.",
  "metadata": {
    "product_id": "vconnex_presence_sensor_vcn_hps",
    "model": "VCN-HPS",
    "category": "sensor",
    "section": "pairing",
    "topic": "automatic_pairing",
    "document_version": "1.0",
    "language": "vi"
  }
}
```

---

## 🧩 Metadata

Metadata không thay thế chunk.

Metadata được sử dụng để:

- Xác định đúng sản phẩm.
- Lọc theo model và nhóm sản phẩm.
- Phân biệt thông số, lắp đặt, reset và tính năng.
- Quản lý phiên bản tài liệu.
- Ngăn truy xuất nhầm giữa các sản phẩm.
- Liên kết sản phẩm với kịch bản hệ sinh thái.
- Hỗ trợ so sánh nhiều sản phẩm.

### Metadata tối thiểu cho mỗi chunk

```json
{
  "product_id": "vconnex_presence_sensor_vcn_hps",
  "product_name": "Cảm biến hiện diện",
  "model": "VCN-HPS",
  "category": "sensor",
  "section": "pairing",
  "topic": "automatic_pairing",
  "document_version": "1.0",
  "effective_date": "2024-06-25",
  "source_file": "Cam_bien_hien_dien.docx",
  "language": "vi"
}
```

Không lưu toàn bộ metadata sản phẩm trong mỗi chunk để tránh:

- Payload quá lớn.
- Lặp dữ liệu.
- Khó cập nhật.
- Không đồng nhất giữa các chunk.

---

## ✂️ Chunking Strategy

Chunk được tạo theo cấu trúc nội dung thay vì chỉ cắt theo số ký tự.

### Các nhóm chunk

```text
overview
specifications
installation
device_control
indicator_states
pairing
device_features
app_features
ecosystem_scenarios
troubleshooting
maintenance
safety
```

### Nguyên tắc chunk

- Một chunk chỉ tập trung vào một chủ đề.
- Không cắt giữa một bước hướng dẫn.
- Giữ nguyên số và đơn vị.
- Giữ tên sản phẩm và model trong metadata.
- Chunk so sánh hoặc kịch bản có thể chứa nhiều `product_ids`.
- Chunk phải có nguồn tài liệu tương ứng.

Ví dụ:

```text
Chunk 1: Điện áp và công suất
Chunk 2: Kích thước và vật liệu
Chunk 3: Kết nối tự động
Chunk 4: Kết nối thủ công
Chunk 5: Cập nhật OTA
```

---

## 🗄️ Vector Database

Hệ thống sử dụng **Qdrant** để lưu:

```text
Vector embedding
Chunk content
Product metadata
Document metadata
Retrieval scores
Source references
```

Collection hiện tại:

```text
vhomenex_products_v2
```

Mỗi point trong Qdrant có dạng:

```json
{
  "id": "vcn_hps_pairing_auto_001",
  "vector": [0.012, -0.028, 0.041],
  "payload": {
    "content": "Nhấn giữ nút trong 3 giây...",
    "product_id": "vconnex_presence_sensor_vcn_hps",
    "model": "VCN-HPS",
    "section": "pairing",
    "topic": "automatic_pairing"
  }
}
```

### Tại sao cần Vector DB?

Vector Search giúp tìm được nội dung gần nghĩa.

Ví dụ:

```text
Câu hỏi: Làm sao ghép cảm biến với app?

Có thể tìm được chunk:
Nhấn giữ nút 3 giây để vào chế độ kết nối tự động.
```

Hai câu không dùng chính xác cùng từ khóa nhưng có ý nghĩa gần nhau.

---

## 🔎 Hybrid Retrieval

Hệ thống kết hợp hai phương pháp tìm kiếm.

### Dense Vector Search

Tìm kiếm theo ngữ nghĩa:

```text
reset thiết bị
khôi phục thiết bị
đặt lại thiết bị
```

Các câu trên có thể được đưa về các vector gần nhau.

### BM25 Search

Tìm kiếm chính xác theo từ khóa:

```text
VCN-HPS
5 VDC
90 giây
Bluetooth Mesh
120 x 72 x 36 mm
```

BM25 đặc biệt hiệu quả với:

- Model
- Mã sản phẩm
- Thông số
- Đơn vị
- Tên tính năng
- Thuật ngữ kỹ thuật

### Reciprocal Rank Fusion

RRF hợp nhất thứ hạng của Dense Search và BM25:

```text
Dense results ─┐
               ├──► RRF ──► Final ranked chunks
BM25 results ──┘
```

RRF giúp hệ thống vừa hiểu ý nghĩa câu hỏi vừa giữ được độ chính xác của thông số.

---

## 🤖 Models

### Generation Model

```text
Model: qwen3.5:4b
Runtime: Ollama
Vai trò: Sinh câu trả lời từ context được truy xuất
```

Model được yêu cầu:

- Chỉ trả lời theo context.
- Giữ nguyên số và đơn vị.
- Không tự suy diễn thông số.
- Trình bày bảng khi so sánh.
- Thông báo rõ khi dữ liệu chưa đầy đủ.

### Embedding Model

```text
Model: qwen3-embedding:0.6b
Runtime: Ollama
Vai trò: Chuyển query và chunk thành vector
```

Embedding được sử dụng cho:

- Semantic Search
- Similarity Search
- Multi-product Retrieval
- Topic Retrieval

---

## 📁 Cấu trúc thư mục

```text
vhomenex_hybrid_rag_v2/
├── README.md
├── docker-compose.yml
├── .env.docker
├── requirements.txt
│
├── backend/
│   ├── app/
│   │   ├── api/                     # FastAPI routes
│   │   ├── core/                    # Settings và cấu hình
│   │   ├── ingestion/               # Parse DOCX và tạo chunks
│   │   ├── retrieval/               # Dense, BM25 và RRF
│   │   ├── generation/              # Prompt và Ollama generation
│   │   ├── catalog/                 # Product resolver
│   │   └── main.py
│   └── Dockerfile
│
├── ui/
│   ├── streamlit_app.py             # Giao diện chatbot
│   └── Dockerfile
│
├── data/
│   ├── documents/                   # Tài liệu DOCX sản phẩm
│   ├── product_catalog.json         # Metadata sản phẩm
│   ├── ecosystem_scenarios.json     # Kịch bản hệ sinh thái
│   ├── metadata_schema.json         # Schema chuẩn hóa
│   └── bm25_index/                  # Chỉ mục tìm kiếm từ khóa
│
├── scripts/
│   ├── ingest_documents.py          # Parse và ingest dữ liệu
│   ├── rebuild_collection.py        # Tạo lại Qdrant collection
│   ├── build_bm25.py                # Tạo chỉ mục BM25
│   └── evaluate_retrieval.py        # Đánh giá retrieval
│
└── tests/
    ├── test_api.py
    ├── test_retrieval.py
    └── test_product_resolver.py
```

> Cấu trúc trên được trình bày rút gọn; tên thư mục thực tế có thể thay đổi theo từng phiên bản dự án.

---

## 🚀 Quickstart

### 1. Clone repository

```bash
git clone https://github.com/your-username/vhomenex-hybrid-rag.git
cd vhomenex-hybrid-rag
```

### 2. Cấu hình môi trường

Cập nhật file `.env.docker`:

```env
OLLAMA_BASE_URL=http://ollama:11434
QDRANT_URL=http://qdrant:6333

OLLAMA_CHAT_MODEL=qwen3.5:4b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b

QDRANT_COLLECTION=vhomenex_products_v2

FINAL_TOP_K=8
OLLAMA_NUM_PREDICT=1400
```

### 3. Khởi động hệ thống

```bash
docker compose up -d --build
```

Kiểm tra container:

```bash
docker compose ps
```

### 4. Tải model Ollama

```bash
docker compose exec ollama ollama pull qwen3.5:4b
docker compose exec ollama ollama pull qwen3-embedding:0.6b
```

### 5. Ingest dữ liệu

```bash
docker compose exec backend python scripts/ingest_documents.py
```

Kết quả mong đợi:

```text
Documents: 10
Chunks: 242
Collection: vhomenex_products_v2
```

### 6. Truy cập giao diện

```text
Frontend: http://localhost:8501
Backend:  http://localhost:8000
API docs: http://localhost:8000/docs
Qdrant:   http://localhost:6333/dashboard
```

Truy cập từ thiết bị khác trong cùng mạng LAN:

```text
http://<SERVER_IP>:8501
```

---

## 🔌 API

### Danh sách sản phẩm

```http
GET /products
```

Ví dụ:

```bash
curl http://localhost:8000/products
```

### Chat

```http
POST /chat
```

Ví dụ payload:

```json
{
  "question": "Cách reset cảm biến hiện diện?",
  "product_filter": "Cảm biến hiện diện",
  "debug": false
}
```

Ví dụ:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Cách reset cảm biến hiện diện?",
    "product_filter": "Cảm biến hiện diện",
    "debug": false
  }'
```

---

## 💬 Ví dụ câu hỏi

### Tra cứu thông số

```text
Cảm biến hiện diện sử dụng nguồn điện bao nhiêu?
```

### Hướng dẫn kết nối

```text
Cách kết nối tự động công tắc Dimmer với ứng dụng?
```

### Reset

```text
Giữ nút bao lâu để reset bộ điều khiển hồng ngoại?
```

### So sánh sản phẩm

```text
So sánh công tơ và bộ điều khiển hồng ngoại.
```

### Hệ sinh thái

```text
Cảm biến hiện diện có thể kết hợp với công tắc đèn như thế nào?
```

### Tìm kiếm nhiều sản phẩm

```text
So sánh phương thức kết nối của cảm biến hiện diện, Dimmer và công tắc cửa cuốn.
```

---

## 🖥️ Giao diện

Frontend được xây dựng bằng Streamlit với thiết kế tối giản:

- Sidebar chọn phạm vi sản phẩm.
- Tạo cuộc trò chuyện mới.
- Câu hỏi gợi ý.
- Hiển thị lịch sử chat.
- Hiển thị nguồn tham khảo.
- Debug retrieval.
- Thời gian phản hồi.
- Hỗ trợ so sánh nhiều sản phẩm.

---

## ⚙️ Cấu hình retrieval

Các tham số chính trong `.env.docker`:

```env
DENSE_TOP_K=20
SPARSE_TOP_K=20
FINAL_TOP_K=8

RRF_K=60

ENABLE_PRODUCT_RESOLVER=true
ENABLE_METADATA_FILTER=true
ENABLE_DEBUG=false
```

Ý nghĩa:

```text
DENSE_TOP_K   Số kết quả lấy từ Vector Search
SPARSE_TOP_K  Số kết quả lấy từ BM25
FINAL_TOP_K   Số chunk cuối đưa vào context
RRF_K         Hằng số điều chỉnh Reciprocal Rank Fusion
```

---

## 🧪 Kiểm thử

### Kiểm tra backend

```bash
curl http://localhost:8000/health
```

### Kiểm tra danh sách sản phẩm

```bash
curl http://localhost:8000/products
```

### Kiểm tra logs

```bash
docker compose logs --tail=100 backend
docker compose logs --tail=100 frontend
```

### Kiểm tra collection Qdrant

```bash
curl http://localhost:6333/collections/vhomenex_products_v2
```

---

## 📊 Đánh giá hệ thống

Các nhóm test nên có:

| Nhóm câu hỏi | Ví dụ |
|---|---|
| Product resolution | “Thông số cảm biến hiện diện” |
| Exact specification | “Điện áp của VCN-HPS” |
| Installation | “Cách lắp Dimmer” |
| Pairing | “Kết nối tự động giữ nút bao lâu?” |
| Reset | “Cách đặt lại thiết bị” |
| Multi-product comparison | “So sánh công tơ và IR Controller” |
| Ecosystem scenario | “Cảm biến kết hợp công tắc thế nào?” |
| Unknown information | “Thiết bị có chuẩn IP68 không?” |

### Chỉ số đánh giá đề xuất

```text
Product Resolution Accuracy
Recall@K
Precision@K
MRR
Answer Groundedness
Citation Accuracy
Specification Exact Match
Hallucination Rate
Response Latency
```

---

## 🔐 Quy tắc an toàn dữ liệu

- Model chạy local qua Ollama.
- Tài liệu sản phẩm không cần gửi lên API bên thứ ba.
- Không sinh thông số không có trong nguồn.
- Câu trả lời phải dựa trên retrieval context.
- Hiển thị nguồn tham khảo cho người dùng.
- Kịch bản hệ sinh thái chỉ được công bố khi `verification_status=approved`.
- Nội dung chưa xác nhận phải được đánh dấu `draft` hoặc `pending`.

---

## 🛠️ Yêu cầu hệ thống

```text
Docker Engine     >= 24
Docker Compose    >= 2.20
Python            >= 3.11
RAM               >= 8 GB
Disk               >= 10 GB

Khuyến nghị:
RAM               >= 16 GB
GPU NVIDIA         >= 8 GB VRAM
```

Hệ thống vẫn có thể chạy bằng CPU, nhưng thời gian sinh câu trả lời sẽ lâu hơn.

---

## 🗺️ Roadmap

- [x] Hybrid Retrieval: Dense + BM25
- [x] Reciprocal Rank Fusion
- [x] Product Resolver
- [x] Multi-product Comparison
- [x] Metadata Filtering
- [x] Source Citation
- [x] Debug Retrieval
- [x] Docker Deployment
- [x] Streamlit Interface
- [ ] Tự động parse template metadata-ready
- [ ] Quản lý ecosystem scenarios riêng
- [ ] Version-aware retrieval
- [ ] Reranker model
- [ ] Retrieval evaluation dashboard
- [ ] Feedback và đánh giá câu trả lời
- [ ] Authentication và phân quyền dữ liệu
- [ ] Đồng bộ dữ liệu từ CMS
- [ ] API quản trị tài liệu sản phẩm

---

## 📝 Ghi chú

- Metadata không thay thế retrieval chunk.
- Product Metadata lưu thông tin tổng thể của sản phẩm.
- Chunk chứa nội dung trực tiếp để RAG tìm kiếm.
- Ecosystem Scenario lưu quan hệ giữa nhiều sản phẩm.
- Không nên sao chép toàn bộ Product Metadata vào từng chunk.
- BM25 phù hợp với model, mã sản phẩm, số và đơn vị.
- Vector Search phù hợp với các câu hỏi diễn đạt khác tài liệu.
- RRF giúp kết hợp ưu điểm của cả hai phương pháp.
- Chất lượng câu trả lời phụ thuộc trực tiếp vào chất lượng template và dữ liệu đầu vào.

---

## 📄 License

MIT License — xem [LICENSE](LICENSE).

# ⭐ If you like this project

Give this repository a star ⭐