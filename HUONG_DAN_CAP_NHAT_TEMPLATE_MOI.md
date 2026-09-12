# Cập nhật tài liệu theo template mô tả sản phẩm mới

## 1. Kiểm tra tài liệu Cảm biến cửa Mesh

```bash
python scripts/validate_template_doc.py \
  data/source_docs/01_Cam_bien_cua_Mesh_v1.0.docx
```

Kết quả hợp lệ phải có `VALIDATION_OK`. File hiện tạo 24 chunk: 7 `spec`,
5 `procedure`, 5 `feature`, 4 `section` và 3 `faq`.

Không lưu preview/chunk sinh sẵn trong gói bàn giao; dữ liệu này phải được tạo
lại từ tài liệu nguồn khi cần kiểm tra.

## 2. Cấu hình staging hợp nhất

```text
QDRANT_COLLECTION=vhomenex_products_unified_v1
DATA_DIRECTORY=/app/data/source_docs
BM25_INDEX_PATH=/app/storage/bm25/unified_v1.joblib
```

Không đổi về `vhomenex_products_v2` trước khi retrieval, citation và câu trả lời
hướng dẫn nhiều bước đạt regression test.

## 3. Dry-run và ingest

```bash
docker exec vhomenex-backend-v2 \
  python scripts/ingest_documents.py \
  --data-dir /app/data/source_docs \
  --dry-run
```

```bash
docker exec vhomenex-backend-v2 \
  python scripts/ingest_documents.py \
  --data-dir /app/data/source_docs \
  --recreate
```

`--recreate` chỉ được chạy khi collection vẫn là
`vhomenex_products_unified_v1`.

## 4. Quy tắc cho tài liệu tiếp theo

- Giữ Heading 1 và Heading 2 theo template.
- Dùng Heading 3 cho từng tính năng hoặc nhóm hướng dẫn.
- Bỏ mục không có nội dung và nhãn hình ảnh trống.
- Viết quy trình theo `B1:`, `B2:`... để parser giữ trong một `procedure` chunk.
- Bảng thông số dùng hai cột thuộc tính/giá trị.
- Bảng FAQ dùng ba cột `Câu hỏi`, `Tình trạng`, `Cách khắc phục`.
- Chạy validator và dry-run trước khi ingest.
