# Báo cáo hợp nhất dự án

## Phạm vi dữ liệu cuối cùng

Bản bàn giao chỉ giữ một tài liệu nguồn:
`data/source_docs/01_Cam_bien_cua_Mesh_v1.0.docx`.

`data/product_catalog.json` chỉ còn một bản ghi cho `cam_bien_cua_mesh`.

## Đã loại bỏ

- 11 tài liệu sản phẩm cũ ngoài Cảm biến cửa Mesh.
- Toàn bộ dữ liệu evaluation cũ.
- Preview/chunk đã sinh trước đó.
- BM25 index và Qdrant storage cũ.
- `.venv`, cache, file backup và script thử nghiệm tạm.

## Mã nguồn đã giữ và sửa

- Parser/chunker/metadata theo template Word mới.
- Phân biệt Cảm biến cửa Mesh với Gateway và USB Converter trong câu hỏi kết nối.
- Bảo toàn quy trình `B1...Bn` khi tạo câu trả lời.
- Citation chỉ trả về khi câu trả lời thực sự dùng nguồn.
- Debug trả `retrieval_ms`, `generation_ms` và `total_ms`.
- Giao diện không hiển thị bullet rỗng dưới nguồn tham khảo.

## Tài liệu Cảm biến cửa Mesh

Bản tài liệu trong gói là bản đã sửa từ tài liệu người dùng cung cấp:

- Sửa `Trạng thái thiết bí` thành `Trạng thái thiết bị`.
- Xóa nhãn `b) Hướng dẫn:` bị dính vào dòng mô tả.
- Làm rõ cảm biến chỉ phát hiện/cập nhật trạng thái, không điều khiển cửa.
- Bỏ nội dung suy đoán về sạc pin.
- Xóa page break rỗng gây trang trắng cuối.

## Triển khai

Collection mặc định là `vhomenex_products_unified_v1`. Khi ingest với
`--recreate`, collection này sẽ chỉ chứa chunk của Cảm biến cửa Mesh.
