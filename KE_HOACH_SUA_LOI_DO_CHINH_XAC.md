# KẾ HOẠCH SỬA LỖI ĐỘ CHÍNH XÁC

Dựa trên đợt test 279 câu (17 sản phẩm) và danh sách lỗi Công tổng hợp.
Mỗi mục dưới đây đều đã đối chiếu trực tiếp với tài liệu gốc.

---

## PHÂN LOẠI: lỗi thuộc về đâu?

Điều quan trọng nhất sau khi kiểm chứng: **3 trong 5 lỗi đã kiểm KHÔNG
phải lỗi chatbot**. Sửa nhầm chỗ sẽ tốn công mà không hết lỗi.

| Nhóm | Số lỗi (đã kiểm) | Sửa ở đâu |
|---|---|---|
| A. Lỗi script test | 1 | Sửa script, 5 phút |
| B. Lỗi tài liệu gốc | 2 | Cần Công/người soạn tài liệu sửa |
| C. Lỗi chatbot thật | 2 | Sửa code |

---

## NHÓM A — LỖI SCRIPT TEST (ưu tiên cao nhất, dễ nhất)

### A1. Khóa cửa gỗ "thiếu nguyên phần 5"  → KHÔNG phải lỗi chatbot
**Đã kiểm chứng:** tài liệu có đủ 15 mục (5.1–5.15), nhưng script test
trích được **0 mục**.

**Nguyên nhân:** file này đặt tên chương là "**Tính năng** với ứng dụng
Vhomenex", trong khi mọi file khác dùng "**Chức năng** với ứng dụng".
Mẫu nhận diện trong script chỉ khớp chữ "Chức năng".

**Hệ quả:** 15 mục quan trọng nhất của sản phẩm khóa cửa gỗ CHƯA TỪNG
được test lần nào. Rất có thể còn lỗi thật đang ẩn trong đó.

**Cách sửa:** đổi mẫu nhận diện thành `(chức năng|tính năng).*ứng dụng`.
Sau đó chạy lại riêng sản phẩm này.

**Việc cần làm:** sửa script → chạy lại khóa cửa gỗ → đánh giá 15 mục mới.

---

## NHÓM B — LỖI TÀI LIỆU GỐC (chatbot trả lời đúng theo tài liệu)

### B1. Bộ điều khiển hồng ngoại — mục 5.4 "Vào chế độ OTA"
**Đã kiểm chứng:** chatbot trả lời **khớp chính xác** tài liệu:
- B1: Truy cập ứng dụng, chọn chế độ Cập nhật OTA
- B2: Theo dõi đèn nhấp nháy 3 lần

Tài liệu mục 5.4 mô tả: *"Cho phép nhận biết thiết bị đang vào trạng thái
cập nhật OTA"* — tức là mục này nói về cách NHẬN BIẾT qua đèn, nhưng bước
đầu tiên lại là "chọn chế độ Cập nhật OTA", nên nghe trùng với mục 6.8
"Cập nhật OTA".

**Kết luận:** chatbot không sai. Muốn hết nhầm lẫn thì phải sửa tài liệu
cho rõ ràng hơn (ví dụ đổi tên mục 5.4 thành "Nhận biết thiết bị đang
cập nhật OTA qua đèn báo").

**Việc cần làm:** Công quyết định — sửa tài liệu hay chấp nhận.

### B2. Cảm biến hiện diện — mục 4.4 "Giám sát thiết bị"
**Đã kiểm chứng:** tài liệu ghi mục 4.4 mô tả *"Theo dõi phát hiện hiện
diện trong phạm vi quét"*, NHƯNG phần hướng dẫn bên dưới lại là:
- B1: Nhấn Thêm thiết bị
- B2: Chọn cảm biến hiện diện
- B3: Chọn loại kết nối Bluetooth mesh
- B4: Chọn bộ điều khiển trung tâm

Đây rõ ràng là hướng dẫn **THÊM/KẾT NỐI** thiết bị, không phải giám sát —
nhiều khả năng bị copy nhầm từ mục 4.3.

**Hệ quả:** mục 4.3 và 4.4 có nội dung gần giống hệt nhau, nên chatbot
trả lời cho 4.3 bằng nội dung của 4.4 — retrieval lẫn lộn là điều tất
yếu, không phải lỗi logic.

**Việc cần làm:** BẮT BUỘC sửa tài liệu. Không sửa thì mọi cải tiến code
đều không giải quyết được, vì hai mục vốn dĩ trùng nội dung.

---

## NHÓM C — LỖI CHATBOT THẬT (sửa code)

### C1. Không tái hiện đúng các bước B1/B2/B3 → trả lời lan man
**Ví dụ đã kiểm:** Cảm biến cửa Mesh, mục 5.1 "Trạng thái thiết bị".

Tài liệu có 3 bước rõ ràng:
- B1: Tại Trang chủ, chọn xem chi tiết Cảm biến cửa Vconnex
- B2: Kiểm tra trạng thái Đóng/Mở trên ứng dụng
- B3: Thực hiện đóng/mở cửa để kiểm tra trạng thái cập nhật

Chatbot trả lời bằng 5 gạch đầu dòng chung chung, trộn nội dung từ **7
nguồn khác nhau** (SOURCE_1,2,6,8,9,10,11), mất hẳn cấu trúc 3 bước.

**Nguyên nhân nghi ngờ:** câu hỏi dạng "...thực hiện như thế nào?" khiến
retrieval kéo về quá nhiều chunk liên quan, rồi model tổng hợp lan man
thay vì bám đúng mục được hỏi.

### C2. Bỏ sót phần sau trong mục có nhiều phần
**Ví dụ đã kiểm:** Khóa điện tử, mục 4.11 "Cài đặt mật mã mở khóa".

Tài liệu có **3 phần**, mỗi phần 7 bước:
1. Thêm mật mã mở khóa
2. Xóa mật mã mở khóa
3. Xóa tất cả mật mã

Chatbot chỉ trả lời phần 1, **thiếu 2/3 nội dung**.

Đây cùng loại với lỗi đã gặp trước đây (bỏ sót phần giữa/cuối danh sách
dài), đã thử 2 vòng vá prompt chưa dứt điểm. Nhiều khả năng mục 4.12,
4.13 cũng bị y hệt (cùng cấu trúc 3 phần).

**Hướng sửa đề xuất — theo thứ tự nên thử:**
1. **Ưu tiên 1:** khi câu hỏi nhắm đúng một mục cụ thể (4.11, 5.1...),
   ưu tiên lấy TRỌN mục đó thay vì trộn nhiều mục liên quan. Hiện có
   `retrieve_full()` làm được việc này nhưng chỉ dùng cho intent
   `section_full`; cần mở rộng để nhận diện được câu hỏi kiểu này.
2. **Ưu tiên 2:** nếu chunk của một mục bị cắt thành nhiều phần, kéo về
   đủ các phần anh em (đã có `_pull_in_sibling_chunks`, cần kiểm tra
   xem có áp dụng cho trường hợp này chưa).
3. **Ưu tiên 3:** nếu 2 cách trên chưa đủ → cân nhắc tăng giới hạn độ
   dài câu trả lời cho câu hỏi dạng liệt kê nhiều phần.

---

## CÁC LỖI CHƯA KIỂM CHỨNG (cần làm tiếp)

Chưa đối chiếu tài liệu, chưa kết luận được thuộc nhóm nào:

- [ ] Công tắc cửa cuốn/cửa cổng V2 — "Chế độ kết nối thủ công qua
      Bluetooth Mesh"
- [ ] Nhầm lẫn giữa Bộ điều khiển trung tâm (Gateway) và Bộ trung tâm
      Gateway khóa  ← đã từng sửa ở patch trước, cần xem còn sót không
- [ ] Ổ cắm thông minh chống giật — mục 4.5
- [ ] Cảm biến chuyển động ánh sáng — mục 4.3
- [ ] Cảm biến hiện diện — mục 5.7
- [ ] Bộ giám sát tiêu thụ điện năng — mục 5.5
- [ ] Khóa điện tử — mục 4.12, 4.13 (nghi cùng lỗi với 4.11)

---

## THỨ TỰ THỰC HIỆN ĐỀ XUẤT

**Bước 1 — Sửa script test (30 phút)**
Sửa mẫu nhận diện chương, chạy lại khóa cửa gỗ. Làm trước vì đang có 15
mục chưa từng được test, có thể lộ thêm lỗi mới ảnh hưởng tới kế hoạch.

**Bước 2 — Kiểm chứng nốt 7 lỗi còn lại (1-2 giờ)**
Đối chiếu tài liệu từng cái, phân loại vào nhóm A/B/C. Không nên sửa
code trước khi biết chắc bao nhiêu lỗi thực sự thuộc về code.

**Bước 3 — Sửa tài liệu (cần người soạn tài liệu)**
Ít nhất mục 4.4 Cảm biến hiện diện. Đây là việc không thể thay thế bằng
sửa code.

**Bước 4 — Sửa code cho nhóm C**
Làm sau cùng, khi đã biết đầy đủ phạm vi. Thử theo thứ tự ưu tiên 1→2→3
ở phần C2, test lại sau mỗi bước.

---

## LƯU Ý QUAN TRỌNG

Trong 5 lỗi đã kiểm, chỉ **2 lỗi thuộc về chatbot**. Nếu lao vào sửa code
ngay từ đầu thì phần lớn công sức sẽ đổ vào những chỗ không phải nguyên
nhân — đúng bài học đã rút ra ở các đợt sửa lỗi trước: kiểm chứng bằng
dữ liệu thật trước khi sửa.
