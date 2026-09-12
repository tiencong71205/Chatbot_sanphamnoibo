# Patch v71: nút ⋮ luôn hiện, bỏ hiệu ứng ẩn/hiện (gồm v69 + v70)

## Thay đổi
Bỏ hẳn việc ẩn nút ⋮ rồi hiện khi rê chuột. Nút giờ luôn hiện, cùng nền
trắng và viền như ô tên đoạn chat bên cạnh.

## Vì sao đây là lựa chọn tốt hơn
Cơ chế ẩn/hiện đã gây ra một loạt lỗi liên tiếp trong mấy bản vá vừa rồi:
- Ẩn nhầm cả menu Tài khoản (v69 phải sửa)
- Nền xám khi rê chuột, mà nút chỉ hiện lúc rê nên lúc nào cũng thấy xám
  (v70 phải sửa)
- Trên điện thoại không có "rê chuột" nên phải viết thêm ngoại lệ riêng

Bỏ đi thì mọi thứ đơn giản hơn hẳn: một trạng thái duy nhất, hiển thị
giống nhau trên máy tính lẫn điện thoại, không cần ngoại lệ nào.

Đổi lại, danh sách trông hơi "nhiều nút" hơn một chút so với Claude/ChatGPT
— nhưng đó là đánh đổi hợp lý cho một công cụ nội bộ, và cũng dễ dùng hơn
cho người chưa quen (nhìn là biết có thể đổi tên/xoá, không phải mò).

Đã dọn sạch cả quy tắc dành riêng cho thiết bị cảm ứng — không còn cần nữa.

## Đã kiểm thử
Syntax hợp lệ, không còn quy tắc opacity nào liên quan tới nút menu,
test suite 58 pass / 16 fail (đúng baseline).

## Triển khai
    docker compose restart frontend
Ctrl+F5.
