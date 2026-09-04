# Patch v45: cho phép TẤT CẢ mạng nội bộ công ty (RD + HO + mạng dây)

Thay thế phần `geo $allowed_network` của patch v43.

## Phát hiện quan trọng dẫn tới thay đổi này
Kiểm tra log nginx thực tế: khi máy người dùng ở dải 192.168.2.54 truy
cập, nginx ghi nhận IP nguồn là **192.168.1.5** — KHÔNG phải IP thật của
máy. Nghĩa là có thiết bị NAT giữa các mạng, thay đổi địa chỉ nguồn.

Hệ quả: việc liệt kê từng dải mạng riêng chỉ tạo CẢM GIÁC an toàn giả.
- Máy ở dải chưa khai báo (192.168.2.x) vẫn vào được bình thường
- Nginx không phân biệt được máy nào là RD, máy nào là HO
- Mỗi lần công ty thêm mạng mới lại phải sửa file + khởi động lại nginx

## Thay đổi
Cho phép toàn bộ dải nội bộ chuẩn (RFC1918):
    192.168.0.0/16
    10.0.0.0/8
    172.16.0.0/12    (gồm cả dải nội bộ Docker)
    127.0.0.1/32     (máy chủ tự gọi)

Đúng với yêu cầu thực tế: người ở RD, HO và mạng dây đều dùng được, thêm
mạng mới sau này không phải sửa gì.

## Vẫn giữ được tác dụng thật
Chặn truy cập từ Internet hoặc mạng lạ không đi qua hạ tầng nội bộ.
Đã kiểm thử: IP ngoài dải nội bộ -> HTTP 403 + trang báo lỗi tiếng Việt.

## Đã kiểm thử thật (nginx + server giả lập)
- tracuu.vconnex.vn -> HTTP 200
- chat / ai         -> HTTP 301 chuyển hướng đúng
- IP ngoài dải nội bộ -> HTTP 403 + trang tiếng Việt

## Điều cấu hình này KHÔNG làm được (cần biết rõ)
Không phân biệt được NGƯỜI DÙNG — mọi người trong công ty đều vào được
như nhau, không đăng nhập, không biết ai đã hỏi gì. Nếu sau này cần giới
hạn theo người hoặc theo phòng ban, phải thêm lớp đăng nhập; lọc theo IP
không làm được điều đó, đặc biệt khi có NAT như hạ tầng hiện tại.
