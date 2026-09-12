# Triển khai Vhomenex AI Assistant lên tên miền nội bộ

Mục tiêu: truy cập bằng tên miền (thay vì `192.168.1.226:8501`), và **chỉ
mạng RD hoặc HO mới vào được**.

---

## Cần chuẩn bị trước — 4 thông tin

Chưa có đủ 4 thông tin này thì chưa triển khai được:

| # | Thông tin | Cách lấy |
|---|---|---|
| 1 | **Dải IP mạng RD** | Trên 1 máy trong mạng RD: `ipconfig` (Windows) hoặc `ip addr` (Linux). Lấy dạng `10.10.20.0/24` |
| 2 | **Dải IP mạng HO** | Tương tự, trên 1 máy ở Head Office |
| 3 | **Tên miền chốt** | `tracuu.vconnex.vn` / `chat.vconnex.vn` / `ai.vconnex.vn` |
| 4 | **Ai quản lý DNS `vconnex.vn`** | Cần người này tạo bản ghi trỏ tên miền về `192.168.1.226` |

---

## Bước 1 — Cài nginx trên máy chủ

```bash
sudo apt update && sudo apt install -y nginx
```

## Bước 2 — Điền 3 chỗ trong file cấu hình

Mở `deploy/nginx-vhomenex.conf`, sửa 3 chỗ đánh dấu `# >>> SỬA <<<`:
dải mạng RD, dải mạng HO, và tên miền.

## Bước 3 — Kích hoạt cấu hình

```bash
sudo cp ~/vhomenex_hybrid_rag/deploy/nginx-vhomenex.conf /etc/nginx/sites-available/vhomenex
sudo ln -sf /etc/nginx/sites-available/vhomenex /etc/nginx/sites-enabled/vhomenex
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t          # phải báo "syntax is ok" và "test is successful"
sudo systemctl reload nginx
```

## Bước 4 — Trỏ tên miền (nhờ người quản lý DNS)

Tạo bản ghi A trong DNS nội bộ:

```
tracuu.vconnex.vn.   A   192.168.1.226
```

**Lưu ý quan trọng**: đây là địa chỉ nội bộ (`192.168.x.x`), nên bản ghi
này phải nằm ở **DNS nội bộ công ty**, không phải DNS công cộng của
`vconnex.vn` — đưa IP nội bộ lên DNS công cộng vừa không hoạt động ngoài
công ty, vừa lộ cấu trúc mạng ra ngoài.

## Bước 5 — Đóng cổng truy cập trực tiếp (quan trọng)

Sau khi tên miền chạy được, phải chặn cổng 8501/8000 để không ai bỏ qua
được lớp kiểm tra mạng của nginx bằng cách gọi thẳng IP:cổng:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 22/tcp          # giữ SSH, nếu không sẽ tự khoá mình ra ngoài
sudo ufw deny 8501/tcp
sudo ufw deny 8000/tcp
sudo ufw enable
sudo ufw status
```

---

## Kiểm tra sau khi cài

```bash
# Từ máy trong mạng RD hoặc HO — phải mở được giao diện
curl -I http://tracuu.vconnex.vn

# Thử gọi thẳng cổng 8501 — phải KHÔNG vào được nữa
curl -m 5 -I http://192.168.1.226:8501
```

Quan trọng nhất: **mở trình duyệt, gõ 1 câu hỏi thật**. Nếu giao diện tải
được nhưng gõ câu hỏi không thấy phản hồi → nginx chưa chuyển tiếp
WebSocket đúng (xem lại phần `proxy_set_header Upgrade` trong file cấu
hình).

---

## Về HTTPS (https://)

Chưa cấu hình trong bản này, vì phụ thuộc câu trả lời cho câu hỏi: tên
miền nội bộ có ra được Internet không?

| Tình huống | Cách làm |
|---|---|
| Tên miền **chỉ dùng nội bộ** (khả năng cao nhất) | Let's Encrypt kiểu thông thường **không dùng được** (cần máy chủ ra được Internet để xác thực). Chọn: (a) dùng HTTP thôi — chấp nhận được trong mạng nội bộ, (b) chứng chỉ tự ký — trình duyệt sẽ cảnh báo đỏ mỗi lần vào, (c) chứng chỉ từ CA nội bộ công ty nếu IT có sẵn |
| Công ty **quản lý được DNS** của `vconnex.vn` | Dùng Let's Encrypt kiểu DNS-01 — có chứng chỉ thật, không cần mở máy chủ ra Internet. Đây là cách tốt nhất |

Cho mình biết công ty thuộc tình huống nào, mình sẽ đưa cấu hình HTTPS
tương ứng.

---

## Cảnh báo về cách giới hạn theo dải IP

Chặn theo dải mạng là lớp bảo vệ **cơ bản**, phù hợp cho công cụ nội bộ —
nhưng cần biết rõ giới hạn của nó:

- Ai cắm được vào mạng RD/HO (kể cả khách, kể cả máy bị nhiễm mã độc)
  đều truy cập được. Không có đăng nhập, không phân biệt được ai là ai.
- Không ghi lại được **ai** đã hỏi gì — chỉ biết IP nào.
- Nếu sau này cần biết chính xác người dùng (để phân quyền, hoặc để thống
  kê feedback theo người), sẽ cần thêm lớp đăng nhập thật. Đây là việc
  riêng, nên làm sau khi phần này chạy ổn.
