"""Quản lý tài khoản đăng nhập cho Vhomenex AI Assistant.

Dùng cho người quản trị, chạy trên máy chủ. Người dùng thường KHÔNG tự
đăng ký được — chủ ý như vậy để chỉ người trong đội mới có tài khoản.

Cách dùng:
    python3 scripts/quan_ly_tai_khoan.py them          # tạo tài khoản (hỏi từng bước)
    python3 scripts/quan_ly_tai_khoan.py danh-sach     # xem tài khoản đang có
    python3 scripts/quan_ly_tai_khoan.py xoa <tên>     # xóa tài khoản
    python3 scripts/quan_ly_tai_khoan.py doi-mk <tên>  # đặt lại mật khẩu
"""
from __future__ import annotations

import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))

import auth  # noqa: E402


def _nhap_mat_khau(nhac: str = "Mật khẩu") -> str:
    """Nhập mật khẩu 2 lần cho khớp. Dùng getpass nên không hiện lên màn hình."""
    while True:
        pw1 = getpass.getpass(f"{nhac}: ")
        pw2 = getpass.getpass("Nhập lại: ")
        if pw1 != pw2:
            print("  [!] Hai lần nhập không khớp, thử lại.\n")
            continue
        if len(pw1) < 5:
            print("  [!] Mật khẩu phải có ít nhất 5 ký tự.\n")
            continue
        return pw1


def _lam_sach(text: str) -> str:
    """Loại bỏ ký tự hỏng do terminal không dùng bảng mã UTF-8.

    Khi gõ tiếng Việt qua `docker exec -it` vào container không đặt locale,
    Python đọc được các "surrogate" không hợp lệ (\\udcc3...) — ghi ra file
    JSON sẽ báo UnicodeEncodeError. Dòng này chuyển chúng về dạng ghi được
    thay vì để lệnh thất bại sau khi người dùng đã nhập xong mật khẩu.
    """
    return text.encode("utf-8", "replace").decode("utf-8", "replace")


def them() -> None:
    print("=== Tạo tài khoản mới ===")
    username = _lam_sach(input("Tên đăng nhập (không dấu, viết liền): ").strip())
    full_name = _lam_sach(input("Họ tên hiển thị: ").strip())
    password = _nhap_mat_khau()

    if "\ufffd" in full_name:
        print("  [!] Họ tên có ký tự bị lỗi bảng mã — nhập lại không dấu,")
        print("      hoặc đặt tên bằng lệnh trực tiếp (xem hướng dẫn).")

    # Không hỏi "có phải quản trị viên không": quyền quản trị được cố định
    # trong auth.ADMIN_USERNAMES, không phải thứ chọn được lúc tạo tài khoản.
    ok, msg = auth.create_user(username, password, full_name)
    print(("  ✓ " if ok else "  ✗ ") + msg)


def danh_sach() -> None:
    users = auth.list_users()
    if not users:
        print("Chưa có tài khoản nào.")
        print("Tạo tài khoản đầu tiên: python3 scripts/quan_ly_tai_khoan.py them")
        return

    print(f"=== {len(users)} tài khoản ===\n")
    print(f"{'TÊN ĐĂNG NHẬP':<20} {'HỌ TÊN':<28} {'QUYỀN':<12} NGÀY TẠO")
    print("-" * 78)
    for name, info in sorted(users.items()):
        quyen = "Quản trị" if info["is_admin"] else "Người dùng"
        ngay = (info.get("created_at") or "")[:10]
        print(f"{name:<20} {info['full_name']:<28} {quyen:<12} {ngay}")


def xoa(username: str) -> None:
    users = auth.list_users()
    if username not in users:
        print(f"  ✗ Không tìm thấy tài khoản '{username}'.")
        return
    xac_nhan = input(f"Xóa tài khoản '{username}'? Gõ 'xoa' để xác nhận: ").strip()
    if xac_nhan != "xoa":
        print("  Đã hủy.")
        return
    ok, msg = auth.delete_user(username)
    print(("  ✓ " if ok else "  ✗ ") + msg)
    print("  Lưu ý: lịch sử hội thoại của tài khoản này vẫn còn trong "
          f"ui/chat_sessions/{username}/ — xóa thủ công nếu cần.")


def doi_mk(username: str) -> None:
    """Đặt lại mật khẩu mà KHÔNG cần mật khẩu cũ — dùng khi người dùng quên.

    Cố ý tách khỏi hàm change_password() (vốn bắt buộc có mật khẩu cũ):
    ở đây người chạy lệnh là quản trị viên đã có quyền trên máy chủ, còn
    trong giao diện web thì luôn phải nhập mật khẩu hiện tại.
    """
    users = auth.load_users()
    if username not in users:
        print(f"  ✗ Không tìm thấy tài khoản '{username}'.")
        return

    print(f"=== Đặt lại mật khẩu cho '{username}' ===")
    password = _nhap_mat_khau("Mật khẩu mới")

    import secrets
    salt = secrets.token_hex(16)
    users[username]["salt"] = salt
    users[username]["password_hash"] = auth._hash_password(password, salt)
    auth.save_users(users)
    print(f"  ✓ Đã đặt lại mật khẩu cho '{username}'.")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return

    lenh = sys.argv[1]
    if lenh == "them":
        them()
    elif lenh in ("danh-sach", "ds"):
        danh_sach()
    elif lenh == "xoa":
        if len(sys.argv) < 3:
            print("Thiếu tên tài khoản. Ví dụ: python3 scripts/quan_ly_tai_khoan.py xoa cong")
            return
        xoa(sys.argv[2].strip().lower())
    elif lenh == "doi-mk":
        if len(sys.argv) < 3:
            print("Thiếu tên tài khoản. Ví dụ: python3 scripts/quan_ly_tai_khoan.py doi-mk cong")
            return
        doi_mk(sys.argv[2].strip().lower())
    else:
        print(f"Không hiểu lệnh '{lenh}'.\n")
        print(__doc__)


if __name__ == "__main__":
    main()
