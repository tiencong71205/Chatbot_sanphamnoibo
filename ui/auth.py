"""Quản lý tài khoản và xác thực cho Vhomenex AI Assistant.

Chỉ dùng thư viện chuẩn của Python — không phụ thuộc bcrypt/passlib, vì
`pip` trên máy chủ đang hỏng (thiếu distutils) và việc sửa nó rủi ro hơn
lợi ích. `hashlib.pbkdf2_hmac` là hàm băm mật khẩu đúng chuẩn, có sẵn
trong Python, đủ an toàn cho công cụ nội bộ.

Mật khẩu KHÔNG BAO GIỜ được lưu dạng thô — chỉ lưu chuỗi băm kèm salt
ngẫu nhiên riêng cho từng tài khoản.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

USERS_FILE = Path(
    os.getenv("USERS_FILE", str(Path(__file__).parent / "users.json"))
)

# 200k vòng lặp: đủ chậm để việc dò mật khẩu hàng loạt trở nên tốn kém,
# nhưng vẫn dưới ~0.2s nên người dùng không cảm thấy chờ khi đăng nhập.
_PBKDF2_ROUNDS = 200_000
_SALT_BYTES = 16

# Danh sách quản trị viên được CỐ ĐỊNH TRONG CODE, không phải tuỳ chọn khi
# tạo tài khoản. Trước đây quyền quản trị là một ô tích lúc tạo tài khoản,
# nghĩa là bất kỳ ai chạy được lệnh tạo tài khoản cũng tự cho mình quyền
# quản trị được. Cố định ở đây thì muốn thêm quản trị viên phải sửa mã
# nguồn rồi khởi động lại — một việc có dấu vết rõ ràng, không xảy ra
# lặng lẽ.
ADMIN_USERNAMES = {"nguyentiencong"}


def is_admin_username(username: str) -> bool:
    return (username or "").strip().lower() in ADMIN_USERNAMES


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS
    ).hex()


def load_users() -> Dict[str, Dict[str, Any]]:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # errors="replace": ký tự hỏng bảng mã (surrogate) từ terminal không đặt
    # locale UTF-8 sẽ được thay bằng ký tự thay thế thay vì làm cả thao tác
    # lưu thất bại — mất dấu tiếng Việt trong tên hiển thị còn hơn mất luôn
    # tài khoản vừa tạo.
    USERS_FILE.write_text(
        json.dumps(users, ensure_ascii=False, indent=2),
        encoding="utf-8",
        errors="replace",
    )


def create_user(
    username: str, password: str, full_name: str = ""
) -> tuple[bool, str]:
    """Tạo tài khoản mới. Trả về (thành công, thông báo).

    KHÔNG có tham số is_admin: quyền quản trị lấy từ ADMIN_USERNAMES cố
    định trong mã nguồn, không phải thứ người tạo tài khoản tự chọn được.
    """
    username = username.strip().lower()
    if not username:
        return False, "Tên đăng nhập không được để trống."

    users = load_users()
    # Kiểm tra trùng tên TRƯỚC khi kiểm tra mật khẩu: nếu ngược lại, người
    # tạo tài khoản trùng tên mà lỡ đặt mật khẩu ngắn sẽ chỉ thấy báo lỗi
    # mật khẩu, sửa mật khẩu xong mới biết tên đã có người dùng.
    if username in users:
        return False, f"Tài khoản '{username}' đã tồn tại."

    if len(password) < 5:
        return False, "Mật khẩu phải có ít nhất 5 ký tự."

    salt = secrets.token_hex(_SALT_BYTES)
    users[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "full_name": full_name.strip() or username,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_users(users)
    quyen = " (quản trị viên)" if is_admin_username(username) else ""
    return True, f"Đã tạo tài khoản '{username}'{quyen}."


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Kiểm tra đăng nhập. Trả về thông tin tài khoản nếu đúng, None nếu sai.

    Cố ý KHÔNG phân biệt "sai tên đăng nhập" với "sai mật khẩu" ở tầng gọi:
    thông báo lỗi chung chung khiến kẻ dò mật khẩu không biết tài khoản nào
    thực sự tồn tại.
    """
    username = (username or "").strip().lower()
    users = load_users()
    user = users.get(username)
    if not user:
        # Vẫn băm một lần với salt giả để thời gian phản hồi tương đương
        # trường hợp tài khoản có thật -- tránh lộ sự tồn tại của tài khoản
        # qua việc trả lời nhanh hơn hẳn.
        _hash_password(password or "", secrets.token_hex(_SALT_BYTES))
        return None

    expected = user.get("password_hash", "")
    actual = _hash_password(password or "", user.get("salt", ""))
    # compare_digest: so sánh trong thời gian không đổi, không "thoát sớm"
    # ở ký tự đầu tiên khác nhau.
    if not secrets.compare_digest(expected, actual):
        return None

    return {
        "username": username,
        "full_name": user.get("full_name", username),
        # Đọc từ ADMIN_USERNAMES, KHÔNG đọc từ file users.json: nếu ai đó
        # sửa tay file để tự thêm "is_admin": true thì cũng vô hiệu.
        "is_admin": is_admin_username(username),
    }


def change_password(username: str, old_password: str, new_password: str) -> tuple[bool, str]:
    if not verify_user(username, old_password):
        return False, "Mật khẩu hiện tại không đúng."
    if len(new_password) < 5:
        return False, "Mật khẩu mới phải có ít nhất 5 ký tự."

    users = load_users()
    username = username.strip().lower()
    salt = secrets.token_hex(_SALT_BYTES)
    users[username]["salt"] = salt
    users[username]["password_hash"] = _hash_password(new_password, salt)
    save_users(users)
    return True, "Đã đổi mật khẩu."


def delete_user(username: str) -> tuple[bool, str]:
    users = load_users()
    username = username.strip().lower()
    if username not in users:
        return False, f"Không tìm thấy tài khoản '{username}'."
    del users[username]
    save_users(users)
    return True, f"Đã xóa tài khoản '{username}'."


def list_users() -> Dict[str, Dict[str, Any]]:
    """Danh sách tài khoản, đã bỏ hết thông tin nhạy cảm."""
    return {
        name: {
            "full_name": info.get("full_name", name),
            "is_admin": is_admin_username(name),
            "created_at": info.get("created_at", ""),
        }
        for name, info in load_users().items()
    }


# ───────────────────── Giữ đăng nhập qua tải lại trang ─────────────────────

SESSIONS_FILE = Path(
    os.getenv("SESSIONS_FILE", str(Path(__file__).parent / "sessions.json"))
)
# 7 ngày: đủ dài để không phải đăng nhập lại mỗi ngày, đủ ngắn để một máy
# bỏ quên không mở phiên vô thời hạn.
SESSION_TTL_SECONDS = 7 * 24 * 3600


def _load_sessions() -> Dict[str, Dict[str, Any]]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sessions(sessions: Dict[str, Dict[str, Any]]) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        SESSIONS_FILE.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2),
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        # Không ghi được phiên thì cùng lắm người dùng phải đăng nhập lại,
        # không được để hỏng cả luồng đăng nhập.
        pass


def create_session(username: str) -> str:
    """Tạo phiếu phiên cho người dùng, trả về mã phiếu.

    Mã phiếu sinh ngẫu nhiên bằng `secrets` (không đoán được), lưu kèm hạn
    dùng. Bản thân mã KHÔNG chứa thông tin gì về tài khoản — lộ mã thì chỉ
    dùng được tới khi hết hạn hoặc người dùng đăng xuất, không lộ mật khẩu.
    """
    sessions = _load_sessions()

    # Dọn phiên hết hạn mỗi lần tạo mới, để file không phình mãi.
    now = time.time()
    sessions = {
        tok: info for tok, info in sessions.items()
        if info.get("expires_at", 0) > now
    }

    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "username": username.strip().lower(),
        "expires_at": now + SESSION_TTL_SECONDS,
    }
    _save_sessions(sessions)
    return token


def verify_session(token: str) -> Optional[Dict[str, Any]]:
    """Kiểm tra mã phiên. Trả về thông tin tài khoản nếu còn hiệu lực."""
    if not token:
        return None
    sessions = _load_sessions()
    info = sessions.get(token)
    if not info:
        return None
    if info.get("expires_at", 0) <= time.time():
        # Hết hạn thì xoá luôn, không để rác lại.
        sessions.pop(token, None)
        _save_sessions(sessions)
        return None

    username = info.get("username", "")
    users = load_users()
    if username not in users:
        # Tài khoản đã bị xoá nhưng phiên còn -> huỷ phiên.
        sessions.pop(token, None)
        _save_sessions(sessions)
        return None

    return {
        "username": username,
        "full_name": users[username].get("full_name", username),
        "is_admin": is_admin_username(username),
    }


def destroy_session(token: str) -> None:
    """Xoá phiên khi đăng xuất."""
    if not token:
        return
    sessions = _load_sessions()
    if sessions.pop(token, None) is not None:
        _save_sessions(sessions)
