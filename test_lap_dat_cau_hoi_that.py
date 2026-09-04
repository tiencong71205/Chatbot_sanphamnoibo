"""
Hỏi chatbot đúng 17 câu hỏi "Cách lắp đặt..." theo cách diễn đạt cụ thể,
in thẳng ra màn hình để đọc và đối chiếu ngay, không tạo file.

Cách chạy (trên server, nơi chatbot đang chạy ở localhost:8000):
    python3 test_lap_dat_cau_hoi_that.py
"""
from __future__ import annotations

import json
import urllib.request
from urllib.error import URLError, HTTPError

BASE_URL = "http://localhost:8000"

QUESTIONS = [
    "Cách lắp đặt Khóa điện tử thông minh cửa gỗ như thế nào?",
    "Cách lắp đặt Bộ điều khiển trung tâm (Gateway) như thế nào?",
    "Cách lắp đặt Cảm biến hiện diện như thế nào?",
    "Cách lắp đặt Bộ giám sát tiêu thụ điện năng thông minh như thế nào?",
    "Cách lắp đặt Công tắc thông minh không dây như thế nào?",
    "Cách lắp đặt Công tắc thông minh cho cửa cuốn, cửa cổng như thế nào?",
    "Cách lắp đặt Khóa thông minh cửa nhôm (Khóa cửa WiFi) như thế nào?",
    "Cách lắp đặt Cảm biến chuyển động, ánh sáng như thế nào?",
    "Cách lắp đặt Công tắc thông minh như thế nào?",
    "Cách lắp đặt Công tắc thông minh Dimmer như thế nào?",
    "Cách lắp đặt Công tắc thông minh chống giật như thế nào?",
    "Cách lắp đặt Bộ trung tâm Gateway khóa BLE như thế nào?",
    "Cách lắp đặt Cảm biến cửa Mesh như thế nào?",
    "Cách lắp đặt Công tắc cơ thông minh như thế nào?",
    "Cách lắp đặt Bộ điều khiển hồng ngoại thông minh như thế nào?",
    "Cách lắp đặt Động cơ rèm thông minh như thế nào?",
    "Cách lắp đặt Ổ cắm thông minh chống giật như thế nào?",
]


def ask_chatbot(question: str, timeout: int = 60) -> dict:
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    total = len(QUESTIONS)
    print(f"Sẽ hỏi {total} câu...\n")
    print("=" * 78)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{total}] Câu hỏi: \"{question}\"")
        print("-" * 78)
        try:
            data = ask_chatbot(question)
            print(data.get("answer", "(không có trường 'answer')"))
        except HTTPError as e:
            print(f"[LỖI HTTP {e.code}] {e.reason}")
        except URLError as e:
            print(f"[LỖI KẾT NỐI] {e.reason} -- chatbot có đang chạy không?")
        except Exception as e:  # noqa: BLE001
            print(f"[LỖI] {e}")
        print("=" * 78)


if __name__ == "__main__":
    main()
