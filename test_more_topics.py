"""
Hỏi chatbot Vhomenex 3 chủ đề mới cho từng sản phẩm: kiểm tra sau lắp
đặt, thông số kỹ thuật, FAQ liên quan kết nối -- in thẳng ra màn hình
để đọc và đối chiếu ngay, không tạo file.

Cách chạy (trên server, nơi chatbot đang chạy ở localhost:8000):
    python3 test_more_topics.py
"""
from __future__ import annotations

import json
import urllib.request
from urllib.error import URLError, HTTPError

BASE_URL = "http://localhost:8000"

PRODUCTS = [
    "Bộ điều khiển hồng ngoại thông minh",
    "Cảm biến cửa Mesh",
    "Cảm biến chuyển động ánh sáng",
    "Cảm biến hiện diện",
    "Công tắc chống giật cho BNN",
    "Công tắc thông minh dimmer",
    "Động Cơ Rèm",
    "Khóa điện tử wifi",
    "Công tắc thông minh",
    "Công tơ",
    "Gateway",
    "Công tắc thông minh cho cửa cuốn cửa cổng",
    "Gateway khóa",
    "Khóa điện tử cửa gỗ",
    "Công tắc cơ thông minh",
    "Công tắc thông minh không dây",
    "Ổ cắm thông minh chống giật",
]

# 3 dạng câu hỏi -- {product} sẽ được thay bằng tên sản phẩm ở trên
QUESTION_TEMPLATES = [
    ("Kiểm tra sau lắp đặt", "hướng dẫn kiểm tra sau khi lắp đặt {product}"),
    ("Thông số kỹ thuật", "thông số kỹ thuật của {product} là gì"),
    ("FAQ kết nối", "{product} bị lỗi không kết nối được thì xử lý sao"),
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
    total = len(PRODUCTS) * len(QUESTION_TEMPLATES)
    print(f"Sẽ hỏi {total} câu ({len(PRODUCTS)} sản phẩm x 3 chủ đề)...\n")
    print("=" * 78)

    count = 0
    for product in PRODUCTS:
        for topic_label, template in QUESTION_TEMPLATES:
            count += 1
            question = template.format(product=product)
            print(f"\n[{count}/{total}] ### {product} -- {topic_label}")
            print(f"    Câu hỏi: \"{question}\"")
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
