"""
Hỏi chatbot Vhomenex "cách lắp đặt" cho từng sản phẩm trong 17 sản phẩm,
in ra câu trả lời để đối chiếu với nội dung tài liệu gốc thật.

Cách chạy (trên server, nơi chatbot đang chạy ở localhost:8000):
    python3 test_installation_answers.py

Chỉ dùng thư viện chuẩn (urllib) -- không cần cài thêm gì.
"""
from __future__ import annotations

import json
import urllib.request
from urllib.error import URLError, HTTPError

BASE_URL = "http://localhost:8000"

# Tên sản phẩm hiển thị + câu hỏi thật gửi cho chatbot -- dùng đúng tên
# sản phẩm để tăng khả năng chatbot resolve đúng ngay, không bị hỏi lại.
PRODUCTS = [
    ("Bộ điều khiển hồng ngoại thông minh", "cho tôi cách lắp đặt bộ điều khiển hồng ngoại thông minh"),
    ("Cảm biến cửa Mesh", "cho tôi cách lắp đặt cảm biến cửa Mesh"),
    ("Cảm biến chuyển động ánh sáng", "cho tôi cách lắp đặt cảm biến chuyển động ánh sáng"),
    ("Cảm biến hiện diện", "cho tôi cách lắp đặt cảm biến hiện diện"),
    ("Công tắc chống giật cho BNN", "cho tôi cách lắp đặt công tắc chống giật cho BNN"),
    ("Công tắc thông minh dimmer", "cho tôi cách lắp đặt công tắc thông minh dimmer"),
    ("Động Cơ Rèm", "cho tôi cách lắp đặt động cơ rèm"),
    ("Khóa điện tử (wifi)", "cho tôi cách lắp đặt khóa điện tử wifi"),
    ("Công tắc thông minh", "cho tôi cách lắp đặt công tắc thông minh"),
    ("Bộ giám sát tiêu thụ điện năng thông minh (Công tơ)", "cho tôi cách lắp đặt công tơ"),
    ("Bộ điều khiển trung tâm - Gateway", "cho tôi cách lắp đặt Gateway"),
    ("Công tắc thông minh cho cửa cuốn, cửa cổng", "cho tôi cách lắp đặt công tắc thông minh cho cửa cuốn cửa cổng"),
    ("Bộ trung tâm Gateway khóa", "cho tôi cách lắp đặt Gateway khóa"),
    ("Khóa Điện Tử Thông Minh (Cửa Gỗ)", "cho tôi cách lắp đặt khóa điện tử cửa gỗ"),
    ("Công tắc cơ thông minh", "cho tôi cách lắp đặt công tắc cơ thông minh"),
    ("Công tắc thông minh không dây", "cho tôi cách lắp đặt công tắc thông minh không dây"),
    ("Ổ cắm thông minh chống giật", "cho tôi cách lắp đặt ổ cắm thông minh chống giật"),
]


def ask_chatbot(question: str, timeout: int = 60) -> dict:
    """Gửi 1 câu hỏi tới /api/chat, trả về JSON response."""
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
    print(f"Sẽ hỏi {len(PRODUCTS)} câu hỏi lắp đặt, mỗi câu có thể mất vài giây...\n")
    print("=" * 78)

    results = []

    for i, (display_name, question) in enumerate(PRODUCTS, 1):
        print(f"\n[{i}/{len(PRODUCTS)}] ### SẢN PHẨM: {display_name}")
        print(f"    Câu hỏi gửi: \"{question}\"")
        print("-" * 78)

        try:
            data = ask_chatbot(question)
            answer = data.get("answer", "(không có trường 'answer' trong response)")
            print(answer)
            results.append({"product": display_name, "question": question, "answer": answer, "error": None})
        except HTTPError as e:
            msg = f"[LỖI HTTP {e.code}] {e.reason}"
            print(msg)
            results.append({"product": display_name, "question": question, "answer": None, "error": msg})
        except URLError as e:
            msg = f"[LỖI KẾT NỐI] Không gọi được {BASE_URL} -- chatbot có đang chạy không? Chi tiết: {e.reason}"
            print(msg)
            results.append({"product": display_name, "question": question, "answer": None, "error": msg})
        except Exception as e:  # noqa: BLE001 -- muốn thấy rõ lỗi bất ngờ, không nuốt âm thầm
            msg = f"[LỖI KHÔNG XÁC ĐỊNH] {e}"
            print(msg)
            results.append({"product": display_name, "question": question, "answer": None, "error": msg})

        print("=" * 78)

    # Tóm tắt cuối: sản phẩm nào bị lỗi, dễ rà soát nhanh
    errors = [r for r in results if r["error"]]
    print(f"\n\nTÓM TẮT: {len(results) - len(errors)}/{len(results)} câu hỏi thành công.")
    if errors:
        print(f"{len(errors)} câu hỏi bị lỗi:")
        for r in errors:
            print(f"  - {r['product']}: {r['error']}")

    # Lưu thêm ra file JSON để tiện xử lý/so sánh tự động sau này nếu cần
    with open("installation_answers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nĐã lưu toàn bộ kết quả (kèm câu hỏi) vào installation_answers.json")


if __name__ == "__main__":
    main()
