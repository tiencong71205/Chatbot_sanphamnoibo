import requests
import json

def test_chat():
    url = "http://localhost:8000/chat"
    questions = [
        "Công tắc chống giật BNN có những tính năng an toàn gì?",
        "Cảm biến hiện diện Vhomenex dùng công nghệ gì và khoảng cách phát hiện tối đa bao xa?",
        "Khóa điện tử Vhomenex hỗ trợ những phương thức mở khóa nào?"
    ]
    
    for q in questions:
        print("\n" + "="*50)
        print("Q:", q)
        resp = requests.post(url, json={"question": q, "debug": True})
        if resp.status_code == 200:
            data = resp.json()
            print("\nANSWER:\n", data.get("answer"))
            print("\nSOURCES:")
            for s in data.get("sources", []):
                print(f"  - [{s.get('source_id')}] {s.get('product_name')} ({s.get('source_document')})")
            print(f"\nLATENCY: {data.get('latency_ms', 0):.0f}ms")
        else:
            print("ERROR:", resp.status_code, resp.text)

if __name__ == "__main__":
    test_chat()
