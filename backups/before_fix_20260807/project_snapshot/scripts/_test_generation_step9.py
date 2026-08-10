import sys, httpx, json

url = "http://127.0.0.1:8000/chat"

test_questions = [
    {"question": "Cảm biến cửa VCN-SEN-DOOR sử dụng loại pin gì và tuổi thọ bao lâu?"},
    {"question": "Bộ điều khiển hồng ngoại VCN-IR-REMOTE có thể học lệnh remote cũ không?"},
    {"question": "Động cơ rèm VCN-CURTAIN-M1 chịu được tải trọng bao nhiêu kg?"},
    {"question": "Bóng đèn RGB VCN-BULB-RGB reset bằng cách nào?"},
    {"question": "Khóa vân tay VCN-LOCK-PRO mở bằng những phương thức nào?"},
]

for idx, item in enumerate(test_questions, 1):
    q = item["question"]
    print("=" * 60)
    print(f"TEST {idx}: {q}")
    print("=" * 60)
    r = httpx.post(url, json={"question": q, "debug": True}, timeout=120)
    if r.status_code == 200:
        data = r.json()
        print("ANSWER:\n", data["answer"])
        print("\nLATENCY:", round(data["latency_ms"]), "ms")
        print("SOURCES COUNT:", len(data["sources"]))
        for s in data["sources"]:
            print(f"  - [{s['source_id']}] {s['product_name']} | {s['source_document']} | {s['content_type']}")
    else:
        print("ERROR:", r.status_code, r.text)
    print()
