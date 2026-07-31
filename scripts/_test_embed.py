import os, httpx

base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
print("OLLAMA_BASE_URL =", base)

try:
    r = httpx.get(f"{base}/api/tags", timeout=10)
    print("tags status =", r.status_code)
    models = [m["name"] for m in r.json().get("models", [])]
    print("available models =", models[:10])
except Exception as e:
    print("tags ERROR:", e)

try:
    r2 = httpx.post(
        f"{base}/api/embed",
        json={"model": "qwen3-embedding:0.6b", "input": ["Kiểm tra embedding cho sản phẩm Vhomenex"]},
        timeout=120
    )
    print("embed status =", r2.status_code)
    d = r2.json()
    v = d.get("embeddings", d.get("embedding", []))
    vec = v[0] if v and isinstance(v[0], list) else v
    print("dimension =", len(vec))
    print("first_5 =", vec[:5])
except Exception as e:
    print("embed ERROR:", e)
