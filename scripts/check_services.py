#!/usr/bin/env python3
"""Check all services are running correctly."""
import sys
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings


def check():
    ok = True
    print("=" * 50)
    print("SERVICE HEALTH CHECK")
    print("=" * 50)

    # Ollama
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        llm_ok = settings.ollama_llm_model in models
        emb_ok = settings.ollama_embedding_model in models
        print(f"✅ Ollama: {settings.ollama_base_url}")
        print(f"   LLM {settings.ollama_llm_model}: {'✅' if llm_ok else '❌'}")
        print(f"   Embed {settings.ollama_embedding_model}: {'✅' if emb_ok else '❌'}")
        if not llm_ok or not emb_ok:
            ok = False
    except Exception as e:
        print(f"❌ Ollama: {e}")
        ok = False

    # Qdrant
    try:
        r = httpx.get(f"{settings.qdrant_url}/collections", timeout=5)
        collections = [c["name"] for c in r.json().get("result", {}).get("collections", [])]
        has_col = settings.qdrant_collection in collections
        print(f"✅ Qdrant: {settings.qdrant_url}")
        print(f"   Collection {settings.qdrant_collection}: {'✅' if has_col else '⚠️ not found'}")
    except Exception as e:
        print(f"❌ Qdrant: {e}")
        ok = False

    # BM25
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        print(f"✅ BM25 index: {bm25_path} ({bm25_path.stat().st_size // 1024} KB)")
    else:
        print(f"⚠️  BM25 index not found: {bm25_path}")

    print("=" * 50)
    print("STATUS:", "OK" if ok else "ISSUES FOUND")
    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
