import sys, os
sys.path.insert(0, os.path.abspath("."))

from app.config import settings
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.hybrid_retriever import HybridRetriever

qdrant = QdrantStore(settings)
bm25 = BM25Store(settings)
emb = OllamaEmbedding(settings)
retriever = HybridRetriever(settings, qdrant, bm25, emb)

queries = [
    "Công tắc cửa cuốn có dùng được khi mất Internet không?",
    "Cách kết nối thủ công công tắc cửa cuốn",
    "Model VCN-WSRGC2 dùng điện áp bao nhiêu?",
    "Cảm biến cửa sử dụng chuẩn kết nối gì?",
    "Bộ điều khiển hồng ngoại có chức năng gì?",
]

for i, q in enumerate(queries, 1):
    print("=" * 60)
    print(f"QUERY {i}: {q}")
    print("=" * 60)
    res = retriever.retrieve(q, debug=True)
    dbg = res.get("debug", {})
    print("\n--- DENSE CANDIDATES (Top 3) ---")
    for dc in dbg.get("dense_candidates", [])[:3]:
        p = dc.get("payload", {})
        print(f"  [Rank {dc.get('dense_rank')}] score={dc.get('dense_score'):.4f} | {p.get('product_name')} | {p.get('title')}")
    print("\n--- SPARSE CANDIDATES (Top 3) ---")
    for sc in dbg.get("sparse_candidates", [])[:3]:
        p = sc.get("payload", {})
        print(f"  [Rank {sc.get('sparse_rank')}] score={sc.get('sparse_score'):.4f} | {p.get('product_name')} | {p.get('title')}")
    print("\n--- RRF & FINAL SELECTED CHUNKS ---")
    for item in res["results"]:
        p = item.get("payload", {})
        print(f"  * Product: {p.get('product_name')}")
        print(f"    File: {p.get('source_document')}.docx | Section: {p.get('section_heading')}")
        print(f"    Dense rank: {item.get('dense_rank')} | Sparse rank: {item.get('sparse_rank')} | RRF Score: {item.get('rrf_score'):.4f}")
        print(f"    Content snippet: {p.get('content')[:120]}...")
        print()
