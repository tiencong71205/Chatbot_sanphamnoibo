"""Test hybrid retrieval from the command line."""

import argparse
import time
from typing import Any

from app.config import settings
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_QUERIES = [
    "Cách reset thiết bị về cài đặt gốc?",
    "Cho tôi tất cả thông số của cảm biến hiện diện",
    "Công tắc dimmer có công suất tối đa bao nhiêu?",
    "Cách kết nối công tắc dimmer bằng WiFi tự động?",
    "Cảm biến cửa Mesh dùng giao thức gì?",
    "Bộ điều khiển hồng ngoại có chức năng gì?",
    "Động cơ rèm có những tính năng nào?",
    "Công tơ có những thông số kỹ thuật nào?",
    "Công tắc chống giật BNN có chức năng gì?",
    "Công tắc cửa cuốn có dùng được khi mất Internet không?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra Hybrid Retrieval")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def extract_results(result: Any) -> list[Any]:
    if isinstance(result, list):
        return result

    if isinstance(result, dict):
        for key in ("results", "final_results", "documents"):
            value = result.get(key)
            if isinstance(value, list):
                return value

    return []


def get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


def main() -> None:
    args = parse_args()
    queries = [args.query] if args.query else DEFAULT_QUERIES

    qdrant = QdrantStore(settings)
    bm25 = BM25Store(settings)
    embedding = OllamaEmbedding(settings)

    retriever = HybridRetriever(
        settings=settings,
        qdrant=qdrant,
        bm25=bm25,
        embedding=embedding,
    )

    label = "query" if len(queries) == 1 else "queries"
    print(f"Running {len(queries)} retrieval {label}...")
    print("=" * 60)

    for query_index, query in enumerate(queries, start=1):
        print(f"\n[Q{query_index}] {query}")

        started_at = time.perf_counter()

        result = retriever.retrieve(
            query=query,
            top_k=args.top_k,
            debug=args.debug,
        )

        latency_ms = (time.perf_counter() - started_at) * 1000
        results = extract_results(result)

        print(
            f"     Found {len(results)} results "
            f"(latency={latency_ms:.0f}ms)"
        )

        for index, item in enumerate(results, start=1):
            payload = (
                get_value(item, "payload", {})
                or get_value(item, "metadata", {})
                or {}
            )

            if not isinstance(payload, dict):
                payload = {}

            content = (
                get_value(item, "content")
                or payload.get("content")
                or get_value(item, "text")
                or ""
            )

            print(f"\n  [{index}]")
            print(
                "      chunk_id:",
                get_value(item, "chunk_id")
                or payload.get("chunk_id")
                or get_value(item, "id")
                or "unknown",
            )
            print(
                "      product:",
                payload.get("product_name")
                or get_value(item, "product_name")
                or "unknown",
            )
            print(
                "      content_type:",
                payload.get("content_type")
                or get_value(item, "content_type")
                or "unknown",
            )
            print(
                "      source_file:",
                payload.get("source_file")
                or get_value(item, "source_file")
                or "unknown",
            )
            print("      dense_rank:", get_value(item, "dense_rank"))
            print("      sparse_rank:", get_value(item, "sparse_rank"))
            print("      rrf_score:", get_value(item, "rrf_score"))
            print("      content:", str(content)[:600])


if __name__ == "__main__":
    main()
