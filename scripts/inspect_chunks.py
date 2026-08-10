#!/usr/bin/env python3
"""Inspect chunks in Qdrant collection."""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings
from app.database.qdrant_store import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--product", default="")
    parser.add_argument("--content-type", default="")
    args = parser.parse_args()

    qdrant = QdrantStore(settings)
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Total chunks: {qdrant.count()}")
    print()

    payloads = qdrant.get_all_payloads(limit=args.limit * 10)
    filtered = payloads
    if args.product:
        filtered = [p for p in filtered if args.product.lower() in p.get("product_name", "").lower()]
    if args.content_type:
        filtered = [p for p in filtered if p.get("content_type") == args.content_type]

    for i, p in enumerate(filtered[: args.limit]):
        print(f"[{i+1}] {p.get('product_name', '?')} | {p.get('content_type', '?')} | {p.get('title', '?'[:60])}")
        print(f"     {p.get('content', '')[:200]}")
        print()


if __name__ == "__main__":
    main()
