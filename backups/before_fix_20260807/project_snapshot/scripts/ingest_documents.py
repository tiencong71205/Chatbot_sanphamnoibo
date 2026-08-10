#!/usr/bin/env python3
"""Ingest DOCX documents into Qdrant and BM25 with detailed output."""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings
from app.logging_config import setup_logging
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.ingestion.ingest_service import IngestService

setup_logging(settings.log_level)


def main():
    parser = argparse.ArgumentParser(description="Ingest DOCX documents")
    parser.add_argument("--data-dir", default=settings.data_directory)
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no writes)")
    args = parser.parse_args()

    qdrant = QdrantStore(settings)
    bm25 = BM25Store(settings)
    embedding = OllamaEmbedding(settings)
    svc = IngestService(settings, qdrant, bm25, embedding)

    data_dir = Path(args.data_dir)
    print(f"=== INGESTION START ===")
    print(f"Data directory: {data_dir}")
    print(f"Recreate collection: {args.recreate}")
    print(f"Dry run mode: {args.dry_run}")
    print("-" * 50)

    result = svc.ingest_all(
        data_dir=data_dir,
        recreate_collection=args.recreate,
        dry_run=args.dry_run,
    )

    print("-" * 50)
    print("=== INGESTION RESULTS ===")
    print(f"Files processed: {result['files']}")
    print(f"Chunks generated: {result['chunks']}")
    print(f"Chunks skipped: {result['skipped']}")
    if result.get("doc_details"):
        print("\nDocument Breakdown:")
        for doc in result["doc_details"]:
            print(f"  - {doc['file']}: {doc['elements_count']} elements -> {doc['chunks_count']} chunks")

    if result.get("preview_json"):
        print(f"\nPreview JSON path: {result['preview_json']}")

    if result["errors"]:
        print(f"\n⚠️ Total Errors: {len(result['errors'])}")
        for err in result["errors"]:
            print(f"  - {err}")
    else:
        print("\n✅ Status: SUCCESS (0 errors)")

    print("=========================")


if __name__ == "__main__":
    main()
