#!/usr/bin/env python3
"""Ingest DOCX documents into Qdrant and BM25 with detailed output."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings
from app.database.bm25_store import BM25Store
from app.database.qdrant_store import QdrantStore
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.ingestion.ingest_service import IngestService
from app.logging_config import setup_logging

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
    print("=== INGESTION START ===")
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
    print(f"Files found: {result['files']}")
    print(f"Total chunks (valid files): {result['chunks']}")
    if result.get("dry_run"):
        print(f"Would embed (new/changed): {result.get('would_embed', 0)}")
        print(f"Would skip (unchanged): {result.get('would_skip_unchanged', 0)}")
        print(f"Would delete (stale): {result.get('would_delete_stale', 0)}")
    else:
        print(f"Embedded (new/changed): {result.get('embedded', 0)}")
        print(f"Skipped (unchanged, saved an embedding call): {result.get('skipped_unchanged', 0)}")
        print(f"Deleted (stale/orphaned points): {result.get('deleted_stale', 0)}")
    print(f"Chunks skipped (validation failed): {result.get('skipped_validation', result.get('skipped', 0))}")

    if result.get("doc_details"):
        print("\nDocument Breakdown:")
        for doc in result["doc_details"]:
            status = doc.get("status", "ok")
            marker = "✅" if status == "ok" else "❌ SKIPPED"
            extra = ""
            if status == "ok":
                extra = (
                    f" (mới/thay đổi: {doc['new_chunks']}, "
                    f"không đổi: {doc['unchanged_chunks']}, "
                    f"xoá cũ: {doc['deleted_stale_chunks']})"
                )
            print(f"  - {marker} {doc['file']}: {doc['elements_count']} elements -> {doc['chunks_count']} chunks{extra}")
            for err in doc.get("validation_errors") or []:
                print(f"      ⛔ ERROR: {err}")
            for warn in doc.get("warnings") or []:
                print(f"      ⚠️  WARN: {warn}")

    if result.get("preview_json"):
        print(f"\nPreview JSON path: {result['preview_json']}")

    if result["errors"]:
        print(f"\n⚠️ Total Errors: {len(result['errors'])}")
        for err in result["errors"]:
            print(f"  - {err}")
    else:
        print("\n✅ Status: SUCCESS (0 lỗi chặn cứng)")

    if result.get("warnings"):
        print(f"\n📋 Total Warnings (không chặn ingest, nên xem lại data): {len(result['warnings'])}")
        for w in result["warnings"]:
            print(f"  - {w}")

    print("\nBáo cáo đầy đủ đã lưu tại: data/processed/last_ingest_report.json")
    print("=========================")


if __name__ == "__main__":
    main()
