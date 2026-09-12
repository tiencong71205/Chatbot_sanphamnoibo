#!/usr/bin/env python3
"""Validate a template-based Vhomenex DOCX without Qdrant or Ollama."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.docx_loader import load_docx  # noqa: E402
from app.ingestion.semantic_chunker import SemanticChunker  # noqa: E402
from app.ingestion.validation import validate_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.docx.is_file():
        parser.error(f"Không tìm thấy file: {args.docx}")

    elements = load_docx(args.docx)
    chunks = SemanticChunker().chunk(elements, str(args.docx))
    errors = validate_chunks(chunks, filename=args.docx.name)

    print(f"File: {args.docx.name}")
    print(f"Sản phẩm: {chunks[0].product_name if chunks else 'Không xác định'}")
    print(f"Phiên bản: {chunks[0].document_version if chunks else ''}")
    print(f"Tổng chunk: {len(chunks)}")
    print("Theo chunk_type:", dict(Counter(chunk.chunk_type for chunk in chunks)))
    print("Theo content_type:", dict(Counter(chunk.content_type for chunk in chunks)))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã ghi preview: {args.json_out}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
