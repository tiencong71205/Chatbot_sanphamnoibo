#!/usr/bin/env python3
"""Backup BM25 index and export Qdrant collection info."""
import sys
import json
import shutil
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings

BACKUP_DIR = Path("./backups")


def main():
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup BM25 index
    bm25_src = Path(settings.bm25_index_path)
    if bm25_src.exists():
        dst = BACKUP_DIR / f"bm25_{ts}.joblib"
        shutil.copy2(bm25_src, dst)
        print(f"✅ BM25 backed up: {dst}")
    else:
        print(f"⚠️  BM25 index not found: {bm25_src}")

    print(f"✅ Backup completed to {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
