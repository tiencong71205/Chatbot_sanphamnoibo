import logging
from pathlib import Path
from app.config import get_settings
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store
from app.models.chunk import Chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    qdrant = QdrantStore(settings)
    bm25 = BM25Store(settings)

    logger.info("Fetching all chunks from Qdrant...")
    records = qdrant.scroll_all()
    logger.info("Retrieved %d points from Qdrant", len(records))

    chunks = []
    for r in records:
        payload = r.payload or {}
        if payload:
            c = Chunk(
                chunk_id=payload.get("chunk_id", ""),
                product_id=payload.get("product_id", ""),
                product_name=payload.get("product_name", ""),
                product_group=payload.get("product_group", ""),
                model=payload.get("model", ""),
                content_type=payload.get("content_type", "feature"),
                section_heading=payload.get("section_heading", ""),
                title=payload.get("title", ""),
                content=payload.get("content", ""),
                source_file=payload.get("source_file", ""),
                source_document=payload.get("source_document", ""),
                source_section=payload.get("source_section", ""),
            )
            chunks.append(c)

    logger.info("Parsed %d valid chunks. Building BM25 index...", len(chunks))
    bm25.build_index(chunks)
    bm25.save_index()
    logger.info("BM25 index successfully saved to %s", settings.bm25_index_path)

if __name__ == "__main__":
    main()
