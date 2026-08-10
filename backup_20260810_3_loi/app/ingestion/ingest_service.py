"""Ingestion orchestration service with detailed batch logging and dry-run preview."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

from app.config import Settings
from app.ingestion.docx_loader import load_docx
from app.ingestion.semantic_chunker import SemanticChunker, Chunk, build_embedding_text
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.database.qdrant_store import QdrantStore
from app.database.bm25_store import BM25Store

logger = logging.getLogger(__name__)


def _infer_product(filename: str) -> tuple[str, str, str]:
    """Infer product_id, name, group from filename."""
    stem = Path(filename).stem
    pretty = stem.replace("_", " ").replace("-", " ").title()
    pid = stem.lower().replace(" ", "_")
    return pid, pretty, "Vhomenex Devices"


class IngestService:
    def __init__(
        self,
        settings: Settings,
        qdrant: QdrantStore,
        bm25: BM25Store,
        embedding: OllamaEmbedding,
    ):
        self.settings = settings
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedding = embedding
        self.chunker = SemanticChunker(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    def ingest_all(
        self,
        data_dir: Optional[Path] = None,
        recreate_collection: bool = False,
        dry_run: bool = False,
    ) -> dict:
        data_dir = data_dir or Path(self.settings.data_directory)
        docx_files = sorted(data_dir.glob("*.docx"))

        if not docx_files:
            logger.warning("No DOCX files found in %s", data_dir)
            return {"files": 0, "chunks": 0, "errors": ["No DOCX files found"], "dry_run": dry_run}

        logger.info("Found %d DOCX files in %s", len(docx_files), data_dir)

        if recreate_collection and not dry_run:
            logger.info("Recreating Qdrant collection: %s", self.settings.qdrant_collection)
            self.qdrant.recreate_collection()

        total_chunks = 0
        all_errors: list[str] = []
        all_chunks: list[Chunk] = []
        skipped_count = 0
        doc_details = []

        for doc_idx, docx_path in enumerate(docx_files, 1):
            logger.info("Document %d/%d: %s", doc_idx, len(docx_files), docx_path.name)
            elements = load_docx(docx_path)
            # Catalog metadata is read by chunker via DocumentMetadataExtractor
            chunks = self.chunker.chunk(
                elements,
                source_file=str(docx_path),
            )
            logger.info("Chunks generated for %s: %d", docx_path.name, len(chunks))
            all_chunks.extend(chunks)
            total_chunks += len(chunks)

            doc_details.append({
                "file": docx_path.name,
                "elements_count": len(elements),
                "chunks_count": len(chunks),
            })

        if dry_run:
            preview_dir = Path("./data/processed")
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / "preview_chunks.json"
            preview_data = [c.to_dict() for c in all_chunks[:10]]
            with open(preview_path, "w", encoding="utf-8") as f:
                json.dump(preview_data, f, ensure_ascii=False, indent=2)
            logger.info("[DRY RUN] Preview saved to %s", preview_path)
            return {
                "files": len(docx_files),
                "chunks": total_chunks,
                "skipped": skipped_count,
                "errors": all_errors,
                "dry_run": True,
                "preview_json": str(preview_path),
                "doc_details": doc_details,
            }

        # Process ingestion in batches
        valid_chunks: list[Chunk] = []
        valid_vectors: list[list[float]] = []
        
        batch_size = 8
        total_chunks_len = len(all_chunks)
        total_batches = (total_chunks_len + batch_size - 1) // batch_size
        
        for b_idx in range(total_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, total_chunks_len)
            batch_chunks = all_chunks[start:end]
            logger.info("Embedding batch %d/%d (%d chunks)...", b_idx + 1, total_batches, len(batch_chunks))
            
            texts = [build_embedding_text(c) for c in batch_chunks]
            for c_idx, (chunk, text) in enumerate(zip(batch_chunks, texts)):
                try:
                    vec = self.embedding.embed_single(text)
                    valid_chunks.append(chunk)
                    valid_vectors.append(vec)
                except Exception as e:
                    msg = f"Failed to embed chunk {chunk.chunk_id} from {chunk.source_file}: {e}"
                    logger.error(msg, exc_info=True)
                    all_errors.append(msg)
            
            logger.info("Embedding batch %d/%d completed", b_idx + 1, total_batches)

            # Per-batch upsert to Qdrant so transaction is not held till end
            if valid_chunks:
                logger.info("Upserting points (%d chunks)...", len(valid_chunks))
                self.qdrant.upsert_chunks(valid_chunks, valid_vectors)
                logger.info("Points inserted: %d total so far", len(valid_chunks))

        # Build BM25 index on valid chunks
        if valid_chunks:
            logger.info("Building BM25 index...")
            self.bm25.build_index(valid_chunks)
            self.bm25.save_index()
            logger.info("BM25 index saved to %s", self.settings.bm25_index_path)

        return {
            "files": len(docx_files),
            "chunks": len(valid_chunks),
            "skipped": skipped_count,
            "errors": all_errors,
            "dry_run": False,
        }