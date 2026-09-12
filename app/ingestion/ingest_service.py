"""Ingestion orchestration service with detailed batch logging, dry-run preview,
per-file validation, non-blocking data-quality warnings, and incremental
(content-hash-diffed) embedding so unchanged chunks are never re-embedded.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.database.bm25_store import BM25Store
from app.database.qdrant_store import QdrantStore
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.ingestion.docx_loader import load_docx
from app.ingestion.document_metadata import update_catalog_entry
from app.ingestion.semantic_chunker import Chunk, SemanticChunker, build_embedding_text
from app.ingestion.validation import validate_chunks, warn_chunks

logger = logging.getLogger(__name__)


def _infer_product(filename: str) -> tuple[str, str, str]:
    """Infer product_id, name, group from filename (fallback only; the
    catalog / in-document table, handled by DocumentMetadataExtractor, is
    always preferred — see app/ingestion/document_metadata.py)."""
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
            # Sibling of data_directory, matching the existing
            # data/raw|processed convention used elsewhere in this service.
            images_dir=Path(settings.data_directory).parent / "images",
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

        all_errors: list[str] = []
        all_warnings: list[str] = []
        # All validated chunks across all files this run — BM25 always needs
        # the *complete* corpus (it isn't an incremental index structure),
        # even for chunks whose embedding was skipped because unchanged.
        all_chunks_for_bm25: list[Chunk] = []
        # Only chunks that are actually new/changed and need an embedding call.
        chunks_to_embed: list[Chunk] = []
        stale_chunk_ids_to_delete: list[str] = []
        skipped_validation_count = 0
        skipped_unchanged_count = 0
        doc_details = []

        for doc_idx, docx_path in enumerate(docx_files, 1):
            logger.info("Document %d/%d: %s", doc_idx, len(docx_files), docx_path.name)
            elements = load_docx(docx_path)
            # Catalog metadata is read (and, for brand-new products, auto-registered
            # into product_catalog.json) by the chunker via DocumentMetadataExtractor.
            chunks = self.chunker.chunk(
                elements,
                source_file=str(docx_path),
            )
            logger.info("Chunks generated for %s: %d", docx_path.name, len(chunks))

            # --- Tier 1: blocking validation. A file that fails is skipped
            # entirely so bad data never reaches Qdrant/BM25 — fix the source
            # .docx and re-run ingest, no separate validation command needed.
            file_errors = validate_chunks(chunks, filename=docx_path.name)
            if file_errors:
                all_errors.extend(file_errors)
                skipped_validation_count += len(chunks)
                doc_details.append({
                    "file": docx_path.name,
                    "elements_count": len(elements),
                    "chunks_count": len(chunks),
                    "status": "skipped_validation_failed",
                    "validation_errors": file_errors,
                    "warnings": [],
                    "new_chunks": 0,
                    "unchanged_chunks": 0,
                    "deleted_stale_chunks": 0,
                })
                logger.error(
                    "Skipping %s: %d validation error(s): %s",
                    docx_path.name, len(file_errors), file_errors,
                )
                continue

            # --- Tier 2: non-blocking data-quality warnings (advisory only).
            file_warnings = warn_chunks(chunks, filename=docx_path.name)
            if getattr(self.chunker, "last_doc_meta", {}).get("name_extraction_failed"):
                file_warnings.append(
                    f"[{docx_path.name}] KHÔNG xác định được tên sản phẩm từ cả bảng thông số "
                    f"lẫn trang bìa tài liệu — đang dùng tên suy ra từ tên file "
                    f"('{chunks[0].product_name if chunks else '?'}'). Kiểm tra lại cấu trúc tài "
                    "liệu (dòng 'Tên sản phẩm' trong bảng B.2, và dòng tiêu đề in hoa ngay dưới "
                    "'TÀI LIỆU MÔ TẢ SẢN PHẨM' ở bìa). Sửa xong nên xoá entry auto_generated tương "
                    "ứng trong product_catalog.json và ingest lại để tên đúng được ghi nhận."
                )
            all_warnings.extend(file_warnings)

            all_chunks_for_bm25.extend(chunks)

            # --- Incremental diff against what's already stored for this file.
            # chunk_id is a content hash (filename + heading_path + title +
            # content), so an unchanged section produces the exact same id run
            # after run. We only need to embed ids Qdrant doesn't have yet, and
            # delete ids that existed before but are no longer produced (e.g.
            # the author edited a section, which changes its hash and would
            # otherwise leave the old version as an orphan point forever).
            new_ids = {c.chunk_id for c in chunks}
            existing_meta = (
                {}
                if recreate_collection and not dry_run
                else self.qdrant.get_existing_chunk_metadata(docx_path.name)
            )
            existing_ids = set(existing_meta.keys())

            # A chunk is only truly "unchanged" (safe to skip embedding) if
            # BOTH its chunk_id (content hash) AND its product_id match what's
            # already stored. chunk_id deliberately excludes product_id (see
            # _make_chunk_id), so a metadata-only correction — e.g. fixing a
            # catalog collision where two different products briefly shared
            # one product_id — would otherwise leave the old wrong product_id
            # silently stuck in Qdrant forever, since the content itself never
            # changed. Confirmed to happen in practice; re-upsert (fast, no
            # new embedding needed, we reuse content unchanged) whenever the
            # stored product_id disagrees with the freshly computed one.
            unchanged_ids = {
                c.chunk_id for c in chunks
                if c.chunk_id in existing_ids and existing_meta.get(c.chunk_id) == c.product_id
            }
            metadata_only_changed = [
                c for c in chunks
                if c.chunk_id in existing_ids and existing_meta.get(c.chunk_id) != c.product_id
            ]
            to_embed = [c for c in chunks if c.chunk_id not in unchanged_ids]
            stale_ids = existing_ids - new_ids

            if metadata_only_changed:
                logger.warning(
                    "%s: %d chunk(s) have a stale product_id in Qdrant "
                    "(content unchanged but metadata was corrected) — "
                    "re-embedding to refresh payload: %s",
                    docx_path.name, len(metadata_only_changed),
                    sorted({c.product_id for c in metadata_only_changed}),
                )

            skipped_unchanged_count += len(unchanged_ids)
            chunks_to_embed.extend(to_embed)
            stale_chunk_ids_to_delete.extend(stale_ids)

            doc_details.append({
                "file": docx_path.name,
                "elements_count": len(elements),
                "chunks_count": len(chunks),
                "status": "ok",
                "validation_errors": [],
                "warnings": file_warnings,
                "new_chunks": len(to_embed),
                "unchanged_chunks": len(unchanged_ids),
                "deleted_stale_chunks": len(stale_ids),
            })

        if dry_run:
            preview_dir = Path("./data/processed")
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / "preview_chunks.json"
            preview_data = [c.to_dict() for c in all_chunks_for_bm25[:10]]
            with open(preview_path, "w", encoding="utf-8") as f:
                json.dump(preview_data, f, ensure_ascii=False, indent=2)
            logger.info("[DRY RUN] Preview saved to %s", preview_path)
            result = {
                "files": len(docx_files),
                "chunks": len(all_chunks_for_bm25),
                "skipped": skipped_validation_count,
                "would_embed": len(chunks_to_embed),
                "would_skip_unchanged": skipped_unchanged_count,
                "would_delete_stale": len(stale_chunk_ids_to_delete),
                "errors": all_errors,
                "warnings": all_warnings,
                "dry_run": True,
                "preview_json": str(preview_path),
                "doc_details": doc_details,
            }
            self._write_report(result)
            return result

        # Post-batch cross-product alias safety sweep. Confirmed real bug:
        # docx_files above is a plain lexical sort ("12_..." sorts before
        # "7_..."), so a product processed early in a multi-file batch
        # computes its distinguishing-word aliases (see
        # build_auto_alias_set) by comparing against
        # meta_extractor.catalog_by_filename's state AT THAT MOMENT --
        # which doesn't yet include products later in this SAME batch,
        # even though the comparison is intended to cover the whole
        # catalog. Confirmed case: "Bộ trung tâm Gateway khóa" (file
        # "12_...") got a standalone alias "khóa" because "Khóa điện tử"
        # (file "7_...") hadn't been registered yet when 12's aliases
        # were computed -- even though both exist in the same finished
        # catalog and "khóa" appears in both products' own names. That
        # false-confident single-word alias then caused a completely
        # unrelated query (just containing the generic word "khóa") to
        # resolve to the wrong product with 100% confidence via exact
        # match.
        #
        # Re-check every product's short (<10 char), single-word
        # auto-generated aliases now that the full batch (and any
        # pre-existing catalog entries) is complete, and drop any that
        # also appear as a standalone word in some OTHER product's own
        # name -- genuinely ambiguous, shared vocabulary, unsafe as a
        # confident exact-match identifier for just one of them. Hand-
        # curated (non-auto_generated) aliases are never touched here.
        catalog_by_filename = self.chunker.meta_extractor.catalog_by_filename
        all_product_names_final = [
            (item.get("product_name") or "").lower()
            for item in catalog_by_filename.values()
        ]
        for basename, entry in catalog_by_filename.items():
            if not entry.get("auto_generated"):
                continue
            own_name = (entry.get("product_name") or "").lower()
            aliases = entry.get("aliases", [])
            pruned = []
            changed = False
            for alias in aliases:
                is_short_bare_word = " " not in alias and len(alias) < 10
                shared_elsewhere = is_short_bare_word and any(
                    other_name != own_name and alias in other_name.split()
                    for other_name in all_product_names_final
                )
                if shared_elsewhere:
                    changed = True
                    continue
                pruned.append(alias)
            if changed:
                entry["aliases"] = pruned
                source_file = entry.get("source_file", basename)
                update_catalog_entry(source_file, {"aliases": pruned})
                logger.warning(
                    "Pruned ambiguous auto-generated alias(es) from '%s' "
                    "after full-batch visibility -- aliases now: %s",
                    entry.get("product_name", basename), pruned,
                )

        # Delete points for sections whose content changed (old hash orphans)
        # or that were removed from the source document entirely.
        if stale_chunk_ids_to_delete:
            logger.info("Deleting %d stale/changed points...", len(stale_chunk_ids_to_delete))
            self.qdrant.delete_by_chunk_ids(stale_chunk_ids_to_delete)

        # Embed + upsert only new/changed chunks, in batches.
        valid_chunks: list[Chunk] = []
        batch_size = 8
        total_to_embed = len(chunks_to_embed)
        total_batches = (total_to_embed + batch_size - 1) // batch_size

        for b_idx in range(total_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, total_to_embed)
            batch_chunks = chunks_to_embed[start:end]
            batch_valid_chunks: list[Chunk] = []
            batch_valid_vectors: list[list[float]] = []
            logger.info("Embedding batch %d/%d (%d chunks)...", b_idx + 1, total_batches, len(batch_chunks))

            texts = [build_embedding_text(c) for c in batch_chunks]
            for chunk, text in zip(batch_chunks, texts):
                try:
                    vec = self.embedding.embed_single(text)
                    valid_chunks.append(chunk)
                    batch_valid_chunks.append(chunk)
                    batch_valid_vectors.append(vec)
                except Exception as e:
                    msg = f"Failed to embed chunk {chunk.chunk_id} from {chunk.source_file}: {e}"
                    logger.error(msg, exc_info=True)
                    all_errors.append(msg)

            logger.info("Embedding batch %d/%d completed", b_idx + 1, total_batches)

            if batch_valid_chunks:
                logger.info("Upserting points (%d chunks)...", len(batch_valid_chunks))
                self.qdrant.upsert_chunks(batch_valid_chunks, batch_valid_vectors)
                logger.info("Points inserted: %d total so far", len(valid_chunks))

        if total_to_embed == 0:
            logger.info("No new/changed chunks to embed — everything already up to date.")

        # BM25 always needs the full corpus (unchanged + newly embedded chunks
        # that passed validation this run) since it isn't incrementally updatable.
        if all_chunks_for_bm25:
            logger.info("Building BM25 index (%d chunks total)...", len(all_chunks_for_bm25))
            self.bm25.build_index(all_chunks_for_bm25)
            self.bm25.save_index()
            logger.info("BM25 index saved to %s", self.settings.bm25_index_path)

        result = {
            "files": len(docx_files),
            "chunks": len(all_chunks_for_bm25),
            "embedded": len(valid_chunks),
            "skipped_unchanged": skipped_unchanged_count,
            "skipped_validation": skipped_validation_count,
            "deleted_stale": len(stale_chunk_ids_to_delete),
            "errors": all_errors,
            "warnings": all_warnings,
            "dry_run": False,
            "doc_details": doc_details,
        }
        self._write_report(result)
        return result

    @staticmethod
    def _write_report(result: dict) -> None:
        """Persist the last ingest run (errors + warnings + per-file stats) to
        disk so data-quality issues are visible any time, not just in console
        output that scrolls away. Also exposed via GET /api/ingest/report.
        """
        report_dir = Path("./data/processed")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "last_ingest_report.json"
        report = {**result, "generated_at": datetime.now(timezone.utc).isoformat()}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Ingest report written to %s", report_path)
