"""Heading-aware semantic chunking for Vhomenex product documents."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from app.ingestion.content_classifier import ContentTypeClassifier
from app.ingestion.document_metadata import DocumentMetadataExtractor
from app.ingestion.section_parser import SectionNode, SectionParser
from app.ingestion.table_normalizer import TableNormalizer
from app.models.chunk import Chunk

INTERNAL_SECTION_PATTERNS = [
    r"metadata sản phẩm", r"quy tắc tạo chunk", r"mẫu metadata chunk",
    r"hướng dẫn biên soạn", r"hướng dẫn chuẩn hóa", r"quy ước dữ liệu",
    r"phụ lục dành cho người soạn", r"nội dung nội bộ", r"do_not_ingest",
]
PLACEHOLDER_RE = re.compile(
    r"^(?:\[.*?\]|<.*?>|todo|tbd|n/?a|\.|mặt trước:|mặt sau:|cạnh bên:)$",
    re.IGNORECASE,
)
NAVIGATION_LABEL_RE = re.compile(r"^-?\s*bảng danh sách (?:tính năng|chức năng)\s*:?$", re.IGNORECASE)
STEP_RE = re.compile(r"^B\d+\s*:", re.IGNORECASE | re.MULTILINE)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


_IMAGE_EXT_BY_CONTENT_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}


def _make_chunk_id(source_file: str, heading_path: str, title: str, content: str) -> str:
    """Stable ID: unchanged content keeps the same Qdrant point identifier."""
    identity = "\n".join((os.path.basename(source_file), heading_path, title, content.strip()))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def build_embedding_text(chunk: Chunk) -> str:
    parts = []
    if chunk.product_name:
        parts.append(f"Sản phẩm: {chunk.product_name}")
    if chunk.heading_path:
        parts.append(f"Vị trí: {chunk.heading_path}")
    if chunk.title:
        parts.append(f"Mục: {chunk.title}")
    parts.append(chunk.content)
    return "\n".join(part for part in parts if part)


class SemanticChunker:
    def __init__(
        self,
        target_tokens: int = 400,
        max_tokens: int = 750,
        overlap_tokens: int = 50,
        images_dir: Optional[Path] = None,
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.parser = SectionParser()
        self.meta_extractor = DocumentMetadataExtractor()
        # Where extracted DOCX images get saved to disk -- only their
        # resulting relative path is ever stored on a Chunk (see
        # models/chunk.py's image_paths field docstring); raw bytes never
        # enter the chunk data that flows into Qdrant's payload.
        self.images_dir = Path(images_dir) if images_dir else Path("./data/images")

    @staticmethod
    def _is_internal_section(title: str) -> bool:
        lowered = title.lower()
        return any(re.search(pattern, lowered) for pattern in INTERNAL_SECTION_PATTERNS)

    @staticmethod
    def _clean_line(text: str) -> str:
        # "a) Mô tả: Không có" / "b) Hướng dẫn: Không áp dụng" used to be
        # dropped here as if they were empty placeholders. Confirmed real
        # bug: these are meaningful negative facts (this feature/field
        # genuinely doesn't apply to this product), not unfilled template
        # noise — dropping them permanently loses the chatbot's ability to
        # answer a direct "does this apply?" question with a confident,
        # sourced "no" instead of a vague "no data in the document" (which
        # reads as "maybe it exists but wasn't captured", a materially
        # weaker and less trustworthy answer). See table_normalizer.py for
        # the matching table-row-level fix — both were dropping the exact
        # same category of real information.
        value = re.sub(r"\s+", " ", text or "").strip()
        if not value or PLACEHOLDER_RE.match(value):
            return ""
        if NAVIGATION_LABEL_RE.match(value):
            return ""
        # A template typo can append an empty second label to the description.
        value = re.sub(r"\s*b\)\s*Hướng dẫn\s*:\s*$", "", value, flags=re.IGNORECASE).strip()
        return value

    @staticmethod
    def _section_name(node: SectionNode) -> str:
        current = node
        while current and current.level > 2:
            current = current.parent
        return current.title if current and current.level == 2 else node.title

    def _save_image(self, image_bytes: bytes, content_type: str) -> str:
        """Persist one extracted DOCX image to disk, deduping by content
        hash so the same picture reused in multiple places (or re-ingested
        unchanged on a later run) is only ever written once. Returns the
        filename actually stored (relative to self.images_dir), which is
        what a Chunk's image_paths field holds -- never raw bytes.
        """
        digest = hashlib.sha256(image_bytes).hexdigest()[:24]
        ext = _IMAGE_EXT_BY_CONTENT_TYPE.get((content_type or "").lower(), ".png")
        filename = f"{digest}{ext}"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.images_dir / filename
        if not filepath.exists():
            filepath.write_bytes(image_bytes)
        return filename

    @staticmethod
    def _infer_chunk_type(node: SectionNode, content: str, content_type: str) -> str:
        if STEP_RE.search(content) or (node.level >= 3 and content_type == "installation"):
            return "procedure"
        if node.level >= 3:
            return "feature"
        return "section"

    def _split_body(self, paragraphs: List[str]) -> List[str]:
        """Split long prose while keeping a B1..Bn procedure as one atomic unit."""
        if not paragraphs:
            return []
        full = "\n\n".join(paragraphs)
        if len(full.split()) <= self.max_tokens:
            return [full]

        units: List[str] = []
        step_group: List[str] = []
        for paragraph in paragraphs:
            if STEP_RE.match(paragraph):
                step_group.append(paragraph)
                continue
            if step_group:
                units.append("\n".join(step_group))
                step_group = []
            units.append(paragraph)
        if step_group:
            units.append("\n".join(step_group))

        output: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for unit in units:
            unit_tokens = len(unit.split())
            if current and current_tokens + unit_tokens > self.target_tokens:
                output.append("\n\n".join(current))
                current, current_tokens = [], 0
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            output.append("\n\n".join(current))
        return output

    @staticmethod
    def _derive_preamble_title(node: SectionNode, content: str) -> str:
        """Build a distinguishing title for a section's own leading body
        content when that section also has child subsections following it.

        Confirmed real bug: a section like "4. Tính năng trên thiết bị" can
        have its own preamble paragraph (e.g. "Lưu ý: ... cần đưa thiết bị
        về trạng thái sẵn sàng kết nối. B1: Nhấn giữ nút Reset...") BEFORE
        its "4.1 Chế độ kết nối..." child section starts. Both chunks
        previously got titled from the same generic parent heading, and the
        embedding text (which weights "Mục: {title}" heavily) made the
        preamble chunk indistinguishable from — and lose ranking to — its
        own child chunk whose title happens to share more surface keywords
        with common queries ("kết nối"). This recurs across multiple
        product docs sharing the same template pattern (a features overview
        section with a reset/prep preamble before per-feature subsections),
        so the fix is structural, not specific to one document.

        Only applies when this is genuinely a "preamble before subsections"
        case (node.children is non-empty) — a section with no children just
        keeps its own title as before, since there's no sibling/child title
        collision risk to disambiguate against.
        """
        if not node.children:
            return node.title or "Mô tả sản phẩm"

        base_title = node.title or "Mô tả sản phẩm"
        first_line = content.split("\n\n", 1)[0].split("\n", 1)[0].strip()
        first_line = re.sub(r"^\*+|\*+$", "", first_line).strip()
        # A bare step marker ("B1: ...") isn't a useful disambiguator on its
        # own — only use genuinely descriptive lead-in text.
        if not first_line or STEP_RE.match(first_line):
            return base_title
        if len(first_line) > 140:
            first_line = first_line[:137].rstrip() + "..."
        return f"{base_title} — {first_line}"

    def _new_chunk(
        self,
        *,
        node: SectionNode,
        doc_meta: Dict[str, object],
        filename: str,
        title: str,
        content: str,
        content_type: str,
        chunk_type: str,
        image_paths: Optional[List[str]] = None,
    ) -> Chunk:
        heading_path = node.full_path()
        locator_parts = [str(doc_meta.get("product_name", "")), heading_path]
        source_locator = " → ".join(part for part in locator_parts if part)
        digest = _content_hash(content)
        feature_name = node.title if node.level >= 3 else ""
        return Chunk(
            chunk_id=_make_chunk_id(filename, heading_path, title, content),
            product_id=str(doc_meta.get("product_id", "")),
            product_name=str(doc_meta.get("product_name", "")),
            product_group=str(doc_meta.get("product_group", "")),
            model=str(doc_meta.get("model", "")),
            content_type=content_type,
            section_number=node.number,
            section_heading=node.title,
            title=title,
            content=content,
            source_file=filename,
            source_document=filename,
            source_section=node.title,
            source_heading=heading_path,
            source_location=source_locator,
            document_version=str(doc_meta.get("document_version", "")),
            effective_date=str(doc_meta.get("effective_date", "")),
            heading_path=heading_path,
            heading_level=node.level,
            chunk_type=chunk_type,
            section_name=self._section_name(node),
            feature_name=feature_name,
            source_locator=source_locator,
            content_hash=digest,
            token_count=len(content.split()),
            image_paths=list(image_paths) if image_paths else [],
        )

    def chunk(
        self,
        elements: list,
        source_file: str = "",
        product_id: str = "",
        product_name: str = "",
        product_group: str = "",
        model: str = "",
    ) -> List[Chunk]:
        filename = os.path.basename(source_file)
        doc_meta = self.meta_extractor.get_metadata_for_file(filename, elements)
        for key, value in {
            "product_id": product_id,
            "product_name": product_name,
            "product_group": product_group,
            "model": model,
        }.items():
            if value:
                doc_meta[key] = value
        # Exposed so ingest_service can surface a warning when the product
        # name couldn't be read from the document's own table (see
        # document_metadata.py's "name_extraction_failed") without having to
        # thread a new field through every Chunk instance.
        self.last_doc_meta = doc_meta

        root = self.parser.parse(elements)
        chunks: List[Chunk] = []

        def traverse(node: SectionNode) -> None:
            if self._is_internal_section(node.title):
                return

            paragraphs: List[str] = []
            tables = []
            image_paths: List[str] = []
            for element in node.elements:
                if getattr(element, "is_front_matter", False) or element.kind == "metadata":
                    continue
                if element.kind == "table":
                    tables.append(element)
                elif element.kind == "image":
                    if getattr(element, "image_bytes", None):
                        image_paths.append(
                            self._save_image(element.image_bytes, element.image_content_type)
                        )
                elif element.kind in {"paragraph", "bullet", "numbered", "caption"}:
                    cleaned = self._clean_line(element.text)
                    if cleaned:
                        paragraphs.append(cleaned)

            body = "\n\n".join(paragraphs)
            content_type = ContentTypeClassifier.classify(
                node.title or "Mô tả sản phẩm",
                body,
                node.parent.title if node.parent else "",
            )
            chunk_type = self._infer_chunk_type(node, body, content_type)
            body_parts = self._split_body(paragraphs)
            if not body_parts and image_paths:
                # A section made up of ONLY an image (no body text at all --
                # e.g. a standalone product photo under its own heading)
                # would otherwise create zero chunks here, silently
                # orphaning the image file already saved to disk above:
                # nothing in Qdrant would ever reference it, so it could
                # never be retrieved. One minimal chunk keeps it
                # discoverable via the section's own title.
                body_parts = [""]
            for part_index, part in enumerate(body_parts, start=1):
                if chunk_type == "procedure":
                    title = self._derive_preamble_title(node, part)
                else:
                    title = node.title or "Mô tả sản phẩm"
                if len(body_parts) > 1:
                    title = f"{title} (phần {part_index})"
                chunks.append(self._new_chunk(
                    node=node,
                    doc_meta=doc_meta,
                    filename=filename,
                    title=title,
                    content=part,
                    content_type=content_type,
                    chunk_type=chunk_type,
                    # Attached to every body-text chunk of this section (not
                    # split further per-part) -- a section's images belong
                    # to the section as a whole, not to one specific
                    # paragraph-length slice of a long body.
                    image_paths=image_paths,
                ))

            for table in tables:
                for record in TableNormalizer.records(table.raw_rows, node.title):
                    record_type = record["chunk_type"]
                    record_content_type = (
                        "troubleshooting" if record_type == "faq"
                        else "specification" if record_type == "spec"
                        else content_type
                    )
                    chunks.append(self._new_chunk(
                        node=node,
                        doc_meta=doc_meta,
                        filename=filename,
                        title=record["title"],
                        content=record["content"],
                        content_type=record_content_type,
                        chunk_type=record_type,
                    ))

            for child in node.children:
                traverse(child)

        traverse(root)
        return chunks
