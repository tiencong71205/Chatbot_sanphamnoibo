"""Chunk model definition."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str = ""
    product_id: str = ""
    product_name: str = ""
    product_group: str = ""
    model: str = ""
    hardware_version: str = ""
    firmware_version: str = ""
    app_version: str = ""
    content_type: str = "feature"
    section_number: str = ""
    section_heading: str = ""
    title: str = ""
    content: str = ""
    conditions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    source_file: str = ""
    source_document: str = ""
    source_page: Optional[int] = None
    source_section: str = ""
    source_heading: str = ""
    source_location: str = ""
    document_version: str = ""
    effective_date: str = ""
    heading_path: str = ""
    heading_level: int = 0
    chunk_type: str = "section"
    section_name: str = ""
    feature_name: str = ""
    source_locator: str = ""
    content_hash: str = ""
    extraction_method: str = "verbatim"
    verification_status: str = "pending_review"
    token_count: int = 0
    # Relative file paths (under the ingest pipeline's images directory) for
    # any pictures embedded in this section of the source DOCX -- e.g. an
    # installation diagram or app-screenshot next to a "Hướng dẫn lắp đặt"
    # procedure. Populated by SemanticChunker; only paths are stored here
    # (never raw bytes), matching how every other file-backed reference in
    # this model works (source_file, source_document).
    image_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
