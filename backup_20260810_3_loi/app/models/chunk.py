"""Chunk model definition."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

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
    extraction_method: str = "verbatim"
    verification_status: str = "pending_review"
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
