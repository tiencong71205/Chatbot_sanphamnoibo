"""Document metadata extractor reading from product_catalog.json and '0. Metadata sản phẩm' table."""
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try multiple catalog paths (Docker vs local)
_CATALOG_PATHS = [
    Path(os.environ.get("PRODUCT_CATALOG_PATH", "/app/data/product_catalog.json")),
    Path("/app/data/product_catalog.json"),
    Path("./data/product_catalog.json"),
    Path("data/product_catalog.json"),
]


def _load_catalog() -> Dict[str, Dict[str, Any]]:
    for p in _CATALOG_PATHS:
        if p.exists():
            logger.info("Loading product catalog from: %s", p)
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = {}
            for item in data:
                sf = item.get("source_file")
                if sf:
                    result[sf] = item
            return result
    logger.warning("Product catalog not found at any expected path: %s", _CATALOG_PATHS)
    return {}


class DocumentMetadataExtractor:
    def __init__(self, catalog_path: Optional[Path] = None):
        if catalog_path and catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.catalog_by_filename: Dict[str, Dict[str, Any]] = {
                item["source_file"]: item for item in data if "source_file" in item
            }
        else:
            self.catalog_by_filename = _load_catalog()

    def get_metadata_for_file(self, filename: str, elements: Optional[list] = None) -> Dict[str, Any]:
        basename = os.path.basename(filename)
        catalog_entry = self.catalog_by_filename.get(basename, {})

        product_id = catalog_entry.get("product_id", "")
        product_name = catalog_entry.get("product_name", "Thiết bị Vhomenex")
        product_group = catalog_entry.get("product_group", "Thiết bị thông minh")
        model = catalog_entry.get("model", "Chưa xác định - cần xác nhận")
        aliases = catalog_entry.get("aliases", [])
        keywords = catalog_entry.get("keywords", [])

        # If no catalog entry found, try to infer from document content
        if not catalog_entry and elements:
            for elem in elements:
                if elem.kind == "table" and hasattr(elem, "raw_rows") and elem.raw_rows:
                    for row in elem.raw_rows:
                        if len(row) >= 2:
                            k, v = row[0].strip().lower(), row[1].strip()
                            if "tên sản phẩm" in k and v:
                                product_name = v
                            elif "nhóm sản phẩm" in k and v:
                                product_group = v
                            elif "model" in k and v:
                                model = v
                            elif "mã sản phẩm" in k and v:
                                product_id = v

        if not product_id:
            # Generate product_id from filename prefix
            stem = Path(basename).stem
            # Remove RAG version suffix
            import re
            stem = re.sub(r'_RAG_v\d+.*$', '', stem, flags=re.IGNORECASE)
            # Remove number prefix
            stem = re.sub(r'^\d+_', '', stem)
            product_id = stem.lower().replace(" ", "_")

        return {
            "product_id": product_id,
            "product_name": product_name,
            "product_group": product_group,
            "model": model,
            "aliases": aliases,
            "keywords": keywords,
            "source_file": basename,
            "source_document": basename,
        }
