"""Document metadata extractor reading from product_catalog.json and '0. Metadata sản phẩm' table."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

CATALOG_PATH = Path("/app/data/product_catalog.json")

class DocumentMetadataExtractor:
    def __init__(self, catalog_path: Optional[Path] = None):
        path = catalog_path or CATALOG_PATH
        self.catalog_by_filename: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    sf = item.get("source_file")
                    if sf:
                        self.catalog_by_filename[sf] = item

    def get_metadata_for_file(self, filename: str, elements: Optional[list] = None) -> Dict[str, Any]:
        basename = os.path.basename(filename)
        catalog_entry = self.catalog_by_filename.get(basename, {})

        product_id = catalog_entry.get("product_id", "")
        product_name = catalog_entry.get("product_name", "Thiết bị Vhomenex")
        product_group = catalog_entry.get("product_group", "Thiết bị thông minh")
        model = catalog_entry.get("model", "Chưa xác định - cần xác nhận")

        if elements:
            for elem in elements:
                if elem.kind == "table" and hasattr(elem, "raw_rows") and elem.raw_rows:
                    for row in elem.raw_rows:
                        if len(row) >= 2:
                            k, v = row[0].strip().lower(), row[1].strip()
                            if "tên sản phẩm" in k and v and not catalog_entry:
                                product_name = v
                            elif "nhóm sản phẩm" in k and v and not catalog_entry:
                                product_group = v
                            elif "model" in k and v and not catalog_entry:
                                model = v

        return {
            "product_id": product_id,
            "product_name": product_name,
            "product_group": product_group,
            "model": model,
            "source_file": basename,
            "source_document": basename,
        }
