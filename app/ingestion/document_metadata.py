"""Document metadata extractor reading from product_catalog.json and '0. Metadata sản phẩm' table."""
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Pre-replace map: đ/Đ → d/D before NFKD normalization.
# unicodedata.normalize("NFKD") cannot decompose đ/Đ (they are standalone
# Unicode codepoints without a canonical decomposition), so without this
# pre-replacement the subsequent ASCII filter drops them entirely, producing
# incorrect slugs such as "bo_ieu_khien" instead of "bo_dieu_khien".
_VIET_D_MAP = str.maketrans("đĐ", "dD")


def _key(value: str) -> str:
    # Step 1: pre-replace đ/Đ so NFKD does not lose them.
    value = (value or "").translate(_VIET_D_MAP)
    # Step 2: decompose combining characters (e.g. ã → a + combining tilde).
    normalized = unicodedata.normalize("NFKD", value)
    # Step 3: drop combining (diacritic) characters, keep base letters.
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    # Step 4: lowercase + collapse non-alphanumeric runs to a single space.
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _slug(value: str) -> str:
    return _key(value).replace(" ", "_")

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


def _writable_catalog_path() -> Optional[Path]:
    """Return the first catalog path whose parent directory is writable."""
    for p in _CATALOG_PATHS:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() or os.access(p.parent, os.W_OK):
                return p
        except OSError:
            continue
    return None


_NAME_STOPWORDS = frozenset({
    "cho", "của", "và", "với", "trong", "các", "những", "là", "để", "theo",
})


def build_auto_alias_set(
    product_name: str,
    sibling_names: Optional[list] = None,
) -> list:
    """Generate a minimal-but-useful alias list for a product with no catalog entry.

    Not a substitute for hand-curated aliases/typos (a human still writes
    better synonyms), but this keeps a brand-new product resolvable by exact
    name and by its accent-free / no-space forms immediately after ingest,
    instead of being completely unmatchable until someone edits the catalog.

    Also splits on structural separators the source docs already use to mark
    an alternate/short name — a dash ("Bộ điều khiển trung tâm - Gateway")
    or parentheses ("Bộ điều khiển trung tâm (Gateway)") — since those are a
    strong, low-risk signal of a colloquial short name real users type
    instead of the full official name (confirmed: "Gateway" alone failed to
    resolve at all without this, since resolve_all only matches substrings/
    aliases and the full official name never appears verbatim in short
    questions).

    Two more real gaps, confirmed via testing against actual user phrasing,
    fixed here:

    1. Small connecting words dropped in casual reference. A catalog name
       like "Công tắc chống giật cho BNN" is very commonly shortened to
       "công tắc chống giật BNN" (no "cho") by an actual user — that's not
       a typo, it's normal spoken-register phrasing. Adds a second alias
       with a short, fixed stopword list stripped out, whenever doing so
       changes the string.

    2. A distinguishing suffix/prefix word for "base name + variant"
       product families, e.g. "Công tắc thông minh" vs. "Công tắc thông
       minh dimmer" — users very commonly say just "dimmer" rather than
       the full compound name. Only trusted when there are 1-2 words
       present in this name but absent from EVERY sibling product's name
       (so a generic shared word like "công tắc" is never added alone —
       that would be a badly over-broad alias matching half the catalog).
    """
    aliases = set()
    lowered = product_name.strip().lower()
    if lowered:
        aliases.add(lowered)
    no_diacritics = _key(product_name)
    if no_diacritics and no_diacritics != lowered:
        aliases.add(no_diacritics)

    # Dash-separated alternate name, e.g. "Bộ điều khiển trung tâm - Gateway"
    if " - " in product_name:
        for part in product_name.split(" - "):
            part = part.strip()
            if len(part) >= 2:
                aliases.add(part.lower())
                key = _key(part)
                if key:
                    aliases.add(key)

    # Parenthetical alternate name, e.g. "Bộ điều khiển trung tâm (Gateway)"
    for match in re.findall(r"\(([^)]+)\)", product_name):
        part = match.strip()
        if len(part) >= 2:
            aliases.add(part.lower())
            key = _key(part)
            if key:
                aliases.add(key)

    # Stopword-stripped form — see rule 1 in the docstring above.
    name_words = lowered.split()
    stripped_words = [w for w in name_words if w not in _NAME_STOPWORDS]
    if stripped_words and len(stripped_words) != len(name_words):
        stripped = " ".join(stripped_words)
        aliases.add(stripped)
        stripped_key = _key(stripped)
        if stripped_key:
            aliases.add(stripped_key)

    # Distinguishing word(s) vs. sibling products — see rule 2 in the
    # docstring above.
    if sibling_names:
        my_words = set(name_words)
        shared_with_any_sibling: set = set()
        for sibling in sibling_names:
            sibling_norm = (sibling or "").strip().lower()
            if not sibling_norm or sibling_norm == lowered:
                continue
            shared_with_any_sibling |= (my_words & set(sibling_norm.split()))
        distinguishing = my_words - shared_with_any_sibling
        if 1 <= len(distinguishing) <= 2:
            frag_words = [w for w in name_words if w in distinguishing]
            frag = " ".join(frag_words)
            if len(frag) >= 3:
                aliases.add(frag)

    return sorted(aliases)


def update_catalog_entry(source_file: str, fields: Dict[str, Any]) -> bool:
    """Merge `fields` into the existing catalog entry matching `source_file`
    and persist to disk. Returns True if the file was found and updated.

    Used by the self-healing alias step so improvements to alias generation
    apply to already-registered products on their next ingest, not just to
    brand-new ones.
    """
    path = _writable_catalog_path()
    if not path or not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Cannot read catalog at %s to update entry: %s", path, e)
        return False

    updated = False
    for item in data:
        if item.get("source_file") == source_file:
            item.update(fields)
            updated = True
            break

    if not updated:
        return False

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def append_catalog_entry(entry: Dict[str, Any]) -> bool:
    """Persist a newly inferred product to product_catalog.json.

    Returns True if written. Skips silently (returns False) if no catalog
    path is writable (e.g. read-only container mount) — ingestion still
    proceeds, but the product will need a manual catalog entry to be
    resolvable by name in chat queries.
    """
    path = _writable_catalog_path()
    if not path:
        logger.warning(
            "Cannot auto-register product '%s' in catalog: no writable path found among %s",
            entry.get("product_id"), _CATALOG_PATHS,
        )
        return False

    data: list = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Cannot read existing catalog at %s to append entry: %s", path, e)
            return False

    if any(item.get("source_file") == entry.get("source_file") for item in data):
        return False  # already present, nothing to do

    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(
        "Auto-registered new product '%s' (%s) in catalog: %s",
        entry.get("product_name"), entry.get("product_id"), path,
    )
    return True


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
        self.catalog_by_product_name = {
            _key(item.get("product_name", "")): item
            for item in self.catalog_by_filename.values()
            if item.get("product_name")
        }

    def get_metadata_for_file(self, filename: str, elements: Optional[list] = None) -> Dict[str, Any]:
        basename = os.path.basename(filename)
        catalog_entry = self.catalog_by_filename.get(basename, {})

        product_id = catalog_entry.get("product_id", "")
        product_name = catalog_entry.get("product_name", "Thiết bị Vhomenex")
        product_group = catalog_entry.get("product_group", "Thiết bị thông minh")
        model = catalog_entry.get("model", "")
        aliases = catalog_entry.get("aliases", [])
        keywords = catalog_entry.get("keywords", [])
        document_version = ""
        effective_date = ""

        # Version/date live in the cover block, independent of whether the
        # filename already has a catalog entry.
        if elements:
            for elem in elements:
                text = getattr(elem, "text", "").strip()
                version_match = re.match(r"^PHIÊN BẢN\s*:\s*(.+)$", text, re.IGNORECASE)
                date_match = re.match(r"^NGÀY HIỆU LỰC\s*:\s*(.+)$", text, re.IGNORECASE)
                if version_match:
                    document_version = version_match.group(1).strip()
                elif date_match:
                    effective_date = date_match.group(1).strip()

        DEFAULT_PLACEHOLDER_NAME = "Thiết bị Vhomenex"

        # --- Self-healing: an existing auto_generated catalog entry with a
        # too-thin alias set (a product added before build_auto_alias_set was
        # improved, or before this healing step existed at all) gets its
        # aliases regenerated and merged in — automatically, on the very next
        # ingest run, no manual catalog editing required. Confirmed in
        # practice: 5 of 8 products were stuck with only 2 generic aliases
        # each (full name + no-diacritics) and 0 keywords, which was the root
        # cause of most product-misresolution failures in real testing (e.g.
        # "Gateway" alone never matched "Bộ điều khiển trung tâm - Gateway"
        # because no alias contained just "gateway"). This makes future
        # improvements to alias generation retroactively apply to already
        # -ingested products too, not just brand-new ones.
        _MIN_HEALTHY_ALIAS_COUNT = 3
        if (
            catalog_entry
            and catalog_entry.get("auto_generated")
            and len(catalog_entry.get("aliases", [])) < _MIN_HEALTHY_ALIAS_COUNT
        ):
            regenerated = build_auto_alias_set(
                product_name,
                sibling_names=[
                    item.get("product_name", "")
                    for item in self.catalog_by_filename.values()
                    if item.get("product_name")
                ],
            )
            merged_aliases = sorted(set(aliases) | set(regenerated))
            if merged_aliases != sorted(aliases):
                aliases = merged_aliases
                catalog_entry["aliases"] = aliases
                self.catalog_by_filename[basename] = catalog_entry
                if update_catalog_entry(basename, {"aliases": aliases}):
                    logger.info(
                        "Self-healed thin alias set for '%s' (%s): %s",
                        product_name, basename, aliases,
                    )

        # If no catalog entry found, try to infer from document content
        auto_generated = False
        name_extracted_from_table = bool(catalog_entry)  # already known if catalog matched
        if not catalog_entry and elements:
            for elem in elements:
                if elem.kind == "table" and hasattr(elem, "raw_rows") and elem.raw_rows:
                    for row in elem.raw_rows:
                        if len(row) >= 2:
                            k, v = row[0].strip().lower(), row[1].strip()
                            if "tên sản phẩm" in k and v:
                                product_name = v
                                name_extracted_from_table = True
                            elif "nhóm sản phẩm" in k and v:
                                product_group = v
                            elif "model" in k and v:
                                model = v
                            elif "mã sản phẩm" in k and v:
                                product_id = v

            # Fallback #2: some source docs omit the "Tên sản phẩm" row from
            # the spec table entirely (confirmed on real files — the table
            # starts straight at "Điện áp hoạt động" with no name row at
            # all), even though the document as a whole is fine. Every file
            # following the standard template still reliably has the
            # product name on the cover page, as the all-caps line right
            # after the "TÀI LIỆU MÔ TẢ SẢN PHẨM" title (appears twice, cover
            # + repeated header block — take the first occurrence). This is
            # a more structural, template-guaranteed source than the spec
            # table, so use it whenever the table lookup above found nothing.
            if not name_extracted_from_table:
                _BOILERPLATE_PREFIXES = (
                    "TÊN DỰ ÁN", "MÃ SỐ", "PHIÊN BẢN", "NGÀY HIỆU LỰC", "CÔNG TY",
                )
                for i, elem in enumerate(elements):
                    if (
                        getattr(elem, "kind", "") == "metadata"
                        and elem.text.strip().upper() == "TÀI LIỆU MÔ TẢ SẢN PHẨM"
                        and i + 1 < len(elements)
                    ):
                        nxt = elements[i + 1]
                        cand = getattr(nxt, "text", "").strip()
                        if (
                            getattr(nxt, "kind", "") == "metadata"
                            and getattr(nxt, "is_front_matter", False)
                            and cand
                            and not cand.upper().startswith(_BOILERPLATE_PREFIXES)
                        ):
                            # Title-case it (cover titles are ALL CAPS by
                            # template convention) so it reads naturally
                            # alongside product names extracted from tables.
                            product_name = cand.title()
                            name_extracted_from_table = True
                        break

            # Only reuse an existing catalog entry's aliases/id when the
            # product name was actually read from the document's own table.
            # If extraction failed, product_name is still the generic
            # DEFAULT_PLACEHOLDER_NAME — matching on that string would wrongly
            # merge every unrelated file whose table also failed to parse
            # into a single fake "product" (confirmed bug: two different new
            # product docs both landed on 'Thiết bị Vhomenex' and the second
            # one silently inherited the first one's product_id/aliases).
            name_entry = (
                self.catalog_by_product_name.get(_key(product_name), {})
                if name_extracted_from_table
                else {}
            )
            if name_entry:
                product_id = name_entry.get("product_id", product_id)
                product_name = name_entry.get("product_name", product_name)
                product_group = name_entry.get("product_group", product_group)
                model = name_entry.get("model", model)
                aliases = name_entry.get("aliases", aliases)
                keywords = name_entry.get("keywords", keywords)
            else:
                auto_generated = True

        if not product_id:
            source_name = product_name if product_name != DEFAULT_PLACEHOLDER_NAME else Path(basename).stem
            source_name = re.sub(r'_RAG_v\d+.*$', '', source_name, flags=re.IGNORECASE)
            source_name = re.sub(r'^\d+[_.\s-]+', '', source_name)
            product_id = _slug(source_name)

        # A file with no catalog entry and no name recognized from any
        # existing product is a brand-new product. Register it automatically
        # so it is resolvable by name/alias in chat on the very next request,
        # without requiring a manual product_catalog.json edit.
        #
        # If the product name couldn't be extracted from the document table
        # at all (name_extracted_from_table is False), still register the
        # entry — using the filename-derived product_id/product_name instead
        # of the generic placeholder — so this product is at least resolvable
        # by its file-derived name, and flag it clearly so a human fixes the
        # source .docx table (see the "Thiếu tên sản phẩm..." warning below).
        if auto_generated and not aliases:
            if not name_extracted_from_table:
                # Don't register the useless literal placeholder string as
                # this product's identity/alias — use the same file-derived
                # name computed for product_id above instead.
                product_name = source_name.replace("_", " ").replace("-", " ").strip() or product_name
            aliases = build_auto_alias_set(
                product_name,
                sibling_names=[
                    item.get("product_name", "")
                    for item in self.catalog_by_filename.values()
                    if item.get("product_name")
                ],
            )
            new_entry = {
                "product_id": product_id,
                "product_name": product_name,
                "product_group": product_group,
                "source_file": basename,
                "business_product_id": "",
                "model": model,
                "aliases": aliases,
                "common_typos": [],
                "keywords": keywords,
                "auto_generated": True,
                "name_extraction_failed": not name_extracted_from_table,
            }
            if append_catalog_entry(new_entry):
                self.catalog_by_filename[basename] = new_entry
                self.catalog_by_product_name[_key(product_name)] = new_entry

        return {
            "product_id": product_id,
            "product_name": product_name,
            "product_group": product_group,
            "model": model,
            "aliases": aliases,
            "keywords": keywords,
            "source_file": basename,
            "source_document": basename,
            "document_version": document_version,
            "effective_date": effective_date,
            "name_extraction_failed": not name_extracted_from_table,
        }
