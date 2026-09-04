"""Resolve one or multiple products from the product catalog."""

import difflib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CATALOG_PATHS = [
    Path(os.environ.get("PRODUCT_CATALOG_PATH", "/app/data/product_catalog.json")),
    Path("/app/data/product_catalog.json"),
    Path("./data/product_catalog.json"),
    Path("data/product_catalog.json"),
]


@dataclass
class ResolvedProduct:
    """Backward-compatible result used by metadata-filter helpers."""

    product_id: Optional[str] = None
    product_name: Optional[str] = None
    model: Optional[str] = None
    confidence: float = 0.0


_MODEL_RE = re.compile(
    r"\b([A-Z]{2,}[-][A-Z0-9]+(?:[-][A-Z0-9]+)*)\b",
    re.IGNORECASE,
)

_CONNECTION_TERMS = (
    "kết nối",
    "ghép nối",
    "thêm thiết bị",
    "đăng ký thiết bị",
)

_RELATION_MARKERS = (
    " thông qua ",
    " với ",
    " qua ",
    " bằng ",
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # Strip common sentence punctuation so it doesn't stick to an adjacent
    # word (e.g. "Dimmer," -> "dimmer") and dilute n-gram window matching in
    # _best_ngram_similarity. Keep "/" since product names sometimes use it
    # (e.g. "Tăng/Giảm" as a button label, not part of a product name).
    text = re.sub(r"[.,;:!?()\[\]\"“”'’]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        # Containment alone isn't enough — a short, generic fragment (e.g.
        # "công tắc", 2 words) is a substring of many different product
        # names ("Công tắc thông minh", "Công tắc thông minh dimmer",
        # "Công tắc chống giật..."), so unconditional high-confidence
        # containment previously caused a short window to false-match
        # whichever unrelated product got iterated first. Scale by how much
        # of the longer string the shorter one actually covers: a near-full
        # match still scores ~0.9, but a trivial fragment scores much lower
        # and won't cross the resolution threshold on its own.
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        coverage = len(shorter) / max(1, len(longer))
        return 0.5 + 0.4 * coverage
    return difflib.SequenceMatcher(None, a, b).ratio()


def _best_ngram_similarity(candidate: str, query_words: List[str]) -> float:
    """Best similarity between `candidate` and any contiguous word-window of
    the query, instead of the whole raw sentence.

    difflib.SequenceMatcher.ratio() over a short candidate name (a few words)
    vs. an entire long question is dominated by whatever characters happen to
    overlap across the whole sentence — mostly noise, and confirmed in
    practice to pick essentially arbitrary wrong products for real questions
    (e.g. "Công tắc Dimmer" fuzzy-matched onto "Cảm biến chuyển động ánh
    sáng"). Comparing against sliding windows sized close to the candidate's
    own word count is far closer to how a person spots a product name inside
    a sentence, and scores the correct product much more reliably (e.g. the
    window "công tắc dimmer" vs. candidate "công tắc thông minh dimmer"
    scores ~0.73, well above threshold, where whole-sentence comparison
    scored too low to ever match).
    """
    if not candidate or not query_words:
        return similarity_ratio(candidate, " ".join(query_words))

    candidate_len = max(1, len(candidate.split()))
    best = 0.0
    # Try windows both shorter and longer than the candidate's own word
    # count: the query may omit a word the official name has (e.g. "Công
    # tắc Dimmer" vs. catalog's "Công tắc thông minh dimmer" — 3 words vs
    # 4), or add one. A window range fixed at exactly candidate_len-and-up
    # misses the shorter, better-matching phrase entirely.
    min_window = max(1, candidate_len - 2)
    max_window = candidate_len + 2
    for window_size in range(min_window, max_window + 1):
        for start in range(0, max(1, len(query_words) - window_size + 1)):
            window = " ".join(query_words[start:start + window_size])
            if not window:
                continue
            score = similarity_ratio(candidate, window)
            if score > best:
                best = score
    return best


class ProductResolver:
    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog: List[Dict[str, Any]] = []
        self._catalog_path = catalog_path
        self.reload()

    def reload(self) -> None:
        """(Re)load the catalog from disk.

        ProductResolver is now created once at app startup (singleton) rather
        than per-request, so it must be explicitly refreshed after ingest
        auto-registers new products into product_catalog.json — otherwise
        the in-memory catalog would silently drift from disk until restart.
        """
        if self._catalog_path and self._catalog_path.exists():
            search_paths = [self._catalog_path]
        else:
            search_paths = _CATALOG_PATHS

        for p in search_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as file:
                    self.catalog = json.load(file)
                logger.info("ProductResolver loaded catalog from: %s (%d products)", p, len(self.catalog))
                return

        logger.warning("ProductResolver: no catalog found at any path")

    @staticmethod
    def _result(
        item: Dict[str, Any],
        confidence: float,
        match_type: str,
    ) -> Dict[str, Any]:
        return {
            "product_id": item["product_id"],
            "product_name": item["product_name"],
            "confidence": confidence,
            "match_type": match_type,
        }

    def resolve_all(self, query: str) -> List[Dict[str, Any]]:
        """Return every product explicitly mentioned in the query."""
        q_norm = normalize_text(query)

        if not q_norm or not self.catalog:
            return []

        matches: Dict[str, Dict[str, Any]] = {}
        # Text still available for the fuzzy pass below — shrinks as exact
        # matches consume their own substring, so the fuzzy pass never
        # re-discovers a close *variant* of an already-matched product off
        # of text that already "belongs" to that match (see fuzzy pass
        # comment for why this matters).
        remaining = q_norm

        # Pass 1: exact name / model / alias / typo matches. May find
        # several distinct products directly in one query.
        #
        # Candidates are gathered FIRST, then applied longest-match-first
        # against `remaining` (which shrinks as each match consumes its
        # own text) -- rather than each product independently testing
        # itself against the untouched query. Confirmed real bug this
        # fixes: "Cách lắp đặt Công tắc thông minh chống giật" matched
        # BOTH the intended product (via its alias "công tắc thông minh
        # chống giật", 0.95) AND an unrelated one whose full product_name
        # "Công tắc thông minh" is merely a PREFIX of that same phrase
        # (exact_name, 1.0) -- the shorter, more generic match even
        # outscoring the specific one, so the answer got built from a mix
        # of both products' installation steps.
        #
        # Longest-first + masking resolves this without discarding
        # legitimately-multiple products: in a genuine comparison query
        # ("so sánh công tắc thông minh và công tắc thông minh dimmer")
        # the longer name consumes only ITS OWN occurrence, leaving the
        # second, independent occurrence of the shorter name still
        # present in `remaining` -- so both products are correctly found.
        exact_candidates = []  # (matched_text, confidence, match_type, item)
        for item in self.catalog:
            product_name = normalize_text(item.get("product_name", ""))
            if product_name and product_name in q_norm:
                exact_candidates.append((product_name, 1.0, "exact_name", item))

            model = normalize_text(item.get("model", ""))
            if (
                model
                and model not in {
                    "chưa xác định - cần xác nhận",
                    "chưa có dữ liệu",
                    "chưa có dữ liệu trong tài liệu",
                }
                and model in q_norm
            ):
                exact_candidates.append((model, 1.0, "model", item))

            for alias in item.get("aliases", []):
                alias_norm = normalize_text(alias)
                if alias_norm and alias_norm in q_norm:
                    exact_candidates.append((alias_norm, 0.95, "exact_alias", item))

            for typo in item.get("common_typos", []):
                typo_norm = normalize_text(typo)
                if typo_norm and typo_norm in q_norm:
                    exact_candidates.append((typo_norm, 0.90, "typo", item))

        # Longest matched text first; higher confidence breaks ties so a
        # product_name still outranks an alias of identical length.
        exact_candidates.sort(key=lambda c: (len(c[0]), c[1]), reverse=True)

        for matched_text, confidence, match_type, item in exact_candidates:
            if matched_text not in remaining:
                # Fully consumed by an earlier, longer match -- this
                # product was never independently named in the query.
                continue
            product_id = item["product_id"]
            if product_id not in matches:
                matches[product_id] = self._result(item, confidence, match_type)
            remaining = remaining.replace(matched_text, " ")

        # Pass 2: fuzzy n-gram fallback for every catalog item NOT already
        # matched in pass 1 — runs unconditionally, not only when pass 1
        # found nothing at all.
        #
        # Confirmed real bug this fixes: a query naming several products
        # where SOME match exactly and one doesn't (e.g. a joining word
        # like "cho" is dropped in casual phrasing: "công tắc chống giật
        # BNN" vs. the catalog's "...cho Bnn") used to return ONLY the
        # exact matches, silently dropping the rest from retrieval
        # entirely — not because the document lacked data, but because
        # that product was never looked up in the first place.
        #
        # Matching against `remaining` (exact-matched substrings masked
        # out) rather than the raw query is what makes this safe: without
        # masking, a product whose full name is a superset of an
        # already-matched sibling's name (e.g. "Công tắc thông minh
        # dimmer" vs. already-matched "Công tắc thông minh") would itself
        # fuzzy-match the leftover "công tắc thông minh" text with a high
        # score purely from that shared prefix, even when the query never
        # actually said "dimmer" — falsely pulling in a second product
        # nobody asked about. Masking removes that shared text before the
        # fuzzy pass ever runs, so a variant only gets pulled in by
        # genuine leftover similarity, not leftover overlap with a
        # sibling that already matched.
        remaining_words = remaining.split()
        if remaining_words:
            scored: List[Tuple[str, float, Dict[str, Any]]] = []
            for item in self.catalog:
                product_id = item["product_id"]
                if product_id in matches:
                    continue

                # product_name + aliases only — NOT keywords. Confirmed
                # real bug: keywords are short, generic topical markers
                # (e.g. "cửa", "ánh sáng", "pin"), not curated product-
                # identity strings, and are exactly the kind of short text
                # that difflib's character-level ratio scores deceptively
                # high against an unrelated word that merely shares most
                # letters — e.g. the keyword "cửa" ("door") scored 0.667
                # (over the 0.65 threshold) against the common Vietnamese
                # possessive particle "của" ("of"), which appears in a huge
                # fraction of ordinary Vietnamese sentences ("công suất
                # tải CỦA công tắc..."). This silently pulled the door
                # sensor into a 3-product switch comparison that never
                # mentioned it. Verified empirically against the real
                # catalog: product_name/aliases for both affected products
                # scored well under 0.65 in every case; only their bare
                # keyword entries crossed the threshold. Aliases stay
                # eligible here since they're curated specifically to
                # identify this product; keywords were never meant for
                # that and should not be trusted as fuzzy identity anchors.
                candidates = [item["product_name"]] + item.get("aliases", [])
                best_score = 0.0
                for candidate in candidates:
                    candidate_norm = normalize_text(candidate)
                    # Defense-in-depth, and confirmed necessary in
                    # practice — not just theoretical: a hand-curated
                    # alias is not automatically safe just for being an
                    # alias. Adding the short, legitimate colloquial
                    # alias "công tơ" ("electricity meter") for the
                    # smart-meter product later caused it to fuzzy-match
                    # "công tắc cửa cuốn" ("roller-shutter switch") at
                    # 0.800 — comfortably past even the high-confidence
                    # bypass — misrouting a roller-shutter-switch question
                    # to the electricity meter. "Động cơ rèm" (11 chars),
                    # a genuinely different product's alias, legitimately
                    # relies on fuzzy matching (confirmed working
                    # correctly earlier) and needs to keep doing so — the
                    # earlier 4-character floor was tuned against the
                    # "cửa" keyword collision alone and wasn't tested
                    # against short *aliases* specifically, which turned
                    # out to need a materially higher floor. 10 characters
                    # cleanly separates the two real cases on record
                    # (7 vs. 11) with margin on both sides; short aliases
                    # under this floor still resolve perfectly well via
                    # the exact-substring pass above — this only removes
                    # them as fuzzy-match anchors, not as valid aliases.
                    if len(candidate_norm) < 10:
                        continue
                    score = _best_ngram_similarity(
                        candidate_norm,
                        remaining_words,
                    )
                    if score > best_score:
                        best_score = score

                if best_score > 0:
                    scored.append((product_id, best_score, item))

            # Confirmed real bug (a second, distinct case beyond the
            # keyword one above): a query mentioning only a GENERIC shared
            # term ("công tắc" / "switch") that several sibling products'
            # names all start with ("Công tắc thông minh", "Công tắc
            # thông minh dimmer", "Công tắc chống giật cho Bnn") produced
            # three close, plausible fuzzy scores (0.696 / 0.610 / 0.633)
            # for the exact same query — the naive ">= 0.65" check picked
            # the single highest one with false 1.0 confidence, when
            # really none of them should have won outright: the query
            # never said which switch it meant. That false-confident
            # resolution also silently bypassed the multi-product
            # ambiguity check downstream (which only runs when NO product
            # resolves), since the caller believed a specific product was
            # confidently identified.
            #
            # Fix: only accept a MODERATE-confidence fuzzy match (score
            # below _HIGH_CONFIDENCE) when the top-scoring candidate is
            # CLEARLY separated from the next-best one — a narrow margin
            # in that zone is a sign several products are genuinely
            # competing for the same generic text, not that this one
            # candidate was truly identified. A HIGH-confidence match
            # (score >= _HIGH_CONFIDENCE) still requires at least a SMALL
            # margin (_HIGH_CONFIDENCE_MARGIN, looser than the standard
            # one) rather than being trusted completely unconditionally
            # — confirmed real case where an unconditional bypass matters:
            # "Rèm không di chuyển dù động cơ hoạt động..." correctly
            # fuzzy-matches "Động Cơ Rèm" at 0.755, with a coincidental,
            # unrelated runner-up ("Cảm biến chuyển động ánh sáng",
            # scoring 0.703) sitting close by raw margin (0.052) — that
            # margin is still wide enough to trust.
            #
            # Only the SINGLE top-scoring candidate is ever considered
            # for a fuzzy match, checked against the single runner-up —
            # deliberately not "reject the top pair, then let 3rd place
            # win instead". Confirmed necessary by two real bugs found
            # together: (1) an unconditional high-confidence bypass let
            # two near-identical top scores (0.774 vs 0.769, "cảm biến
            # chống giật") each independently clear 0.73 and both resolve
            # confidently, since neither check looked at the other's
            # score. (2) Fixing that by making even high-confidence
            # matches require a small margin then broke a genuinely
            # different case: when the top TWO scores are an EXACT tie
            # (0.767 vs 0.767, "liệt kê các loại cảm biến"), rejecting
            # only the first-sorted of the pair against its neighbour
            # left the second one with no lower neighbour to compare
            # against, so it fell through and resolved anyway — the
            # opposite of the intended fix. Restricting fuzzy matching to
            # a single up-or-down decision about the overall top
            # candidate sidesteps both failure modes at once: a
            # rejected top pair yields NO fuzzy match here rather than a
            # partial, unpredictable one, deferring to the downstream
            # unrestricted-search + relevance-ratio filtering (see
            # _relevant_products_for_ambiguity_check) to find a genuinely
            # relevant product like "Ổ cắm thông minh chống giật" instead
            # — already confirmed to work correctly for that case.
            scored.sort(key=lambda entry: entry[1], reverse=True)
            if scored:
                top_id, top_score, top_item = scored[0]
                if top_score >= 0.65:
                    _FUZZY_MARGIN = 0.08
                    _HIGH_CONFIDENCE = 0.73
                    _HIGH_CONFIDENCE_MARGIN = 0.03
                    runner_up_score = scored[1][1] if len(scored) > 1 else 0.0
                    required_margin = (
                        _HIGH_CONFIDENCE_MARGIN if top_score >= _HIGH_CONFIDENCE else _FUZZY_MARGIN
                    )
                    too_close = (
                        runner_up_score >= 0.55
                        and (top_score - runner_up_score) < required_margin
                    )
                    if not too_close:
                        matches[top_id] = self._result(top_item, round(top_score, 2), "fuzzy")

        return list(matches.values())

    def catalog_item(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return one catalog row without exposing catalog traversal to callers."""
        return next(
            (
                item
                for item in self.catalog
                if str(item.get("product_id", "")) == product_id
            ),
            None,
        )

    def resolve_primary(self, query: str) -> Dict[str, Any]:
        """Resolve the product being operated on in a relational question.

        For example, in ``kết nối Cảm biến cửa Mesh với Gateway`` the sensor
        is the primary product while Gateway is the connection target.  A
        normal multi-product fact or comparison still remains multi-product
        through :meth:`resolve_all`.
        """
        matches = self.resolve_all(query)
        if len(matches) <= 1:
            return matches[0] if matches else {
                "product_id": "",
                "product_name": "",
                "confidence": 0.0,
                "match_type": "none",
            }

        normalized = normalize_text(query)
        if any(term in normalized for term in _CONNECTION_TERMS):
            marker_positions = [
                normalized.find(marker)
                for marker in _RELATION_MARKERS
                if normalized.find(marker) >= 0
            ]
            if marker_positions:
                prefix = normalized[: min(marker_positions)]
                prefix_matches = [
                    item
                    for item in self.resolve_all(prefix)
                    if item.get("product_id")
                    and item.get("confidence", 0.0) >= 0.65
                ]
                if len(prefix_matches) == 1:
                    return prefix_matches[0]

        return {
            "product_id": "",
            "product_name": "",
            "confidence": 0.0,
            "match_type": "multiple",
        }

    def resolve(self, query: str) -> Dict[str, Any]:
        """Backward-compatible single-product resolution."""
        return self.resolve_primary(query)


def resolve_product(
    query: str,
    product_id_hint: Optional[str] = None,
) -> ResolvedProduct:
    """Compatibility API retained for older tests and metadata filters."""
    if product_id_hint:
        return ResolvedProduct(product_id=product_id_hint, confidence=1.0)

    match = ProductResolver().resolve_primary(query)
    model_match = _MODEL_RE.search(query)
    model = model_match.group(1).upper() if model_match else None

    if match.get("product_id"):
        item = ProductResolver().catalog_item(str(match["product_id"]))
        return ResolvedProduct(
            product_id=str(match["product_id"]),
            product_name=str(match.get("product_name") or ""),
            model=str(item.get("model") or "") if item else model,
            confidence=float(match.get("confidence", 0.0)),
        )

    return ResolvedProduct(model=model, confidence=0.6 if model else 0.0)
