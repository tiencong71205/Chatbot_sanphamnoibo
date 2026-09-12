"""Hybrid Retriever integrating resolver, dense, sparse, RRF fusion, and full-coverage scroll retrieval."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, List, Optional

from app.config import Settings
from app.database.bm25_store import BM25Store
from app.database.qdrant_store import QdrantStore
from app.embeddings.ollama_embedding import OllamaEmbedding
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.product_resolver import ProductResolver, normalize_text
from app.retrieval.rrf_fusion import rrf_fusion

logger = logging.getLogger(__name__)

# Maps common Vietnamese section-name keywords (as passed via
# section_full's requested_sections) to the content_type category
# ContentTypeClassifier assigns at ingest time -- lets retrieve_full()
# actively prefer chunks genuinely labeled that category over ones that
# only surface via incidental keyword overlap in an unrelated section's
# own title. Kept in sync with app/ingestion/content_classifier.py's
# SECTION_MAP categories; only include keywords unambiguous enough that a
# section_name match strongly implies wanting that specific category.
_SECTION_HINT_CONTENT_TYPES = {
    "lắp đặt": "installation",
    "kết nối": "connection",
    "cấu hình": "configuration",
    "sự cố": "troubleshooting",
    "khôi phục": "reset",
}

# Content_type pairs confirmed to genuinely conflict via incidental
# keyword overlap (not just theoretical) -- "installation" vs
# "configuration" specifically confirmed with real data: a query hinting
# "installation" (via the word "lắp đặt") can retrieve a "configuration"
# chunk (an app-setting named "Chế độ lắp đặt" — "installation MODE",
# i.e. a setting, not an instruction to physically install) ranked ABOVE
# the genuinely correct "installation" chunk, purely because the setting
# name repeats "lắp đặt" densely. Deliberately narrow — only pairs with
# confirmed real evidence are listed, to avoid excluding content types
# that merely differ without genuinely conflicting.
_CONFLICTING_CONTENT_TYPES = {
    "installation": {"configuration"},
}


class _TTLCache:
    """Small thread-safe TTL+LRU cache, in-process only.

    This is a single-instance FastAPI deployment and the values being
    cached (a retrieve() result: a handful of chunk payloads) are cheap to
    hold in memory — the point is purely to skip redundant Ollama-embedding
    + Qdrant + BM25 round trips when the same or a repeated question comes
    in again shortly after (common for FAQ-style product questions across
    concurrent users), not to build a distributed cache.
    """

    def __init__(self, maxsize: int, ttl_seconds: float):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class HybridRetriever:
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
        self.resolver = ProductResolver()

        self.dense = DenseRetriever(settings, qdrant, embedding)
        self.sparse = BM25Retriever(settings, bm25)

        # HybridRetriever is now a process-lifetime singleton (built once in
        # app/api.py's lifespan handler), so this pool is created once and
        # reused across requests rather than spun up per call. It overlaps
        # the network-bound Qdrant dense search with the CPU-bound in-memory
        # BM25 sparse search — previously these ran strictly sequentially
        # per product, which is pure wasted wall-clock time since neither
        # depends on the other's result.
        self._io_pool = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="hybrid-retrieve-io"
        )

        # Retrieval-result cache, keyed on (query, top_k, filters,
        # product_id). Deliberately caches the *retrieved chunks*, not the
        # final LLM answer — every request still gets a fresh generation
        # call, so caching here can never make an answer stale or generic;
        # it only skips re-running embed+dense+sparse+fusion for a question
        # (or its exact repeat) that was just answered. 2 minute TTL is
        # short enough that a missed invalidate_cache() call after ingest
        # self-heals quickly; invalidate_cache() is also called explicitly
        # from the /api/ingest and /api/reindex endpoints so a rebuilt
        # index is never masked by a stale cached result in practice.
        self._retrieval_cache = _TTLCache(maxsize=512, ttl_seconds=120)

    @staticmethod
    def _round_robin(
        result_groups: List[List[Dict[str, Any]]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Merge result groups while preserving product diversity."""
        merged: List[Dict[str, Any]] = []
        seen_chunk_ids = set()
        index = 0

        while len(merged) < limit:
            added = False

            for group in result_groups:
                if index >= len(group):
                    continue

                item = group[index]
                payload = item.get("payload", {})
                chunk_id = payload.get("chunk_id")

                if chunk_id not in seen_chunk_ids:
                    merged.append(item)
                    seen_chunk_ids.add(chunk_id)
                    added = True

                if len(merged) >= limit:
                    break

            if not added:
                break

            index += 1

        return merged

    def retrieve_full(
        self,
        product_id: Optional[str] = None,
        content_type: Optional[str] = None,
        section_name: Optional[str] = None,
        heading_path: Optional[str] = None,
        source_document: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve ALL matching chunks without top_k semantic limits via Qdrant scroll.

        Used for intent: product_full, section_full, compare_full.
        """
        filters: Dict[str, Any] = {}
        if product_id:
            filters["product_id"] = product_id
        if content_type:
            filters["content_type"] = content_type
        if source_document:
            filters["source_document"] = source_document

        records = self.qdrant.scroll_by_filter(filters)

        # Post-filter by section_name / heading_path if requested
        filtered_records = []
        for rec in records:
            p = rec.get("payload", {})
            sec = str(p.get("source_section", "")).lower()
            head = str(p.get("heading_path", "")).lower()

            if section_name and section_name.lower() not in sec and section_name.lower() not in head:
                continue
            if heading_path and heading_path.lower() not in head:
                continue

            filtered_records.append(rec)

        if section_name:
            sec_lower = section_name.lower()

            # A section_name that names a common document category (e.g.
            # "lắp đặt") almost always means the user wants THAT category
            # of content specifically -- and ContentTypeClassifier already
            # assigns each chunk exactly this kind of category label at
            # ingest time. When the keyword-substring match above pulled
            # in a mix of content_type values, actively drop chunks whose
            # content_type doesn't match the hinted category, rather than
            # only reordering them.
            #
            # Confirmed necessary, not just reordering, by real evidence:
            # a prior fix that only SORTED matching content later (never
            # excluding it) still let the model pick the wrong chunk over
            # genuinely correct ones earlier in the list -- its clean,
            # step-numbered formatting reads as more "answer-shaped" than
            # the correct chunks' prose, regardless of position. Only
            # applied when at least one chunk actually carries the hinted
            # content_type, so a real (if unusually-labeled) match is
            # never discarded outright.
            hinted_type = next(
                (ct for kw, ct in _SECTION_HINT_CONTENT_TYPES.items() if kw in sec_lower),
                None,
            )
            if hinted_type:
                matching = [
                    r for r in filtered_records
                    if r.get("payload", {}).get("content_type") == hinted_type
                ]
                if matching:
                    filtered_records = matching

            # Secondary signal, still useful for content_type values that
            # aren't in the hint map, or ties within the same
            # content_type: chunks whose TOP-LEVEL heading_path segment
            # (the chapter, not a specific leaf section) contains
            # section_name come first -- these are chunks genuinely filed
            # under the requested chapter, not just incidentally
            # mentioning it in an unrelated section's own title.
            def _chapter_match_priority(rec: Dict[str, Any]) -> int:
                head = str(rec.get("payload", {}).get("heading_path", "")).lower()
                chapter = head.split(">", 1)[0].strip()
                return 0 if sec_lower in chapter else 1

            filtered_records.sort(key=_chapter_match_priority)

        logger.info(
            "retrieve_full (filters=%s, section=%s) returned %d chunks",
            filters, section_name, len(filtered_records)
        )
        return filtered_records

    def _pull_in_sibling_chunks(
        self,
        selected: List[Dict[str, Any]],
        pool: List[Dict[str, Any]],
        max_extra: int = 3,
    ) -> List[Dict[str, Any]]:
        """Add back sibling chunks (same parent heading) that the top-k cut
        left out, when at least one of their siblings did make the cut.

        Confirmed real bug: a product's installation chapter documented two
        wiring diagrams as separate sibling sections ("Kết nối cửa cuốn tự
        động - Loại nút bấm Stop là NC" and "... Stop là NO") whose text is
        near-identical apart from NC/NO. BM25 ranked them adjacently (7 and
        8) -- correctly, since both are equally relevant -- but the dense
        retriever surfaced only the NC one, so RRF scored NO on a single
        channel and it fell from rank 5 to rank 12, outside the top-k. The
        answer then showed the NC diagram only, leaving an installer whose
        door uses an NO stop button with no diagram at all.

        Ranking these siblings against EACH OTHER is meaningless here: for a
        "how do I install this" question both diagrams are needed, and which
        one wins is decided by an arbitrary dense-retrieval cutoff rather
        than by relevance. So instead of re-tuning scores, pull the missing
        siblings back in. Capped by max_extra so a section with many
        children can't flood the context.
        """
        if not selected:
            return selected

        def parent_of(rec: Dict[str, Any]) -> str:
            head = str(rec.get("payload", {}).get("heading_path", ""))
            if ">" not in head:
                return ""
            return head.rsplit(">", 1)[0].strip().lower()

        selected_parents = {p for p in (parent_of(r) for r in selected) if p}
        if not selected_parents:
            return selected

        selected_ids = {
            r.get("id") or id(r) for r in selected
        }
        extras = []
        for rec in pool:
            if len(extras) >= max_extra:
                break
            rec_id = rec.get("id") or id(rec)
            if rec_id in selected_ids:
                continue
            if parent_of(rec) in selected_parents:
                extras.append(rec)

        return selected + extras

    def _prefer_exactly_named_section(
        self, candidates: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Khi câu hỏi gọi gần đúng TÊN một mục, ưu tiên hẳn chunk của mục đó.

        Confirmed real bug this fixes — a whole family of wrong answers, all
        with the same shape: a product documents two near-identical sections
        that differ only in a few words ("Chế độ kết nối TỰ ĐỘNG qua
        Bluetooth Mesh" vs "... THỦ CÔNG qua Bluetooth Mesh"). Asking about
        one of them returned the other's numbers:

        - Cửa cuốn V2, mục 4.2 (thủ công): answered with 4.1's values —
          "giữ 3s, đèn xanh" instead of the correct "giữ 7s, đèn vàng"
        - Ổ cắm chống giật, mục 4.5: answered "đèn xanh" instead of "đèn đỏ"

        These are not cosmetic: an installer following "3 giây" on a device
        that needs 7 seconds simply never gets it into pairing mode.

        RRF alone can't separate these — the two sections share almost every
        word, so their scores land within noise of each other and which one
        wins is effectively arbitrary. But the question itself contains the
        distinguishing word ("thủ công"), so matching the query against the
        SECTION TITLE settles it directly.

        Only fires when one section title is clearly the best match (see the
        margin check below); otherwise the normal ranking is left untouched.
        """
        q_norm = normalize_text(query or "")
        if not q_norm or not candidates:
            return candidates

        q_words = set(q_norm.split())
        if len(q_words) < 3:
            return candidates  # câu hỏi quá ngắn, không đủ căn cứ

        best_section: Optional[str] = None
        best_score = 0.0
        runner_up = 0.0

        seen_sections: Dict[str, float] = {}
        for item in candidates:
            section = str(item.get("payload", {}).get("source_section", "")).strip()
            if not section or section in seen_sections:
                continue
            s_words = set(normalize_text(section).split())
            if not s_words:
                continue
            # Tỉ lệ từ trong tên mục xuất hiện trong câu hỏi. Dùng tên mục
            # làm mẫu số (không phải câu hỏi) vì câu hỏi luôn dài hơn do có
            # thêm tên sản phẩm và phần "thực hiện như thế nào".
            score = len(s_words & q_words) / len(s_words)
            seen_sections[section] = score
            if score > best_score:
                runner_up = best_score
                best_score, best_section = score, section
            elif score > runner_up:
                runner_up = score

        # Chỉ can thiệp khi câu hỏi phủ gần trọn tên một mục VÀ mục đó tách
        # biệt rõ với mục đứng nhì.
        #
        # Hai ngưỡng khác nhau, vì tên sản phẩm trong câu hỏi có thể vô tình
        # trùng từ với tên mục và làm điểm của mục SAI cao lên. Trường hợp
        # thật đã gặp: hỏi "Chế độ kết nối TỰ ĐỘNG ... của CÔNG tắc thông
        # minh cho cửa cuốn" — mục đúng ("...tự động...") khớp 100%, nhưng
        # mục sai ("...thủ CÔNG...") vẫn được 89% chỉ vì chữ "công" trùng
        # với "Công tắc" trong tên sản phẩm. Chênh lệch 0.11 không đủ vượt
        # ngưỡng thường, nên thứ tự sai được giữ nguyên và câu trả lời lấy
        # nhầm số giây của mục kia.
        #
        # Khi câu hỏi chứa TRỌN VẸN tên một mục (>=0.99) thì đó là căn cứ
        # rất mạnh — người dùng gọi đúng tên mục — nên chỉ cần tách biệt
        # tối thiểu là đủ.
        if best_section is None or best_score < 0.8:
            return candidates
        required_margin = 0.02 if best_score >= 0.99 else 0.15
        if (best_score - runner_up) < required_margin:
            return candidates

        matching = [
            item for item in candidates
            if str(item.get("payload", {}).get("source_section", "")).strip() == best_section
        ]
        others = [
            item for item in candidates
            if str(item.get("payload", {}).get("source_section", "")).strip() != best_section
        ]
        # Đưa lên đầu chứ không loại bỏ phần còn lại: mục được hỏi có thể
        # tham chiếu sang mục khác, và giữ lại phần đuôi vẫn có ích khi câu
        # trả lời cần bối cảnh.
        return matching + others

    def _apply_content_type_conflict_filter(
        self, fused: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Drop candidates whose content_type is a confirmed-conflicting
        category for this query, before the caller slices to top-k.

        Confirmed real bug this exists to fix: retrieve() (the plain
        "fact"-intent path) fuses purely on BM25+dense+RRF, with no
        awareness of content_type at all -- unlike retrieve_full() (the
        "section_full"-intent path), which already got this protection.
        A query like "cho tôi cách lắp đặt cảm biến hiện diện" ranked a
        "Chế độ lắp đặt cảm biến" chunk (content_type="configuration", an
        app-SETTING named "installation mode") at #1, while the
        genuinely correct "Tiến hành lắp đặt" chunk (content_type=
        "installation") sat at #11 — one position outside the default
        top-10 cutoff — purely because the setting's name and body
        densely repeat "lắp đặt" and the product's own name. Confirmed
        by inspecting an extended (top-30) candidate pool: the correct
        chunk was present, just narrowly excluded by the top-k slice.
        Dropping the one confirmed-conflicting chunk ahead of it was
        enough to pull it inside the window in this case — a narrow
        margin, not a comfortable one; a query where the correct chunk
        sits further outside top-k than this would need more than this
        filter alone to recover it.
        """
        q_norm = normalize_text(query or "")
        hinted_type = next(
            (ct for kw, ct in _SECTION_HINT_CONTENT_TYPES.items() if kw in q_norm),
            None,
        )
        if not hinted_type:
            return fused
        conflicting = _CONFLICTING_CONTENT_TYPES.get(hinted_type)
        if not conflicting:
            return fused
        has_matching = any(
            item.get("payload", {}).get("content_type") == hinted_type
            for item in fused
        )
        if not has_matching:
            return fused  # no confirmed-relevant chunk exists — don't blindly exclude
        return [
            item for item in fused
            if item.get("payload", {}).get("content_type") not in conflicting
        ]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        product_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Cache-checking wrapper around _retrieve_uncached.

        debug=True bypasses the cache entirely — a debug response carries
        the full dense/sparse candidate dump for troubleshooting, so it
        should always reflect a live retrieval, and caching that much
        larger payload would push smaller, more useful entries out of the
        cache for no benefit (debug requests aren't hot-path traffic).
        """
        if kwargs.get("debug"):
            return self._retrieve_uncached(
                query, top_k=top_k, filters=filters, product_id=product_id, **kwargs
            )

        product_match_type = str(kwargs.get("product_match_type", "explicit_filter"))
        cache_key = self._cache_key(query, top_k, filters, product_id, product_match_type)

        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            # Return a deep copy: the cached entry is shared across
            # concurrent requests, and downstream code (context builder,
            # source-reference building) only reads these payload dicts —
            # but a deep copy costs microseconds next to the network round
            # trips it replaces, and removes any need to audit every
            # current and future caller for in-place mutation.
            return deepcopy(cached)

        result = self._retrieve_uncached(
            query, top_k=top_k, filters=filters, product_id=product_id, **kwargs
        )
        self._retrieval_cache.set(cache_key, result)
        return result

    @staticmethod
    def _cache_key(
        query: str,
        top_k: Optional[int],
        filters: Optional[Dict[str, Any]],
        product_id: Optional[str],
        product_match_type: str,
    ) -> str:
        filters_repr = tuple(sorted((filters or {}).items()))
        raw = repr((
            (query or "").strip().lower(),
            top_k,
            filters_repr,
            product_id,
            product_match_type,
        ))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def invalidate_cache(self) -> None:
        """Drop every cached retrieval result.

        Called after /api/ingest and /api/reindex complete — a rebuilt
        Qdrant collection or BM25 index must never be masked by a stale
        cached retrieve() result for the remainder of the cache's TTL.
        """
        cleared = len(self._retrieval_cache)
        self._retrieval_cache.clear()
        logger.info("Retrieval cache invalidated (%d entries dropped)", cleared)

    def _retrieve_uncached(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        product_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        debug = bool(kwargs.get("debug", False))
        product_match_type = str(
            kwargs.get("product_match_type", "explicit_filter")
        )
        # Callers that already resolved products for this exact query text
        # (e.g. ChatbotService._handle_multi_product_fact, which loops this
        # method once per product) can pass the query embedding they've
        # already computed, instead of every retrieve() call re-embedding
        # identical query text via a fresh Ollama round trip.
        caller_precomputed_vector = kwargs.get("precomputed_vector")
        k = top_k or self.settings.final_top_k

        resolved_products = self.resolver.resolve_all(query)

        if product_id:
            target_product_ids = [product_id]
            catalog_item = self.resolver.catalog_item(product_id) or {}
            resolved_products = [
                {
                    "product_id": product_id,
                    "product_name": catalog_item.get("product_name", ""),
                    "confidence": 1.0,
                    "match_type": product_match_type,
                }
            ]
        else:
            target_product_ids = [
                item["product_id"]
                for item in resolved_products
                if item.get("confidence", 0.0) >= 0.65
            ]

        all_dense_results: List[Dict[str, Any]] = []
        all_sparse_results: List[Dict[str, Any]] = []
        fused_groups: List[List[Dict[str, Any]]] = []
        applied_filters: List[Dict[str, Any]] = []

        # Embed the query text once. When a query mentions 2-3 products, the
        # loop below issues one Qdrant search per product_id, but the query
        # vector itself never changes — only the filter does. Re-embedding it
        # per product was N redundant Ollama round-trips for identical input.
        if caller_precomputed_vector is not None:
            precomputed_vector = caller_precomputed_vector
        else:
            precomputed_vector = self.embedding.embed_query(query) if query else None

        # Multi-product retrieval: retrieve and fuse each product separately.
        if target_product_ids:
            # Build the per-product filter list first, then fire every dense
            # and every sparse call concurrently. Fusion/round-robin below is
            # unchanged and deterministic — only the *order in which the
            # underlying Qdrant/BM25 I/O executes* changes (concurrent
            # instead of one full dense+sparse round trip per product before
            # starting the next), so results are identical to before, just
            # faster for 2+ product queries (compare/multi-product-fact).
            per_product_filters: List[Dict[str, Any]] = []
            for target_pid in target_product_ids:
                req_filters = dict(filters) if filters else {}
                req_filters["product_id"] = target_pid
                per_product_filters.append(req_filters)
                applied_filters.append(req_filters)
                logger.info("Applying hard product filter for '%s'", target_pid)

            dense_futures = [
                self._io_pool.submit(
                    self.dense.retrieve,
                    query,
                    top_k=self.settings.dense_top_k,
                    filters=req_filters,
                    query_vector=precomputed_vector,
                )
                for req_filters in per_product_filters
            ]
            sparse_futures = [
                self._io_pool.submit(
                    self.sparse.retrieve,
                    query,
                    top_k=self.settings.sparse_top_k,
                    filters=req_filters,
                )
                for req_filters in per_product_filters
            ]

            for target_pid, dense_future, sparse_future in zip(
                target_product_ids, dense_futures, sparse_futures
            ):
                dense_results = dense_future.result()
                sparse_results = sparse_future.result()

                fused = rrf_fusion(
                    dense_results,
                    sparse_results,
                    rrf_k=self.settings.rrf_k,
                    target_product_id=target_pid,
                    query=query,
                )

                all_dense_results.extend(dense_results)
                all_sparse_results.extend(sparse_results)
                fused_groups.append(fused)

            if len(fused_groups) > 1:
                final_results = self._round_robin(fused_groups, k)
            else:
                filtered = self._apply_content_type_conflict_filter(
                    fused_groups[0], query
                )
                # Đặt TRƯỚC bước cắt top-k: mục được gọi đúng tên phải nằm
                # trong phần được giữ lại, không thì việc ưu tiên vô nghĩa.
                filtered = self._prefer_exactly_named_section(filtered, query)
                final_results = self._pull_in_sibling_chunks(
                    filtered[:k], filtered
                )

        # No product resolved: search the entire collection.
        else:
            req_filters = dict(filters) if filters else {}
            applied_filters.append(req_filters)

            dense_future = self._io_pool.submit(
                self.dense.retrieve,
                query,
                top_k=self.settings.dense_top_k,
                filters=req_filters or None,
                query_vector=precomputed_vector,
            )
            sparse_future = self._io_pool.submit(
                self.sparse.retrieve,
                query,
                top_k=self.settings.sparse_top_k,
                filters=req_filters or None,
            )
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()

            fused = rrf_fusion(
                dense_results,
                sparse_results,
                rrf_k=self.settings.rrf_k,
                target_product_id=None,
                query=query,
            )

            all_dense_results = dense_results
            all_sparse_results = sparse_results
            fused_groups = [fused]
            filtered = self._apply_content_type_conflict_filter(fused, query)
            filtered = self._prefer_exactly_named_section(filtered, query)
            final_results = self._pull_in_sibling_chunks(filtered[:k], filtered)

        latency_ms = (time.perf_counter() - started_at) * 1000

        resolved_product = (
            resolved_products[0]
            if resolved_products
            else {
                "product_id": "",
                "product_name": "",
                "confidence": 0.0,
                "match_type": "none",
            }
        )

        response: Dict[str, Any] = {
            "query": query,
            "resolved_product": resolved_product,
            "resolved_products": resolved_products,
            "results": final_results,
            "dense_count": len(all_dense_results),
            "sparse_count": len(all_sparse_results),
            "latency_ms": latency_ms,
        }

        if debug:
            response["debug"] = {
                "dense_candidates": all_dense_results,
                "sparse_candidates": all_sparse_results,
                "rrf_results": [
                    item
                    for group in fused_groups
                    for item in group
                ],
                "applied_filters": applied_filters,
                "target_product_ids": target_product_ids,
                "resolved_products": resolved_products,
            }

        return response
