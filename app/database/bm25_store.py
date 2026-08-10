"""BM25 index store with Vietnamese tokenizer and technical token preservation."""
from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import List, Optional

import joblib
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.ingestion.semantic_chunker import Chunk

logger = logging.getLogger(__name__)

# Technical token pattern — preserve these during tokenization
_TECH_RE = re.compile(
    r"(?:"
    r"[A-Z]{2,}[-][A-Z0-9]+(?:[-][A-Z0-9]+)*"  # VCN-WSRGC2, Stop NC
    r"|\d{1,3}[-]\d{1,3}V"                     # 110-240V
    r"|\d{2,3}[-]\d{2}Hz"                       # 50-60Hz
    r"|[A-Z]{1,3}\d{3,5}"                        # CR2450
    r"|FW\d+\.\d+\.\d+"                      # FW2.3.0
    r"|v?\d+\.\d+\.\d+"                      # 2.3.0
    r"|Bluetooth\s+Mesh"                          # Bluetooth Mesh
    r")",
    re.IGNORECASE,
)

_SPLIT_RE = re.compile(r"[\s,;:.!?()\[\]{}/\\]+")


def _try_underthesea(text: str) -> Optional[List[str]]:
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").split()
    except ImportError:
        return None
    except Exception as e:
        logger.debug("underthesea failed: %s", e)
        return None


def tokenize_vi(text: str) -> List[str]:
    """
    Vietnamese tokenizer:
    1. Extract and preserve technical tokens
    2. Try underthesea word tokenizer
    3. Fallback: regex split
    4. Lowercase non-technical tokens
    """
    # Extract technical tokens and replace with placeholders
    tech_tokens: List[str] = []
    placeholders: dict[str, str] = {}

    def replace_tech(m: re.Match) -> str:
        tok = m.group(0)
        ph = f"TECH_{len(tech_tokens)}"
        tech_tokens.append(tok)
        placeholders[ph] = tok
        return f" {ph} "

    processed = _TECH_RE.sub(replace_tech, text)

    # Try Vietnamese tokenizer
    tokens = _try_underthesea(processed) or _SPLIT_RE.split(processed)

    result: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in placeholders:
            result.append(placeholders[tok])  # keep original case
        elif tok.startswith("TECH_"):
            # Remaining tech placeholders
            idx_str = tok[5:]
            if idx_str.isdigit() and int(idx_str) < len(tech_tokens):
                result.append(tech_tokens[int(idx_str)])
            else:
                result.append(tok.lower())
        else:
            result.append(tok.lower())

    return [t for t in result if len(t) > 0]


class BM25Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_path = Path(settings.bm25_index_path)
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[dict] = []  # Stored chunk metadata
        self._corpus: List[List[str]] = []

    def build_index(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from chunks."""
        logger.info("Building BM25 index from %d chunks", len(chunks))
        self._chunks = [c.to_dict() for c in chunks]
        self._corpus = [tokenize_vi(c.content) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index built with vocabulary of %d terms", len(self._bm25.idf))

    def save_index(self) -> None:
        """Save BM25 index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
            "corpus": self._corpus,
        }
        joblib.dump(payload, self.index_path)
        logger.info("BM25 index saved to %s", self.index_path)

    def load_index(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if not self.index_path.exists():
            logger.warning("BM25 index not found at %s", self.index_path)
            return False
        try:
            payload = joblib.load(self.index_path)
            self._bm25 = payload["bm25"]
            self._chunks = payload["chunks"]
            self._corpus = payload.get("corpus", [])
            logger.info("BM25 index loaded: %d documents", len(self._chunks))
            return True
        except Exception as e:
            logger.error("Failed to load BM25 index: %s", e)
            return False

    def search(self, query: str, top_k: int = 15) -> List[dict]:
        """Search BM25 index. Returns list of result dicts with scores."""
        if self._bm25 is None:
            if not self.load_index():
                logger.warning("BM25 index not available for search")
                return []

        tokens = tokenize_vi(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # Get top-k indices sorted by score descending
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(indexed[:top_k]):
            if score <= 0:
                break
            chunk_dict = self._chunks[idx].copy()
            results.append({
                "chunk": chunk_dict,
                "score": float(score),
                "rank": rank,
            })
        return results

    def is_loaded(self) -> bool:
        return self._bm25 is not None and len(self._chunks) > 0
