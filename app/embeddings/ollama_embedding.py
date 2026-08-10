"""Ollama embedding client with batching, retry, and dimension validation."""
from __future__ import annotations
import logging
import time
from typing import List

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import Settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 8
REQUEST_TIMEOUT = 180.0


class EmbeddingDimensionError(Exception):
    pass


class OllamaEmbedding:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_embedding_model
        self.expected_dim = settings.embedding_dimension
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text with timeout=180s and max 3 retries."""
        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") or data.get("embedding")
        if isinstance(embeddings[0], list):
            vec = embeddings[0]
        else:
            vec = embeddings
        if self.expected_dim and len(vec) != self.expected_dim:
            logger.warning(
                f"Dimension mismatch: expected {self.expected_dim}, got {len(vec)} from model {self.model}"
            )
        return vec

    def embed_batch(self, texts: List[str]) -> tuple[List[List[float]], int]:
        """
        Embed a list of texts in batches of 8.
        Logs every batch step clearly.
        Returns (vectors, error_count).
        """
        if not texts:
            return [], 0
        results: List[List[float]] = []
        total = len(texts)
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        error_count = 0

        for b_idx in range(total_batches):
            i = b_idx * BATCH_SIZE
            batch = texts[i : i + BATCH_SIZE]
            logger.info("Embedding batch %d/%d (%d texts)...", b_idx + 1, total_batches, len(batch))
            
            for j, text in enumerate(batch):
                try:
                    vec = self.embed_single(text)
                    results.append(vec)
                except Exception as e:
                    error_count += 1
                    logger.error("Embedding failed for item %d: %s", i + j, e, exc_info=True)
                    # Zero vector placeholder or skip
                    results.append([])
            logger.info("Embedding batch %d/%d completed", b_idx + 1, total_batches)

        return results, error_count

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_single(query)

    def close(self) -> None:
        self._client.close()
