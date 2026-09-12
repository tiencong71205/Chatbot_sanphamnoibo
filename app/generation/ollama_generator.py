"""Ollama LLM generation client with history and streaming support."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterator, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 300.0


class _ThinkBlockFilter:
    """Strip <think>...</think> chain-of-thought spans from a stream of text
    deltas, even when a tag is split across two deltas.

    generate() already strips these post-hoc with a regex over the full
    response; a true token stream can't wait for the full response, so this
    keeps the same "never leak chain-of-thought to the user" guarantee
    delta-by-delta instead. Pure string bookkeeping — negligible cost next
    to the network I/O it sits alongside.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buf += delta
        out = []
        while True:
            if not self._in_think:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    # Keep back a tail long enough that a split "<think>"
                    # tag can't be missed across the next feed() call.
                    safe_len = max(0, len(self._buf) - (len(self._OPEN) - 1))
                    if safe_len:
                        out.append(self._buf[:safe_len])
                        self._buf = self._buf[safe_len:]
                    break
                if idx:
                    out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_think = True
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    # A split closing tag (e.g. "</th" + "ink>") must not be
                    # discarded outright — keep a tail long enough to catch
                    # it on the next feed(), same as the opening-tag case.
                    safe_len = max(0, len(self._buf) - (len(self._CLOSE) - 1))
                    self._buf = self._buf[safe_len:]
                    break
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_think = False
        return "".join(out)

    def flush(self) -> str:
        if self._in_think:
            return ""
        remaining, self._buf = self._buf, ""
        return remaining


class OllamaGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_llm_model
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT)

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build messages list with optional conversation history."""
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            # Add last few turns (to avoid context overflow)
            max_history = 6  # 3 exchanges
            for turn in history[-max_history:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a response from Ollama. Returns dict with answer and metadata.

        `max_tokens` overrides settings.ollama_num_predict for this call —
        used by ChatbotService to give simple factual questions a smaller
        generation ceiling than open-ended full-product/comparison answers,
        cutting decode time (the dominant per-request GPU cost) without
        touching what evidence gets retrieved or how it's selected.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT
        t0 = time.perf_counter()

        messages = self._build_messages(user_message, sys_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.settings.ollama_temperature,
                "num_ctx": self.settings.ollama_num_ctx,
                "num_predict": max_tokens or self.settings.ollama_num_predict,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
            "think": False,  # Disable chain-of-thought for qwen3
        }

        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        latency = (time.perf_counter() - t0) * 1000
        message = data.get("message", {})
        content = message.get("content", "")

        # Strip any <think>...</think> blocks (chain-of-thought)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        logger.info(
            "Generated response in %.0fms (%d chars) with %s",
            latency, len(content), self.model,
        )
        return {
            "answer": content,
            "latency_ms": latency,
            "model": self.model,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "eval_count": data.get("eval_count", 0),
        }

    def generate_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream text deltas from Ollama as they're generated.

        No @retry here: retrying a stream that already emitted partial
        output to the caller would duplicate visible text, so a mid-stream
        failure just propagates — the caller (ChatbotService.chat_stream)
        surfaces it as a stream error event instead of silently retrying.
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT
        messages = self._build_messages(user_message, sys_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.settings.ollama_temperature,
                "num_ctx": self.settings.ollama_num_ctx,
                "num_predict": max_tokens or self.settings.ollama_num_predict,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
            "think": False,
        }

        think_filter = _ThinkBlockFilter()
        with self._client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = data.get("message", {}).get("content", "")
                cleaned = think_filter.feed(delta)
                if cleaned:
                    yield cleaned
                if data.get("done"):
                    break
        tail = think_filter.flush()
        if tail:
            yield tail

    def check_model(self) -> bool:
        """Check if LLM model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return self.model in models
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
