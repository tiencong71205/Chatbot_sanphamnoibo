"""Ollama LLM generation client with history and streaming support."""
from __future__ import annotations
import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import Settings
from app.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 300.0


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
    ) -> Dict[str, Any]:
        """Generate a response from Ollama. Returns dict with answer and metadata."""
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
                "num_predict": self.settings.ollama_num_predict,
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
