"""vLLM OpenAI-compatible generation client."""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.generation.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 300.0


class VLLMGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.vllm_base_url
        self.model = settings.vllm_llm_model
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT)

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            max_history = 6
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
        sys_prompt = system_prompt or SYSTEM_PROMPT
        messages = self._build_messages(user_message, sys_prompt, history)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.ollama_temperature,
            "top_p": 0.9,
            "max_tokens": self.settings.ollama_num_predict,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        }

        t0 = time.perf_counter()

        resp = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        latency = (time.perf_counter() - t0) * 1000

        choices = data.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "") or ""

        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

        usage = data.get("usage", {})

        logger.info(
            "Generated response via vLLM in %.0fms (%d chars) with %s",
            latency,
            len(content),
            self.model,
        )

        return {
            "answer": content,
            "latency_ms": latency,
            "model": self.model,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    def check_model(self) -> bool:
        try:
            resp = self._client.get(
                f"{self.base_url}/v1/models",
                timeout=5.0,
            )
            resp.raise_for_status()

            models = [
                m.get("id", "")
                for m in resp.json().get("data", [])
            ]
            return self.model in models
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
