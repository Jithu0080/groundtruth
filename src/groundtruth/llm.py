"""Minimal provider-agnostic LLM client (Anthropic or any OpenAI-compatible endpoint).

Standard library only. Keys come from the environment and are never logged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from groundtruth import GroundtruthError

TIMEOUT_S = 60
MAX_ATTEMPTS = 3
RETRYABLE = {429, 500, 502, 503, 504, 529}


class LLM(Protocol):
    def complete(self, prompt: str, max_tokens: int = 800) -> str: ...


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str | None, base_url: str | None = None):
        if provider not in {"anthropic", "openai"}:
            raise GroundtruthError("GT_LLM_PROVIDER must be 'anthropic' or 'openai'")
        if not model:
            raise GroundtruthError("set GT_LLM_MODEL to the model name to use")
        if provider == "anthropic" and not api_key:
            raise GroundtruthError("set ANTHROPIC_API_KEY in your environment")
        self.provider, self.model, self._key = provider, model, api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    @classmethod
    def from_env(cls, model_var: str = "GT_LLM_MODEL") -> LLMClient:
        provider = os.environ.get("GT_LLM_PROVIDER", "anthropic")
        model = os.environ.get(model_var) or ""
        if model_var == "GT_JUDGE_MODEL" and not model:
            model = os.environ.get("GT_LLM_MODEL", "")
            print("warning: GT_JUDGE_MODEL not set; judging with the generator model (self-preference bias)")
        key = os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
        return cls(provider, model, key, os.environ.get("OPENAI_BASE_URL"))

    def _request(self, prompt: str, max_tokens: int) -> tuple[str, dict[str, str], dict[str, Any]]:
        msgs = [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "temperature": 0, "messages": msgs}
        if self.provider == "anthropic":
            headers = {"x-api-key": self._key or "", "anthropic-version": "2023-06-01"}
            return "https://api.anthropic.com/v1/messages", headers, body
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        return f"{self.base_url}/chat/completions", headers, body

    def complete(self, prompt: str, max_tokens: int = 800) -> str:
        url, headers, body = self._request(prompt, max_tokens)
        if not url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise GroundtruthError("refusing non-HTTPS LLM endpoint (only localhost may use http)")
        data = json.dumps(body).encode()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            req = urllib.request.Request(  # noqa: S310 - scheme checked above
                url, data=data, method="POST", headers={"content-type": "application/json", **headers}
            )
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
                    payload = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                if e.code in RETRYABLE and attempt < MAX_ATTEMPTS:
                    time.sleep(2**attempt)
                    continue
                raise GroundtruthError(f"LLM call failed: HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < MAX_ATTEMPTS:
                    time.sleep(2**attempt)
                    continue
                raise GroundtruthError(f"LLM call failed: {e}") from e
        if self.provider == "anthropic":
            return "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
        return str(payload["choices"][0]["message"]["content"] or "")
