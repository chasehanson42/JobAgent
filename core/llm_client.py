"""Zero-cost LLM client: local Ollama or Gemini's free tier.

Both backends expose the same `chat()` call so the rest of the app never
cares which one is configured. Only the standard library is used for HTTP,
so there is no SDK to install or pay for.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None) -> str: ...


class CallStats:
    """Where the AI's time goes: reading the prompt vs. writing the answer (Ollama reports both)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.prompt_seconds = 0.0
        self.output_tokens = 0
        self.output_seconds = 0.0
        self.wall_seconds = 0.0

    def add(self, data: dict[str, Any], wall: float) -> None:
        with self._lock:
            self.calls += 1
            self.wall_seconds += wall
            self.prompt_tokens += int(data.get("prompt_eval_count") or 0)
            self.prompt_seconds += (data.get("prompt_eval_duration") or 0) / 1e9
            self.output_tokens += int(data.get("eval_count") or 0)
            self.output_seconds += (data.get("eval_duration") or 0) / 1e9

    def summary(self) -> str:
        if not self.calls:
            return ""
        parts = [f"{self.calls} AI call(s), {self.wall_seconds / self.calls:.0f}s each on average"]
        if self.prompt_seconds and self.output_seconds:
            parts.append(f"reading {self.prompt_tokens / self.prompt_seconds:.0f} words/s, "
                         f"writing {self.output_tokens / self.output_seconds:.1f} words/s")
        return " · ".join(parts)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str],
               timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Request to {url} timed out after {timeout:.0f}s") from exc


def _with_retries(fn, retries: int):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except LLMError as exc:
            last = exc
            if "timed out" in str(exc):
                raise  # retrying a slow model just multiplies the wait
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise LLMError(f"LLM call failed after {retries + 1} attempts: {last}")


@dataclass(eq=False)  # compared/hashed by identity, so clients can be dict keys
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    timeout: float = 180
    max_retries: int = 2
    num_ctx: int = 8192
    keep_alive: str = "30m"
    name: str = "ollama"
    stats: CallStats = field(default_factory=CallStats, repr=False, compare=False)

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": self.temperature if temperature is None else temperature,
                        "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        url = self.base_url.rstrip("/") + "/api/chat"
        t0 = time.monotonic()
        data = _with_retries(lambda: _post_json(url, payload, {}, self.timeout), self.max_retries)
        self.stats.add(data, time.monotonic() - t0)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Ollama response: {str(data)[:300]}") from exc


@dataclass(eq=False)
class GeminiFreeTierClient:
    """Gemini via its OpenAI-compatible endpoint (free developer tier)."""
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.2
    timeout: float = 120
    max_retries: int = 2
    min_interval: float = 4.5
    name: str = "gemini"
    stats: CallStats = field(default_factory=CallStats, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _throttle(self) -> None:
        # Stay within free-tier RPM limits so we never get pushed to billing.
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float | None = None, max_tokens: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        def call() -> dict[str, Any]:
            self._throttle()
            return _post_json(url, payload, headers, self.timeout)

        t0 = time.monotonic()
        data = _with_retries(call, self.max_retries)
        self.stats.add({}, time.monotonic() - t0)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response: {str(data)[:300]}") from exc


def build_client(settings: dict[str, Any], provider: str | None = None,
                 purpose: str = "general") -> LLMClient:
    """Create the configured client. `provider` overrides settings.yaml.

    purpose="scoring" uses llm.ollama.scoring_model when one is set: a smaller,
    faster model for the quick first-pass ratings.
    """
    llm = settings.get("llm", {})
    provider = (provider or os.environ.get("FJA_LLM_PROVIDER") or llm.get("provider", "ollama")).lower()
    common = {
        "temperature": float(llm.get("temperature", 0.2)),
        "timeout": float(llm.get("timeout_seconds", 180)),
        "max_retries": int(llm.get("max_retries", 2)),
    }
    if provider == "ollama":
        cfg = llm.get("ollama", {})
        model = cfg.get("model", "qwen2.5:7b-instruct")
        if purpose == "scoring" and cfg.get("scoring_model"):
            model = cfg["scoring_model"]
        return OllamaClient(base_url=cfg.get("base_url", "http://localhost:11434"),
                            model=model,
                            num_ctx=int(cfg.get("num_ctx", 8192)),
                            keep_alive=str(cfg.get("keep_alive", "30m")), **common)
    if provider == "gemini":
        cfg = llm.get("gemini", {})
        env = cfg.get("api_key_env", "GEMINI_API_KEY")
        key = os.environ.get(env, "").strip()
        if not key:
            raise LLMError(f"Gemini selected but ${env} is not set. Get a free key at "
                           "https://aistudio.google.com/apikey or switch provider to ollama.")
        return GeminiFreeTierClient(api_key=key,
                                    base_url=cfg.get("base_url", GeminiFreeTierClient.base_url),
                                    model=cfg.get("model", "gemini-2.0-flash"),
                                    min_interval=float(cfg.get("min_seconds_between_calls", 4.5)),
                                    **common)
    raise LLMError(f"Unknown LLM provider {provider!r}; use 'ollama' or 'gemini'.")
