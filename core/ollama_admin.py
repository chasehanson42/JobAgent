"""Small helpers for managing a local Ollama server from the web UI.

Standard library only. Errors a user can hit raise :class:`OllamaAdminError`
with a plain-English message.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import urllib.error
import urllib.request
from typing import Any, Iterator

__all__ = [
    "OllamaAdminError",
    "is_running",
    "list_models",
    "has_model",
    "pull_model",
    "recommended_model",
    "total_ram_gb",
]

SMALL_MODEL = "qwen2.5:3b-instruct"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaAdminError(Exception):
    """A problem talking to Ollama, with a message meant for end users."""


def _url(base_url: str, path: str) -> str:
    return (base_url or "http://localhost:11434").rstrip("/") + path


def _get_json(base_url: str, path: str, timeout: float) -> Any:
    with urllib.request.urlopen(_url(base_url, path), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_running(base_url: str, timeout: float = 2) -> bool:
    """True if an Ollama server answers at ``base_url``."""
    for path in ("/api/version", "/api/tags"):
        try:
            _get_json(base_url, path, timeout)
            return True
        except Exception:
            continue
    return False


def list_models(base_url: str, timeout: float = 5) -> list[str]:
    """Names of locally installed models; [] if Ollama can't be reached."""
    try:
        data = _get_json(base_url, "/api/tags", timeout)
        return [m["name"] for m in data.get("models", [])
                if isinstance(m, dict) and isinstance(m.get("name"), str)]
    except Exception:
        return []


def _names_for(model: str) -> set[str]:
    model = (model or "").strip()
    if not model:
        return set()
    if ":" not in model.rsplit("/", 1)[-1]:
        return {model, model + ":latest"}
    if model.endswith(":latest"):
        return {model, model[: -len(":latest")]}
    return {model}


def has_model(base_url: str, model: str) -> bool:
    """True if ``model`` is installed (a missing tag means ``:latest``)."""
    wanted = _names_for(model)
    return any(name in wanted for name in list_models(base_url))


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def pull_model(base_url: str, model: str, timeout: float = 60) -> Iterator[dict]:
    """Download ``model``, yielding progress dicts as Ollama streams them.

    Each item is ``{"status": str, "completed": int | None, "total": int | None}``.
    ``timeout`` applies to each network read, not the whole download.
    """
    body = json.dumps({"model": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(_url(base_url, "/api/pull"), data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:
            pass
        msg = f"Ollama couldn't download the model \"{model}\""
        raise OllamaAdminError(msg + (f": {detail}" if detail else ".") +
                               " Check the model name and your internet connection.") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise OllamaAdminError("We couldn't reach Ollama. Make sure Ollama is installed and "
                               "running, then try again.") from exc

    with resp:
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("error"):
                    raise OllamaAdminError(f"Ollama couldn't download the model \"{model}\": "
                                           f"{item['error']}")
                yield {"status": str(item.get("status", "")),
                       "completed": _int_or_none(item.get("completed")),
                       "total": _int_or_none(item.get("total"))}
        except OllamaAdminError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            raise OllamaAdminError("The model download was interrupted. Check your internet "
                                   "connection and try again; it will resume where it "
                                   "left off.") from exc


def recommended_model(total_ram_gb: float | None) -> str:
    """Pick a model that fits comfortably in this machine's memory."""
    if total_ram_gb is not None and total_ram_gb < 8:
        return SMALL_MODEL
    return DEFAULT_MODEL


def _ram_windows() -> float | None:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
        return None
    return stat.ullTotalPhys / 1024 ** 3


def _ram_linux() -> float | None:
    with open("/proc/meminfo", encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 ** 2  # kB -> GiB
    return None


def _ram_sysconf() -> float | None:
    pages = os.sysconf("SC_PHYS_PAGES")
    size = os.sysconf("SC_PAGE_SIZE")
    if pages > 0 and size > 0:
        return pages * size / 1024 ** 3
    return None


def _ram_sysctl() -> float | None:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                         timeout=5, check=True).stdout.strip()
    return int(out) / 1024 ** 3


def total_ram_gb() -> float | None:
    """Total physical memory in GiB, or None if it can't be determined."""
    system = platform.system()
    if system == "Windows":
        probes = [_ram_windows]
    elif system == "Linux":
        probes = [_ram_linux, _ram_sysconf]
    elif system == "Darwin":
        probes = [_ram_sysconf, _ram_sysctl]
    else:
        probes = [_ram_sysconf]
    for probe in probes:
        try:
            value = probe()
        except Exception:
            continue
        if value and value > 0:
            return round(value, 1)
    return None
