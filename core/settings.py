"""Load YAML config files relative to the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def resolve_path(relative: str | Path) -> Path:
    """Resolve a settings path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    return _load_yaml(Path(path) if path else DEFAULT_SETTINGS)


def load_profile(settings: dict[str, Any]) -> dict[str, Any]:
    rel = settings.get("paths", {}).get("candidate_profile", "config/candidate_profile.yaml")
    return _load_yaml(resolve_path(rel))
