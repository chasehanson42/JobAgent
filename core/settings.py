"""Load YAML config files relative to the project root.

config/settings.yaml holds the shipped defaults. Choices made in the web app
are written to config/user_settings.yaml and config/my_profile.yaml (both
git-ignored), so updating the code never overwrites a user's data.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"
USER_SETTINGS = PROJECT_ROOT / "config" / "user_settings.yaml"
USER_PROFILE = PROJECT_ROOT / "config" / "my_profile.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(relative: str | Path) -> Path:
    """Resolve a settings path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Explicit path: that file only. Default: settings.yaml + user_settings.yaml."""
    if path:
        return _load_yaml(Path(path))
    settings = _load_yaml(DEFAULT_SETTINGS)
    if USER_SETTINGS.exists():
        settings = deep_merge(settings, _load_yaml(USER_SETTINGS))
    return settings


def load_user_overrides() -> dict[str, Any]:
    return _load_yaml(USER_SETTINGS) if USER_SETTINGS.exists() else {}


def save_user_overrides(overrides: dict[str, Any]) -> None:
    """Merge `overrides` into config/user_settings.yaml."""
    _write_yaml(USER_SETTINGS, deep_merge(load_user_overrides(), overrides))


def load_profile(settings: dict[str, Any]) -> dict[str, Any]:
    rel = settings.get("paths", {}).get("candidate_profile", "config/candidate_profile.yaml")
    return _load_yaml(resolve_path(rel))


def save_profile(profile: dict[str, Any]) -> Path:
    """Save the user's own profile and point settings at it."""
    _write_yaml(USER_PROFILE, profile)
    try:
        rel = USER_PROFILE.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = str(USER_PROFILE)
    save_user_overrides({"paths": {"candidate_profile": rel}})
    return USER_PROFILE


def has_user_profile(settings: dict[str, Any]) -> bool:
    rel = settings.get("paths", {}).get("candidate_profile", "")
    return bool(rel) and resolve_path(rel) == USER_PROFILE and USER_PROFILE.exists()
