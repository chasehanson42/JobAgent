"""Turn the AI's quick ratings plus plain facts into one score with a visible breakdown.

A small local model asked for a single 0-100 number tends to answer with the
same number for every decent job (users saw 78 everywhere). Here the AI only
gives two coarse 1-5 ratings, which small models handle well, and the rest of
the score comes from facts that need no AI: how well the title matches, how
far the job is from home, and whether it's a small local employer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.models import Job

DEFAULT_WEIGHTS = {"skills": 35, "experience": 25, "title": 15, "location": 15, "local": 10}
_WORD = re.compile(r"[a-z0-9+#]+")
_FILLER = {"and", "of", "the", "a", "an", "to", "for", "in", "senior", "junior", "sr", "jr", "lead",
           "i", "ii", "iii", "iv", "level", "entry", "mid", "remote", "hybrid", "full", "part", "time"}


@dataclass
class ScoreParts:
    skills: int                    # AI rating 1-5
    experience: int                # AI rating 1-5
    title: float                   # 0-1
    location: float                # 0-1
    local: float                   # 0-1
    location_label: str = ""
    employer_label: str = ""


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _FILLER}


def title_match(job_title: str, target_titles: list[str]) -> float:
    """Best share of a wanted title's words that appear in the job title."""
    job_words = _words(job_title)
    best = 0.0
    for t in target_titles or []:
        want = _words(t)
        if want:
            best = max(best, len(want & job_words) / len(want))
    return best


def weights_for(settings: dict[str, Any], profile: dict[str, Any]) -> dict[str, float]:
    w = {**DEFAULT_WEIGHTS, **(settings.get("ranking", {}).get("weights") or {})}
    if not (profile.get("target") or {}).get("prefer_local", True):
        w["local"] = 0
    return {k: float(v) for k, v in w.items()}


def combine(parts: ScoreParts, weights: dict[str, float]) -> tuple[int, dict[str, float]]:
    """Return (score 0-100, points per part) — the points are shown on each match."""
    shares = {"skills": (parts.skills - 1) / 4, "experience": (parts.experience - 1) / 4,
              "title": parts.title, "location": parts.location, "local": parts.local}
    total_w = sum(weights.values()) or 1.0
    points = {k: round(100 * weights[k] * max(0.0, min(1.0, v)) / total_w, 1) for k, v in shares.items()}
    return max(0, min(100, round(sum(points.values())))), points


def recommend(score: int, parts: ScoreParts, min_fit_score: int) -> str:
    # Being close to home can lift a decent job, but never a job you aren't qualified for.
    if parts.skills <= 2:
        return "Skip"
    return "Apply" if score >= min_fit_score else "Skip"


def location_parts(job: Job, profile: dict[str, Any]) -> tuple[float, float, str, str]:
    """(location 0-1, local employer 0-1, location label, employer label)."""
    try:
        from matching.location import local_employer_score, location_fit
    except ImportError:  # location support not installed: stay neutral
        return 0.5, 0.0, "", ""
    target = profile.get("target") or {}
    fit = location_fit(profile.get("location", ""), job.location, job.is_remote,
                       commute_miles=float(target.get("commute_miles") or 25),
                       remote_ok=bool(target.get("remote_ok", True)))
    local, employer_label = local_employer_score(job, fit)
    return fit.score, local, fit.label, employer_label
