"""Fast, deterministic keyword pre-screen (no LLM).

Scoring a job with the local LLM is slow, so before that step every scraped
job gets a cheap relevance score computed from the candidate profile's target
titles, headline, past job titles and skills. Jobs that share nothing with the
profile (e.g. a software role for a nurse) can then be skipped.

Usage::

    terms = profile_terms(profile)          # once per run
    score = relevance(job, terms)           # per job, pure Python
    if is_relevant(score, terms, min_score=2): ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.models import Job

__all__ = ["ProfileTerms", "profile_terms", "relevance", "is_relevant", "MAX_RELEVANCE"]

MAX_RELEVANCE = 50.0
TITLE_HIT = 3.0
DESCRIPTION_TITLE_HIT = 1.0
SKILL_HIT = 1.0

# Stopwords and generic seniority / employment filler that say nothing about
# the kind of work. Role nouns ("manager", "engineer", "nurse") are kept.
_TITLE_STOPWORDS = frozenset({
    "a", "an", "and", "or", "of", "the", "for", "in", "at", "to", "on", "with",
    "by", "as", "from", "into", "my", "our", "your", "who", "focused",
    "senior", "junior", "lead", "sr", "jr", "i", "ii", "iii", "iv", "v",
    "mid", "level", "entry", "principal", "staff", "head", "chief",
    "remote", "hybrid", "onsite", "full", "part", "time", "contract",
    "temporary", "permanent", "intern", "internship", "trainee",
})

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9+#]*")


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    # Word-boundary aware even for phrases that start/end with symbols
    # ("c++", ".net", "ci/cd"): require no word character on either side.
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")


@dataclass
class ProfileTerms:
    title_terms: list[str] = field(default_factory=list)
    skill_terms: list[str] = field(default_factory=list)
    # Precompiled, matched against lowercased text.
    title_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list, repr=False)
    skill_patterns: list[tuple[str, re.Pattern[str]]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.title_patterns:
            self.title_patterns = [(t, _phrase_pattern(t)) for t in self.title_terms]
        if not self.skill_patterns:
            self.skill_patterns = [(s, _phrase_pattern(s)) for s in self.skill_terms]

    @property
    def is_empty(self) -> bool:
        return not self.title_terms and not self.skill_terms


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _title_words(text: str) -> list[str]:
    out = []
    for word in _WORD_RE.findall(str(text or "").lower()):
        if len(word) >= 2 and word not in _TITLE_STOPWORDS:
            out.append(word)
    return out


def _collect_skills(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _collect_skills(v, out)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            _collect_skills(v, out)
    elif value is not None:
        out.append(str(value))


def profile_terms(profile: dict) -> ProfileTerms:
    """Extract title words and skill phrases from a candidate profile."""
    profile = profile or {}
    title_sources: list[Any] = []
    target = profile.get("target")
    if isinstance(target, dict):
        title_sources.extend(_as_list(target.get("titles")))
    title_sources.append(profile.get("headline"))
    for exp in _as_list(profile.get("experience")):
        if isinstance(exp, dict):
            title_sources.append(exp.get("title"))

    title_terms: list[str] = []
    seen: set[str] = set()
    for src in title_sources:
        if src is None or isinstance(src, (dict, list)):
            continue
        for word in _title_words(str(src)):
            if word not in seen:
                seen.add(word)
                title_terms.append(word)

    raw_skills: list[str] = []
    _collect_skills(profile.get("skills"), raw_skills)
    skill_terms: list[str] = []
    seen = set()
    for skill in raw_skills:
        phrase = " ".join(skill.lower().split())
        if phrase and phrase not in seen:
            seen.add(phrase)
            skill_terms.append(phrase)

    return ProfileTerms(title_terms=title_terms, skill_terms=skill_terms)


def relevance(job: Job, terms: ProfileTerms) -> float:
    """Deterministic keyword relevance of ``job`` to the profile, 0..50."""
    title = str(job.title or "").lower()
    description = str(job.description or "").lower()
    score = 0.0
    for _, pat in terms.title_patterns:
        if pat.search(title):
            score += TITLE_HIT
        elif pat.search(description):
            score += DESCRIPTION_TITLE_HIT
        if score >= MAX_RELEVANCE:
            return MAX_RELEVANCE
    for _, pat in terms.skill_patterns:
        if pat.search(title) or pat.search(description):
            score += SKILL_HIT
            if score >= MAX_RELEVANCE:
                return MAX_RELEVANCE
    return min(score, MAX_RELEVANCE)


def is_relevant(score: float, terms: ProfileTerms, min_score: float) -> bool:
    """True if the job clears ``min_score`` (always True for an empty profile)."""
    if terms.is_empty:
        return True
    return score >= min_score
