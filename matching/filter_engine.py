"""Deterministic, zero-LLM job filtering driven by settings["filters"]."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core.models import Job

MIN_DESCRIPTION_CHARS = 50

_INTERVAL_MULTIPLIER: dict[str, float] = {
    "hourly": 2080, "hour": 2080, "hr": 2080,
    "daily": 260, "day": 260,
    "weekly": 52, "week": 52,
    "monthly": 12, "month": 12,
    "yearly": 1, "year": 1, "annual": 1, "annually": 1, "yr": 1,
}


def annualize(amount: float | None, interval: str | None) -> float | None:
    """Convert a salary amount to an annual figure. Unknown interval = yearly."""
    if amount is None:
        return None
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    key = (interval or "yearly").strip().lower()
    return value * _INTERVAL_MULTIPLIER.get(key, 1)


def best_annual_salary(job: Job) -> float | None:
    """Best-case annual salary for the job (salary_max, else salary_min)."""
    raw = job.salary_max if job.salary_max is not None else job.salary_min
    return annualize(raw, job.salary_interval)


def _norm_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(v).lower() for v in values if str(v).strip()]


def _first_match(haystack: str, needles: Iterable[str]) -> str | None:
    for needle in needles:
        if needle in haystack:
            return needle
    return None


@dataclass
class FilterDecision:
    keep: bool
    reasons: list[str] = field(default_factory=list)


class FilterEngine:
    def __init__(self, filters: dict, profile: dict | None = None) -> None:
        filters = filters or {}
        self.title_blacklist = _norm_list(filters.get("title_blacklist"))
        self.title_keywords_required = _norm_list(filters.get("title_keywords_required"))
        self.company_blacklist = _norm_list(filters.get("company_blacklist"))
        self.location_allowlist = _norm_list(filters.get("location_allowlist"))
        self.location_blacklist = _norm_list(filters.get("location_blacklist"))
        self.remote_only = bool(filters.get("remote_only", False))
        self.drop_if_salary_unknown = bool(filters.get("drop_if_salary_unknown", False))
        self.description_blacklist = _norm_list(filters.get("description_blacklist"))

        min_salary = filters.get("min_salary")
        if min_salary is None and profile:
            min_salary = (profile.get("target") or {}).get("min_salary")
        try:
            self.min_salary: float | None = float(min_salary) if min_salary is not None else None
        except (TypeError, ValueError):
            self.min_salary = None

    @staticmethod
    def _is_remote(job: Job) -> bool:
        return job.is_remote is True or "remote" in (job.location or "").lower()

    def evaluate(self, job: Job) -> FilterDecision:
        reasons: list[str] = []
        title = (job.title or "").strip()
        title_l = title.lower()
        company_l = (job.company or "").lower()
        location_l = (job.location or "").lower()
        desc = job.description or ""
        desc_l = desc.lower()
        remote = self._is_remote(job)

        if not title:
            reasons.append("empty_title")
        if len(desc.strip()) < MIN_DESCRIPTION_CHARS:
            reasons.append("description_too_short")

        hit = _first_match(title_l, self.title_blacklist)
        if hit:
            reasons.append(f"title_blacklist:{hit.strip()}")

        if self.title_keywords_required and not _first_match(title_l, self.title_keywords_required):
            reasons.append("title_missing_required_keyword")

        hit = _first_match(company_l, self.company_blacklist)
        if hit:
            reasons.append(f"company_blacklist:{hit.strip()}")

        if self.location_allowlist:
            allowed = _first_match(location_l, self.location_allowlist) is not None
            if not allowed and remote and "remote" in self.location_allowlist:
                allowed = True
            if not allowed:
                reasons.append("location_not_allowed")

        hit = _first_match(location_l, self.location_blacklist)
        if hit:
            reasons.append(f"location_blacklist:{hit.strip()}")

        if self.remote_only and not remote:
            reasons.append("not_remote")

        if self.min_salary is not None:
            annual = best_annual_salary(job)
            if annual is None:
                if self.drop_if_salary_unknown:
                    reasons.append("salary_unknown")
            elif annual < self.min_salary:
                reasons.append(f"salary_below_min:{int(annual)}<{int(self.min_salary)}")
        elif self.drop_if_salary_unknown and best_annual_salary(job) is None:
            reasons.append("salary_unknown")

        hit = _first_match(desc_l, self.description_blacklist)
        if hit:
            reasons.append(f"description_blacklist:{hit.strip()}")

        return FilterDecision(keep=not reasons, reasons=reasons)

    def apply(self, jobs: list[Job]) -> tuple[list[Job], list[tuple[Job, list[str]]]]:
        kept: list[Job] = []
        dropped: list[tuple[Job, list[str]]] = []
        for job in jobs:
            decision = self.evaluate(job)
            if decision.keep:
                kept.append(job)
            else:
                dropped.append((job, decision.reasons))
        return kept, dropped
