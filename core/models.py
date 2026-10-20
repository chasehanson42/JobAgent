"""Shared data models passed between pipeline stages."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


def make_job_id(company: str, title: str, url: str) -> str:
    key = f"{company.strip().lower()}|{title.strip().lower()}|{url.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class Job:
    id: str
    title: str
    company: str
    location: str = ""
    url: str = ""
    source: str = ""
    description: str = ""
    is_remote: bool | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_interval: str | None = None   # yearly, monthly, hourly ...
    date_posted: str | None = None
    scraped_at: str = ""
    # Set by the sanitizer: names of injection/malware patterns that were removed.
    sanitizer_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FitResult:
    job_id: str
    fit_score: int
    matched_skills: list[str]
    skill_gaps: list[str]
    recommendation: Literal["Apply", "Skip"]
    rationale: str = ""
    model: str = ""
    evaluated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FitResult":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})
