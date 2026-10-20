"""Shared prompt-building helpers for the document generators."""
from __future__ import annotations

import re
from typing import Any

import yaml

from core.models import Job
from scrapers.sanitizer import sanitize_text, wrap_untrusted

# Keys in the profile that are not candidate facts (job-search preferences).
_NON_FACT_KEYS = {"target"}


def render_profile(profile: dict[str, Any]) -> str:
    """Render the master profile as compact YAML inside <candidate_profile> tags."""
    facts = {k: v for k, v in profile.items() if k not in _NON_FACT_KEYS}
    body = yaml.safe_dump(facts, sort_keys=False, allow_unicode=True, width=1000).strip()
    return (
        "<candidate_profile>\n"
        "# This profile is the ONLY source of truth about the candidate. Every fact in\n"
        "# the output (employers, titles, dates, degrees, certifications, numbers,\n"
        "# tools, skills) must come from here.\n"
        f"{body}\n"
        "</candidate_profile>"
    )


def render_job(job: Job, max_chars: int = 8000) -> str:
    """Sanitize job fields and wrap them as untrusted data for a prompt."""
    title = sanitize_text(job.title or "", max_chars=300).text
    company = sanitize_text(job.company or "", max_chars=200).text
    location = sanitize_text(job.location or "", max_chars=200).text
    description = sanitize_text(job.description or "", max_chars=max_chars).text
    text = (
        f"Title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Description:\n{description}"
    )
    return wrap_untrusted(text, tag="job_description")


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)
_CHATTER_RE = re.compile(
    r"^\s*(?:sure[,!.]?\s*|certainly[,!.]?\s*|of course[,!.]?\s*|okay[,!.]?\s*)?"
    r"(?:here(?:'s| is| are)\b[^\n]*:|below is\b[^\n]*:)\s*\n",
    re.IGNORECASE,
)


def strip_fences(text: str) -> str:
    """Remove leading chatter ("Here is your resume:") and one surrounding code fence."""
    out = (text or "").strip()
    for _ in range(2):
        before = out
        m = _CHATTER_RE.match(out + "\n")
        if m:
            out = (out + "\n")[m.end():].strip()
        m = _FENCE_RE.match(out)
        if m:
            out = m.group(1).strip()
        if out == before:
            break
    return out


def profile_skill_list(profile: dict[str, Any]) -> list[str]:
    """Flatten profile skills (dict of categories or plain list) into one list."""
    skills = profile.get("skills") or []
    flat: list[str] = []
    if isinstance(skills, dict):
        for vals in skills.values():
            if isinstance(vals, (list, tuple)):
                flat.extend(str(v) for v in vals)
            elif vals:
                flat.append(str(vals))
    elif isinstance(skills, (list, tuple)):
        flat.extend(str(v) for v in skills)
    return flat


def contact_line(profile: dict[str, Any]) -> str:
    parts = [str(profile.get(k)) for k in ("location", "email", "phone") if profile.get(k)]
    parts.extend(str(link) for link in (profile.get("links") or []))
    return " | ".join(parts)
