"""Cut a job posting down to what decides fit, for the quick first-pass rating.

On a laptop the AI spends most of its time *reading*, so the first pass sees
only the opening lines plus the lines that state duties and requirements.
Benefits, equal-opportunity text, "about us" and application instructions
are dropped. The full posting is still used for "Why you fit" and documents.
"""
from __future__ import annotations

import re

_REQUIREMENT = re.compile(
    r"\b(require|qualif|must|experience|years?|degree|diploma|licen[cs]e|certif|skills?|"
    r"proficien|knowledge|ability|able to|familiar|responsib|you will|duties|preferred|"
    r"background|training|bachelor|associate|master|clearance)", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"(equal opportunity|eeo\b|affirmative action|without regard to|veteran|disabilit|"
    r"401\s*\(?k|dental insurance|vision insurance|health insurance|paid time off|\bpto\b|"
    r"benefits? (include|package)|we offer|perks|reasonable accommodation|e-?verify|"
    r"background check|drug (test|screen)|privacy (policy|notice)|how to apply|click apply|"
    r"apply now|about (us|the company)|our mission|follow us)", re.IGNORECASE)
_BULLET = re.compile(r"^\s*([-*•·▪●◦]|\d+[.)])\s+")


def job_essentials(description: str, max_chars: int = 1500, intro_chars: int = 300) -> str:
    text = (description or "").strip()
    if len(text) <= max_chars:
        return text
    lines = [ln.strip() for ln in re.split(r"\n+|(?<=[.!?])\s{2,}", text) if ln.strip()]
    intro, used = [], 0
    for i, ln in enumerate(lines):
        if used >= intro_chars or _BOILERPLATE.search(ln):
            break
        intro.append(i)
        used += len(ln)

    scored = []
    for i, ln in enumerate(lines):
        if i in intro or _BOILERPLATE.search(ln):
            continue
        score = len(_REQUIREMENT.findall(ln)) * 2 + (1 if _BULLET.match(ln) else 0)
        if score:
            scored.append((score, i))

    keep = set(intro)
    budget = max_chars - sum(len(lines[i]) + 1 for i in intro)
    for _, i in sorted(scored, key=lambda x: (-x[0], x[1])):
        if len(lines[i]) + 1 > budget:
            continue
        keep.add(i)
        budget -= len(lines[i]) + 1
    out = "\n".join(lines[i] for i in sorted(keep))
    return out[:max_chars]
