"""Write a three-paragraph cover letter grounded only in the master profile."""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any

from core.models import FitResult, Job
from generators._prompting import contact_line, render_job, render_profile, strip_fences
from scrapers.sanitizer import UNTRUSTED_DATA_POLICY, sanitize_text

WORD_CAP = 350

_SYSTEM = """You are an expert cover letter writer.

Write EXACTLY three body paragraphs separated by one blank line:
1. Hook: the role and why this company / role is a fit.
2. Evidence: concrete, verified accomplishments from the candidate's experience.
3. Close: enthusiasm and a call to action.

RULES (non-negotiable):
- Use ONLY facts in <candidate_profile>. NEVER add employers, titles, dates,
  degrees, certifications, metrics/numbers, tools or skills that are not in it.
- Skill gaps must not be claimed.
- Do not include a greeting, date, address block, sign-off or signature; only
  the three paragraphs. No headings, bullets, tables, emojis or placeholders
  like [Company].
- At most {cap} words in total.

{policy}
"""

_GREETING_RE = re.compile(r"^(dear\b[^\n]*|to whom it may concern[^\n]*)$", re.IGNORECASE)
_SIGNOFF_RE = re.compile(r"^(sincerely|best regards|regards|kind regards|thank you)[,.!]?\s*$",
                         re.IGNORECASE)


def _paragraphs(text: str) -> list[str]:
    paras = []
    for block in re.split(r"\n\s*\n", strip_fences(text)):
        lines = block.strip().split("\n")
        first_line = lines[0].strip()
        if _GREETING_RE.match(first_line):
            lines = lines[1:]  # drop "Dear ...," but keep any text after it
        block = " ".join(" ".join(lines).split())
        if not block or block.startswith("#"):
            continue
        if _SIGNOFF_RE.match(first_line):
            break  # everything from the sign-off on is the signature
        paras.append(block)
    return paras


def _cap_words(paras: list[str], cap: int) -> list[str]:
    total = sum(len(p.split()) for p in paras)
    if total <= cap:
        return paras
    # Trim whole sentences from the longest paragraph until under the cap.
    paras = list(paras)
    while sum(len(p.split()) for p in paras) > cap:
        idx = max(range(len(paras)), key=lambda i: len(paras[i].split()))
        sentences = re.split(r"(?<=[.!?])\s+", paras[idx])
        if len(sentences) > 1:
            paras[idx] = " ".join(sentences[:-1])
        else:
            words = paras[idx].split()
            excess = sum(len(p.split()) for p in paras) - cap
            paras[idx] = " ".join(words[: max(1, len(words) - excess)])
    return paras


class CoverLetterWriter:
    def __init__(self, client: Any, profile: dict[str, Any]) -> None:
        self.client = client
        self.profile = profile

    def build_prompt(self, job: Job, fit: FitResult | None = None) -> tuple[str, str]:
        system = _SYSTEM.format(cap=WORD_CAP, policy=UNTRUSTED_DATA_POLICY)
        parts = [render_profile(self.profile), "", render_job(job), ""]
        if fit is not None:
            if fit.matched_skills:
                parts.append("Emphasize these verified matching skills: "
                             + ", ".join(fit.matched_skills))
            if fit.skill_gaps:
                parts.append("Skill gaps - do NOT claim these: " + ", ".join(fit.skill_gaps))
        parts.append("Write the three paragraphs now.")
        return system, "\n".join(parts)

    def write(self, job: Job, fit: FitResult | None = None, *,
              today: _dt.date | None = None) -> str:
        system, user = self.build_prompt(job, fit)
        paras = _paragraphs(self.client.chat(system, user))
        if len(paras) < 3:
            retry_user = (user + "\n\nYour previous answer did not contain exactly three "
                          "paragraphs separated by blank lines. Rewrite it as exactly three.")
            retry = _paragraphs(self.client.chat(system, retry_user))
            if len(retry) >= len(paras):
                paras = retry
        if len(paras) > 3:
            paras = paras[:2] + [" ".join(paras[2:])]
        paras = _cap_words(paras, WORD_CAP)
        return self.assemble(job, paras, today=today)

    def assemble(self, job: Job, paras: list[str], *, today: _dt.date | None = None) -> str:
        p = self.profile
        name = str(p.get("name", "")).strip()
        date = (today or _dt.date.today()).strftime("%B %d, %Y").replace(" 0", " ")
        company = " ".join(sanitize_text(job.company or "", max_chars=200).text.split())
        lines = [f"# {name}"]
        contact = contact_line(p)
        if contact:
            lines.append(contact)
        lines += ["", date, "", f"Hiring Team, {company}" if company else "Hiring Team", ""]
        for para in paras:
            lines += [para, ""]
        lines += ["Sincerely,", name]
        return "\n".join(lines) + "\n"
