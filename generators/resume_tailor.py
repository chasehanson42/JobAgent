"""Tailor the master profile into an ATS-friendly Markdown resume for one job.

Only the Summary, Skills and Experience bullets are written by the LLM; the
header, job titles, dates, education and certifications are rendered
deterministically from the profile so they can never be fabricated.
"""
from __future__ import annotations

import logging
from typing import Any

from core.jsonutil import extract_json_object
from core.models import FitResult, Job
from generators._prompting import contact_line, profile_skill_list, render_job, render_profile
from scrapers.sanitizer import UNTRUSTED_DATA_POLICY

log = logging.getLogger(__name__)

_SYSTEM = """You are an expert resume writer producing ATS-friendly resumes.

RULES (non-negotiable):
- Use ONLY facts found in <candidate_profile>. You may rewrite, reorder and
  emphasize those facts toward the job's keywords.
- NEVER add employers, job titles, dates, degrees, certifications,
  metrics/numbers, tools or skills that are not in the profile. Keep every
  number exactly as written in the profile; do not invent new ones.
- Skills listed as gaps must NOT be claimed anywhere.
- Plain text only: no tables, columns, images, emojis or markdown inside values.
- Bullets start with a strong verb and stay to one sentence.

{policy}

Return ONLY a JSON object with this shape:
{{"summary": "2-3 sentence summary",
  "skills": ["skill", ...],
  "experience": [{{"company": "exact company name from profile", "bullets": ["...", ...]}}]}}
"""


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


class ResumeTailor:
    def __init__(self, client: Any, profile: dict[str, Any]) -> None:
        self.client = client
        self.profile = profile

    # ---------------------------------------------------------------- prompt
    def build_prompt(self, job: Job, fit: FitResult | None = None) -> tuple[str, str]:
        system = _SYSTEM.format(policy=UNTRUSTED_DATA_POLICY)
        parts = [render_profile(self.profile), "", render_job(job), ""]
        if fit is not None:
            if fit.matched_skills:
                parts.append("Emphasize these verified matching skills: "
                             + ", ".join(fit.matched_skills))
            if fit.skill_gaps:
                parts.append("Skill gaps - do NOT claim these anywhere: "
                             + ", ".join(fit.skill_gaps))
        companies = [e.get("company", "") for e in self.profile.get("experience") or []]
        parts.append("Write bullets for each of these companies: " + "; ".join(companies))
        parts.append("Return the JSON object now.")
        return system, "\n".join(parts)

    # ---------------------------------------------------------------- public
    def tailor(self, job: Job, fit: FitResult | None = None) -> str:
        system, user = self.build_prompt(job, fit)
        raw = self.client.chat(system, user, json_mode=True)
        try:
            data = extract_json_object(raw)
        except ValueError:
            log.warning("Resume LLM output was not JSON; falling back to profile text")
            data = {}
        return self.assemble(data, fit)

    # -------------------------------------------------------------- assembly
    def _clean_skills(self, skills: Any, fit: FitResult | None) -> list[str]:
        allowed = {_norm(s): s for s in profile_skill_list(self.profile)}
        gaps = {_norm(g) for g in (fit.skill_gaps if fit else [])}
        out: list[str] = []
        seen: set[str] = set()
        if isinstance(skills, list):
            for s in skills:
                key = _norm(s) if isinstance(s, str) else ""
                if key in allowed and key not in gaps and key not in seen:
                    seen.add(key)
                    out.append(allowed[key])
        # Never return an empty skills list: fall back to the profile order.
        if not out:
            out = [s for s in allowed.values() if _norm(s) not in gaps]
        return out

    def assemble(self, data: dict[str, Any], fit: FitResult | None = None) -> str:
        p = self.profile
        lines: list[str] = [f"# {p.get('name', '').strip()}"]
        contact = contact_line(p)
        if contact:
            lines.append(contact)
        lines.append("")

        summary = data.get("summary") if isinstance(data.get("summary"), str) else ""
        summary = " ".join(summary.split()) or " ".join(str(p.get("summary", "")).split())
        lines += ["## Summary", summary, ""]

        skills = self._clean_skills(data.get("skills"), fit)
        lines += ["## Skills", ", ".join(skills), ""]

        llm_exp: dict[str, list[str]] = {}
        for entry in data.get("experience") or []:
            if not isinstance(entry, dict):
                continue
            key = _norm(entry.get("company", ""))
            bullets = [" ".join(str(b).split()).lstrip("-*• ").strip()
                       for b in entry.get("bullets") or [] if str(b).strip()]
            if key and bullets and key not in llm_exp:
                llm_exp[key] = bullets

        lines.append("## Experience")
        for exp in p.get("experience") or []:
            company = str(exp.get("company", "")).strip()
            title = str(exp.get("title", "")).strip()
            lines.append(f"### {title} — {company}")
            dates = f"{exp.get('start', '')} – {exp.get('end', '')}"
            loc = str(exp.get("location", "")).strip()
            lines.append(f"{loc} | {dates}" if loc else dates)
            bullets = llm_exp.get(_norm(company)) or [str(b) for b in exp.get("bullets") or []]
            lines += [f"- {b}" for b in bullets]
            lines.append("")

        lines.append("## Education")
        for ed in p.get("education") or []:
            bits = [str(ed.get(k)) for k in ("degree", "school", "year") if ed.get(k)]
            lines.append("- " + ", ".join(bits))
        lines.append("")

        lines.append("## Certifications")
        for cert in p.get("certifications") or []:
            lines.append(f"- {cert}")
        return "\n".join(lines).strip() + "\n"
