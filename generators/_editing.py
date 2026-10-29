"""Small, parallel AI edits shared by the resume and cover letter writers.

Instead of one long call that rewrites a whole document (slow on a laptop),
each piece the AI tailors (the summary, one job's bullets, one letter
paragraph) is its own short call, and the calls run side by side. Every
call starts with the same rules, the same job excerpt and the same fact
sheet, so Ollama can reuse what it already read for the earlier calls.

The fact sheet holds only what is relevant to this job (chosen without the
AI by generators.relevance), so unrelated jobs and bullets never reach it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from core.jsonutil import extract_json_object
from core.models import FitResult, Job
from generators._prompting import strip_fences
from generators.relevance import ExperiencePlan, JobKeywords, job_keywords, plan_experience, select_skills
from matching.essentials import job_essentials
from scrapers.sanitizer import UNTRUSTED_DATA_POLICY, sanitize_text, wrap_untrusted

SYSTEM = f"""You tailor parts of one candidate's job application to a job posting.

RULES (non-negotiable):
- Use ONLY facts from the candidate fact sheet or the text you are asked to edit.
- NEVER add employers, job titles, dates, degrees, certifications, numbers, tools or
  skills that are not there. Keep every number exactly as written.
- Never claim anything listed under "not claimed".
- Plain text only: no markdown, bullets symbols, emojis, quotes or placeholders.
- Do exactly the task at the end of the message and nothing else.

{UNTRUSTED_DATA_POLICY}"""


@dataclass
class Context:
    """Everything that stays the same across one document's AI calls."""
    job: Job
    kw: JobKeywords
    plans: list[ExperiencePlan]
    skills: list[str]
    gaps: list[str]
    prefix: str          # job excerpt + fact sheet: identical in every call, so it can be cached
    allowed: str         # text a rewrite may draw words from (profile facts used here)


def build_context(profile: dict[str, Any], job: Job, fit: FitResult | None, settings: dict[str, Any]) -> Context:
    cfg = (settings.get("generation") or {}).get("resume") or {}
    kw = job_keywords(job)
    plans = plan_experience(profile, kw, detailed_jobs=int(cfg.get("detailed_jobs", 3)),
                            bullets_per_job=int(cfg.get("bullets_per_job", 3)))
    _top_up(plans, int(cfg.get("min_bullets", 2)))
    skills = select_skills(profile, kw, max_skills=int(cfg.get("max_skills", 10)))
    gaps = [g for g in (fit.skill_gaps if fit else []) if g]

    meta = (f"title: {sanitize_text(job.title or '', max_chars=300).text}\n"
            f"company: {sanitize_text(job.company or '', max_chars=200).text}")
    essentials = sanitize_text(job_essentials(job.description or "", 1500)).text
    sheet = [f"name: {profile.get('name', '')}"]
    for key in ("headline", "summary"):
        if profile.get(key):
            sheet.append(f"{key}: {' '.join(str(profile[key]).split())}")
    sheet.append("skills: " + ", ".join(skills))
    for plan in plans:
        e = plan.entry
        sheet.append(f"job: {e.get('title', '')} at {e.get('company', '')} ({e.get('start', '')} to {e.get('end', '')})")
        sheet += [f"  - {b}" for b in plan.bullets]
    for ed in profile.get("education") or []:
        if isinstance(ed, dict):
            sheet.append(f"education: {ed.get('degree', '')}, {ed.get('school', '')}")
    if gaps:
        sheet.append("not claimed: " + ", ".join(gaps))

    prefix = (f"{wrap_untrusted(meta + chr(10) + essentials, 'job_posting')}\n\n"
              "<candidate_fact_sheet>\n" + "\n".join(sheet) + "\n</candidate_fact_sheet>\n\n")
    companies = [str(p.entry.get("company", "")) for p in plans]
    allowed = " ".join(sheet + companies + [str(job.company or ""), str(job.title or "")])
    return Context(job, kw, plans, skills, gaps, prefix, allowed)


def _top_up(plans: list[ExperiencePlan], min_bullets: int) -> None:
    """A detailed job with only one matching bullet looks thin: add its next bullets, in order."""
    for plan in plans:
        if not plan.detailed:
            continue
        own = [str(b) for b in plan.entry.get("bullets") or [] if str(b).strip()]
        if len(plan.bullets) >= min(min_bullets, len(own)):
            continue
        keep = set(plan.bullets)
        for b in own:
            if len(keep) >= min_bullets:
                break
            keep.add(b)
        plan.bullets = [b for b in own if b in keep]


def ask_text(client: Any, ctx: Context, task: str, max_tokens: int) -> str:
    raw = client.chat(SYSTEM, ctx.prefix + "TASK: " + task, max_tokens=max_tokens, temperature=0.3)
    return " ".join(strip_fences(raw).replace("\n", " ").split()).strip(' "')


def ask_list(client: Any, ctx: Context, task: str, max_tokens: int) -> list[str]:
    raw = client.chat(SYSTEM, ctx.prefix + "TASK: " + task + '\nReturn JSON: {"items": ["...", ...]}',
                      json_mode=True, max_tokens=max_tokens, temperature=0.3)
    try:
        items = extract_json_object(raw).get("items")
    except ValueError:
        return []
    return [" ".join(str(x).split()).lstrip("-*• ").strip() for x in items or [] if str(x).strip()]


def mentions_gap(text: str, gaps: list[str]) -> bool:
    low = text.lower()
    return any(g.lower() in low for g in gaps if len(g) > 2)


def run_parallel(jobs: list[Callable[[], Any]], workers: int) -> list[Any]:
    """Run the document's small AI calls side by side; results come back in order."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        return list(pool.map(lambda fn: fn(), jobs))
