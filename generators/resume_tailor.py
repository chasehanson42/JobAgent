"""Fill the resume template for one job.

The header, titles, dates, education and certifications come straight from
the profile. Skills and which jobs/bullets to show are chosen without the AI
(generators.relevance), so unrelated experience is condensed to one line.
The AI only rewrites the summary and the chosen bullets, each job in its own
short call, all side by side. Any rewritten line that adds numbers or facts
not in the original is replaced by the original line.
"""
from __future__ import annotations

import logging
from typing import Any

from core.models import FitResult, Job
from generators import templates
from generators._editing import Context, ask_list, ask_text, build_context, mentions_gap, run_parallel
from generators._prompting import contact_line
from generators.relevance import ExperiencePlan, is_grounded_rewrite

log = logging.getLogger(__name__)


class ResumeTailor:
    def __init__(self, client: Any, profile: dict[str, Any], settings: dict[str, Any] | None = None,
                 template: str | None = None) -> None:
        self.client = client
        self.profile = profile
        self.settings = settings or {}
        self.template = template if template is not None else templates.load_template("resume")
        self.workers = int((self.settings.get("parallel") or {}).get("llm_workers", 2))

    def tailor(self, job: Job, fit: FitResult | None = None, ctx: Context | None = None) -> str:
        ctx = ctx or build_context(self.profile, job, fit, self.settings)
        want_summary = "{summary}" in self.template
        want_bullets = "{experience}" in self.template
        detailed = [p for p in ctx.plans if p.detailed and p.bullets] if want_bullets else []

        calls = ([lambda: self._summary(ctx)] if want_summary else []) + \
                [lambda p=p: self._bullets(ctx, p) for p in detailed]
        results = run_parallel(calls, self.workers)
        summary = results.pop(0) if want_summary else ""
        rewritten = {id(p): r for p, r in zip(detailed, results)}

        values = {
            "name": str(self.profile.get("name", "")).strip(),
            "contact": contact_line(self.profile),
            "summary": summary,
            "skills": ", ".join(ctx.skills),
            "experience": self._experience(ctx.plans, rewritten),
            "education": "\n".join("- " + ", ".join(str(ed[k]) for k in ("degree", "school", "year") if ed.get(k))
                                   for ed in self.profile.get("education") or [] if isinstance(ed, dict)),
            "certifications": "\n".join(f"- {c}" for c in self.profile.get("certifications") or []),
        }
        return _drop_empty_sections(templates.fill(self.template, values)).strip() + "\n"

    # ---------------------------------------------------------------- AI parts

    def _summary(self, ctx: Context) -> str:
        original = " ".join(str(self.profile.get("summary") or self.profile.get("headline") or "").split())
        try:
            text = ask_text(self.client, ctx, "Write a 2-sentence professional summary for this candidate, "
                            "aimed at this job. Reword the profile summary; add nothing new.", max_tokens=120)
        except Exception as exc:  # noqa: BLE001 - keep the profile summary
            if _is_stop(exc):
                raise
            log.warning("Summary rewrite failed: %s", exc)
            return original
        if text and is_grounded_rewrite(original or ctx.allowed, text, ctx.allowed) and not mentions_gap(text, ctx.gaps):
            return text
        return original

    def _bullets(self, ctx: Context, plan: ExperiencePlan) -> list[str]:
        e = plan.entry
        numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(plan.bullets))
        try:
            items = ask_list(self.client, ctx,
                             f"Rewrite these {len(plan.bullets)} resume bullets from the candidate's job as "
                             f"{e.get('title', '')} at {e.get('company', '')} so they speak to this job: same "
                             "facts, strong verb first, one sentence each, same order, same count.\n" + numbered,
                             max_tokens=70 * len(plan.bullets) + 40)
        except Exception as exc:  # noqa: BLE001 - keep the original bullets
            if _is_stop(exc):
                raise
            log.warning("Bullet rewrite failed for %s: %s", e.get("company"), exc)
            return list(plan.bullets)
        out = []
        for i, original in enumerate(plan.bullets):
            new = items[i] if i < len(items) else ""
            ok = new and is_grounded_rewrite(original, new, ctx.allowed) and not mentions_gap(new, ctx.gaps)
            out.append(new if ok else original)
        return out

    # ---------------------------------------------------------------- layout

    @staticmethod
    def _experience(plans: list[ExperiencePlan], rewritten: dict[int, list[str]]) -> str:
        blocks = []
        for plan in plans:
            e = plan.entry
            lines = [f"### {str(e.get('title', '')).strip()} — {str(e.get('company', '')).strip()}"]
            dates = f"{e.get('start', '')} – {e.get('end', '')}"
            loc = str(e.get("location", "")).strip()
            lines.append(f"{loc} | {dates}" if loc else dates)
            if plan.detailed:
                lines += [f"- {b}" for b in rewritten.get(id(plan), plan.bullets)]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _is_stop(exc: Exception) -> bool:
    from core.llm_client import LLMCancelled
    return isinstance(exc, LLMCancelled)


def _drop_empty_sections(text: str) -> str:
    """Remove a "## Heading" whose section ended up empty (e.g. no certifications)."""
    out: list[str] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("## "):
            rest = []
            for nxt in lines[i + 1:]:
                if nxt.startswith("## ") or nxt.startswith("# "):
                    break
                rest.append(nxt)
            if not any(r.strip() for r in rest):
                continue
        out.append(line)
    return "\n".join(out)
