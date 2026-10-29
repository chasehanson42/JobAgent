"""Fill the cover letter template for one job.

The user's template keeps its own wording and layout. {company}, {job_title},
the date and contact details are filled in by code. Each body paragraph is
tailored by the AI in its own short call, all side by side: it swaps in this
company, role and the candidate's most relevant skills and experience, and
follows any [note] the user left. A paragraph whose rewrite adds facts that
aren't in the profile or the template is kept as the user wrote it.
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from core.models import FitResult, Job
from generators import templates
from generators._editing import Context, ask_text, build_context, mentions_gap, run_parallel
from generators._prompting import contact_line
from generators.relevance import is_grounded_rewrite
from scrapers.sanitizer import sanitize_text

log = logging.getLogger(__name__)

WORD_CAP = 400


class CoverLetterWriter:
    def __init__(self, client: Any, profile: dict[str, Any], settings: dict[str, Any] | None = None,
                 template: str | None = None) -> None:
        self.client = client
        self.profile = profile
        self.settings = settings or {}
        self.template = template if template is not None else templates.load_template("cover_letter")
        self.workers = int((self.settings.get("parallel") or {}).get("llm_workers", 2))

    def values(self, job: Job, today: _dt.date | None = None) -> dict[str, str]:
        p = self.profile
        return {
            "name": str(p.get("name", "")).strip(),
            "contact": contact_line(p),
            "date": (today or _dt.date.today()).strftime("%B %d, %Y").replace(" 0", " "),
            "company": " ".join(sanitize_text(job.company or "", max_chars=200).text.split()) or "your company",
            "job_title": " ".join(sanitize_text(job.title or "", max_chars=200).text.split()) or "this role",
            "email": str(p.get("email", "")), "phone": str(p.get("phone", "")),
        }

    def write(self, job: Job, fit: FitResult | None = None, *, today: _dt.date | None = None,
              ctx: Context | None = None) -> str:
        ctx = ctx or build_context(self.profile, job, fit, self.settings)
        values = self.values(job, today)
        blocks = templates.split_cover_letter(self.template)
        edits = [(i, templates.fill(text, values)) for i, (kind, text) in enumerate(blocks) if kind == "edit"]
        results = run_parallel([lambda t=t: self._paragraph(ctx, t) for _, t in edits], self.workers)
        tailored = {i: r for (i, _), r in zip(edits, results)}

        out = []
        for i, (kind, text) in enumerate(blocks):
            para = tailored.get(i) if kind == "edit" else templates.fill(text, values)
            if para:
                out.append(para)
        return "\n\n".join(out).strip() + "\n"

    def _paragraph(self, ctx: Context, original: str) -> str:
        fallback = templates.strip_notes(original)
        has_notes = "[" in original and "]" in original
        try:
            text = ask_text(self.client, ctx,
                            "Tailor this cover letter paragraph to the job. Keep the writer's voice and "
                            "roughly the same length; mention the most relevant skills and experience from "
                            "the fact sheet where it fits. Text in [square brackets] is a note telling you "
                            "what to write there: replace it. Return only the paragraph.\n\nPARAGRAPH: "
                            + original, max_tokens=220)
        except Exception as exc:  # noqa: BLE001 - keep the user's own words
            from core.llm_client import LLMCancelled
            if isinstance(exc, LLMCancelled):
                raise
            log.warning("Paragraph rewrite failed: %s", exc)
            return fallback
        if (text and "[" not in text and len(text.split()) <= WORD_CAP // 2
                and is_grounded_rewrite(fallback or original, text, ctx.allowed + " " + original)
                and not mentions_gap(text, ctx.gaps)):
            return text
        if has_notes:
            log.info("Tailored paragraph rejected; using the template text without its notes")
        return fallback
