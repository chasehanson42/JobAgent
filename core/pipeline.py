"""Pipeline steps shared by the CLI (orchestrator.py) and the web app (app.py)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.llm_client import LLMClient
from core.models import FitResult, Job, make_job_id
from core.settings import load_profile
from core.storage import Storage

log = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]


class PipelineError(Exception):
    """A user-facing problem (bad job id, nothing to do, ...)."""


# ---------------------------------------------------------------- scrape

@dataclass
class ScrapeSummary:
    scraped: int
    kept: int
    dropped: list[tuple[Job, list[str]]]
    injection_flagged: int
    total_stored: int


def jobs_from_rows(rows: list[dict[str, Any]], source: str = "file") -> list[Job]:
    now = datetime.now(timezone.utc).isoformat()
    jobs = []
    for row in rows:
        row = dict(row)
        row.setdefault("id", make_job_id(row.get("company", ""), row.get("title", ""), row.get("url", "")))
        row.setdefault("scraped_at", now)
        row.setdefault("source", source)
        jobs.append(Job.from_dict(row))
    return jobs


def load_jobs_file(path: Path) -> list[Job]:
    """Import jobs from a JSON list/JSONL file (manual or offline input)."""
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    return jobs_from_rows(rows)


def scrape(settings: dict[str, Any], storage: Storage, raw: list[Job] | None = None) -> ScrapeSummary:
    """Scrape (or take `raw`), sanitize, rule-filter and store jobs."""
    from matching.filter_engine import FilterEngine
    from scrapers.sanitizer import sanitize_job

    if raw is None:
        from scrapers.job_scraper import scrape_jobs
        raw = scrape_jobs(settings)
    log.info("Collected %d raw listings", len(raw))
    storage.save_jobs(raw, storage.raw_jobs_path)

    max_chars = settings.get("ranking", {}).get("max_description_chars", 12000)
    clean = [sanitize_job(j, max_chars=max_chars) for j in raw]
    flagged = sum(any(f.startswith("prompt_injection") for f in j.sanitizer_flags) for j in clean)
    if flagged:
        log.warning("%d listing(s) contained prompt-injection attempts (neutralized)", flagged)

    kept, dropped = FilterEngine(settings.get("filters", {}), load_profile(settings)).apply(clean)

    # Merge with previously kept jobs so repeated scrapes accumulate.
    existing = {j.id: j for j in storage.load_jobs()}
    existing.update({j.id: j for j in kept})
    storage.save_jobs(list(existing.values()))
    storage.write_jsonl(storage.data_dir / "dropped_jobs.jsonl",
                        [{"id": j.id, "title": j.title, "company": j.company, "reasons": r}
                         for j, r in dropped])
    return ScrapeSummary(len(raw), len(kept), dropped, flagged, len(existing))


# ---------------------------------------------------------------- rank

def unranked_jobs(storage: Storage, rescore: bool = False) -> list[Job]:
    ranked = {r.job_id for r in storage.load_rankings()}
    return [j for j in storage.load_jobs() if rescore or j.id not in ranked]


def rank(settings: dict[str, Any], storage: Storage, client: LLMClient, *,
         limit: int | None = None, rescore: bool = False,
         progress: Progress | None = None) -> list[FitResult]:
    """Score unranked jobs; returns the newly scored results."""
    from matching.fit_evaluator import FitEvaluator

    if not storage.load_jobs():
        raise PipelineError("No jobs found yet. Search for jobs first.")
    ranking_cfg = settings.get("ranking", {})
    todo = unranked_jobs(storage, rescore)[:limit or int(ranking_cfg.get("max_jobs_per_run", 25))]
    if not todo:
        return []

    evaluator = FitEvaluator(client, load_profile(settings),
                             min_fit_score=int(ranking_cfg.get("min_fit_score", 65)),
                             max_description_chars=int(ranking_cfg.get("max_description_chars", 12000)))
    cb = (lambda i, n, job: progress(i, n, f"{job.title} @ {job.company}")) if progress else None
    results = evaluator.evaluate_many(todo, progress=cb)
    merged = {r.job_id: r for r in storage.load_rankings()}
    merged.update({r.job_id: r for r in results})
    storage.save_rankings(list(merged.values()))
    return results


def ranked_rows(storage: Storage, include_skip: bool = False) -> list[tuple[FitResult, Job | None]]:
    jobs = {j.id: j for j in storage.load_jobs()}
    rows = sorted(storage.load_rankings(), key=lambda r: r.fit_score, reverse=True)
    return [(r, jobs.get(r.job_id)) for r in rows if include_skip or r.recommendation == "Apply"]


# ---------------------------------------------------------------- generate

@dataclass
class GenerateResult:
    job: Job
    out_dir: Path
    files: list[Path]
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r["passed"] for r in self.reports.values())


def generate(settings: dict[str, Any], storage: Storage, client: LLMClient, job_id: str, *,
             force: bool = False, progress: Progress | None = None) -> GenerateResult:
    """Tailored resume + cover letter, critic loop, export to build/<job_id>/."""
    from exporters.pdf_builder import export_documents
    from generators.cover_letter_writer import CoverLetterWriter
    from generators.critic_agent import CriticAgent
    from generators.resume_tailor import ResumeTailor

    job = storage.get_job(job_id)
    if job is None:
        raise PipelineError(f"Unknown job id {job_id!r}.")
    fit = next((r for r in storage.load_rankings() if r.job_id == job.id), None)
    if fit and fit.recommendation == "Skip" and not force:
        raise PipelineError(f"This job scored {fit.fit_score} (Skip). Generate anyway with force.")

    gen_cfg = settings.get("generation", {})
    rounds = int(gen_cfg.get("critic_max_rounds", 2))
    formats = list(gen_cfg.get("export_formats", ["md", "pdf", "docx"]))
    profile = load_profile(settings)
    critic = CriticAgent(client, profile)
    out_dir = storage.job_build_dir(job.id)
    result = GenerateResult(job, out_dir, [])

    steps = (("resume", lambda: ResumeTailor(client, profile).tailor(job, fit)),
             ("cover_letter", lambda: CoverLetterWriter(client, profile).write(job, fit)))
    for i, (kind, make) in enumerate(steps, start=1):
        if progress:
            progress(i, len(steps), kind.replace("_", " "))
        final, report = critic.refine(make(), job, kind, max_rounds=rounds)
        result.reports[kind] = {"passed": report.passed, "issues": report.issues,
                                "hallucinations": report.hallucinations}
        result.files += export_documents(final, out_dir / kind, formats)

    (out_dir / "critic_report.json").write_text(json.dumps(result.reports, indent=2), encoding="utf-8")
    (out_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return result
