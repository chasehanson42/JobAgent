"""Pipeline steps shared by the CLI (orchestrator.py) and the web app (app.py)."""
from __future__ import annotations

import json
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
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
         progress: Progress | None = None, scoring_client: LLMClient | None = None) -> list[FitResult]:
    """Score stored jobs that have no score yet (parallel scoring agents); returns new results."""
    from core.agents import HuntPipeline

    if not storage.load_jobs():
        raise PipelineError("No jobs found yet. Search for jobs first.")
    if not unranked_jobs(storage, rescore):
        return []
    hunt_ = HuntPipeline(settings, storage, client, scoring_client=scoring_client,
                         on_progress=_progress_adapter(progress))
    if limit:
        hunt_.budget = limit
    return hunt_.run(score_stored=True, rescore=rescore).results


def hunt(settings: dict[str, Any], storage: Storage, client: LLMClient | None, *,
         new_jobs: list[Job] | None = None, search: bool = True,
         on_progress: Callable[[Any], None] | None = None,
         scoring_client: LLMClient | None = None,
         should_stop: Callable[[], bool] | None = None):
    """Search the job sites and score what fits, all in parallel. Returns a HuntSummary."""
    from core.agents import HuntPipeline
    from scrapers.job_scraper import build_queries

    queries = build_queries(settings.get("scrape", {})) if search else []
    return HuntPipeline(settings, storage, client, scoring_client=scoring_client, on_progress=on_progress,
                        should_stop=should_stop or (lambda: False)).run(queries, new_jobs or [])


def explain(settings: dict[str, Any], storage: Storage, client: LLMClient, job_id: str) -> FitResult:
    """Write the detailed "Why you fit" for one match on request; its score stays the same."""
    from core.agents import merge_explanation
    from matching.fit_evaluator import FitEvaluator

    job = storage.get_job(job_id)
    if job is None:
        raise PipelineError("That job is no longer in your list.")
    rank_cfg = settings.get("ranking", {})
    detail = FitEvaluator(client, load_profile(settings),
                          min_fit_score=int(rank_cfg.get("min_fit_score", 65)),
                          max_description_chars=int(rank_cfg.get("prompt_description_chars", 4000))
                          ).evaluate(job)
    rankings = {r.job_id: r for r in storage.load_rankings()}
    result = merge_explanation(rankings.get(job_id), detail)
    rankings[job_id] = result
    storage.save_rankings(list(rankings.values()))
    return result


def _progress_adapter(progress: Progress | None):
    """Map agent progress onto the simple (done, total, label) callback the CLI uses."""
    if progress is None:
        return None
    last = {"scored": -1}

    def cb(p) -> None:
        if p.scoring and p.scored != last["scored"]:
            last["scored"] = p.scored
            progress(p.scored + 1, p.scored + p.queued + len(p.scoring), p.scoring[-1])
    return cb


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
    bundle: Path | None = None

    @property
    def passed(self) -> bool:
        return all(r["passed"] for r in self.reports.values())


DOCUMENT_KINDS = ("resume", "cover_letter")


def generate(settings: dict[str, Any], storage: Storage, client: LLMClient, job_id: str, *,
             force: bool = False, progress: Progress | None = None,
             kinds: tuple[str, ...] = DOCUMENT_KINDS) -> GenerateResult:
    """Fill the resume and/or cover letter templates for one job; export to build/<job_id>/.

    Each document is a handful of small AI edits run side by side (see
    generators._editing), checked instantly for invented facts. The slow AI
    review is optional (generation.llm_review).
    """
    from exporters.pdf_builder import export_documents
    from generators._editing import build_context
    from generators.cover_letter_writer import CoverLetterWriter
    from generators.critic_agent import CriticAgent
    from generators.resume_tailor import ResumeTailor

    job = storage.get_job(job_id)
    if job is None:
        raise PipelineError(f"Unknown job id {job_id!r}.")
    fit = next((r for r in storage.load_rankings() if r.job_id == job.id), None)
    if fit and fit.recommendation == "Skip" and not force:
        raise PipelineError(f"This job scored {fit.fit_score} (Skip). Generate anyway with force.")
    kinds = tuple(k for k in DOCUMENT_KINDS if k in kinds)

    gen_cfg = settings.get("generation", {})
    llm_review = bool(gen_cfg.get("llm_review", False))
    rounds = int(gen_cfg.get("critic_max_rounds", 1))
    formats = list(gen_cfg.get("export_formats", ["md", "pdf", "docx"]))
    profile = load_profile(settings)
    critic = CriticAgent(client, profile)
    out_dir = storage.job_build_dir(job.id)
    result = GenerateResult(job, out_dir, [])
    # One context for both documents: the same job excerpt and fact sheet open every
    # AI call, so Ollama can reuse what it has already read.
    ctx = build_context(profile, job, fit, settings)

    def write(kind: str) -> tuple[str, Any]:
        draft = (ResumeTailor(client, profile, settings).tailor(job, fit, ctx) if kind == "resume"
                 else CoverLetterWriter(client, profile, settings).write(job, fit, ctx=ctx))
        if llm_review:
            return critic.refine(draft, job, kind, max_rounds=rounds, recheck_with_llm=False)
        return draft, critic.deterministic_checks(draft, kind)

    if progress:
        progress(1, 2, " and ".join(k.replace("_", " ") for k in kinds))
    with ThreadPoolExecutor(max_workers=len(kinds) if gen_cfg.get("parallel", True) else 1) as pool:
        outcomes = dict(zip(kinds, pool.map(write, kinds)))

    reports_path = out_dir / "critic_report.json"
    try:
        result.reports = json.loads(reports_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result.reports = {}
    for kind in kinds:
        final, report = outcomes[kind]
        result.reports[kind] = {"passed": report.passed, "issues": report.issues,
                                "hallucinations": report.hallucinations}
        result.files += export_documents(final, out_dir / kind, formats)
    result.bundle = bundle_application(out_dir, profile.get("name", ""), job.company)

    reports_path.write_text(json.dumps(result.reports, indent=2), encoding="utf-8")
    (out_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return result


def bundle_application(out_dir: Path, name: str, company: str) -> Path:
    """One download with both documents, as PDF and Word, named for the application."""
    def slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")[:40]

    stem = "_".join(x for x in (slug(name), slug(company)) if x) or "application"
    path = out_dir / "application.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for kind, label in (("resume", "Resume"), ("cover_letter", "Cover_Letter")):
            for ext in ("pdf", "docx"):
                src = out_dir / f"{kind}.{ext}"
                if src.exists():
                    zf.write(src, f"{stem}_{label}.{ext}")
    return path
