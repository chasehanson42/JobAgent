#!/usr/bin/env python3
"""FreeJobAgent CLI.

    python orchestrator.py scrape [--from-file jobs.json]
    python orchestrator.py rank [--limit N] [--rescore]
    python orchestrator.py list [--all]
    python orchestrator.py generate --job-id ID [--force]
    python orchestrator.py check

Pipeline: scrape -> sanitize -> deterministic filter (data/jobs.jsonl)
          -> LLM fit scoring (data/rankings.json)
          -> tailored resume + cover letter + critic loop -> build/<job_id>/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.llm_client import LLMError, build_client  # noqa: E402
from core.models import Job, make_job_id  # noqa: E402
from core.settings import load_profile, load_settings  # noqa: E402
from core.storage import Storage  # noqa: E402

log = logging.getLogger("freejobagent")


# ---------------------------------------------------------------- scrape

def _load_jobs_file(path: Path) -> list[Job]:
    """Import jobs from a JSON list/JSONL file (manual or offline input)."""
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    now = datetime.now(timezone.utc).isoformat()
    jobs = []
    for row in rows:
        row.setdefault("id", make_job_id(row.get("company", ""), row.get("title", ""), row.get("url", "")))
        row.setdefault("scraped_at", now)
        row.setdefault("source", "file")
        jobs.append(Job.from_dict(row))
    return jobs


def cmd_scrape(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    from matching.filter_engine import FilterEngine
    from scrapers.sanitizer import sanitize_job

    if args.from_file:
        raw = _load_jobs_file(Path(args.from_file))
    else:
        from scrapers.job_scraper import scrape_jobs
        raw = scrape_jobs(settings)
    log.info("Collected %d raw listings", len(raw))
    storage.save_jobs(raw, storage.raw_jobs_path)

    max_chars = settings.get("ranking", {}).get("max_description_chars", 12000)
    clean = [sanitize_job(j, max_chars=max_chars) for j in raw]
    flagged = [j for j in clean if any(f.startswith("prompt_injection") for f in j.sanitizer_flags)]
    if flagged:
        log.warning("%d listing(s) contained prompt-injection attempts (neutralized)", len(flagged))

    profile = load_profile(settings)
    kept, dropped = FilterEngine(settings.get("filters", {}), profile).apply(clean)

    # Merge with previously kept jobs so repeated scrapes accumulate.
    existing = {j.id: j for j in storage.load_jobs()}
    existing.update({j.id: j for j in kept})
    storage.save_jobs(list(existing.values()))
    storage.write_jsonl(storage.data_dir / "dropped_jobs.jsonl",
                        [{"id": j.id, "title": j.title, "company": j.company, "reasons": r}
                         for j, r in dropped])

    print(f"Scraped {len(raw)} | kept {len(kept)} | dropped {len(dropped)} by rules "
          f"| {len(existing)} total in {storage.jobs_path}")
    return 0


# ---------------------------------------------------------------- rank

def cmd_rank(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    from matching.fit_evaluator import FitEvaluator

    jobs = storage.load_jobs()
    if not jobs:
        print("No jobs found. Run `python orchestrator.py scrape` first.")
        return 1

    ranking_cfg = settings.get("ranking", {})
    previous = {r.job_id: r for r in storage.load_rankings()}
    todo = jobs if args.rescore else [j for j in jobs if j.id not in previous]
    limit = args.limit or int(ranking_cfg.get("max_jobs_per_run", 25))
    todo = todo[:limit]
    if not todo:
        print("All jobs already ranked (use --rescore to re-evaluate).")
        _print_rankings(list(previous.values()), {j.id: j for j in jobs}, show_all=False)
        return 0

    client = build_client(settings, args.provider)
    evaluator = FitEvaluator(client, load_profile(settings),
                             min_fit_score=int(ranking_cfg.get("min_fit_score", 65)),
                             max_description_chars=int(ranking_cfg.get("max_description_chars", 12000)))

    def progress(i: int, n: int, job: Job) -> None:
        print(f"  [{i}/{n}] {job.title} @ {job.company}", flush=True)

    print(f"Scoring {len(todo)} job(s) with {client.name}...")
    for result in evaluator.evaluate_many(todo, progress=progress):
        previous[result.job_id] = result
    storage.save_rankings(list(previous.values()))
    _print_rankings(list(previous.values()), {j.id: j for j in jobs}, show_all=False)
    return 0


def _print_rankings(results, jobs_by_id: dict[str, Job], show_all: bool) -> None:
    rows = sorted(results, key=lambda r: r.fit_score, reverse=True)
    if not show_all:
        rows = [r for r in rows if r.recommendation == "Apply"]
    if not rows:
        print("No recommended jobs yet (use `list --all` to see everything).")
        return
    print(f"\n{'JOB ID':<13} {'SCORE':>5}  {'REC':<5}  TITLE @ COMPANY")
    for r in rows:
        job = jobs_by_id.get(r.job_id)
        label = f"{job.title} @ {job.company}" if job else "(job no longer stored)"
        print(f"{r.job_id:<13} {r.fit_score:>5}  {r.recommendation:<5}  {label[:70]}")


def cmd_list(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    _print_rankings(storage.load_rankings(), {j.id: j for j in storage.load_jobs()}, args.all)
    return 0


# ---------------------------------------------------------------- generate

def cmd_generate(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    from exporters.pdf_builder import export_documents
    from generators.cover_letter_writer import CoverLetterWriter
    from generators.critic_agent import CriticAgent
    from generators.resume_tailor import ResumeTailor

    job = storage.get_job(args.job_id)
    if job is None:
        print(f"Unknown job id {args.job_id!r}. Run `python orchestrator.py list --all`.")
        return 1
    fit = next((r for r in storage.load_rankings() if r.job_id == job.id), None)
    if fit and fit.recommendation == "Skip" and not args.force:
        print(f"Job scored {fit.fit_score} (Skip). Re-run with --force to generate anyway.")
        return 1

    gen_cfg = settings.get("generation", {})
    rounds = int(gen_cfg.get("critic_max_rounds", 2))
    formats = list(gen_cfg.get("export_formats", ["md", "pdf", "docx"]))
    profile = load_profile(settings)
    client = build_client(settings, args.provider)
    critic = CriticAgent(client, profile)
    out_dir = storage.job_build_dir(job.id)

    print(f"Generating application for {job.title} @ {job.company} with {client.name}...")
    outputs: list[Path] = []
    reports: dict[str, Any] = {}
    for kind, make in (("resume", lambda: ResumeTailor(client, profile).tailor(job, fit)),
                       ("cover_letter", lambda: CoverLetterWriter(client, profile).write(job, fit))):
        print(f"  drafting {kind.replace('_', ' ')}...", flush=True)
        draft = make()
        final, report = critic.refine(draft, job, kind, max_rounds=rounds)
        reports[kind] = {"passed": report.passed, "issues": report.issues,
                         "hallucinations": report.hallucinations}
        outputs += export_documents(final, out_dir / kind, formats)
        status = "passed" if report.passed else f"{len(report.issues) + len(report.hallucinations)} open issue(s)"
        print(f"  critic: {status}")

    (out_dir / "critic_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (out_dir / "job.json").write_text(json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    print("\nWrote:")
    for p in outputs:
        print(f"  {p}")
    if not all(r["passed"] for r in reports.values()):
        print(f"Review {out_dir / 'critic_report.json'} before sending.")
    return 0


# ---------------------------------------------------------------- check

def cmd_check(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    client = build_client(settings, args.provider)
    reply = client.chat("Reply with JSON only.", 'Return {"ok": true}.', json_mode=True)
    print(f"{client.name} backend reachable. Reply: {reply.strip()[:200]}")
    return 0


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchestrator.py", description="FreeJobAgent: local-first job pipeline")
    p.add_argument("--settings", help="path to settings.yaml (default: config/settings.yaml)")
    p.add_argument("--provider", choices=["ollama", "gemini"], help="override llm.provider")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scrape", help="scrape, sanitize and rule-filter listings")
    s.add_argument("--from-file", help="import listings from a JSON/JSONL file instead of scraping")
    s.set_defaults(func=cmd_scrape)

    r = sub.add_parser("rank", help="LLM fit-score unranked jobs")
    r.add_argument("--limit", type=int, help="max jobs to score this run")
    r.add_argument("--rescore", action="store_true", help="re-score already ranked jobs")
    r.set_defaults(func=cmd_rank)

    ls = sub.add_parser("list", help="show ranked jobs")
    ls.add_argument("--all", action="store_true", help="include Skip recommendations")
    ls.set_defaults(func=cmd_list)

    g = sub.add_parser("generate", help="tailored resume + cover letter for one job")
    g.add_argument("--job-id", required=True)
    g.add_argument("--force", action="store_true", help="generate even if ranked Skip")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("check", help="verify the LLM backend is reachable")
    c.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.settings)
    storage = Storage(settings)
    try:
        return args.func(args, settings, storage)
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        if "Cannot reach" in str(exc):
            print("Is Ollama running? Try `ollama serve` and `ollama pull "
                  f"{settings.get('llm', {}).get('ollama', {}).get('model', 'qwen2.5:7b-instruct')}`.",
                  file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
