#!/usr/bin/env python3
"""FreeJobAgent CLI. (Prefer a browser? Run `streamlit run app.py` or the Start launcher.)

    python orchestrator.py hunt                      # search + rank, in parallel
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
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import pipeline  # noqa: E402
from core.llm_client import LLMError, build_client  # noqa: E402
from core.settings import load_settings  # noqa: E402
from core.storage import Storage  # noqa: E402

log = logging.getLogger("freejobagent")


def cmd_scrape(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    raw = pipeline.load_jobs_file(Path(args.from_file)) if args.from_file else None
    s = pipeline.scrape(settings, storage, raw)
    print(f"Scraped {s.scraped} | kept {s.kept} | dropped {len(s.dropped)} by rules "
          f"| {s.total_stored} total in {storage.jobs_path}")
    return 0


def cmd_hunt(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    client = build_client(settings, args.provider)
    last = {"msg": ""}

    def show(p) -> None:
        if p.message != last["msg"]:
            last["msg"] = p.message
            print(f"  {p.message}", flush=True)

    sm = pipeline.hunt(settings, storage, client, on_progress=show)
    p = sm.progress
    print(f"\nDone in {sm.seconds:.0f}s: {p.found} found, {p.filtered} filtered, "
          f"{p.not_relevant} not relevant, {p.scored} scored, {p.apply} good matches"
          + (f", {sm.unscored_left} left for `rank`" if sm.unscored_left else ""))
    if sm.ai_speed:
        print(f"AI speed: {sm.ai_speed}")
    for label in p.failed_queries:
        print(f"  search failed: {label}")
    _print_rankings(storage, show_all=False)
    return 0


def cmd_rank(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    if not pipeline.unranked_jobs(storage, args.rescore) and storage.load_jobs():
        print("All jobs already ranked (use --rescore to re-evaluate).")
    else:
        client = build_client(settings, args.provider)
        print(f"Scoring with {client.name}...")
        pipeline.rank(settings, storage, client, limit=args.limit, rescore=args.rescore,
                      progress=lambda i, n, label: print(f"  [{i}/{n}] {label}", flush=True))
    _print_rankings(storage, show_all=False)
    return 0


def _print_rankings(storage: Storage, show_all: bool) -> None:
    rows = pipeline.ranked_rows(storage, include_skip=show_all)
    if not rows:
        print("No recommended jobs yet (use `list --all` to see everything).")
        return
    print(f"\n{'JOB ID':<13} {'SCORE':>5}  {'REC':<5}  TITLE @ COMPANY")
    for r, job in rows:
        label = f"{job.title} @ {job.company}" if job else "(job no longer stored)"
        print(f"{r.job_id:<13} {r.fit_score:>5}  {r.recommendation:<5}  {label[:70]}")


def cmd_list(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    _print_rankings(storage, args.all)
    return 0


def cmd_generate(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    client = build_client(settings, args.provider)
    print(f"Generating application with {client.name}...")
    result = pipeline.generate(settings, storage, client, args.job_id, force=args.force,
                               progress=lambda i, n, label: print(f"  drafting {label}...", flush=True))
    for kind, rep in result.reports.items():
        open_issues = len(rep["issues"]) + len(rep["hallucinations"])
        print(f"  {kind}: critic {'passed' if rep['passed'] else f'{open_issues} open issue(s)'}")
    print("\nWrote:")
    for p in result.files:
        print(f"  {p}")
    if not result.passed:
        print(f"Review {result.out_dir / 'critic_report.json'} before sending.")
    return 0


def cmd_check(args: argparse.Namespace, settings: dict[str, Any], storage: Storage) -> int:
    client = build_client(settings, args.provider)
    reply = client.chat("Reply with JSON only.", 'Return {"ok": true}.', json_mode=True)
    print(f"{client.name} backend reachable. Reply: {reply.strip()[:200]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchestrator.py", description="FreeJobAgent: local-first job pipeline")
    p.add_argument("--settings", help="path to settings.yaml (default: config/settings.yaml + user_settings.yaml)")
    p.add_argument("--provider", choices=["ollama", "gemini"], help="override llm.provider")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    h = sub.add_parser("hunt", help="search all sites and rank the results, in parallel")
    h.set_defaults(func=cmd_hunt)

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
    except pipeline.PipelineError as exc:
        print(exc)
        return 1
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
