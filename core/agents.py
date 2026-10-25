"""Parallel search and scoring agents.

    search agents (one per job site) ──► sanitize ─► rule filter ─► pre-screen
                                                                      │
                          priority queue, most relevant first ◄───────┘
                                       │
    scoring agents (N at once) ◄───────┘ ─► rankings.json, saved as each job finishes

Searching is network-bound, so every job site gets its own agent; queries to
the same site still run one after another, with a pause, to stay polite.
Scoring starts as soon as the first jobs arrive instead of waiting for the
whole search, and the pre-screen keeps jobs that plainly don't match the
profile away from the (slow, local) AI entirely.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from core.llm_client import LLMClient, LLMError
from core.models import FitResult, Job
from core.settings import load_profile
from core.storage import Storage
from matching.filter_engine import FilterEngine
from matching.fit_evaluator import FitEvaluator
from matching.prescreen import is_relevant, profile_terms, relevance
from scrapers import job_scraper
from scrapers.job_scraper import SearchQuery
from scrapers.sanitizer import sanitize_job

log = logging.getLogger(__name__)

SAME_SITE_DELAY_SECONDS = 2.0
MANUAL_PRIORITY = 1000.0      # jobs the user added by hand are scored first


@dataclass
class HuntProgress:
    queries_total: int = 0
    queries_done: int = 0
    found: int = 0             # listings returned by the job sites
    filtered: int = 0          # removed by the user's rules
    not_relevant: int = 0      # removed by the pre-screen
    queued: int = 0            # waiting for the AI
    scoring: list[str] = field(default_factory=list)
    scored: int = 0
    apply: int = 0             # scored as a good match
    failed_queries: list[str] = field(default_factory=list)
    search_done: bool = False

    @property
    def fraction(self) -> float:
        expected = self.scored + self.queued + len(self.scoring)
        score_frac = self.scored / expected if expected else (1.0 if self.search_done else 0.0)
        if not self.queries_total:
            return score_frac
        return 0.25 * (self.queries_done / self.queries_total) + 0.75 * score_frac

    @property
    def message(self) -> str:
        parts = []
        if self.queries_total:
            parts.append(f"Searched {self.queries_done} of {self.queries_total}")
        expected = self.scored + self.queued + len(self.scoring)
        if expected or self.search_done:
            parts.append(f"scored {self.scored} of {expected}")
        if self.apply:
            parts.append(f"{self.apply} good match{'es' if self.apply != 1 else ''}")
        return " · ".join(parts) or "Starting..."


@dataclass
class HuntSummary:
    progress: HuntProgress
    results: list[FitResult]
    dropped: list[tuple[Job, list[str]]]
    unscored_left: int
    injection_flagged: int
    seconds: float


class HuntPipeline:
    """Run search agents and scoring agents concurrently.

    `client=None` means search only. Every write to storage happens under one
    lock, so the page can read partial results while the hunt is running.
    """

    def __init__(self, settings: dict[str, Any], storage: Storage, client: LLMClient | None = None, *,
                 query_fn: Callable[[SearchQuery], list[Job]] | None = None,
                 on_progress: Callable[[HuntProgress], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        self.storage = storage
        self.query_fn = query_fn or (lambda q: job_scraper.run_query(q))
        self.on_progress = on_progress
        self.sleep = sleep

        par, rank = settings.get("parallel", {}), settings.get("ranking", {})
        self.search_workers = max(1, int(par.get("search_workers", 4)))
        self.llm_workers = max(1, int(par.get("llm_workers", 2)))
        self.budget = int(rank.get("max_jobs_per_run", 30))
        self.min_relevance = float(rank.get("prescreen_min_score", 2))
        self.store_chars = int(rank.get("max_description_chars", 12000))

        profile = load_profile(settings)
        self.filter = FilterEngine(settings.get("filters", {}), profile)
        self.terms = profile_terms(profile)
        self.evaluator = None if client is None else FitEvaluator(
            client, profile,
            min_fit_score=int(rank.get("min_fit_score", 65)),
            max_description_chars=int(rank.get("prompt_description_chars", 4000)))

        self.progress = HuntProgress()
        self._lock = threading.Lock()
        self._queue: queue.PriorityQueue[tuple[float, int, Job]] = queue.PriorityQueue()
        self._seq = 0
        self._started = 0
        self._seen: set[str] = set()
        self._ranked_ids = {r.job_id for r in storage.load_rankings()}
        self._raw: list[Job] = []
        self._dropped: list[tuple[Job, list[str]]] = []
        self._results: list[FitResult] = []
        self._injection = 0
        self._search_done = threading.Event()
        self._fatal: BaseException | None = None

    # ---------------------------------------------------------------- public

    def run(self, queries: Iterable[SearchQuery] = (), new_jobs: Iterable[Job] = (), *,
            score_stored: bool = True, rescore: bool = False) -> HuntSummary:
        """Search `queries`, add `new_jobs` (always scored, e.g. pasted by the user) and,
        with `score_stored`, also score stored jobs that have no score yet."""
        t0 = time.time()
        queries = list(queries)
        self.progress.queries_total = len(queries)

        if score_stored and self.evaluator:
            stored = self.storage.load_jobs()
            self._seen.update(j.id for j in stored)
            for job in stored:
                if rescore or job.id not in self._ranked_ids:
                    self._enqueue(job, relevance(job, self.terms))
        new_jobs = list(new_jobs)
        if new_jobs:
            self._ingest(new_jobs, manual=True)

        workers = [threading.Thread(target=self._score_agent, name=f"score-{i}", daemon=True)
                   for i in range(self.llm_workers if self.evaluator else 0)]
        for w in workers:
            w.start()

        by_site: dict[str, list[SearchQuery]] = defaultdict(list)
        for q in queries:
            by_site[q.site].append(q)
        if by_site:
            with ThreadPoolExecutor(max_workers=min(self.search_workers, len(by_site)),
                                    thread_name_prefix="search") as pool:
                list(pool.map(self._search_agent, by_site.values()))
        self._search_done.set()
        with self._lock:
            self.progress.search_done = True
        self._report()

        for w in workers:
            w.join()
        self._finish()
        if self._fatal is not None:
            raise self._fatal
        return HuntSummary(replace(self.progress), list(self._results), list(self._dropped),
                           self._queue.qsize(), self._injection, time.time() - t0)

    # ---------------------------------------------------------------- search

    def _search_agent(self, queries: list[SearchQuery]) -> None:
        for i, q in enumerate(queries):
            if self._fatal is not None:
                return
            if i:
                self.sleep(SAME_SITE_DELAY_SECONDS)
            try:
                self._ingest(self.query_fn(q))
            except Exception as exc:  # noqa: BLE001 - one failing site must not stop the hunt
                log.warning("Search failed (%s): %s", q.label, exc)
                with self._lock:
                    self.progress.failed_queries.append(q.label)
            finally:
                with self._lock:
                    self.progress.queries_done += 1
                self._report()

    def _ingest(self, raw: list[Job], manual: bool = False) -> None:
        # Sanitizing and rule filtering are pure; do them outside the lock.
        clean = [sanitize_job(j, max_chars=self.store_chars) for j in raw]
        kept, dropped = self.filter.apply(clean)
        with self._lock:
            self._raw.extend(raw)
            self.progress.found += len(raw)
            self.progress.filtered += len(dropped)
            self._dropped.extend(dropped)
            self._injection += sum(any(f.startswith("prompt_injection") for f in j.sanitizer_flags)
                                   for j in clean)
            fresh: list[tuple[Job, float]] = []
            for job in kept:
                if job.id in self._seen:
                    continue
                self._seen.add(job.id)
                score = relevance(job, self.terms)
                if not manual and not is_relevant(score, self.terms, self.min_relevance):
                    self.progress.not_relevant += 1
                    self._dropped.append((job, ["not_relevant_to_profile"]))
                    continue
                fresh.append((job, MANUAL_PRIORITY if manual else score))
            if fresh:
                stored = {j.id: j for j in self.storage.load_jobs()}
                stored.update({j.id: j for j, _ in fresh})
                self.storage.save_jobs(list(stored.values()))
        if self.evaluator:
            for job, score in fresh:
                if job.id not in self._ranked_ids:
                    self._enqueue(job, score)
        self._report()

    # ---------------------------------------------------------------- scoring

    def _enqueue(self, job: Job, priority: float) -> None:
        with self._lock:
            self._seq += 1
            self._queue.put((-priority, self._seq, job))
            self.progress.queued = self._queue.qsize()

    def _next_job(self) -> Job | None:
        while self._fatal is None:
            with self._lock:
                if self._started >= self.budget:
                    return None
                try:
                    _, _, job = self._queue.get_nowait()
                except queue.Empty:
                    job = None
                else:
                    self._started += 1
                    self.progress.queued = self._queue.qsize()
            if job is not None:
                return job
            if self._search_done.is_set() and self._queue.empty():
                return None
            time.sleep(0.05)
        return None

    def _score_agent(self) -> None:
        assert self.evaluator is not None
        while (job := self._next_job()) is not None:
            label = f"{job.title} @ {job.company}"
            with self._lock:
                self.progress.scoring.append(label)
            self._report()
            try:
                result = self.evaluator.evaluate(job)
            except LLMError as exc:
                # The AI itself is unreachable or too slow: stop instead of
                # filling the list with failed scores. The job stays unscored.
                with self._lock:
                    self._fatal = self._fatal or exc
                    self.progress.scoring.remove(label)
                return
            except Exception as exc:  # noqa: BLE001 - one odd reply must not stop the rest
                result = FitResult(
                    job_id=job.id, fit_score=0, matched_skills=[], skill_gaps=[], recommendation="Skip",
                    rationale=f"evaluation_error: {type(exc).__name__}: {str(exc)[:200]}",
                    model=self.evaluator.model_label,
                    evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            with self._lock:
                self.progress.scoring.remove(label)
                self.progress.scored += 1
                self.progress.apply += result.recommendation == "Apply"
                self._results.append(result)
                merged = {r.job_id: r for r in self.storage.load_rankings()}
                merged[result.job_id] = result
                self.storage.save_rankings(list(merged.values()))
            self._report()

    # ---------------------------------------------------------------- helpers

    def _report(self) -> None:
        if self.on_progress is None:
            return
        with self._lock:
            snapshot = replace(self.progress, scoring=list(self.progress.scoring),
                               failed_queries=list(self.progress.failed_queries))
        self.on_progress(snapshot)

    def _finish(self) -> None:
        with self._lock:
            if self._raw:
                self.storage.save_jobs(self._raw, self.storage.raw_jobs_path)
            self.storage.write_jsonl(self.storage.data_dir / "dropped_jobs.jsonl",
                                     [{"id": j.id, "title": j.title, "company": j.company, "reasons": r}
                                      for j, r in self._dropped])
            self.progress.queued = self._queue.qsize()
