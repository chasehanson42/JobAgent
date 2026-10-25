"""Parallel search + scoring agents (core.agents), with fake job sites and a fake AI."""
import json
import threading
import time
from collections import defaultdict

import pytest
import yaml

from core.agents import HuntPipeline
from core.llm_client import LLMError
from core.models import Job, make_job_id
from core.storage import Storage
from scrapers.job_scraper import SearchQuery

PROFILE = {
    "name": "Sam Rivera",
    "headline": "Office manager for busy clinics",
    "target": {"titles": ["Office Manager"], "min_salary": None, "remote_ok": True},
    "skills": {"practices": ["Scheduling", "QuickBooks", "Customer service"]},
    "experience": [{"company": "Bluebird Dental", "title": "Office Manager",
                    "bullets": ["Scheduled 40 patients per day across 3 dentists"]}],
}
DESC = ("We need someone organised to handle scheduling, QuickBooks bookkeeping and "
        "customer service for a growing practice. ")


def job(title, company="Acme", desc=DESC):
    return Job(id=make_job_id(company, title, ""), title=title, company=company,
               location="Denver, CO", description=desc * 2)


class FakeLLM:
    name, model = "fake", "fake"

    def __init__(self, delay=0.0, fail_with=None):
        self.delay, self.fail_with = delay, fail_with
        self.lock = threading.Lock()
        self.active = self.max_active = 0
        self.order = []

    def chat(self, system, user, *, json_mode=False, temperature=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.fail_with:
                raise self.fail_with
            meta = user.split("<untrusted_job_metadata", 1)[1]
            title = meta.split("title: ", 1)[1].split("\n", 1)[0]
            with self.lock:
                self.order.append(title)
            score = 90 if "Office Manager" in title else 50
            return json.dumps({
                "fit_score": score, "matched_skills": ["Scheduling"], "skill_gaps": [],
                "recommendation": "Apply" if score >= 65 else "Skip", "rationale": "ok",
                "fit_highlights": [{"requirement": "Scheduling",
                                    "evidence": "Scheduled 40 patients per day at Bluebird Dental."}]})
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def env(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(PROFILE))
    settings = {
        "paths": {"data_dir": str(tmp_path / "data"), "build_dir": str(tmp_path / "build"),
                  "candidate_profile": str(profile_path)},
        "filters": {"title_blacklist": ["intern"]},
        "ranking": {"min_fit_score": 65, "max_jobs_per_run": 50, "prescreen_min_score": 2},
        "parallel": {"search_workers": 4, "llm_workers": 3},
    }
    return settings, Storage(settings)


def queries(sites=("indeed", "linkedin", "google"), terms=("office manager", "clinic manager")):
    return [SearchQuery(site, term) for site in sites for term in terms]


def test_sites_are_searched_in_parallel_but_each_site_sequentially(env):
    settings, storage = env
    active = defaultdict(int)
    max_per_site = defaultdict(int)
    lock = threading.Lock()

    def search(q):
        with lock:
            active[q.site] += 1
            max_per_site[q.site] = max(max_per_site[q.site], active[q.site])
        time.sleep(0.3)
        with lock:
            active[q.site] -= 1
        return [job(f"Office Manager {q.site} {q.term}")]

    t0 = time.time()
    summary = HuntPipeline(settings, storage, None, query_fn=search, sleep=lambda s: None).run(queries())
    elapsed = time.time() - t0
    assert elapsed < 1.2, f"6 queries x 0.3s should overlap across 3 sites, took {elapsed:.2f}s"
    assert set(max_per_site.values()) == {1}            # polite: one request per site at a time
    assert summary.progress.queries_done == 6 and summary.progress.found == 6
    assert len(storage.load_jobs()) == 6


def test_scoring_runs_in_parallel_and_starts_before_search_finishes(env):
    settings, storage = env
    llm = FakeLLM(delay=0.3)
    first_result_at = {}
    search_finished_at = {}

    def search(q):
        time.sleep(0.2 if q.term == "office manager" else 1.0)
        if q.term != "office manager":
            search_finished_at[q.site] = time.time()
        return [job(f"Office Manager {q.site} {q.term} {i}") for i in range(2)]

    def on_progress(p):
        if p.scored and "t" not in first_result_at:
            first_result_at["t"] = time.time()

    t0 = time.time()
    pipe = HuntPipeline(settings, storage, llm, query_fn=search, on_progress=on_progress,
                        sleep=lambda s: None)
    summary = pipe.run(queries())
    elapsed = time.time() - t0
    assert summary.progress.scored == 12 and summary.progress.apply == 12
    assert llm.max_active == 3                            # three scoring agents at once
    assert first_result_at["t"] < max(search_finished_at.values())   # pipelined, not "search then score"
    assert elapsed < 12 * 0.3, f"took {elapsed:.2f}s; sequential scoring alone would be 3.6s"
    ranked = storage.load_rankings()
    assert len(ranked) == 12 and ranked[0].fit_highlights[0]["requirement"] == "Scheduling"


def test_prescreen_skips_irrelevant_jobs_and_scores_best_first(env):
    settings, storage = env
    settings["parallel"]["llm_workers"] = 1
    llm = FakeLLM()
    found = [
        job("Forklift Operator", desc="Operate forklifts in a warehouse, lift 50 pounds, night shifts. " * 3),
        job("Receptionist", desc=DESC),                         # skills only -> lower relevance
        job("Office Manager", company="Smile Clinic"),          # title + skills -> highest
        job("Summer Intern", company="Smile Clinic"),           # removed by the user's rules
    ]
    summary = HuntPipeline(settings, storage, llm, query_fn=lambda q: found,
                           sleep=lambda s: None).run([SearchQuery("indeed", "office manager")])
    assert llm.order == ["Office Manager", "Receptionist"]
    assert summary.progress.not_relevant == 1 and summary.progress.filtered == 1
    reasons = {j.title: r for j, r in summary.dropped}
    assert reasons["Forklift Operator"] == ["not_relevant_to_profile"]
    assert "Forklift Operator" not in {j.title for j in storage.load_jobs()}


def test_budget_caps_ai_calls_and_leaves_the_rest_for_later(env):
    settings, storage = env
    settings["ranking"]["max_jobs_per_run"] = 2
    found = [job(f"Office Manager {i}") for i in range(5)]
    summary = HuntPipeline(settings, storage, FakeLLM(), query_fn=lambda q: found,
                           sleep=lambda s: None).run([SearchQuery("indeed", "x")])
    assert summary.progress.scored == 2 and summary.unscored_left == 3
    assert len(storage.load_jobs()) == 5 and len(storage.load_rankings()) == 2

    # The next run picks up the leftovers without searching again.
    settings["ranking"]["max_jobs_per_run"] = 10
    again = HuntPipeline(settings, storage, FakeLLM()).run()
    assert again.progress.scored == 3 and len(storage.load_rankings()) == 5


def test_unreachable_ai_stops_the_hunt_without_saving_fake_scores(env):
    settings, storage = env
    found = [job(f"Office Manager {i}") for i in range(4)]
    pipe = HuntPipeline(settings, storage, FakeLLM(fail_with=LLMError("Cannot reach http://localhost:11434")),
                        query_fn=lambda q: found, sleep=lambda s: None)
    with pytest.raises(LLMError, match="Cannot reach"):
        pipe.run([SearchQuery("indeed", "x")])
    assert storage.load_rankings() == []
    assert len(storage.load_jobs()) == 4                  # search results are kept for later


def test_failed_site_does_not_stop_other_sites(env):
    settings, storage = env

    def search(q):
        if q.site == "linkedin":
            raise RuntimeError("blocked")
        return [job(f"Office Manager {q.site} {q.term}")]

    summary = HuntPipeline(settings, storage, FakeLLM(), query_fn=search, sleep=lambda s: None).run(queries())
    assert len(summary.progress.failed_queries) == 2
    assert summary.progress.scored == 4


def test_pasted_jobs_skip_prescreen_and_go_first(env):
    settings, storage = env
    settings["parallel"]["llm_workers"] = 1
    llm = FakeLLM()
    pasted = job("Forklift Operator", company="Mine", desc="Warehouse work with forklifts, lifting and loading. " * 3)
    found = [job("Office Manager", company="Clinic")]
    HuntPipeline(settings, storage, llm, query_fn=lambda q: found, sleep=lambda s: None).run(
        [SearchQuery("indeed", "x")], new_jobs=[pasted])
    assert llm.order[0] == "Forklift Operator"             # the user asked for this one explicitly
    assert {r.job_id for r in storage.load_rankings()} == {pasted.id, found[0].id}


def test_progress_reaches_the_end(env):
    settings, storage = env
    seen = []
    HuntPipeline(settings, storage, FakeLLM(), query_fn=lambda q: [job(f"Office Manager {q.site}")],
                 on_progress=seen.append, sleep=lambda s: None).run(queries(terms=("x",)))
    assert seen[-1].fraction == pytest.approx(1.0)
    assert "Searched 3 of 3" in seen[-1].message and "scored 3 of 3" in seen[-1].message
