"""Pressing Stop: AI calls are cut off at once, and searches keep what they already did."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from core.agents import HuntPipeline
from core.llm_client import LLMCancelled, OllamaClient
from core.models import Job, make_job_id
from core.storage import Storage
from scrapers.job_scraper import SearchQuery

DELAY = 1.5


@pytest.fixture
def slow_ollama():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(DELAY)                                   # a slow laptop model
            quick = "fast first pass" in body["messages"][0]["content"]
            content = json.dumps({"skills": 4, "experience": 4, "missing": []} if quick else
                                 {"fit_score": 80, "matched_skills": [], "skill_gaps": [],
                                  "recommendation": "Apply", "rationale": "ok", "fit_highlights": []})
            data = json.dumps({"message": {"content": content}}).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass                                            # client hung up (Stop)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_stop_cuts_off_an_ai_call_in_progress(slow_ollama):
    client = OllamaClient(base_url=slow_ollama, max_retries=0)
    errors = []
    t = threading.Thread(target=lambda: errors.append(
        pytest.raises(LLMCancelled, client.chat, "s", "u")))
    t0 = time.time()
    t.start()
    time.sleep(0.2)
    client.abort()
    t.join(5)
    assert time.time() - t0 < DELAY, "Stop should not wait for the AI to finish"
    assert errors
    with pytest.raises(LLMCancelled):                           # and nothing new starts
        client.chat("s", "u")


def test_stopped_search_keeps_found_and_scored_jobs(slow_ollama, tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"name": "Sam", "target": {"titles": ["Office Manager"]},
                                       "skills": {"practices": ["Scheduling"]}}))
    settings = {"paths": {"data_dir": str(tmp_path / "d"), "build_dir": str(tmp_path / "b"),
                          "candidate_profile": str(profile)},
                "ranking": {"max_jobs_per_run": 50, "explain_top": 0},
                "parallel": {"llm_workers": 1, "search_workers": 2}}
    storage = Storage(settings)
    desc = "Office manager needed for scheduling at a busy clinic. " * 3
    jobs = [Job(id=make_job_id("c", f"Office Manager {i}", ""), title=f"Office Manager {i}", company="c",
                description=desc) for i in range(6)]
    stop = threading.Event()
    client = OllamaClient(base_url=slow_ollama, max_retries=0)

    def on_progress(p):
        if p.scored >= 1 and not stop.is_set():
            stop.set()                                          # the user presses Stop
            client.abort()

    t0 = time.time()
    summary = HuntPipeline(settings, storage, client, query_fn=lambda q: jobs, on_progress=on_progress,
                           sleep=lambda s: None, should_stop=stop.is_set).run([SearchQuery("indeed", "x")])
    elapsed = time.time() - t0
    assert summary.progress.stopped
    assert elapsed < DELAY * 2 + 1, f"took {elapsed:.1f}s; scoring all 6 would take {6 * DELAY}s"
    assert len(storage.load_rankings()) == 1                    # the score finished before Stop is kept
    assert len(storage.load_jobs()) == 6                        # every job found is kept
    assert summary.unscored_left >= 4                           # the rest wait for "Score"
