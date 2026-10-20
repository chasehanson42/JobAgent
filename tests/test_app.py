"""Drive the Streamlit app headlessly: setup -> profile import -> add job -> score -> generate."""
import json
import threading

import pytest
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import core.llm_client
import core.settings
from core import ollama_admin
from core.background import MANAGER
from core.settings import PROJECT_ROOT

from test_orchestrator_e2e import ScriptedLLM

APP = str(PROJECT_ROOT / "app.py")
RESUME = """Sam Rivera
sam@example.com | Denver, CO
Office Manager, Bluebird Dental, 2019-04 to present
- Scheduled 40 patients per day across 3 dentists
- Cut supply costs 15% by renegotiating vendor contracts
Skills: Scheduling, QuickBooks, Customer service
B.A. Communications, Colorado State University, 2018
"""


class AppLLM(ScriptedLLM):
    def chat(self, system, user, *, json_mode=False, temperature=None):
        if "resume into a structured json profile" in system.lower():
            return json.dumps({
                "name": "Sam Rivera", "email": "sam@example.com", "location": "Denver, CO",
                "skills": {"practices": ["Scheduling", "QuickBooks", "Customer service"]},
                "experience": [{"company": "Bluebird Dental", "title": "Office Manager",
                                "start": "2019-04", "end": "present",
                                "bullets": ["Scheduled 40 patients per day across 3 dentists",
                                            "Cut supply costs 15% by renegotiating vendor contracts"]},
                               {"company": "Invented Corp", "title": "CEO", "bullets": []}],
                "education": [{"school": "Colorado State University", "degree": "B.A. Communications",
                               "year": 2018}],
            })
        if "fit_score" in system.lower():
            return json.dumps({"fit_score": 82, "matched_skills": ["Scheduling", "QuickBooks"],
                               "skill_gaps": ["Medical billing"], "recommendation": "Apply",
                               "rationale": "Strong office management fit."})
        if json_mode and "hallucinations" not in system.lower():
            return json.dumps({"summary": "Office manager who keeps a busy dental practice running.",
                               "skills": ["Scheduling", "QuickBooks"],
                               "experience": [{"company": "Bluebird Dental", "bullets": [
                                   "Scheduled 40 patients per day across 3 dentists"]}]})
        return super().chat(system, user, json_mode=json_mode, temperature=temperature)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text())
    settings["paths"].update(data_dir=str(tmp_path / "data"), build_dir=str(tmp_path / "build"))
    defaults = tmp_path / "settings.yaml"
    defaults.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(core.settings, "DEFAULT_SETTINGS", defaults)
    monkeypatch.setattr(core.settings, "USER_SETTINGS", tmp_path / "user_settings.yaml")
    monkeypatch.setattr(core.settings, "USER_PROFILE", tmp_path / "my_profile.yaml")

    models = []
    monkeypatch.setattr(ollama_admin, "is_running", lambda base, timeout=2: True)
    monkeypatch.setattr(ollama_admin, "list_models", lambda base, timeout=5: list(models))
    monkeypatch.setattr(ollama_admin, "has_model", lambda base, m: m in models)
    monkeypatch.setattr(ollama_admin, "total_ram_gb", lambda: 16.0)

    def fake_pull(base, model, timeout=60):
        yield {"status": "pulling", "completed": 50, "total": 100}
        models.append(model)
        yield {"status": "success", "completed": None, "total": None}

    monkeypatch.setattr(ollama_admin, "pull_model", fake_pull)
    llm = AppLLM()
    monkeypatch.setattr(core.llm_client, "build_client", lambda s, provider=None: llm)
    MANAGER._tasks.clear()
    yield tmp_path, llm
    MANAGER.wait()
    MANAGER._tasks.clear()


def _page(name: str) -> AppTest:
    """Run one page with navigation registered, as app.py would."""
    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
import streamlit as st
import webapp
st.navigation(webapp.make_pages())
webapp.page_{name}()
"""
    return AppTest.from_string(script, default_timeout=30)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _finish(at):
    """Background task started by the last click: wait for it, then refresh the page."""
    MANAGER.wait()
    return at.run()


def test_full_flow(app_env):
    tmp_path, _ = app_env

    # Setup: Ollama "running", model missing -> download it.
    at = _page("setup").run()
    assert not at.exception
    _finish(_button(at, "Download").click().run())
    assert not at.exception
    saved = yaml.safe_load((tmp_path / "user_settings.yaml").read_text())
    assert saved["llm"]["ollama"]["model"] == "qwen2.5:7b-instruct"

    # Profile: import a resume, the invented company is flagged, save it.
    at = _page("profile").run()
    assert not at.exception
    at.file_uploader[0].upload("resume.txt", RESUME.encode(), "text/plain").run()
    _finish(_button(at, "Read my resume").click().run())
    assert not at.exception, at.exception
    assert any("Invented Corp" in w.value for w in at.warning)
    assert at.text_input(key="p2_name").value == "Sam Rivera"
    # Remove the invented job (second experience entry), then save.
    remove = [b for b in at.button if b.label == "Remove this job"]
    remove[1].click().run()
    assert not at.exception
    _button(at, "Save my profile").click().run()
    assert not at.exception
    profile = yaml.safe_load((tmp_path / "my_profile.yaml").read_text())
    assert profile["name"] == "Sam Rivera"
    assert [e["company"] for e in profile["experience"]] == ["Bluebird Dental"]

    # Find jobs: paste a job by hand.
    at = _page("find").run()
    assert not at.exception
    locations = next(t for t in at.text_area if t.label.startswith("Locations"))
    assert "Denver, CO" in locations.value       # pre-filled from the profile, not the examples
    titles = next(t for t in at.text_area if t.label.startswith("Job titles"))
    assert "python developer" not in titles.value
    form_inputs = {t.label: t for t in at.text_input}
    form_inputs["Job title"].input("Office Manager")
    form_inputs["Company"].input("Smile Clinic")
    form_inputs["Location"].input("Denver, CO")
    next(t for t in at.text_area if t.label == "Full job description").input(
        "Busy dental clinic needs an office manager for scheduling, QuickBooks bookkeeping, "
        "patient customer service and vendor management.")
    next(b for b in at.button if b.label == "Add job").click().run()
    assert not at.exception, at.exception
    assert any("kept **1**" in s.value for s in at.success)

    # Matches: score, then generate documents.
    at = _page("ranked").run()
    _finish(_button(at, "Score 1 job(s) with AI").click().run())
    assert not at.exception, at.exception
    assert at.metric[0].value == "82"
    _finish(_button(at, "Create resume & cover letter").click().run())
    assert not at.exception, at.exception
    labels = [d.label for d in at.get("download_button")]
    assert labels.count("Download PDF") == 2 and labels.count("Download DOCX") == 2
    build = next((tmp_path / "build").iterdir())
    resume = (build / "resume.md").read_text()
    assert resume.startswith("# Sam Rivera") and "Invented Corp" not in resume


def test_task_keeps_running_when_user_clicks_elsewhere(app_env):
    """Regression: clicking anything used to abort scoring mid-way."""
    tmp_path, llm = app_env
    release = threading.Event()
    real_chat = llm.chat

    def slow_chat(system, user, **kw):
        release.wait(10)
        return real_chat(system, user, **kw)

    llm.chat = slow_chat
    storage_jobs = tmp_path / "data" / "jobs.jsonl"
    storage_jobs.parent.mkdir(parents=True)
    storage_jobs.write_text(json.dumps({
        "id": "job1", "title": "Office Manager", "company": "Smile Clinic", "location": "Denver, CO",
        "description": "Office manager for scheduling and QuickBooks bookkeeping at a dental clinic."}) + "\n")

    at = _page("ranked").run()
    _button(at, "Score 1 job(s) with AI").click().run()
    assert MANAGER.running() is not None
    assert not any(b.label == "Score 1 job(s) with AI" for b in at.button)   # progress shown instead

    # Click around: toggle a switch and open other pages while the AI is busy.
    at.toggle[0].set_value(True).run()
    other = _page("find").run()
    assert not other.exception
    assert _button(other, "Save and search").disabled                       # one AI task at a time
    assert MANAGER.running() is not None

    release.set()
    _finish(at)
    assert not at.exception, at.exception
    assert MANAGER.get("rank").status == "done"
    assert json.loads((tmp_path / "data" / "rankings.json").read_text())[0]["fit_score"] == 82
    assert at.metric[0].value == "82"
