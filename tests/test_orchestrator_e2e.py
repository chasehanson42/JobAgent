"""Drive the CLI end to end with a scripted fake LLM: no network needed."""
import json
from pathlib import Path

import yaml

import orchestrator
from core.settings import PROJECT_ROOT, load_settings

SAMPLE = PROJECT_ROOT / "examples" / "sample_jobs.json"


class ScriptedLLM:
    """Answers by recognising which component is calling from the system prompt."""
    name = "fake"
    model = "scripted"

    def __init__(self):
        self.calls = []

    def chat(self, system, user, *, json_mode=False, temperature=None):
        self.calls.append((system, user))
        s = system.lower()
        if "fit_score" in s:
            return json.dumps({"fit_score": 88, "matched_skills": ["Python", "FastAPI", "Cobol"],
                               "skill_gaps": ["Rust"], "recommendation": "Apply",
                               "rationale": "Strong backend overlap."})
        if "hallucinations" in s:
            return json.dumps({"hallucinations": [], "issues": []})
        if json_mode:  # resume tailor
            return json.dumps({
                "summary": "Backend engineer building Python APIs and data pipelines.",
                "skills": ["Python", "FastAPI", "PostgreSQL", "Rust"],
                "experience": [{"company": "Acme Analytics", "bullets": [
                    "Designed a FastAPI ingestion service handling 40M events/day on PostgreSQL and Redis."]}],
            })
        return ("Your team's FastAPI and PostgreSQL work matches what I have been doing.\n\n"
                "At Acme Analytics I designed an ingestion service and cut p95 latency.\n\n"
                "I would welcome a conversation about the role.")


def _settings_file(tmp_path: Path) -> Path:
    settings = load_settings()
    settings["paths"] = {"data_dir": str(tmp_path / "data"), "build_dir": str(tmp_path / "build"),
                         "candidate_profile": str(PROJECT_ROOT / "config" / "candidate_profile.yaml")}
    settings["generation"]["export_formats"] = ["md", "pdf", "docx"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


def test_scrape_rank_generate(tmp_path, monkeypatch, capsys):
    cfg = str(_settings_file(tmp_path))
    llm = ScriptedLLM()
    monkeypatch.setattr(orchestrator, "build_client", lambda settings, provider=None: llm)

    assert orchestrator.main(["--settings", cfg, "scrape", "--from-file", str(SAMPLE)]) == 0
    jobs = [json.loads(l) for l in (tmp_path / "data" / "jobs.jsonl").read_text().splitlines()]
    titles = {j["title"] for j in jobs}
    assert "Engineering Intern" not in titles          # dropped by title_blacklist
    contoso = next(j for j in jobs if j["company"] == "Contoso Labs")
    assert "<script>" not in contoso["description"]
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" not in contoso["description"]
    assert any(f.startswith("prompt_injection") for f in contoso["sanitizer_flags"])

    assert orchestrator.main(["--settings", cfg, "rank"]) == 0
    rankings = json.loads((tmp_path / "data" / "rankings.json").read_text())
    assert len(rankings) == len(jobs)
    assert all("Cobol" not in r["matched_skills"] for r in rankings)   # ungrounded skill dropped
    for _, user in llm.calls:
        assert "IGNORE ALL PREVIOUS INSTRUCTIONS" not in user

    job_id = rankings[0]["job_id"]
    assert orchestrator.main(["--settings", cfg, "generate", "--job-id", job_id]) == 0
    out = tmp_path / "build" / job_id
    for kind in ("resume", "cover_letter"):
        for ext in ("md", "pdf", "docx"):
            assert (out / f"{kind}.{ext}").stat().st_size > 0
    resume = (out / "resume.md").read_text()
    assert resume.startswith("# Jane Doe") and "Rust" not in resume
    assert "Widget Co" in resume
    report = json.loads((out / "critic_report.json").read_text())
    assert set(report) == {"resume", "cover_letter"}

    assert orchestrator.main(["--settings", cfg, "generate", "--job-id", "nope"]) == 1
