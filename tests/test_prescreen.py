import time

from core.models import Job
from matching.prescreen import ProfileTerms, is_relevant, profile_terms, relevance

NURSE = {
    "headline": "Registered Nurse with ICU experience",
    "target": {"titles": ["Registered Nurse", "ICU Nurse", "Charge Nurse"]},
    "experience": [{"company": "St. Mary", "title": "Senior Staff Nurse II"}],
    "skills": {"clinical": ["Patient care", "IV therapy", "Critical care"],
               "soft": ["Customer service", "Communication"]},
}

DEV = {
    "target": {"titles": ["Backend Developer"]},
    "skills": {"languages": ["Go", "C++", "Python"], "practices": ["CI/CD"]},
}


def job(title, description="", jid="j"):
    return Job(id=jid, title=title, company="X", description=description)


def test_profile_terms_extraction():
    t = profile_terms(NURSE)
    assert "nurse" in t.title_terms and "registered" in t.title_terms and "icu" in t.title_terms
    for filler in ("senior", "ii", "with", "staff"):
        assert filler not in t.title_terms
    assert "customer service" in t.skill_terms and "iv therapy" in t.skill_terms
    assert t.title_terms.count("nurse") == 1


def test_nurse_vs_software_job():
    t = profile_terms(NURSE)
    nurse_job = job("ICU Registered Nurse", "Provide critical care and patient care on night shift.")
    sw_job = job("Software Engineer", "Build microservices in Java and Kubernetes.")
    s_nurse, s_sw = relevance(nurse_job, t), relevance(sw_job, t)
    assert s_nurse >= 9 + 2
    assert s_sw == 0
    assert is_relevant(s_nurse, t, 3) and not is_relevant(s_sw, t, 3)


def test_title_hit_outweighs_description_hit():
    t = ProfileTerms(title_terms=["nurse"], skill_terms=[])
    assert relevance(job("Nurse"), t) == 3
    assert relevance(job("Caregiver", "Works with the nurse team."), t) == 1
    # Counted once per distinct term, even if in both.
    assert relevance(job("Nurse", "nurse nurse"), t) == 3


def test_go_does_not_match_google():
    t = profile_terms(DEV)
    assert relevance(job("Analyst", "Experience with Google Sheets"), t) == 0
    assert relevance(job("Analyst", "We write services in Go."), t) == 1


def test_cpp_and_cicd_match():
    t = profile_terms(DEV)
    assert relevance(job("Analyst", "Strong C++ skills"), t) == 1
    assert relevance(job("Analyst", "Own our CI/CD pipelines"), t) == 1
    assert relevance(job("Analyst", "Knows C and C#"), t) == 0


def test_score_capped():
    t = ProfileTerms(title_terms=[f"w{i}" for i in range(30)], skill_terms=[])
    assert relevance(job(" ".join(t.title_terms)), t) == 50


def test_empty_profile_always_relevant():
    t = profile_terms({})
    assert t.title_terms == [] and t.skill_terms == []
    assert is_relevant(relevance(job("Anything"), t), t, 5)
    assert profile_terms(None).is_empty


def test_performance_sanity():
    t = profile_terms(NURSE)
    desc = ("We are a fast growing company looking for motivated people. " * 60)[:3000]
    jobs = [job(f"Position {i}", desc, jid=str(i)) for i in range(500)]
    start = time.perf_counter()
    for j in jobs:
        relevance(j, t)
    assert time.perf_counter() - start < 1.0
