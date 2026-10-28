import json

import pytest

from core.models import Job
from matching.essentials import job_essentials
from matching.fit_evaluator import FitEvaluator, QuickRating
from matching.scoring import DEFAULT_WEIGHTS, ScoreParts, combine, recommend, title_match, weights_for


def parts(**kw):
    base = dict(skills=4, experience=4, title=1.0, location=0.5, local=0.0)
    base.update(kw)
    return ScoreParts(**base)


def test_scores_spread_out_instead_of_clustering():
    w = {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}
    scores = {combine(parts(skills=s, experience=e, location=loc, local=lc), w)[0]
              for s in (3, 4, 5) for e in (3, 4, 5) for loc in (0.0, 0.5, 1.0) for lc in (0.0, 1.0)}
    assert len(scores) >= 15


def test_points_add_up_and_local_jobs_rank_higher():
    w = {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}
    far, far_pts = combine(parts(location=0.0, local=0.0), w)
    near, near_pts = combine(parts(location=1.0, local=1.0), w)
    assert near - far == 25 and abs(sum(near_pts.values()) - near) <= 1
    assert near_pts["location"] == 15 and near_pts["local"] == 10


def test_local_bonus_never_rescues_an_unqualified_match():
    w = {k: float(v) for k, v in DEFAULT_WEIGHTS.items()}
    p = parts(skills=2, experience=5, location=1.0, local=1.0)
    score, _ = combine(p, w)
    assert recommend(score, p, min_fit_score=50) == "Skip"


def test_prefer_local_off_removes_the_employer_weight():
    w = weights_for({}, {"target": {"prefer_local": False}})
    assert w["local"] == 0
    s, pts = combine(parts(local=1.0), w)
    assert pts["local"] == 0


def test_title_match():
    assert title_match("Office Manager - Dental", ["Office Manager"]) == 1.0
    assert title_match("Senior Office Administrator", ["Office Manager"]) == 0.5
    assert title_match("Forklift Driver", ["Office Manager", "Practice Manager"]) == 0.0


def test_essentials_keep_requirements_and_drop_boilerplate():
    desc = ("Smile Clinic is a family dental practice in Denver.\n"
            + "We offer dental insurance, 401(k) and paid time off.\n" * 20
            + "- 3+ years of office management experience required\n"
            + "- Proficient with QuickBooks and scheduling software\n"
            + "Smile Clinic is an equal opportunity employer without regard to race.\n" * 20)
    out = job_essentials(desc, max_chars=400)
    assert len(out) <= 400
    assert "QuickBooks" in out and "3+ years" in out
    assert "401(k)" not in out and "equal opportunity" not in out
    assert out.startswith("Smile Clinic is a family dental practice")
    assert job_essentials("short posting") == "short posting"


class OneShot:
    name, model = "fake", "m"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def chat(self, system, user, *, json_mode=False, temperature=None, max_tokens=None):
        self.calls.append((system, user, max_tokens))
        return self.replies.pop(0)


PROFILE = {"headline": "Office manager", "summary": "Runs busy clinics.",
           "target": {"titles": ["Office Manager"]}, "skills": {"practices": ["Scheduling"]},
           "experience": [{"company": "Bluebird Dental", "title": "Office Manager", "start": "2019-04",
                           "end": "present", "bullets": ["A very long bullet " * 30]}]}
JOB = Job(id="j", title="Office Manager", company="Smile", description="Needs scheduling. " * 300)


def test_quick_rate_is_short_compact_and_clamped():
    llm = OneShot('{"skills": 9, "experience": "2", "missing": ["Billing", "Spanish", "Excel", "CPR"]}')
    r = FitEvaluator(llm, PROFILE).quick_rate(JOB, essentials_chars=800)
    assert r == QuickRating(skills=5, experience=2, missing=["Billing", "Spanish", "Excel"])
    system, user, max_tokens = llm.calls[0]
    assert max_tokens and max_tokens <= 100
    assert "A very long bullet" not in user                     # compact profile, no bullets
    assert "Office Manager at Bluebird Dental" in user
    job_part = user.split("<untrusted_job_description", 1)[1]
    assert len(job_part) < 1000                                 # essentials only, not 5000 chars


def test_quick_rate_retries_once_then_fails():
    llm = OneShot("not json", json.dumps({"skills": 3, "experience": 4, "missing": []}))
    assert FitEvaluator(llm, PROFILE).quick_rate(JOB).skills == 3
    with pytest.raises(ValueError):
        FitEvaluator(OneShot("nope", "still nope"), PROFILE).quick_rate(JOB)
