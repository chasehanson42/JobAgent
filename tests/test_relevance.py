"""Tests for deterministic resume content selection (no network, no LLM)."""
from __future__ import annotations

import time

from core.models import Job
from generators.relevance import (
    ExperiencePlan,
    JobKeywords,
    is_grounded_rewrite,
    job_keywords,
    plan_experience,
    relevance,
    select_skills,
)

OFFICE = {
    "skills": {
        "software": ["QuickBooks", "Microsoft Excel", "Outlook", "Photoshop"],
        "office": ["Scheduling", "Payroll", "Customer service", "Knife skills"],
    },
    "experience": [
        {
            "company": "Brightside Dental",
            "title": "Office Manager",
            "location": "Austin, TX",
            "start": "2021-01",
            "end": "present",
            "bullets": [
                "Greeted patients and answered a busy front desk phone line.",
                "Managed staff schedules for 14 employees using a shared calendar.",
                "Decorated the break room for holiday parties.",
                "Reconciled accounts payable and payroll in QuickBooks every week.",
                "Ordered office supplies and kept vendor invoices organized.",
            ],
        },
        {
            "company": "Corner Diner",
            "title": "Line Cook",
            "location": "Austin, TX",
            "start": "2018-05",
            "end": "2020-12",
            "bullets": [
                "Prepped vegetables and grilled burgers during the lunch rush.",
                "Kept the grill station clean to health code standards.",
            ],
        },
        {
            "company": "Lone Star Realty",
            "title": "Administrative Assistant",
            "location": "Austin, TX",
            "start": "2016-01",
            "end": "2018-04",
            "bullets": [
                "Filed closing documents in alphabetical cabinets.",
                "Scheduled client appointments and coordinated agent calendars.",
                "Prepared monthly invoices in QuickBooks.",
            ],
        },
    ],
}

OFFICE_JOB = Job(
    id="om1", title="Office Manager", company="Hill Country Clinic",
    description=(
        "We are seeking an Office Manager to run our busy clinic.\n"
        "- Manage scheduling for providers and staff\n"
        "- Process payroll and accounts payable in QuickBooks\n"
        "- Order supplies and track vendor invoices\n"
        "- Excellent customer service skills required\n"
        "- 3+ years of experience preferred"
    ),
)

DEV = {
    "skills": {
        "languages": ["Python", "Go", "C++", "SQL"],
        "tools": ["PostgreSQL", "Docker", "Figma"],
    },
    "experience": [
        {
            "company": "Acme",
            "title": "Software Engineer",
            "start": "2021-01", "end": "present",
            "bullets": [
                "Organized the office ping pong tournament.",
                "Designed REST API endpoints in Python backed by PostgreSQL.",
                "Created pixel-perfect landing page mockups in Figma.",
                "Tuned slow database queries, cutting p95 latency from 900ms to 180ms.",
            ],
        },
    ],
}

BACKEND_JOB = Job(
    id="be1", title="Backend Engineer", company="Initech",
    description="Build REST API services in Python. Design PostgreSQL database schemas "
                "and optimize queries. Experience with C++ a plus. We use Google Cloud.",
)


def _kw(title, description=""):
    return job_keywords(Job(id="x", title=title, company="X", description=description))


# --------------------------------------------------------------------------- #
# Keywords / stemming
# --------------------------------------------------------------------------- #

def test_job_keywords_drop_generic_words_and_keep_title():
    kw = job_keywords(OFFICE_JOB)
    assert isinstance(kw, JobKeywords)
    assert kw.title_words <= kw.words
    assert "offic" in kw.title_words and "manag" in kw.title_words
    for generic in ("experience", "years", "excellent", "required", "skills", "the", "and"):
        assert generic not in kw.words
    assert "quickbooks" in kw.text


def test_stemming_matches_word_forms():
    kw = _kw("Scheduler", "Responsible for scheduling and managing calendars.")
    assert relevance("Scheduled meetings", kw) > 0
    assert relevance("Managed the calendar", kw) >= 2
    assert relevance("Office manager", _kw("Office Manager")) == 4
    assert relevance("management of office", _kw("Office Manager")) == 4


def test_relevance_weights_title_words():
    kw = _kw("Payroll Specialist", "Handle invoices.")
    assert relevance("payroll", kw) == 2
    assert relevance("invoices", kw) == 1
    assert relevance("payroll invoices payroll", kw) == 3
    assert relevance("baked bread", kw) == 0


def test_short_tech_tokens():
    kw = _kw("IT Support Technician", "Support RN staff with EHR access; go the extra mile.")
    assert "it" in kw.words and "rn" in kw.words and "ehr" in kw.words
    assert "go" not in kw.words  # lower-case "go" is an ordinary verb
    assert relevance("Ran the IT help desk", kw) >= 2


# --------------------------------------------------------------------------- #
# Experience planning
# --------------------------------------------------------------------------- #

def test_office_plan_keeps_cook_condensed_and_picks_relevant_bullets():
    kw = job_keywords(OFFICE_JOB)
    plans = plan_experience(OFFICE, kw)
    assert all(isinstance(p, ExperiencePlan) for p in plans)
    # Every job kept, in profile order, entries untouched.
    assert [p.entry["company"] for p in plans] == ["Brightside Dental", "Corner Diner",
                                                   "Lone Star Realty"]
    assert plans[0].entry is OFFICE["experience"][0]

    cook = plans[1]
    assert not cook.detailed and cook.bullets == []
    assert cook.score < plans[0].score and cook.score < plans[2].score

    om = plans[0]
    assert om.detailed
    assert om.bullets == [
        "Managed staff schedules for 14 employees using a shared calendar.",
        "Reconciled accounts payable and payroll in QuickBooks every week.",
        "Ordered office supplies and kept vendor invoices organized.",
    ]
    assert "Decorated the break room for holiday parties." not in om.bullets

    admin = plans[2]
    assert admin.detailed
    assert admin.bullets == [
        "Scheduled client appointments and coordinated agent calendars.",
        "Prepared monthly invoices in QuickBooks.",
    ]


def test_bullets_per_job_honored_and_original_order():
    kw = job_keywords(OFFICE_JOB)
    plans = plan_experience(OFFICE, kw, bullets_per_job=2)
    assert len(plans[0].bullets) == 2
    src = OFFICE["experience"][0]["bullets"]
    idx = [src.index(b) for b in plans[0].bullets]
    assert idx == sorted(idx)
    assert all(len(p.bullets) <= 2 for p in plans)


def test_most_recent_always_detailed():
    # A posting for cooks: the office job (most recent) still gets detailed,
    # and with detailed_jobs=1 the cook job is the other detailed one.
    kw = _kw("Line Cook", "Grill burgers and prep vegetables during the lunch rush.")
    plans = plan_experience(OFFICE, kw, detailed_jobs=1)
    assert plans[0].detailed
    assert plans[1].detailed and plans[1].bullets
    assert not plans[2].detailed and plans[2].bullets == []


def test_zero_score_fallback_keeps_first_bullet():
    kw = _kw("Line Cook", "Grill burgers and prep vegetables during the lunch rush.")
    plans = plan_experience(OFFICE, kw, detailed_jobs=0)
    assert plans[0].detailed
    assert plans[0].bullets == ["Greeted patients and answered a busy front desk phone line."]
    assert [p.detailed for p in plans] == [True, False, False]


def test_plan_empty_profile():
    assert plan_experience({}, job_keywords(OFFICE_JOB)) == []


def test_developer_backend_job_prefers_api_and_database_bullets():
    kw = job_keywords(BACKEND_JOB)
    plan = plan_experience(DEV, kw, bullets_per_job=2)[0]
    assert plan.bullets == [
        "Designed REST API endpoints in Python backed by PostgreSQL.",
        "Tuned slow database queries, cutting p95 latency from 900ms to 180ms.",
    ]


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #

def test_select_skills_matched_first_in_profile_order():
    kw = job_keywords(OFFICE_JOB)
    skills = select_skills(OFFICE, kw)
    assert skills[:4] == ["QuickBooks", "Scheduling", "Payroll", "Customer service"]
    assert set(skills) == {s for bucket in OFFICE["skills"].values() for s in bucket}


def test_select_skills_cap_and_never_empty():
    kw = job_keywords(OFFICE_JOB)
    assert select_skills(OFFICE, kw, max_skills=2) == ["QuickBooks", "Scheduling"]
    unrelated = _kw("Astronaut", "Fly to the moon.")
    assert len(select_skills(OFFICE, unrelated, max_skills=3)) == 3
    assert select_skills(OFFICE, unrelated, max_skills=0)  # never empty
    assert select_skills({"skills": {}}, unrelated) == []


def test_select_skills_go_vs_google_and_cpp():
    kw = job_keywords(BACKEND_JOB)
    skills = select_skills(DEV, kw)
    assert skills[:3] == ["Python", "C++", "PostgreSQL"]  # "SQL" alone is not in the text
    assert skills.index("Go") >= 3
    go_kw = _kw("Go Developer", "Write services in Go.")
    assert select_skills(DEV, go_kw)[0] == "Go"
    verb_kw = _kw("Developer", "Willing to go the extra mile.")
    assert select_skills(DEV, verb_kw, max_skills=20)[0] != "Go"


def test_select_skills_never_invents():
    kw = job_keywords(BACKEND_JOB)
    profile_skills = {s for b in DEV["skills"].values() for s in b}
    assert set(select_skills(DEV, kw, max_skills=50)) <= profile_skills


# --------------------------------------------------------------------------- #
# Grounding check
# --------------------------------------------------------------------------- #

ORIG = "Managed staff schedules for 14 employees using a shared calendar."
ALLOWED = "QuickBooks, Outlook, Scheduling, Payroll. Brightside Dental"


def test_grounded_accepts_faithful_rephrase():
    assert is_grounded_rewrite(ORIG, "Managed scheduling for 14 staff employees with a shared calendar.",
                               ALLOWED)
    assert is_grounded_rewrite(ORIG, "Managed shared calendar schedules for 14 employees in Outlook.",
                               ALLOWED)
    assert is_grounded_rewrite("Processed 2,500 invoices monthly.",
                               "Processed 2500 invoices each month.")


def test_grounded_rejects_invented_number():
    assert not is_grounded_rewrite(ORIG, "Managed staff schedules for 14 employees, "
                                         "increased revenue 40%.", ALLOWED)
    assert not is_grounded_rewrite(ORIG, "Managed schedules for $1M budget and 14 employees.",
                                   ALLOWED)


def test_grounded_rejects_invented_claim():
    assert not is_grounded_rewrite(
        ORIG, "Spearheaded enterprise transformation at Google, driving cloud strategy globally.",
        ALLOWED)


def test_grounded_rejects_overlong_and_multiline_and_empty():
    long = ORIG + " " + " ".join(["Managed staff schedules for employees."] * 6)
    assert len(long) > 2 * len(ORIG) + 80
    assert not is_grounded_rewrite(ORIG, long, ALLOWED)
    assert not is_grounded_rewrite(ORIG, "Managed staff schedules.\nUsed a shared calendar.", ALLOWED)
    assert not is_grounded_rewrite(ORIG, "   ", ALLOWED)


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #

def test_fast_enough():
    big = Job(id="b", title="Office Manager", company="X",
              description=OFFICE_JOB.description * 50)
    start = time.perf_counter()
    for _ in range(50):
        kw = job_keywords(big)
        plan_experience(OFFICE, kw)
        select_skills(OFFICE, kw)
    assert time.perf_counter() - start < 2.0
