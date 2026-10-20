from core.models import Job
from matching.filter_engine import FilterDecision, FilterEngine, annualize, best_annual_salary

DESC = "We build Python backend services with FastAPI and PostgreSQL for analytics customers."


def make_job(**kw) -> Job:
    base = dict(id="j1", title="Backend Engineer", company="Acme", location="Austin, TX",
                description=DESC)
    base.update(kw)
    return Job(**base)


def test_passes_with_no_filters():
    d = FilterEngine({}).evaluate(make_job())
    assert isinstance(d, FilterDecision)
    assert d.keep and d.reasons == []


def test_empty_title_and_short_description():
    d = FilterEngine({}).evaluate(make_job(title="  ", description="too short"))
    assert not d.keep
    assert "empty_title" in d.reasons
    assert "description_too_short" in d.reasons


def test_title_blacklist_case_insensitive():
    eng = FilterEngine({"title_blacklist": ["intern"]})
    d = eng.evaluate(make_job(title="Software INTERN"))
    assert not d.keep and any(r.startswith("title_blacklist") for r in d.reasons)


def test_title_keywords_required():
    eng = FilterEngine({"title_keywords_required": ["python", "backend"]})
    assert eng.evaluate(make_job(title="Backend Engineer")).keep
    d = eng.evaluate(make_job(title="Frontend Engineer"))
    assert not d.keep and "title_missing_required_keyword" in d.reasons
    assert FilterEngine({"title_keywords_required": []}).evaluate(make_job(title="Chef")).keep


def test_company_blacklist():
    d = FilterEngine({"company_blacklist": ["evilcorp"]}).evaluate(make_job(company="EvilCorp Inc"))
    assert not d.keep and any(r.startswith("company_blacklist") for r in d.reasons)


def test_location_allowlist_and_remote():
    eng = FilterEngine({"location_allowlist": ["Austin", "Remote"]})
    assert eng.evaluate(make_job(location="Austin, TX")).keep
    assert eng.evaluate(make_job(location="Anywhere", is_remote=True)).keep
    d = eng.evaluate(make_job(location="Boston, MA"))
    assert not d.keep and "location_not_allowed" in d.reasons
    # Remote job does not pass if "remote" isn't in the allowlist.
    eng2 = FilterEngine({"location_allowlist": ["Austin"]})
    assert not eng2.evaluate(make_job(location="Denver", is_remote=True)).keep


def test_location_blacklist():
    d = FilterEngine({"location_blacklist": ["new york"]}).evaluate(make_job(location="New York, NY"))
    assert not d.keep and any(r.startswith("location_blacklist") for r in d.reasons)


def test_remote_only():
    eng = FilterEngine({"remote_only": True})
    assert eng.evaluate(make_job(is_remote=True)).keep
    assert eng.evaluate(make_job(location="Remote - US")).keep
    d = eng.evaluate(make_job(location="Austin, TX", is_remote=False))
    assert not d.keep and "not_remote" in d.reasons


def test_salary_annualization():
    assert annualize(50, "hourly") == 104000
    assert annualize(400, "daily") == 104000
    assert annualize(2000, "weekly") == 104000
    assert annualize(10000, "monthly") == 120000
    assert annualize(100000, "yearly") == 100000
    assert annualize(100000, None) == 100000
    assert annualize(None, "hourly") is None
    assert best_annual_salary(make_job(salary_min=40, salary_max=60, salary_interval="hourly")) == 124800
    assert best_annual_salary(make_job(salary_min=40, salary_interval="hourly")) == 83200


def test_min_salary_rule():
    eng = FilterEngine({"min_salary": 100000})
    assert eng.evaluate(make_job(salary_max=60, salary_interval="hourly")).keep  # 124,800
    d = eng.evaluate(make_job(salary_max=40, salary_interval="hourly"))  # 83,200
    assert not d.keep and any(r.startswith("salary_below_min") for r in d.reasons)
    assert not eng.evaluate(make_job(salary_max=8000, salary_interval="monthly")).keep  # 96k


def test_min_salary_falls_back_to_profile():
    eng = FilterEngine({}, profile={"target": {"min_salary": 110000}})
    assert not eng.evaluate(make_job(salary_max=100000, salary_interval="yearly")).keep
    assert eng.evaluate(make_job(salary_max=120000, salary_interval="yearly")).keep
    # Explicit filter value wins over the profile.
    eng2 = FilterEngine({"min_salary": 90000}, profile={"target": {"min_salary": 110000}})
    assert eng2.evaluate(make_job(salary_max=100000)).keep


def test_unknown_salary():
    assert FilterEngine({"min_salary": 90000}).evaluate(make_job()).keep
    d = FilterEngine({"min_salary": 90000, "drop_if_salary_unknown": True}).evaluate(make_job())
    assert not d.keep and "salary_unknown" in d.reasons


def test_description_blacklist():
    eng = FilterEngine({"description_blacklist": ["security clearance required"]})
    d = eng.evaluate(make_job(description=DESC + " Active Security Clearance Required."))
    assert not d.keep and any(r.startswith("description_blacklist") for r in d.reasons)


def test_multiple_reasons_collected():
    eng = FilterEngine({
        "title_blacklist": ["intern"],
        "company_blacklist": ["acme"],
        "remote_only": True,
        "min_salary": 100000,
        "description_blacklist": ["ts/sci"],
    })
    job = make_job(title="Intern", is_remote=False, salary_max=20, salary_interval="hourly",
                   description=DESC + " TS/SCI needed.")
    d = eng.evaluate(job)
    assert not d.keep
    prefixes = {r.split(":")[0] for r in d.reasons}
    assert {"title_blacklist", "company_blacklist", "not_remote", "salary_below_min",
            "description_blacklist"} <= prefixes


def test_apply_splits_kept_and_dropped():
    eng = FilterEngine({"title_blacklist": ["intern"]})
    good, bad = make_job(id="a"), make_job(id="b", title="Intern")
    kept, dropped = eng.apply([good, bad])
    assert kept == [good]
    assert len(dropped) == 1 and dropped[0][0] is bad and dropped[0][1]
