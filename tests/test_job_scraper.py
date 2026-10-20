import builtins
import logging
import math
import sys
import types
from datetime import date

import pytest

from core.models import make_job_id
from scrapers import job_scraper
from scrapers.job_scraper import _career_link_to_url, _dedupe, _row_to_job, scrape_jobs

NAN = float("nan")


def _row(**over):
    row = {
        "title": "Backend Engineer",
        "company": "Acme",
        "location": "Austin, TX, US",
        "job_url": "https://www.indeed.com/viewjob?jk=1",
        "site": "indeed",
        "description": "Build APIs.",
        "is_remote": False,
        "min_amount": 100000.0,
        "max_amount": 140000.0,
        "currency": "USD",
        "interval": "yearly",
        "date_posted": date(2026, 10, 1),
    }
    row.update(over)
    return row


def test_row_to_job_full_mapping():
    job = _row_to_job(_row(), scraped_at="2026-10-20T00:00:00+00:00")
    assert job is not None
    assert job.id == make_job_id("Acme", "Backend Engineer", "https://www.indeed.com/viewjob?jk=1")
    assert job.title == "Backend Engineer"
    assert job.company == "Acme"
    assert job.location == "Austin, TX, US"
    assert job.url == "https://www.indeed.com/viewjob?jk=1"
    assert job.source == "indeed"
    assert job.description == "Build APIs."
    assert job.is_remote is False
    assert job.salary_min == 100000.0 and job.salary_max == 140000.0
    assert job.salary_currency == "USD"
    assert job.salary_interval == "yearly"
    assert job.date_posted == "2026-10-01"
    assert job.scraped_at == "2026-10-20T00:00:00+00:00"
    assert job.sanitizer_flags == []


def test_row_to_job_nan_values_become_none():
    job = _row_to_job(_row(location=NAN, description=NAN, is_remote=NAN, min_amount=NAN,
                           max_amount=None, currency=NAN, interval=NAN, date_posted=NAN, site=NAN))
    assert job is not None
    assert job.location == ""
    assert job.description == ""
    assert job.is_remote is None
    assert job.salary_min is None and job.salary_max is None
    assert job.salary_currency is None and job.salary_interval is None
    assert job.date_posted is None
    assert job.source == ""
    assert job.scraped_at.endswith("+00:00")
    for v in (job.salary_min, job.salary_max):
        assert not (isinstance(v, float) and math.isnan(v))


def test_row_to_job_rejects_unusable_rows():
    assert _row_to_job(_row(title=NAN)) is None
    assert _row_to_job(_row(title="")) is None
    assert _row_to_job(_row(company=NAN, job_url=NAN)) is None
    assert _row_to_job("not a dict") is None  # type: ignore[arg-type]


def test_row_to_job_enum_like_site_and_string_amounts():
    class Site:
        value = "linkedin"
    job = _row_to_job(_row(site=Site(), min_amount="90,000", is_remote="True"))
    assert job.source == "linkedin"
    assert job.salary_min == 90000.0
    assert job.is_remote is True


def test_dedupe_keeps_first():
    a = _row_to_job(_row(description="first"))
    b = _row_to_job(_row(description="second"))
    c = _row_to_job(_row(title="Other"))
    out = _dedupe([a, b, c])
    assert [j.description for j in out] == ["first", "Build APIs."]


def test_career_link_to_url():
    base = "https://example.com/careers/"
    assert _career_link_to_url(base, "/jobs/1#apply") == "https://example.com/jobs/1"
    assert _career_link_to_url(base, "role-2") == "https://example.com/careers/role-2"
    assert _career_link_to_url(base, "javascript:void(0)") is None
    assert _career_link_to_url(base, "mailto:x@example.com") is None
    assert _career_link_to_url(base, None) is None


class _FakeDF:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return self.rows


def test_scrape_jobs_with_fake_jobspy(monkeypatch):
    calls = []

    def fake_scrape(**kwargs):
        calls.append(kwargs)
        if kwargs["search_term"] == "boom":
            raise RuntimeError("board down")
        return _FakeDF([_row(), _row(), _row(title=f"Dev {kwargs['search_term']}", location=NAN)])

    monkeypatch.setitem(sys.modules, "jobspy", types.SimpleNamespace(scrape_jobs=fake_scrape))
    monkeypatch.setattr(job_scraper, "JOBSPY_DELAY_SECONDS", 0)
    settings = {"scrape": {"search_terms": ["python", "boom"], "locations": ["Remote"],
                           "sites": ["indeed"], "results_wanted": 5, "hours_old": 24,
                           "career_pages": []}}
    jobs = scrape_jobs(settings)
    assert len(calls) == 2
    assert calls[0]["is_remote"] is True and calls[0]["hours_old"] == 24
    assert calls[0]["site_name"] == ["indeed"] and calls[0]["results_wanted"] == 5
    assert sorted(j.title for j in jobs) == ["Backend Engineer", "Dev python"]
    assert len({j.id for j in jobs}) == len(jobs)


def test_scrape_jobs_missing_libraries(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("jobspy", "playwright.sync_api", "playwright"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "jobspy", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    settings = {"scrape": {"search_terms": ["x"], "locations": ["Remote"], "sites": ["indeed"],
                           "career_pages": [{"url": "https://example.com/careers"}]}}
    with caplog.at_level(logging.WARNING, logger="scrapers.job_scraper"):
        assert scrape_jobs(settings) == []
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "pip install python-jobspy" in msgs
    assert "pip install playwright" in msgs


def test_scrape_jobs_empty_settings():
    assert scrape_jobs({}) == []
