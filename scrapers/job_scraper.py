"""Collect raw job postings from free sources.

* Job boards via ``python-jobspy`` (Indeed, LinkedIn, Glassdoor, Google, ...).
* Optional company career pages via Playwright (headless Chromium).

Third-party libraries are imported lazily so the rest of the app (and the
tests) work without them. Output is NOT sanitized here; the orchestrator runs
``scrapers.sanitizer.sanitize_job`` on every job before storing or prompting.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit

from core.models import Job, make_job_id

log = logging.getLogger(__name__)

JOBSPY_DELAY_SECONDS = 2.0
CAREER_PAGE_MAX_JOBS = 20
CAREER_PAGE_TIMEOUT_MS = 30_000
CAREER_PAGE_DELAY_SECONDS = 1.0


# --------------------------------------------------------------------------
# Pure helpers (testable without network)
# --------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # pandas NA / NaT and numpy NaN without importing pandas.
    try:
        if value != value:  # noqa: PLR0124 - NaN check
            return True
    except Exception:  # pd.NA raises on bool()
        return True
    type_name = type(value).__name__
    return type_name in ("NAType", "NaTType")


def _clean(value: Any) -> Any:
    return None if _is_missing(value) else value


def _str(value: Any) -> str:
    value = _clean(value)
    if value is None:
        return ""
    return str(value).strip()


def _opt_str(value: Any) -> str | None:
    s = _str(value)
    return s or None


def _float(value: Any) -> float | None:
    value = _clean(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(str(value).replace(",", "").strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _bool(value: Any) -> bool | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1", "remote"):
            return True
        if v in ("false", "no", "0"):
            return False
        return None
    try:
        return bool(value)
    except Exception:
        return None


def _date(value: Any) -> str | None:
    value = _clean(value)
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    iso = getattr(value, "isoformat", None)  # pandas Timestamp
    if callable(iso):
        try:
            return str(iso())
        except Exception:
            pass
    return _opt_str(value)


def _enum_str(value: Any) -> str | None:
    """jobspy sometimes yields enum members (e.g. ``Site.INDEED``)."""
    value = _clean(value)
    if value is None:
        return None
    inner = getattr(value, "value", None)
    if isinstance(inner, str):
        return inner.strip() or None
    s = str(value).strip()
    if "." in s and s.split(".", 1)[0][:1].isupper() and " " not in s:
        s = s.split(".", 1)[1].lower()
    return s or None


def _location(value: Any) -> str:
    value = _clean(value)
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = [_str(value.get(k)) for k in ("city", "state", "country")]
        return ", ".join(p for p in parts if p)
    disp = getattr(value, "display_location", None)  # jobspy Location model
    if callable(disp):
        try:
            return _str(disp())
        except Exception:
            pass
    return _str(value)


def _row_to_job(row: dict[str, Any], scraped_at: str | None = None) -> Job | None:
    """Map one jobspy DataFrame row (as a dict) to a Job, or None if unusable."""
    if not isinstance(row, dict):
        return None
    title = _str(row.get("title"))
    company = _str(row.get("company")) or _str(row.get("company_name"))
    url = _str(row.get("job_url")) or _str(row.get("job_url_direct")) or _str(row.get("url"))
    if not title or not (company or url):
        return None
    return Job(
        id=make_job_id(company, title, url),
        title=title,
        company=company,
        location=_location(row.get("location")),
        url=url,
        source=_enum_str(row.get("site")) or "",
        description=_str(row.get("description")),
        is_remote=_bool(row.get("is_remote")),
        salary_min=_float(row.get("min_amount")),
        salary_max=_float(row.get("max_amount")),
        salary_currency=_opt_str(row.get("currency")),
        salary_interval=_enum_str(row.get("interval")),
        date_posted=_date(row.get("date_posted")),
        company_employees=_opt_str(row.get("company_num_employees")),
        scraped_at=scraped_at or _now_iso(),
    )


def _dedupe(jobs: Iterable[Job]) -> list[Job]:
    seen: set[str] = set()
    out: list[Job] = []
    for job in jobs:
        if job.id in seen:
            continue
        seen.add(job.id)
        out.append(job)
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v not in (None, "")]
    return [value]


def _is_http_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _career_link_to_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    full = urljoin(base_url, href.strip())
    full = full.split("#", 1)[0]
    return full if _is_http_url(full) else None


# --------------------------------------------------------------------------
# python-jobspy
# --------------------------------------------------------------------------
def _dataframe_rows(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    to_dict = getattr(df, "to_dict", None)
    if callable(to_dict):
        try:
            return list(to_dict(orient="records"))
        except TypeError:
            pass
    if isinstance(df, list):
        return [r for r in df if isinstance(r, dict)]
    return []


@dataclass(frozen=True)
class SearchQuery:
    """One job-board search: a single site, title and location.

    Queries are the unit of work for the parallel search agents
    (core.agents); queries for the same site run one after another.
    """
    site: str
    term: str
    location: str = ""
    results_wanted: int = 20
    hours_old: int | None = None

    @property
    def label(self) -> str:
        where = f" in {self.location}" if self.location else ""
        return f"{self.site.replace('_', ' ').title()}: {self.term}{where}"


def build_queries(cfg: dict[str, Any]) -> list[SearchQuery]:
    terms = [str(t) for t in _as_list(cfg.get("search_terms")) if str(t).strip()]
    locations = [str(loc) for loc in _as_list(cfg.get("locations")) if str(loc).strip()] or [""]
    sites = [str(s) for s in _as_list(cfg.get("sites")) if str(s).strip()]
    results_wanted = int(cfg.get("results_wanted") or 20)
    hours_old = int(cfg["hours_old"]) if cfg.get("hours_old") else None
    return [SearchQuery(site, term, location, results_wanted, hours_old)
            for site in sites for term in terms for location in locations]


def jobspy_available() -> bool:
    try:
        import jobspy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def run_query(query: SearchQuery) -> list[Job]:
    """Run one job-board search. Raises on failure; callers decide how to continue."""
    from jobspy import scrape_jobs as jobspy_scrape  # type: ignore[import-not-found]

    kwargs: dict[str, Any] = {
        "site_name": [query.site],
        "search_term": query.term,
        "results_wanted": query.results_wanted,
    }
    if query.location:
        kwargs["location"] = query.location
        if query.location.strip().lower() == "remote":
            kwargs["is_remote"] = True
    if query.hours_old:
        kwargs["hours_old"] = query.hours_old
    if query.site == "google":
        kwargs["google_search_term"] = (
            f"{query.term} jobs near {query.location}" if query.location else f"{query.term} jobs")
    df = jobspy_scrape(**kwargs)
    scraped_at = _now_iso()
    jobs: list[Job] = []
    for row in _dataframe_rows(df):
        try:
            job = _row_to_job(row, scraped_at)
        except Exception as exc:
            log.warning("Skipping malformed jobspy row: %s", exc)
            continue
        if job:
            jobs.append(job)
    log.info("jobspy: %d jobs for %s", len(jobs), query.label)
    return jobs


def _scrape_jobspy(cfg: dict[str, Any]) -> list[Job]:
    """Sequential search (CLI `scrape`). The web app uses core.agents instead."""
    queries = build_queries(cfg)
    if not queries:
        log.info("No search_terms or sites configured; skipping job boards.")
        return []
    if not jobspy_available():
        log.warning("python-jobspy is not installed; skipping job boards. "
                    "Install it with: pip install python-jobspy")
        return []
    jobs: list[Job] = []
    for i, query in enumerate(queries):
        if i:
            time.sleep(JOBSPY_DELAY_SECONDS)
        try:
            jobs.extend(run_query(query))
        except Exception as exc:  # one failing board must not kill the run
            log.warning("jobspy query failed (%s): %s", query.label, exc)
    return jobs


# --------------------------------------------------------------------------
# Playwright career pages
# --------------------------------------------------------------------------
def _scrape_career_pages(pages: list[Any]) -> list[Job]:
    pages = [p for p in pages if isinstance(p, dict) and p.get("url")]
    if not pages:
        return []
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        log.warning("playwright is not installed; skipping career pages. Install it with: "
                    "pip install playwright && python -m playwright install chromium")
        return []

    jobs: list[Job] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context()
                for page_cfg in pages:
                    try:
                        jobs.extend(_scrape_one_career_page(context, page_cfg))
                    except Exception as exc:
                        log.warning("Career page %s failed: %s", page_cfg.get("url"), exc)
            finally:
                browser.close()
    except Exception as exc:
        log.warning("Playwright could not start (try: python -m playwright install chromium): %s", exc)
    return jobs


def _scrape_one_career_page(context: Any, cfg: dict[str, Any]) -> list[Job]:
    base_url = str(cfg["url"])
    if not _is_http_url(base_url):
        log.warning("Skipping career page with non-http(s) URL: %r", base_url)
        return []
    company = str(cfg.get("company") or urlsplit(base_url).netloc)
    selector = str(cfg.get("job_link_selector") or "a")
    max_jobs = int(cfg.get("max_jobs") or CAREER_PAGE_MAX_JOBS)

    page = context.new_page()
    try:
        page.goto(base_url, timeout=CAREER_PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
        try:
            page.wait_for_selector(selector, timeout=10_000)
        except Exception:
            log.info("Selector %r not found on %s", selector, base_url)
        links: list[tuple[str, str]] = []
        seen: set[str] = set()
        for el in page.query_selector_all(selector):
            url = _career_link_to_url(base_url, el.get_attribute("href"))
            if not url or url in seen:
                continue
            seen.add(url)
            links.append((url, " ".join((el.inner_text() or "").split())))
            if len(links) >= max_jobs:
                break

        jobs: list[Job] = []
        for url, link_text in links:
            time.sleep(CAREER_PAGE_DELAY_SECONDS)
            try:
                page.goto(url, timeout=CAREER_PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
                description = page.inner_text("body") or ""
                title = link_text
                if not title:
                    h1 = page.query_selector("h1")
                    title = " ".join((h1.inner_text() if h1 else page.title() or "").split())
            except Exception as exc:
                log.warning("Career job page %s failed: %s", url, exc)
                continue
            if not title:
                continue
            jobs.append(Job(
                id=make_job_id(company, title, url),
                title=title,
                company=company,
                url=url,
                source="career_page",
                description=description,
                scraped_at=_now_iso(),
            ))
        log.info("career page %s: %d jobs", base_url, len(jobs))
        return jobs
    finally:
        page.close()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def scrape_jobs(settings: dict[str, Any]) -> list[Job]:
    """Scrape all configured sources and return de-duplicated, UNSANITIZED jobs."""
    cfg = (settings or {}).get("scrape") or {}
    jobs: list[Job] = []
    try:
        jobs.extend(_scrape_jobspy(cfg))
    except Exception as exc:
        log.warning("Job board scraping failed: %s", exc)
    try:
        jobs.extend(_scrape_career_pages(_as_list(cfg.get("career_pages"))))
    except Exception as exc:
        log.warning("Career page scraping failed: %s", exc)
    unique = _dedupe(jobs)
    log.info("Scraped %d jobs (%d unique)", len(jobs), len(unique))
    return unique
