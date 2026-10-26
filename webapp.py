"""Pages of the JobAgent web app. app.py is the entry point that wires them together."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from core import __version__ as APP_VERSION, llm_client, ollama_admin, pipeline
from core.background import MANAGER, Task, TaskBusy
from core.settings import (has_user_profile, load_profile, load_settings, load_user_overrides,
                           save_profile, save_user_overrides)
from core.storage import Storage
from scrapers.job_scraper import jobspy_available
from importers.resume_importer import (ResumeImportError, draft_profile, empty_profile,
                                       extract_text, validate_profile)

MODEL_SIZES = {"qwen2.5:7b-instruct": "4.7 GB", "qwen2.5:3b-instruct": "1.9 GB"}
JOB_SITES = ["indeed", "linkedin", "glassdoor", "zip_recruiter", "google"]
DOC_LABELS = {"resume": "Resume", "cover_letter": "Cover letter"}


def build_client(s: dict[str, Any], purpose: str = "general"):
    return llm_client.build_client(s, purpose=purpose)


# ---------------------------------------------------------------- helpers

def ollama_settings(s: dict[str, Any]) -> tuple[str, str]:
    cfg = s.get("llm", {}).get("ollama", {})
    return cfg.get("base_url", "http://localhost:11434"), cfg.get("model", "qwen2.5:7b-instruct")


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "Cannot reach" in msg:
        return ("JobAgent can't reach Ollama. Open the Ollama app (Start menu on Windows, "
                "Applications on Mac), wait a few seconds, then try again.")
    if "HTTP 404" in msg and "model" in msg.lower():
        return "The AI model isn't downloaded yet. Go to **Setup** and click **Download**."
    if "timed out" in msg:
        return ("The AI took too long to answer. Close other heavy programs and try again, "
                "or choose the smaller model on the **Setup** page.")
    if isinstance(exc, (pipeline.PipelineError, ResumeImportError, ollama_admin.OllamaAdminError)):
        return msg
    return f"Something went wrong: {msg[:300]}"


def show_error(exc: Exception) -> None:
    st.error(friendly_error(exc))
    if not isinstance(exc, (pipeline.PipelineError, ResumeImportError)):
        with st.expander("Technical details (for whoever set this up)"):
            st.code(f"{type(exc).__name__}: {exc}")


def _md(text: str) -> str:
    """Escape Markdown in text that came from a job posting or the AI."""
    return "".join("\\" + ch if ch in "\\`*_[]<>#|~$" else ch for ch in str(text))


def lines(text: str) -> list[str]:
    return [x.strip() for x in text.splitlines() if x.strip()]


def commas(text: str) -> list[str]:
    return [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]


def setup_complete(s: dict[str, Any]) -> bool:
    base, model = ollama_settings(s)
    return has_user_profile(s) and ollama_admin.has_model(base, model)


# ---------------------------------------------------------------- background tasks

def start_task(key: str, label: str, fn) -> None:
    """Start fn(task) in the background, then refresh so its progress shows."""
    try:
        MANAGER.start(key, label, fn)
    except TaskBusy as exc:
        st.warning(str(exc))
        return
    st.rerun()


def busy_reason() -> str | None:
    t = MANAGER.running()
    return f"Waiting for “{t.label}” to finish" if t else None


def _elapsed(t: Task) -> str:
    import time
    secs = int((t.finished or time.time()) - t.started)
    return f"{secs // 60}:{secs % 60:02d}"


def _refresh_when_finished(t: Task) -> None:
    """Re-run the whole page once after a task ends, so its results appear."""
    seen = st.session_state.setdefault("seen_tasks", set())
    if not t.running and t.id not in seen:
        seen.add(t.id)
        st.rerun()


@st.fragment(run_every=1.5)
def task_progress(key: str) -> None:
    """Live progress for one task; refreshes itself without re-running the page."""
    t = MANAGER.get(key)
    if t is None:
        return
    if t.running:
        st.progress(t.progress or 0.0, text=f"{t.message or t.label} · {_elapsed(t)}")
        st.caption("You can keep using JobAgent while this runs. It stops only if you close "
                   "the black JobAgent window.")
    _refresh_when_finished(t)


@st.fragment(run_every=2)
def sidebar_status() -> None:
    """Shown on every page, so a running task is never 'lost' after navigating away."""
    t = MANAGER.latest()
    if t is None:
        return
    if t.running:
        st.info(f"⏳ **{t.label}**  \n{t.message or 'Working...'}")
        st.progress(t.progress or 0.0, text=_elapsed(t))
    elif t.status == "done":
        st.success(f"✅ Finished: {t.label}")
    else:
        st.error(f"Couldn't finish: {t.label}")
    _refresh_when_finished(t)


def task_error(key: str) -> None:
    t = MANAGER.get(key)
    if t is not None and t.status == "error":
        show_error(t.error)


# ---------------------------------------------------------------- setup

def page_setup() -> None:
    s = load_settings()
    base, model = ollama_settings(s)
    st.title("Welcome to JobAgent")
    if setup_complete(s):
        st.success("You're all set up.")
        st.page_link(PAGES["ranked"], label="Go to your matches", icon="⭐")
    st.write("JobAgent finds jobs, tells you which ones fit you, and writes a matching resume "
             "and cover letter. Everything stays on your computer and it's free.")

    st.subheader("Step 1 · The AI engine (Ollama)")
    running = ollama_admin.is_running(base)
    if running:
        st.success("Ollama is running.")
    else:
        st.error("Ollama isn't running on this computer.")
        st.markdown(
            "- **Don't have it yet?** Download and install it from "
            "[ollama.com/download](https://ollama.com/download), like any other program.\n"
            "- **Already installed?** Open **Ollama** from the Start menu (Windows) or "
            "Applications (Mac). It runs quietly in the background.")
        st.button("I've opened it, check again")

    st.subheader("Step 2 · The AI model")
    if running and ollama_admin.has_model(base, model):
        st.success(f"The model **{model}** is downloaded and ready.")
    else:
        ram = ollama_admin.total_ram_gb()
        recommended = ollama_admin.recommended_model(ram)
        if ram:
            st.write(f"Your computer has about **{ram:.0f} GB** of memory, so we recommend "
                     f"**{recommended}**.")
        options = [recommended] + [m for m in MODEL_SIZES if m != recommended]
        choice = st.radio("Model to download", options,
                          format_func=lambda m: f"{m} ({MODEL_SIZES.get(m, '?')} download)"
                          + ("  · recommended" if m == recommended else ""))
        st.caption("This is a one-time download. It can take 5–30 minutes depending on your internet.")
        pulling = MANAGER.get("pull")
        if pulling and pulling.running:
            task_progress("pull")
        elif st.button("Download", type="primary", disabled=not running or bool(busy_reason()),
                       help=busy_reason()):
            start_task("pull", f"Downloading {choice}", lambda t: _pull(t, base, choice))
        task_error("pull")

    if running:
        others = [m for m in ollama_admin.list_models(base) if m != model]
        if others:
            with st.expander("Use a different model you already have"):
                pick = st.selectbox("Installed models", others)
                if st.button("Use this model"):
                    save_user_overrides({"llm": {"provider": "ollama", "ollama": {"model": pick}}})
                    st.rerun()

    if running and ollama_admin.has_model(base, model):
        _fast_scoring_option(s, base, model)

    st.subheader("Step 3 · Your resume")
    if has_user_profile(s):
        st.success(f"Profile saved for **{load_profile(s).get('name') or 'you'}**.")
    else:
        st.info("Upload your current resume so JobAgent knows your experience.")
    st.page_link(PAGES["profile"], label="Go to My profile", icon="👤")


def _pull(task: Task, base: str, model: str, role: str = "model") -> None:
    """Download a model; role "model" makes it the main model, "scoring_model" the quick-rating one."""
    for ev in ollama_admin.pull_model(base, model):
        done, total = ev.get("completed"), ev.get("total")
        if done and total:
            task.update(done / total, f"{ev['status']} · {done / 1e9:.1f} of {total / 1e9:.1f} GB")
        else:
            task.update(None, ev.get("status", ""))
    save_user_overrides({"llm": {"provider": "ollama", "ollama": {role: model}}})


FAST_MODEL = "qwen2.5:3b-instruct"


def _fast_scoring_option(s: dict[str, Any], base: str, model: str) -> None:
    if model == FAST_MODEL:
        return
    current = s.get("llm", {}).get("ollama", {}).get("scoring_model") or ""
    with st.expander("Optional · Faster job scoring"):
        st.write(f"Use the small model ({MODEL_SIZES[FAST_MODEL]} download) for the quick first-pass "
                 "ratings: roughly twice as fast. Explanations, resumes and cover letters still use "
                 f"**{model}**.")
        pulling = MANAGER.get("pull_fast")
        if pulling and pulling.running:
            task_progress("pull_fast")
            return
        on = st.toggle("Use the small model for quick ratings", value=bool(current))
        if on and not current:
            if ollama_admin.has_model(base, FAST_MODEL):
                save_user_overrides({"llm": {"ollama": {"scoring_model": FAST_MODEL}}})
                st.rerun()
            elif st.button("Download the small model", disabled=bool(busy_reason()), help=busy_reason()):
                start_task("pull_fast", f"Downloading {FAST_MODEL}",
                           lambda t: _pull(t, base, FAST_MODEL, role="scoring_model"))
        elif not on and current:
            save_user_overrides({"llm": {"ollama": {"scoring_model": ""}}})
            st.rerun()
        task_error("pull_fast")


# ---------------------------------------------------------------- profile

def _read_resume(task: Task, client, text: str) -> tuple[dict[str, Any], list[str]]:
    task.update(None, "Reading your resume. This can take a minute or two...")
    return draft_profile(client, text)


def _load_draft(profile: dict[str, Any], warnings: list[str] | None = None) -> None:
    st.session_state.draft = profile
    st.session_state.import_warnings = warnings or []
    st.session_state.ver = st.session_state.get("ver", 0) + 1


def page_profile() -> None:
    s = load_settings()
    if "draft" not in st.session_state:
        _load_draft(validate_profile(load_profile(s))[0] if has_user_profile(s) else empty_profile())

    st.title("My profile")
    st.write("JobAgent only uses what's on this page. It rewords and highlights your real "
             "experience for each job, but never invents anything.")

    reading = MANAGER.get("resume")
    if reading and reading.status == "done" and st.session_state.get("resume_loaded") != reading.id:
        st.session_state.resume_loaded = reading.id
        _load_draft(*reading.result)

    with st.container(border=True):
        st.subheader("Fill this in from your resume")
        up = st.file_uploader("Your current resume (PDF, Word or text)", type=["pdf", "docx", "txt", "md"])
        if reading and reading.running:
            task_progress("resume")
        elif up is not None and st.button("Read my resume", type="primary", disabled=bool(busy_reason()),
                                          help=busy_reason()):
            try:
                text = extract_text(up.name, up.getvalue())
            except ResumeImportError as exc:
                show_error(exc)
            else:
                client = build_client(s)
                start_task("resume", "Reading your resume", lambda t: _read_resume(t, client, text))
        task_error("resume")

    if st.session_state.import_warnings:
        st.warning("**Please double-check these before saving:**\n\n"
                   + "\n".join(f"- {w}" for w in st.session_state.import_warnings))

    d, v = st.session_state.draft, st.session_state.ver

    def k(name: str) -> str:
        return f"p{v}_{name}"

    cur: dict[str, Any] = {}
    st.subheader("About you")
    c1, c2 = st.columns(2)
    cur["name"] = c1.text_input("Full name", d["name"], key=k("name"))
    cur["email"] = c2.text_input("Email", d["email"], key=k("email"))
    cur["phone"] = c1.text_input("Phone", d["phone"], key=k("phone"))
    cur["location"] = c2.text_input("City, State", d["location"], key=k("location"))
    cur["links"] = lines(st.text_area("Links (LinkedIn, portfolio...), one per line",
                                      "\n".join(d["links"]), key=k("links"), height=80))
    cur["headline"] = st.text_input("Headline", d["headline"], key=k("headline"),
                                    placeholder="e.g. Registered nurse with 8 years in pediatric care")
    cur["summary"] = st.text_area("Summary", d["summary"], key=k("summary"), height=100)

    st.subheader("What you're looking for")
    t = d["target"]
    titles = st.text_input("Job titles you want (comma separated)", ", ".join(t["titles"]), key=k("titles"))
    c1, c2 = st.columns(2)
    min_salary = c1.number_input("Minimum yearly salary (0 = any)", min_value=0, step=5000,
                                 value=int(t["min_salary"] or 0), key=k("min_salary"))
    remote_ok = c2.checkbox("Open to remote work", bool(t["remote_ok"]), key=k("remote_ok"))
    c1, c2 = st.columns(2)
    commute = c1.number_input("How far will you travel to work? (miles)", min_value=1, max_value=500,
                              value=int(t.get("commute_miles") or 25), key=k("commute"))
    prefer_local = c2.checkbox("Rank small local employers higher", bool(t.get("prefer_local", True)),
                               key=k("prefer_local"),
                               help="Small businesses near you often favour people who live in the area.")
    cur["target"] = {"titles": commas(titles), "min_salary": int(min_salary) or None, "remote_ok": remote_ok,
                     "commute_miles": int(commute), "prefer_local": prefer_local}

    st.subheader("Skills")
    st.caption("Comma separated. Use whichever boxes fit your field; leave the rest empty.")
    sk = d["skills"]
    labels = {"languages": "Tools & software", "frameworks": "Methods & specialties",
              "infrastructure": "Systems & equipment", "practices": "Other skills"}
    cur["skills"] = {key: commas(st.text_input(label, ", ".join(sk.get(key, [])), key=k(f"sk_{key}")))
                     for key, label in labels.items()}

    st.subheader("Work experience")
    action: tuple[str, int] | None = None
    cur["experience"] = []
    for i, job in enumerate(d["experience"]):
        heading = f"{job['title']} at {job['company']}" if job["company"] else "New job"
        with st.expander(heading, expanded=not job["company"]):
            c1, c2 = st.columns(2)
            e = {"company": c1.text_input("Company", job["company"], key=k(f"x{i}_company")),
                 "title": c2.text_input("Job title", job["title"], key=k(f"x{i}_title")),
                 "location": c1.text_input("Location", job["location"], key=k(f"x{i}_location"))}
            c3, c4 = c2.columns(2)
            e["start"] = c3.text_input("Start (YYYY-MM)", str(job["start"]), key=k(f"x{i}_start"))
            e["end"] = c4.text_input("End (YYYY-MM or present)", str(job["end"]), key=k(f"x{i}_end"))
            e["bullets"] = lines(st.text_area("What you did and achieved, one per line",
                                              "\n".join(job["bullets"]), key=k(f"x{i}_bullets"), height=120))
            cur["experience"].append(e)
            if st.button("Remove this job", key=k(f"x{i}_remove")):
                action = ("remove_job", i)
    if st.button("➕ Add a job"):
        action = ("add_job", 0)

    st.subheader("Education")
    cur["education"] = []
    for i, ed in enumerate(d["education"]):
        c1, c2, c3, c4 = st.columns([3, 3, 1.2, 0.8], vertical_alignment="bottom")
        cur["education"].append({"school": c1.text_input("School", ed["school"], key=k(f"e{i}_school")),
                                 "degree": c2.text_input("Degree", ed["degree"], key=k(f"e{i}_degree")),
                                 "year": c3.text_input("Year", str(ed["year"] or ""), key=k(f"e{i}_year"))})
        if c4.button("✖", key=k(f"e{i}_remove"), help="Remove"):
            action = ("remove_edu", i)
    if st.button("➕ Add education"):
        action = ("add_edu", 0)

    cur["certifications"] = lines(st.text_area("Certifications and licenses, one per line",
                                               "\n".join(d["certifications"]), key=k("certs"), height=80))

    if action:
        kind, idx = action
        if kind == "remove_job":
            cur["experience"].pop(idx)
        elif kind == "add_job":
            cur["experience"].append({"company": "", "title": "", "location": "", "start": "",
                                      "end": "present", "bullets": []})
        elif kind == "remove_edu":
            cur["education"].pop(idx)
        else:
            cur["education"].append({"school": "", "degree": "", "year": ""})
        _load_draft(cur, st.session_state.import_warnings)
        st.rerun()

    st.divider()
    if st.button("Save my profile", type="primary"):
        profile, warnings = validate_profile(cur)
        if not profile["name"]:
            st.error("Please enter your name before saving.")
        else:
            save_profile(profile)
            _load_draft(profile)
            st.session_state.saved_warnings = warnings
            st.rerun()
    if st.session_state.pop("saved_warnings", None) is not None:
        st.success("Profile saved.")
        st.page_link(PAGES["find"], label="Next: find jobs", icon="🔎")


# ---------------------------------------------------------------- find jobs

def page_find() -> None:
    s = load_settings()
    storage = Storage(s)
    sc, f = s.get("scrape", {}), s.get("filters", {})
    if "scrape" not in load_user_overrides() and has_user_profile(s):
        # First visit: start from the user's own targets rather than the shipped examples.
        target = load_profile(s).get("target", {})
        sc = {**sc, "search_terms": target.get("titles") or [],
              "locations": [x for x in [load_profile(s).get("location", "")] if x]
              + (["Remote"] if target.get("remote_ok") else [])}
        f = {**f, "min_salary": target.get("min_salary")}
    st.title("Find jobs")
    if not has_user_profile(s):
        st.warning("Add your profile first, so the filters and scoring are about you.")
        st.page_link(PAGES["profile"], label="Go to My profile", icon="👤")

    st.metric("Jobs saved so far", len(storage.load_jobs()))

    with st.container(border=True):
        st.subheader("Search job sites")
        terms = st.text_area("Job titles to search for, one per line", "\n".join(sc.get("search_terms", [])))
        locs = st.text_area("Locations, one per line (e.g. Remote, or Austin, TX)",
                            "\n".join(sc.get("locations", [])))
        sites = st.multiselect("Job sites", JOB_SITES, [x for x in sc.get("sites", []) if x in JOB_SITES],
                               format_func=lambda x: x.replace("_", " ").title())
        c1, c2 = st.columns(2)
        per_search = c1.slider("Results per search", 10, 100, int(sc.get("results_wanted", 30)), step=10)
        days = c2.slider("Posted within the last … days", 1, 30, max(1, int(sc.get("hours_old", 72)) // 24))

        with st.expander("Filters (applied instantly, before the AI)"):
            min_salary = st.number_input("Skip jobs paying less than (per year, 0 = no limit)", min_value=0,
                                         step=5000, value=int(f.get("min_salary") or 0))
            remote_only = st.checkbox("Remote jobs only", bool(f.get("remote_only")))
            title_bl = st.text_area("Skip jobs whose title contains, one per line",
                                    "\n".join(f.get("title_blacklist", [])))
            company_bl = st.text_area("Skip these companies, one per line",
                                      "\n".join(f.get("company_blacklist", [])))

        st.caption("JobAgent searches every site at the same time, skips jobs that don't mention your "
                   "titles or skills, and has the AI score the best ones first. Matches appear on "
                   "**Your matches** as soon as each one is scored.")
        if st.button("Search and rank", type="primary", disabled=bool(busy_reason()), help=busy_reason()):
            save_user_overrides({
                "scrape": {"search_terms": lines(terms), "locations": lines(locs), "sites": sites,
                           "results_wanted": per_search, "hours_old": days * 24},
                "filters": {"min_salary": int(min_salary) or None, "remote_only": remote_only,
                            "title_blacklist": lines(title_bl), "company_blacklist": lines(company_bl)},
            })
            if not jobspy_available():
                st.error("The job-site search add-on isn't installed. Close JobAgent and start it again "
                         "with the Start launcher; it will try to install it. You can still paste jobs below.")
            elif not lines(terms) or not sites:
                st.error("Enter at least one job title and pick at least one job site.")
            else:
                fresh = load_settings()
                client = build_client(fresh)
                start_task("hunt", "Searching and ranking jobs", lambda t: _hunt(t, fresh, client))
        _hunt_status(search=True)

    with st.container(border=True):
        st.subheader("Found a job yourself? Paste it here")
        with st.form("manual_job", clear_on_submit=True):
            c1, c2 = st.columns(2)
            title = c1.text_input("Job title")
            company = c2.text_input("Company")
            location = c1.text_input("Location")
            url = c2.text_input("Link to the posting")
            desc = st.text_area("Full job description", height=200)
            if st.form_submit_button("Add and score", disabled=bool(busy_reason())):
                if not (title and company and desc):
                    st.error("Please fill in the title, company and description.")
                else:
                    row = {"title": title, "company": company, "location": location, "url": url,
                           "description": desc, "is_remote": "remote" in location.lower() or None}
                    client = build_client(s)
                    pasted = pipeline.jobs_from_rows([row], "manual")
                    start_task("hunt", f"Scoring {title} at {company}",
                               lambda t: _hunt(t, s, client, new_jobs=pasted, search=False))
        _hunt_status(search=False)

    if storage.load_jobs():
        st.page_link(PAGES["ranked"], label="See your matches", icon="⭐")
    with st.expander("Start over"):
        if st.checkbox("Yes, delete all saved jobs and scores (your resumes in build/ are kept)"):
            if st.button("Delete saved jobs"):
                for p in (storage.jobs_path, storage.rankings_path, storage.raw_jobs_path):
                    p.unlink(missing_ok=True)
                st.rerun()


def _hunt(task: Task, settings: dict[str, Any], client, *, new_jobs=None, search: bool = True):
    scoring = build_client(settings, purpose="scoring")
    same = getattr(scoring, "model", None) == getattr(client, "model", None)
    return pipeline.hunt(settings, Storage(settings), client, new_jobs=new_jobs, search=search,
                         scoring_client=None if same else scoring,
                         on_progress=lambda p: task.update(p.fraction, p.message))


def _hunt_status(search: bool) -> None:
    """Progress or result of the last hunt, shown under the control that started it."""
    t = MANAGER.get("hunt")
    if t is None or (t.label == "Searching and ranking jobs") != search:
        return
    if t.running:
        task_progress("hunt")
    elif t.status == "done":
        _show_hunt_summary(t.result, t)
    task_error("hunt")


REASONS = {
    "not_relevant_to_profile": "doesn't mention your job titles or skills",
    "description_too_short": "posting has almost no description",
    "empty_title": "posting has no title",
    "not_remote": "not remote",
    "location_not_allowed": "outside your locations",
    "salary_unknown": "no salary listed",
    "title_missing_required_keyword": "title is missing a required word",
}


def _reason_text(reason: str) -> str:
    key, _, detail = reason.partition(":")
    if key == "salary_below_min":
        return "pays below your minimum"
    if key.endswith("_blacklist"):
        return f"{key.split('_')[0]} contains “{detail}”"
    return REASONS.get(reason, reason.replace("_", " "))


def _show_hunt_summary(sm, task: Task) -> None:
    p = sm.progress
    if p.queries_total and p.found == 0:
        st.warning("No jobs came back. Try broader titles, more locations, or more days.")
    else:
        took = _elapsed(task)
        lines_ = []
        if p.queries_total:
            lines_.append(f"Searched **{p.queries_total}** time(s) across your sites in {took}: "
                          f"**{p.found}** listings found.")
        removed = p.filtered + p.not_relevant
        if removed:
            lines_.append(f"Skipped **{removed}** that didn't fit your filters or profile.")
        lines_.append(f"The AI scored **{p.scored}** job(s): **{p.apply}** good match(es)"
                      + (f", and explained the top {p.explained}." if p.explained else "."))
        st.success("  \n".join(lines_))
        if p.apply:
            st.page_link(PAGES["ranked"], label="See your matches", icon="⭐")
    if sm.ai_speed:
        st.caption(f"AI speed on this computer: {sm.ai_speed}")
    if sm.unscored_left:
        st.info(f"{sm.unscored_left} more relevant job(s) are waiting. Score them on **Your matches**.")
    if p.failed_queries:
        st.warning("Some searches didn't work (the site may be busy or blocking searches): "
                   + "; ".join(p.failed_queries[:5]))
    if sm.injection_flagged:
        st.info(f"{sm.injection_flagged} posting(s) contained hidden text trying to manipulate the AI. "
                "It was removed.")
    if sm.dropped:
        with st.expander(f"Why {len(sm.dropped)} job(s) were skipped"):
            for job, reasons in sm.dropped[:100]:
                st.markdown(f"- **{_md(job.title)}** at {_md(job.company)}: "
                            + _md(", ".join(_reason_text(r) for r in reasons)))


# ---------------------------------------------------------------- ranked

def page_ranked() -> None:
    s = load_settings()
    storage = Storage(s)
    st.title("Your matches")
    total = len(storage.load_jobs())
    if total == 0:
        st.info("No jobs yet.")
        st.page_link(PAGES["find"], label="Find jobs", icon="🔎")
        return

    live = next((t for t in (MANAGER.get("hunt"), MANAGER.get("rank")) if t and t.running), None)
    waiting = pipeline.unranked_jobs(storage)
    per_run = int(s.get("ranking", {}).get("max_jobs_per_run", 30))
    if live:
        task_progress("hunt" if live is MANAGER.get("hunt") else "rank")
    elif waiting:
        n = min(len(waiting), per_run)
        st.write(f"**{len(waiting)}** job(s) haven't been scored yet.")
        if st.button(f"Score {n} job(s) with AI", type="primary", disabled=bool(busy_reason()),
                     help=busy_reason()):
            client = build_client(s)
            start_task("rank", f"Scoring {n} job(s)", lambda t: _hunt(t, s, client, search=False))
    task_error("rank")

    c1, c2 = st.columns([3, 1])
    show_skip = c1.toggle("Also show jobs the AI suggests skipping")
    with c2.popover("Clear list", disabled=bool(live), use_container_width=True):
        st.write(f"Remove all **{total}** job(s) from your list? They won't be shown or scored again in "
                 "future searches. Resumes you already created are kept.")
        if st.button("Yes, clear the list", type="primary"):
            storage.hide({j.id for j in storage.load_jobs()})
            st.rerun()
    if live:
        _live_match_list(show_skip)       # refreshes itself as each job is scored
    else:
        _match_list(s, storage, show_skip)


def _match_list(s: dict[str, Any], storage: Storage, show_skip: bool) -> None:
    rows = pipeline.ranked_rows(storage, include_skip=show_skip)
    if not rows:
        st.info("No recommended jobs yet." + ("" if show_skip else " Turn on the switch above to see all scores."))
    for fit, job in rows:
        if job is not None:
            _job_card(s, storage, fit, job)


@st.fragment(run_every=3)
def _live_match_list(show_skip: bool) -> None:
    s = load_settings()
    _match_list(s, Storage(s), show_skip)


SCORE_PART_LABELS = {"skills": "Skills", "experience": "Experience", "title": "Job title match",
                     "location": "Distance from home", "local": "Small local employer"}


def _score_line(fit) -> str:
    bits = []
    if fit.ratings:
        bits.append(f"Skills {fit.ratings.get('skills', '?')}/5")
        bits.append(f"Experience {fit.ratings.get('experience', '?')}/5")
    if fit.score_points.get("title", 0) >= 10:
        bits.append("Title match")
    bits += [_md(x) for x in fit.labels]
    return " · ".join(bits)


def _job_card(s: dict[str, Any], storage: Storage, fit, job) -> None:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 5, 1.3])
        c1.metric("Fit", fit.fit_score)
        c2.markdown(f"**{_md(job.title)}**  \n{_md(job.company)} · {_md(job.location) or 'Location not listed'}")
        if job.url:
            c2.markdown(f"[Open the posting]({job.url.replace('(', '%28').replace(')', '%29')})")
        if c3.button("Not interested", key=f"hide_{job.id}", help="Remove it and never show it again"):
            storage.hide({job.id})
            st.rerun()
        line = _score_line(fit)
        if line:
            st.caption(line)
        if fit.score_points:
            with st.expander("How this score was worked out"):
                st.markdown("  \n".join(f"{SCORE_PART_LABELS.get(k, k)}: **{v:g}** points"
                                         for k, v in fit.score_points.items()))
        if fit.rationale:
            st.caption(_md(fit.rationale))
        if fit.fit_highlights:
            st.markdown("**Why you fit**  \n" + "  \n".join(
                f"✅ **{_md(h['requirement'])}**: {_md(h['evidence'])}" for h in fit.fit_highlights))
        elif fit.matched_skills:
            st.markdown("✅ **You have:** " + ", ".join(_md(x) for x in fit.matched_skills))
        if fit.skill_gaps:
            st.markdown("➖ **They also want:** " + ", ".join(_md(x) for x in fit.skill_gaps))

        explain_key = f"explain:{job.id}"
        explaining = MANAGER.get(explain_key)
        out_dir = storage.build_dir / job.id
        key = f"generate:{job.id}"
        writing = MANAGER.get(key)
        if explaining and explaining.running:
            task_progress(explain_key)
        if writing and writing.running:
            task_progress(key)
        else:
            b1, b2 = st.columns(2)
            label = "Create again" if (out_dir / "application.zip").exists() else "Create resume & cover letter"
            if b1.button(label, key=f"gen_{job.id}", type="primary" if fit.recommendation == "Apply" else "secondary",
                         disabled=bool(busy_reason()), help=busy_reason(), use_container_width=True):
                client = build_client(s)
                start_task(key, f"Writing documents for {job.title} at {job.company}", lambda t: pipeline.generate(
                    s, storage, client, job.id, force=True,
                    progress=lambda i, n, what: t.update(None, f"Writing the {what} at the same time...")))
            if not fit.explained and not (explaining and explaining.running) and b2.button(
                    "Explain this match", key=f"explain_{job.id}", disabled=bool(busy_reason()),
                    help=busy_reason() or "Have the AI write out why you fit (about a minute)",
                    use_container_width=True):
                client = build_client(s)
                start_task(explain_key, f"Explaining {job.title} at {job.company}",
                           lambda t: pipeline.explain(s, storage, client, job.id))
        task_error(explain_key)
        task_error(key)
        if not (writing and writing.running):
            _downloads(out_dir, job)


def _downloads(out_dir: Path, job) -> None:
    report_path = out_dir / "critic_report.json"
    bundle = out_dir / "application.zip"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if bundle.exists():
        st.download_button("⬇️ Download resume + cover letter (PDF and Word)", bundle.read_bytes(),
                           file_name=f"Application_{''.join(c for c in job.company if c.isalnum())[:30] or 'job'}.zip",
                           mime="application/zip", key=f"dl_{out_dir.name}_bundle", type="primary")
    with st.expander("Download one file at a time"):
        for kind, label in DOC_LABELS.items():
            cols = st.columns(3)
            cols[0].write(f"**{label}**")
            for col, ext in zip(cols[1:], ("pdf", "docx")):
                path = out_dir / f"{kind}.{ext}"
                if path.exists():
                    col.download_button(f"{ext.upper()}", path.read_bytes(),
                                        file_name=f"{label.replace(' ', '_')}.{ext}",
                                        key=f"dl_{out_dir.name}_{path.name}")
    for kind, label in DOC_LABELS.items():
        rep = report.get(kind, {})
        problems = rep.get("hallucinations", []) + rep.get("issues", [])
        if problems:
            with st.expander(f"⚠️ Please check the {label.lower()} before sending"):
                for p in problems:
                    st.write(f"- {p}")


# ---------------------------------------------------------------- navigation

PAGES: dict[str, Any] = {}


def make_pages() -> list[Any]:
    """Create the st.Page objects (page_link needs them). Setup is the home page at "/"."""
    PAGES.clear()
    PAGES["setup"] = st.Page(page_setup, title="Setup", icon="⚙️", default=True)
    PAGES["profile"] = st.Page(page_profile, title="My profile", icon="👤", url_path="profile")
    PAGES["find"] = st.Page(page_find, title="Find jobs", icon="🔎", url_path="find")
    PAGES["ranked"] = st.Page(page_ranked, title="Your matches", icon="⭐", url_path="matches")
    return list(PAGES.values())
