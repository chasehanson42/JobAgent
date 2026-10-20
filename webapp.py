"""Pages of the JobAgent web app. app.py is the entry point that wires them together."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import streamlit as st

from core import llm_client, ollama_admin, pipeline
from core.llm_client import LLMError
from core.settings import (has_user_profile, load_profile, load_settings, load_user_overrides,
                           save_profile, save_user_overrides)
from core.storage import Storage
from importers.resume_importer import (ResumeImportError, draft_profile, empty_profile,
                                       extract_text, validate_profile)

MODEL_SIZES = {"qwen2.5:7b-instruct": "4.7 GB", "qwen2.5:3b-instruct": "1.9 GB"}
JOB_SITES = ["indeed", "linkedin", "glassdoor", "zip_recruiter", "google"]
DOC_LABELS = {"resume": "Resume", "cover_letter": "Cover letter"}


def build_client(s: dict[str, Any]):
    return llm_client.build_client(s)


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


def lines(text: str) -> list[str]:
    return [x.strip() for x in text.splitlines() if x.strip()]


def commas(text: str) -> list[str]:
    return [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]


def setup_complete(s: dict[str, Any]) -> bool:
    base, model = ollama_settings(s)
    return has_user_profile(s) and ollama_admin.has_model(base, model)


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
        if st.button("Download", type="primary", disabled=not running):
            bar = st.progress(0.0, text="Starting download...")
            try:
                for ev in ollama_admin.pull_model(base, choice):
                    done, total = ev.get("completed"), ev.get("total")
                    if done and total:
                        bar.progress(min(done / total, 1.0),
                                     text=f"{ev['status']} · {done / 1e9:.1f} of {total / 1e9:.1f} GB")
                    else:
                        bar.progress(0.0 if ev.get("status") != "success" else 1.0, text=ev.get("status", ""))
                save_user_overrides({"llm": {"provider": "ollama", "ollama": {"model": choice}}})
                st.success("Model downloaded.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - show any failure to the user
                show_error(exc)

    if running:
        others = [m for m in ollama_admin.list_models(base) if m != model]
        if others:
            with st.expander("Use a different model you already have"):
                pick = st.selectbox("Installed models", others)
                if st.button("Use this model"):
                    save_user_overrides({"llm": {"provider": "ollama", "ollama": {"model": pick}}})
                    st.rerun()

    st.subheader("Step 3 · Your resume")
    if has_user_profile(s):
        st.success(f"Profile saved for **{load_profile(s).get('name') or 'you'}**.")
    else:
        st.info("Upload your current resume so JobAgent knows your experience.")
    st.page_link(PAGES["profile"], label="Go to My profile", icon="👤")


# ---------------------------------------------------------------- profile

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

    with st.container(border=True):
        st.subheader("Fill this in from your resume")
        up = st.file_uploader("Your current resume (PDF, Word or text)", type=["pdf", "docx", "txt", "md"])
        if up is not None and st.button("Read my resume", type="primary"):
            try:
                text = extract_text(up.name, up.getvalue())
                with st.spinner("Reading your resume. This can take a minute or two..."):
                    profile, warnings = draft_profile(build_client(s), text)
                _load_draft(profile, warnings)
                st.rerun()
            except (LLMError, ResumeImportError) as exc:
                show_error(exc)

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
    cur["target"] = {"titles": commas(titles), "min_salary": int(min_salary) or None, "remote_ok": remote_ok}

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

        if st.button("Save and search", type="primary"):
            save_user_overrides({
                "scrape": {"search_terms": lines(terms), "locations": lines(locs), "sites": sites,
                           "results_wanted": per_search, "hours_old": days * 24},
                "filters": {"min_salary": int(min_salary) or None, "remote_only": remote_only,
                            "title_blacklist": lines(title_bl), "company_blacklist": lines(company_bl)},
            })
            if importlib.util.find_spec("jobspy") is None:
                st.error("The job-site search add-on isn't installed. Close JobAgent and start it again "
                         "with the Start launcher; it will try to install it. You can still paste jobs below.")
            elif not lines(terms) or not sites:
                st.error("Enter at least one job title and pick at least one job site.")
            else:
                with st.spinner("Searching job sites. This usually takes 1–5 minutes..."):
                    try:
                        _show_scrape_summary(pipeline.scrape(load_settings(), storage))
                    except Exception as exc:  # noqa: BLE001
                        show_error(exc)

    with st.container(border=True):
        st.subheader("Found a job yourself? Paste it here")
        with st.form("manual_job", clear_on_submit=True):
            c1, c2 = st.columns(2)
            title = c1.text_input("Job title")
            company = c2.text_input("Company")
            location = c1.text_input("Location")
            url = c2.text_input("Link to the posting")
            desc = st.text_area("Full job description", height=200)
            if st.form_submit_button("Add job"):
                if not (title and company and desc):
                    st.error("Please fill in the title, company and description.")
                else:
                    row = {"title": title, "company": company, "location": location, "url": url,
                           "description": desc, "is_remote": "remote" in location.lower() or None}
                    _show_scrape_summary(pipeline.scrape(s, storage, pipeline.jobs_from_rows([row], "manual")))

    if storage.load_jobs():
        st.page_link(PAGES["ranked"], label="Next: score your jobs", icon="⭐")
    with st.expander("Start over"):
        if st.checkbox("Yes, delete all saved jobs and scores (your resumes in build/ are kept)"):
            if st.button("Delete saved jobs"):
                for p in (storage.jobs_path, storage.rankings_path, storage.raw_jobs_path):
                    p.unlink(missing_ok=True)
                st.rerun()


def _show_scrape_summary(sm: pipeline.ScrapeSummary) -> None:
    if sm.scraped == 0:
        st.warning("No jobs came back. Try broader titles, more locations, or more days.")
        return
    st.success(f"Found {sm.scraped} job(s): kept **{sm.kept}**, filtered out {len(sm.dropped)}.")
    if sm.injection_flagged:
        st.info(f"{sm.injection_flagged} posting(s) contained hidden text trying to manipulate the AI. "
                "It was removed.")
    if sm.dropped:
        with st.expander("Why some jobs were filtered out"):
            for job, reasons in sm.dropped:
                st.write(f"- **{job.title}** at {job.company}: {', '.join(reasons)}")


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

    waiting = pipeline.unranked_jobs(storage)
    per_run = int(s.get("ranking", {}).get("max_jobs_per_run", 25))
    if waiting:
        n = min(len(waiting), per_run)
        st.write(f"**{len(waiting)}** job(s) haven't been scored yet. The AI takes roughly "
                 "20–60 seconds per job on a laptop.")
        if st.button(f"Score {n} job(s) with AI", type="primary"):
            bar = st.progress(0.0, text="Starting...")
            try:
                pipeline.rank(s, storage, build_client(s), progress=lambda i, n, label: bar.progress(
                    (i - 1) / n, text=f"Scoring {i} of {n}: {label}"))
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                show_error(exc)

    show_skip = st.toggle("Also show jobs the AI suggests skipping")
    rows = pipeline.ranked_rows(storage, include_skip=show_skip)
    if not rows:
        st.info("No recommended jobs yet." + ("" if show_skip else " Turn on the switch above to see all scores."))
    for fit, job in rows:
        if job is None:
            continue
        _job_card(s, storage, fit, job)


def _job_card(s: dict[str, Any], storage: Storage, fit, job) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([1, 5])
        c1.metric("Fit", fit.fit_score)
        c2.markdown(f"**{job.title}**  \n{job.company} · {job.location or 'Location not listed'}")
        if job.url:
            c2.markdown(f"[Open the posting]({job.url})")
        if fit.rationale:
            st.caption(fit.rationale)
        if fit.matched_skills:
            st.markdown("✅ **You have:** " + ", ".join(fit.matched_skills))
        if fit.skill_gaps:
            st.markdown("➖ **They also want:** " + ", ".join(fit.skill_gaps))

        out_dir = storage.build_dir / job.id
        label = "Create again" if (out_dir / "resume.pdf").exists() else "Create resume & cover letter"
        if st.button(label, key=f"gen_{job.id}", type="primary" if fit.recommendation == "Apply" else "secondary"):
            with st.status("Writing your documents. This can take a few minutes...", expanded=True) as status:
                try:
                    pipeline.generate(s, storage, build_client(s), job.id, force=True,
                                      progress=lambda i, n, what: st.write(f"Writing the {what} and checking it..."))
                    status.update(label="Done!", state="complete")
                except Exception as exc:  # noqa: BLE001
                    status.update(label="Couldn't finish", state="error")
                    show_error(exc)
        _downloads(out_dir)


def _downloads(out_dir: Path) -> None:
    report_path = out_dir / "critic_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for kind, label in DOC_LABELS.items():
        files = [out_dir / f"{kind}.{ext}" for ext in ("pdf", "docx")]
        cols = st.columns(3)
        cols[0].write(f"**{label}**")
        for col, path in zip(cols[1:], files):
            if path.exists():
                col.download_button(f"Download {path.suffix[1:].upper()}", path.read_bytes(), file_name=
                                    f"{label.replace(' ', '_')}{path.suffix}", key=f"dl_{out_dir.name}_{path.name}")
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
