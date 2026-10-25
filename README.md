# FreeJobAgent

A local-first, zero-cost job pipeline: scrape listings, filter them with plain
Python rules, fit-score them with a local or free-tier LLM, and generate a
tailored ATS resume and cover letter grounded only in your own profile.

## Quick start (no command line)

1. Install **Python 3.11+** from [python.org](https://www.python.org/downloads/).
   On Windows, tick **"Add python.exe to PATH"** in the installer.
2. Install **Ollama** from [ollama.com/download](https://ollama.com/download).
3. Double-click **`Start JobAgent.bat`** (Windows) or **`Start JobAgent.command`** (Mac).
   The first start takes a few minutes; then JobAgent opens in your browser.
4. Follow the **Setup** page: download the AI model, upload your resume, check it, save.
5. **Find jobs** → **Search and rank** (or paste a posting). Matches appear on
   **Your matches** as they're scored, each with "Why you fit" notes →
   **Create resume & cover letter** → download PDF or Word.

Keep the black launcher window open while you use JobAgent; close it to stop.
On a Mac, the first time macOS may refuse to open the file: right-click it → **Open** → **Open**.
`HOW TO START.txt` has the same steps in plain text for sharing.

Your profile and settings are saved in `config/my_profile.yaml` and
`config/user_settings.yaml`, jobs in `data/`, documents in `build/`. None of
these are committed to git, and the web page only accepts connections from
this computer.

No paid APIs. The LLM is either **Ollama on localhost** (default) or
**Google Gemini's free developer tier** through its OpenAI-compatible endpoint.
HTTP calls use only the Python standard library.

## Layout

```
free_job_agent/
├── Start JobAgent.bat/.command  # double-click launchers (set up .venv, then run launch.py)
├── launch.py                  # starts Streamlit, waits for it, opens the browser
├── app.py / webapp.py         # browser UI: Setup, My profile, Find jobs, Your matches
├── orchestrator.py            # CLI: hunt | scrape | rank | list | generate | check
├── config/
│   ├── settings.yaml          # LLM provider toggle, scrape targets, filters
│   └── candidate_profile.yaml # master profile: the only source of truth
├── core/
│   ├── settings.py            # YAML loading, project-relative paths
│   ├── llm_client.py          # OllamaClient / GeminiFreeTierClient + build_client()
│   ├── models.py              # Job, FitResult dataclasses
│   ├── storage.py             # data/*.jsonl, rankings.json, build/<job_id>/
│   ├── pipeline.py            # hunt/scrape/rank/generate steps shared by CLI and web UI
│   ├── agents.py              # parallel search + scoring agents (HuntPipeline)
│   ├── background.py          # runs long tasks off the page so clicks don't cancel them
│   ├── ollama_admin.py        # is Ollama running, which models, pull with progress
│   └── jsonutil.py            # robust JSON extraction from LLM output
├── scrapers/
│   ├── job_scraper.py         # python-jobspy + optional Playwright career pages
│   └── sanitizer.py           # HTML/script/injection stripping, untrusted wrappers
├── matching/
│   ├── filter_engine.py       # deterministic rules (title, location, salary, ...)
│   ├── prescreen.py           # instant keyword relevance vs. the profile
│   └── fit_evaluator.py       # LLM scoring -> fit_score, matched_skills, gaps, rec
├── generators/
│   ├── resume_tailor.py       # rewrites bullets from verified profile facts only
│   ├── cover_letter_writer.py # 3-paragraph letter
│   └── critic_agent.py        # deterministic + LLM hallucination/format checks
├── importers/
│   └── resume_importer.py     # PDF/DOCX/TXT resume -> draft profile (+ what to double-check)
├── exporters/
│   └── pdf_builder.py         # Markdown -> .md / .pdf (ReportLab) / .docx
├── data/                      # raw_jobs.jsonl, jobs.jsonl, rankings.json (git-ignored)
└── build/                     # build/<job_id>/resume.pdf, cover_letter.pdf, ...
```

## Command-line setup

```bash
pip install -r requirements.txt -r requirements-scraping.txt
playwright install chromium        # only if you use career_pages

# Option A: local LLM (default)
ollama serve &
ollama pull qwen2.5:7b-instruct

# Option B: Gemini free tier (key from https://aistudio.google.com/apikey)
export GEMINI_API_KEY=...
# then set llm.provider: gemini in config/settings.yaml, or pass --provider gemini
```

Edit `config/candidate_profile.yaml` with your real history and
`config/settings.yaml` with your searches and filters.

## Usage

```bash
python orchestrator.py check                       # is the LLM reachable?
python orchestrator.py scrape                      # or: scrape --from-file examples/sample_jobs.json
python orchestrator.py rank                        # scores unranked jobs, prints Apply list
python orchestrator.py list --all
python orchestrator.py generate --job-id <ID>      # writes build/<ID>/resume.{md,pdf,docx}, ...
```

`--provider ollama|gemini` (or `FJA_LLM_PROVIDER`) overrides the configured backend.

## How search and ranking run in parallel

```
search agents (one per job site) ─► sanitize ─► your filters ─► instant pre-screen
                                                                      │
                     priority queue, most relevant first ◄────────────┘
                                  │
scoring agents (N at once) ◄──────┘ ─► rankings.json, saved as each job finishes
```

- **Search agents** (`core/agents.py`): every job site gets its own agent, so
  sites are searched at the same time. Queries to the *same* site still run
  one after another, with a pause, to avoid being blocked.
- **Pre-screen** (`matching/prescreen.py`): a keyword check of each job against
  your titles and skills, with no AI. Jobs that mention none of them never
  reach the AI. Pasted jobs are always scored.
- **Scoring agents** start as soon as the first jobs arrive, take the most
  relevant job first, and save each result straight away, so the matches page
  fills in while the search is still running. `ranking.max_jobs_per_run` caps
  AI calls per search; the rest wait for a "Score" click.
- **Speed settings** (`config/settings.yaml`): `ranking.prompt_description_chars`
  (default 4000) is how much of each posting the AI reads. On a laptop CPU,
  reading the prompt is most of the time per job, so this matters most.
  `parallel.llm_workers` (default 2) is how many jobs are scored at once.
  Ollama only runs them together if `OLLAMA_NUM_PARALLEL` allows it; the
  launchers set it to 2 when *they* start Ollama. If Ollama was already running,
  quit it from the tray/menu bar and start JobAgent again. On a CPU without a
  GPU, parallel scoring helps less than on a GPU.
- **"Why you fit"**: the AI pairs each job requirement with the matching fact
  from your profile. A note is dropped if it cites a number that isn't in
  your profile, if it is mostly made of words that aren't in your profile, or
  if it repeats manipulative text from the posting.

CLI equivalent: `python orchestrator.py hunt`.

## Safety model

- Every scraped field is untrusted. `sanitizer.py` removes scripts, hidden
  Unicode, chat-template tokens and prompt-injection phrases, and records what
  it removed in `Job.sanitizer_flags`.
- Job text reaches the LLM only inside nonce-tagged
  `<untrusted_job_description id="...">` blocks, with `<`/`>` escaped so the
  content cannot close its own boundary. The system prompt says that content is
  data, never instructions.
- LLM output is validated in Python: scores are clamped, recommendations below
  `min_fit_score` are forced to Skip, and matched skills that are not in your
  profile are dropped.
- Generated documents assemble names, dates, employers and education
  from the profile in code; the LLM only rewrites wording. The critic flags
  numbers, companies and skills that are not in the profile, and keeps the
  earlier draft when a revision makes things worse.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q        # no network or LLM needed; fake LLM, stub Ollama, headless UI test
```
