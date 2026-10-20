# FreeJobAgent

A local-first, zero-cost job pipeline: scrape listings, filter them with plain
Python rules, fit-score them with a local or free-tier LLM, and generate a
tailored ATS resume and cover letter grounded only in your own profile.

No paid APIs. The LLM is either **Ollama on localhost** (default) or
**Google Gemini's free developer tier** through its OpenAI-compatible endpoint.
HTTP calls use only the Python standard library.

## Layout

```
free_job_agent/
├── orchestrator.py            # CLI: scrape | rank | list | generate | check
├── config/
│   ├── settings.yaml          # LLM provider toggle, scrape targets, filters
│   └── candidate_profile.yaml # master profile: the only source of truth
├── core/
│   ├── settings.py            # YAML loading, project-relative paths
│   ├── llm_client.py          # OllamaClient / GeminiFreeTierClient + build_client()
│   ├── models.py              # Job, FitResult dataclasses
│   ├── storage.py             # data/*.jsonl, rankings.json, build/<job_id>/
│   └── jsonutil.py            # robust JSON extraction from LLM output
├── scrapers/
│   ├── job_scraper.py         # python-jobspy + optional Playwright career pages
│   └── sanitizer.py           # HTML/script/injection stripping, untrusted wrappers
├── matching/
│   ├── filter_engine.py       # deterministic rules (title, location, salary, ...)
│   └── fit_evaluator.py       # LLM scoring -> fit_score, matched_skills, gaps, rec
├── generators/
│   ├── resume_tailor.py       # rewrites bullets from verified profile facts only
│   ├── cover_letter_writer.py # 3-paragraph letter
│   └── critic_agent.py        # deterministic + LLM hallucination/format checks
├── exporters/
│   └── pdf_builder.py         # Markdown -> .md / .pdf (ReportLab) / .docx
├── data/                      # raw_jobs.jsonl, jobs.jsonl, rankings.json (git-ignored)
└── build/                     # build/<job_id>/resume.pdf, cover_letter.pdf, ...
```

## Setup

```bash
cd free_job_agent
pip install -r requirements.txt
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
python -m pytest -q        # no network or LLM needed; uses fake clients and a stub server
```
