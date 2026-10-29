"""Deterministic, fast selection of what goes into a tailored resume (no LLM).

The local LLM is slow and tends to keep everything it is shown, so before any
rewriting we pick, in pure Python, which experience bullets and skills are
relevant to the job. The LLM then only rephrases what was selected, and
:func:`is_grounded_rewrite` checks each rewrite did not invent facts.

Works for any field (nursing, office work, software): it relies on word
overlap between the job title/essentials and the candidate's own text, with
light suffix stemming so "scheduling" matches "schedule" and "managed"
matches "manager".

Usage::

    kw = job_keywords(job)
    plans = plan_experience(profile, kw)        # every job kept, in order
    skills = select_skills(profile, kw)
    ok = is_grounded_rewrite(bullet, llm_output, allowed=skills_and_companies)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.models import Job
from matching.essentials import job_essentials

__all__ = [
    "JobKeywords",
    "ExperiencePlan",
    "job_keywords",
    "relevance",
    "plan_experience",
    "select_skills",
    "is_grounded_rewrite",
    "stem",
]

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each etc few for from further had has have having he her here hers him
his how however i if in into is it its itself just me more most my no nor not
now of off on once only or other our ours out over own per same she should so
some such than that the their theirs them then there these they this those
through to too under until up upon us very via was we were what when where
which while who whom why will with within without would you your yours
yourself across along among around well like may might must shall get gets
one two three four five six seven eight nine ten new using use used
""".split())

# Words that appear in nearly every job ad and say nothing about the work.
_GENERIC_JOB_WORDS = frozenset("""
experience experienced work working works team teams ability able years year
role roles position positions company companies candidate candidates required
require requires requirement requirements preferred prefer responsibilities
responsibility responsible including include includes strong excellent skill
skills job jobs opportunity opportunities apply applicant applicants looking
seeking join ideal plus must minimum qualifications qualification qualified
knowledge understanding environment demonstrated proven proficiency proficient
familiarity familiar various related relevant similar duties duty day days
other others great good best highly self motivated detail oriented successful
across within based level levels full part time hours hour week weeks month
months benefits salary pay competitive equal employer employment employee
employees offer offers provide provides providing ensure ensures ensuring
support supports supporting help helps helping perform performs performing
assist assists assisting want need needs needed make makes making take takes
new key using use etc
""".split())

# Short tokens that are meaningful (tech, roles, licences). Lower-cased.
_SHORT_TOKENS = frozenset({
    "c#", "c++", "go", "r", "ai", "ui", "ux", "qa", "hr", "rn", "lpn", "cna",
    "cpa", "it", "bi", "ml", "js", "ts", "db", "ar", "ap", "er", "or", "pt",
    "ot", "ma", "ehr", "emr", "f#", "vb", "ios", "aws", "gcp", "sql", "api",
})
# Short tokens that are also ordinary English words: only count them when the
# source writes them with a capital letter ("IT support", "Go services").
_AMBIGUOUS_SHORT = frozenset({"it", "go", "r", "or", "er", "ar", "ap", "pt", "ot", "ma", "ts"})

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#]*")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Suffix rules, tried in order; the first that leaves a stem of >= 3 chars wins.
_SUFFIXES = (
    ("ations", "ate"), ("ation", "ate"), ("ments", ""), ("ment", ""),
    ("ings", ""), ("ing", ""),
    ("ies", "y"), ("ied", "y"),
    ("ers", ""), ("er", ""), ("ors", ""), ("or", ""),
    ("ed", ""),
    ("es", ""), ("s", ""),
)


def stem(word: str) -> str:
    """Very light suffix stemmer: manage/managed/managing/manager -> 'manag'."""
    w = word.lower()
    if len(w) <= 3 or not w.isalpha():
        return w
    for suffix, repl in _SUFFIXES:
        if not w.endswith(suffix):
            continue
        if suffix == "s" and w.endswith(("ss", "us", "is")):
            break
        cand = w[: -len(suffix)] + repl
        if len(cand) >= 3:
            w = cand
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    # Collapse a doubled final consonant left by -ed/-ing ("planned" -> "plan").
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    return w


def _tokens(text: str) -> list[str]:
    """Raw tokens with their original case, trailing '+'/'#' kept only for c++/c#/f#."""
    out = []
    for tok in _TOKEN_RE.findall(text or ""):
        low = tok.lower()
        if low not in _SHORT_TOKENS:
            tok = tok.rstrip("+#")
        if tok:
            out.append(tok)
    return out


def _content_stems(text: str, *, drop_generic: bool) -> set[str]:
    """Normalized, stemmed content words of ``text``."""
    out: set[str] = set()
    for tok in _tokens(text):
        low = tok.lower()
        if low.isdigit():
            continue
        if len(low) < 3 or low in _SHORT_TOKENS:
            if low not in _SHORT_TOKENS:
                continue
            if low in _AMBIGUOUS_SHORT and tok == low:
                continue  # plain lower-case "it", "go", "or": an ordinary word
            out.add(low)
            continue
        if low in _STOPWORDS or (drop_generic and low in _GENERIC_JOB_WORDS):
            continue
        s = stem(low)
        if drop_generic and s in _GENERIC_STEMS:
            continue
        out.add(s)
    return out


_GENERIC_STEMS = frozenset(stem(w) for w in _GENERIC_JOB_WORDS)


# --------------------------------------------------------------------------- #
# Job keywords and scoring
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class JobKeywords:
    words: frozenset[str]          # normalized content words from job title + essentials
    title_words: frozenset[str]    # from the job title only (weighted higher)
    text: str                      # lowercased essentials text, for phrase matching


def job_keywords(job: Job, essentials_chars: int = 2500) -> JobKeywords:
    """Extract the content words of a job's title and essential description."""
    title = str(getattr(job, "title", "") or "")
    essentials = job_essentials(str(getattr(job, "description", "") or ""), essentials_chars)
    title_words = _content_stems(title, drop_generic=True)
    words = title_words | _content_stems(essentials, drop_generic=True)
    text = " ".join(f"{title}\n{essentials}".lower().split())
    return JobKeywords(words=frozenset(words), title_words=frozenset(title_words), text=text)


def relevance(text: str, kw: JobKeywords) -> float:
    """+2 per distinct job-title word in ``text``, +1 per other distinct job word."""
    found = _content_stems(str(text or ""), drop_generic=False) & kw.words
    title_hits = len(found & kw.title_words)
    return float(2 * title_hits + (len(found) - title_hits))


# --------------------------------------------------------------------------- #
# Experience selection
# --------------------------------------------------------------------------- #

@dataclass
class ExperiencePlan:
    entry: dict                    # the original profile experience entry (unchanged)
    bullets: list[str]             # selected ORIGINAL bullets, kept in their original order
    detailed: bool                 # True: show bullets; False: condensed (title/company/dates only)
    score: float


# A non-recent job is only detailed if it shares at least a title word (or two
# job words) with the posting, or scores at least half the best job; a job
# with a single incidental word in common stays condensed.
_MIN_DETAIL_SCORE = 2.0
_MIN_SHARE_OF_BEST = 0.5


def _bullets_of(entry: dict) -> list[str]:
    raw = entry.get("bullets") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(b) for b in raw if b is not None and str(b).strip()]


def plan_experience(profile: dict, kw: JobKeywords, *, detailed_jobs: int = 3,
                    bullets_per_job: int = 3) -> list[ExperiencePlan]:
    """Plan every experience entry (profile order): detailed with chosen bullets, or condensed."""
    entries = [e for e in ((profile or {}).get("experience") or []) if isinstance(e, dict)]
    if not entries:
        return []

    scored = []
    for entry in entries:
        bullets = _bullets_of(entry)
        text = " ".join([str(entry.get("title") or "")] + bullets)
        scored.append((entry, bullets, relevance(text, kw)))

    best = max(s for _, _, s in scored)
    ranked = sorted(range(len(scored)), key=lambda i: (-scored[i][2], i))
    detailed: set[int] = {0}
    for i in ranked[: max(0, detailed_jobs)]:
        s = scored[i][2]
        if s > 0 and (s >= _MIN_DETAIL_SCORE or s >= best * _MIN_SHARE_OF_BEST):
            detailed.add(i)

    plans = []
    for i, (entry, bullets, score) in enumerate(scored):
        chosen: list[str] = []
        if i in detailed and bullets:
            b_scores = [relevance(b, kw) for b in bullets]
            order = sorted(range(len(bullets)), key=lambda j: (-b_scores[j], j))
            keep = [j for j in order[: max(0, bullets_per_job)] if b_scores[j] > 0]
            if not keep and bullets_per_job > 0:
                keep = [0]
            chosen = [bullets[j] for j in sorted(keep)]
        plans.append(ExperiencePlan(entry=entry, bullets=chosen,
                                    detailed=i in detailed, score=score))
    return plans


# --------------------------------------------------------------------------- #
# Skills selection
# --------------------------------------------------------------------------- #

def _collect(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _collect(v, out)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            _collect(v, out)
    elif value is not None and str(value).strip():
        out.append(str(value).strip())


def _profile_skills(profile: dict) -> list[str]:
    raw: list[str] = []
    _collect((profile or {}).get("skills"), raw)
    seen: set[str] = set()
    out = []
    for s in raw:
        key = " ".join(s.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _skill_in_job(skill: str, kw: JobKeywords) -> bool:
    phrase = " ".join(skill.lower().split())
    if phrase in _AMBIGUOUS_SHORT:
        # "Go" / "R" / "IT": only when the posting used it as a term, not as "go above".
        return phrase in kw.words
    pat = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
    return bool(pat.search(kw.text))


def select_skills(profile: dict, kw: JobKeywords, max_skills: int = 12) -> list[str]:
    """Profile skills named in the job first (profile order), then the rest by relevance."""
    skills = _profile_skills(profile)
    if not skills:
        return []
    matched = [s for s in skills if _skill_in_job(s, kw)]
    matched_set = set(matched)
    rest = [s for s in skills if s not in matched_set]
    rest_scores = {s: relevance(s, kw) for s in rest}
    rest.sort(key=lambda s: -rest_scores[s])  # stable: ties keep profile order
    return (matched + rest)[: max(1, max_skills)]


# --------------------------------------------------------------------------- #
# Anti-fabrication check
# --------------------------------------------------------------------------- #

_MIN_GROUNDED_SHARE = 0.6


def _numbers(text: str) -> set[str]:
    out = set()
    for n in _NUMBER_RE.findall(text or ""):
        n = n.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def _grounding_words(text: str) -> set[str]:
    out = set()
    for tok in _TOKEN_RE.findall(text or ""):
        low = tok.lower().rstrip("+#")
        if len(low) < 4 or low in _STOPWORDS or low.isdigit():
            continue
        out.add(stem(low))
    return out


def is_grounded_rewrite(original: str, rewritten: str, allowed: str = "") -> bool:
    """True if an LLM rewrite of one bullet/sentence adds no new facts.

    Checks: no new numbers; >= 60% of content words come from the original or
    ``allowed`` (skills, employer names); single line; not much longer.
    """
    original = str(original or "")
    rewritten = str(rewritten or "")
    if not rewritten.strip() or "\n" in rewritten or "\r" in rewritten:
        return False
    if len(rewritten) > 2 * len(original) + 80:
        return False
    source = f"{original}\n{allowed or ''}"
    if not _numbers(rewritten) <= _numbers(source):
        return False
    words = _grounding_words(rewritten)
    if words:
        grounded = len(words & _grounding_words(source))
        if grounded / len(words) < _MIN_GROUNDED_SHARE:
            return False
    return True
