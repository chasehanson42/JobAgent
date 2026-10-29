"""Resume and cover letter templates the user can edit.

A template is plain text with {placeholders}. JobAgent fills in facts from the
profile and the job in code; the AI only edits the parts marked for it:

Resume: {summary} and the bullet points inside {experience}.
Cover letter: each body paragraph, unless it starts with "KEEP:". Text in
[square brackets] is a note to the AI saying what to write there.

Saved templates live in config/my_templates/ (git-ignored); the defaults below
are used until the user saves their own.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.settings import PROJECT_ROOT

TEMPLATE_DIR = PROJECT_ROOT / "config" / "my_templates"

RESUME_DEFAULT = """# {name}
{contact}

## Summary
{summary}

## Skills
{skills}

## Experience
{experience}

## Education
{education}

## Certifications
{certifications}
"""

COVER_LETTER_DEFAULT = """# {name}
{contact}

{date}

Hiring Team, {company}

Dear Hiring Team,

I am writing to apply for the {job_title} position at {company}. [Add one sentence on why this role at {company} suits you, based on your background.]

[Describe your most relevant experience and skills for this job in three or four sentences, using only real facts from your profile.]

KEEP: Thank you for considering my application. I would welcome the chance to talk about how I can help {company}.

Sincerely,
{name}
"""

RESUME_FIELDS = {
    "name": "Your name", "contact": "City, email, phone and links", "summary": "Summary (AI tailors it)",
    "skills": "Your skills that matter for this job", "experience": "Jobs; the AI rewrites the relevant bullets",
    "education": "Education", "certifications": "Certifications and licenses",
}
COVER_LETTER_FIELDS = {
    "name": "Your name", "contact": "City, email, phone and links", "date": "Today's date",
    "company": "The employer", "job_title": "The job title", "email": "Your email", "phone": "Your phone",
}

_DEFAULTS = {"resume": RESUME_DEFAULT, "cover_letter": COVER_LETTER_DEFAULT}
_FIELDS = {"resume": RESUME_FIELDS, "cover_letter": COVER_LETTER_FIELDS}
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _path(kind: str, directory: Path | None = None) -> Path:
    return (directory or TEMPLATE_DIR) / f"{kind}.md"


def load_template(kind: str, directory: Path | None = None) -> str:
    path = _path(kind, directory)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if text.strip():
            return text
    return _DEFAULTS[kind]


def is_custom(kind: str, directory: Path | None = None) -> bool:
    return _path(kind, directory).exists()


def save_template(kind: str, text: str, directory: Path | None = None) -> list[str]:
    """Save a user template; returns plain-English warnings about it (it is saved anyway)."""
    path = _path(kind, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8")
    return check_template(kind, text)


def reset_template(kind: str, directory: Path | None = None) -> None:
    _path(kind, directory).unlink(missing_ok=True)


def check_template(kind: str, text: str) -> list[str]:
    known = _FIELDS[kind]
    unknown = sorted({m for m in _PLACEHOLDER.findall(text) if m not in known})
    warnings = [f"{{{u}}} isn't a field JobAgent knows, so it will be left as is." for u in unknown]
    if kind == "resume" and "{experience}" not in text:
        warnings.append("There's no {experience} in the template, so your jobs won't appear.")
    if kind == "cover_letter" and "{company}" not in text:
        warnings.append("The letter never mentions {company}; employers notice generic letters.")
    return warnings


def fill(text: str, values: dict[str, str]) -> str:
    """Replace known {placeholders}; anything else (including stray braces) stays as typed."""
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


# ---------------------------------------------------------------- cover letter parts

_GREETING = re.compile(r"^(dear\b|to whom it may concern|hello\b|hi\b)", re.IGNORECASE)
_SIGNOFF = re.compile(r"^(sincerely|best regards|kind regards|regards|warm regards|respectfully|"
                      r"thank you,|yours (truly|sincerely))\b", re.IGNORECASE)
KEEP_PREFIX = "KEEP:"


def split_cover_letter(template: str) -> list[tuple[str, str]]:
    """Split a letter template into blocks: ("edit", text) for paragraphs the AI tailors,
    ("fixed", text) for everything else (header, date, greeting, KEEP paragraphs, sign-off)."""
    blocks: list[tuple[str, str]] = []
    after_signoff = False
    for block in re.split(r"\n\s*\n", template.strip("\n")):
        stripped = block.strip()
        if not stripped:
            continue
        if after_signoff or _SIGNOFF.match(stripped):
            after_signoff = True
            blocks.append(("fixed", block))
        elif stripped.startswith(KEEP_PREFIX):
            blocks.append(("fixed", stripped[len(KEEP_PREFIX):].strip()))
        elif (stripped.startswith("#") or _GREETING.match(stripped) or _is_address_block(stripped)
              or len(_PLACEHOLDER.sub("", stripped).split()) < 6):
            blocks.append(("fixed", block))           # header/date/address lines, greetings
        else:
            blocks.append(("edit", " ".join(stripped.split())))   # a paragraph, even if wrapped
    return blocks


def _is_address_block(text: str) -> bool:
    """Several short lines that don't read as a sentence: a name/address/contact block."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return (len(lines) > 1 and all(len(ln.split()) < 9 for ln in lines)
            and not re.search(r"[.!?]\s*$", text.strip()))


def strip_notes(text: str) -> str:
    """Remove [notes to the AI] when the AI's version can't be used."""
    return " ".join(re.sub(r"\[[^\]]*\]", "", text).split())
