"""How far a job is from home, worked out offline.

Place names are looked up in the GeoNames "cities500" list that ships with the
``geonamescache`` package (every town with at least 500 people), so small towns
work and nothing is sent over the internet. The full list is a large file, so a
compact index is built the first time it is needed and saved to
``data/places_index.json``; later starts load that file instead.

If ``geonamescache`` is not installed, everything still works with plain text
matching: a job whose location mentions your home town counts as local.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.models import Job

log = logging.getLogger(__name__)

INDEX_FORMAT = 2
_ALT_MIN_POPULATION = 15000   # alternate names are only indexed for bigger places


def _default_index_path() -> Path:
    try:
        from core.settings import PROJECT_ROOT
    except Exception:  # pragma: no cover - settings always importable in the app
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return Path(PROJECT_ROOT) / "data" / "places_index.json"


# Tests may point this somewhere else (see set_index_path).
INDEX_PATH: Path = _default_index_path()

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}

_COUNTRIES = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "america": "US", "canada": "CA", "ca": "CA", "uk": "GB", "united kingdom": "GB",
    "great britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB", "gb": "GB",
    "ireland": "IE", "australia": "AU", "new zealand": "NZ", "mexico": "MX", "india": "IN",
    "germany": "DE", "france": "FR", "spain": "ES", "italy": "IT", "netherlands": "NL",
    "philippines": "PH", "brazil": "BR", "japan": "JP", "singapore": "SG",
}


# --------------------------------------------------------------------------
# Public data types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Place:
    name: str
    admin1: str
    country: str
    lat: float
    lon: float
    population: int


@dataclass(frozen=True)
class LocationFit:
    score: float          # 0-1
    miles: float | None
    label: str            # short plain English, e.g. "12 mi from home"
    is_local: bool        # within the person's commute


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
_ABBREV = {"st": "saint", "ste": "sainte", "ft": "fort", "mt": "mount"}
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, expand St./Ft./Mt."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("'", "").replace("’", "")
    words = _NON_WORD.sub(" ", s).split()
    return " ".join(_ABBREV.get(w, w) for w in words)


_STATE_BY_NAME = {normalize(name): code for code, name in US_STATES.items()}
_STATE_BY_NAME["washington dc"] = "DC"
_STATE_CODES = {code.lower(): code for code in US_STATES}


def _state_code(part: str) -> str | None:
    n = normalize(part)
    return _STATE_CODES.get(n) or _STATE_BY_NAME.get(n)


def _country_code(part: str) -> str | None:
    return _COUNTRIES.get(normalize(part))


# --------------------------------------------------------------------------
# Index (built once per process, cached on disk)
# --------------------------------------------------------------------------
_lock = threading.Lock()
_index: dict[str, Any] | None = None     # None = not loaded yet
_last_load: dict[str, Any] = {}


def _load_geonames() -> tuple[str, dict[str, dict[str, Any]]]:
    """Import geonamescache and return (version, cities500). Raises ImportError if missing."""
    import geonamescache  # noqa: PLC0415 - lazy: optional and heavy

    try:
        from importlib.metadata import version
        ver = version("geonamescache")
    except Exception:
        ver = str(getattr(geonamescache, "__version__", "unknown"))
    return ver, geonamescache.GeonamesCache(min_city_population=500).get_cities()


def _geonames_version() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("geonamescache")
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover
        return None


def _build_index(version: str, cities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pack cities into a few long strings: they load far faster than nested JSON."""
    display: list[str] = []
    region: list[str] = []
    coords: list[str] = []
    names: dict[str, list[int]] = {}
    alts: dict[str, list[int]] = {}
    for city in cities.values():
        try:
            name = " ".join(str(city["name"]).split())
            lat, lon = round(float(city["latitude"]), 4), round(float(city["longitude"]), 4)
            pop = int(city.get("population") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        key = normalize(name)
        if not key:
            continue
        idx = len(display)
        display.append(name)
        region.append(f"{city.get('countrycode') or ''} {city.get('admin1code') or ''}".strip())
        coords.append(f"{lat},{lon},{pop}")
        names.setdefault(key, []).append(idx)
        if pop < _ALT_MIN_POPULATION:
            continue
        seen = {key}
        for alt in city.get("alternatenames") or []:
            # Skip codes like "NYC"/"WAS" and non-Latin spellings.
            if not isinstance(alt, str) or not alt.isascii() or alt.isupper() or len(alt) < 4:
                continue
            akey = normalize(alt)
            if akey and akey not in seen:
                seen.add(akey)
                alts.setdefault(akey, []).append(idx)

    def pack(table: dict[str, list[int]]) -> dict[str, str]:
        return {"keys": "\n".join(table), "ids": " ".join(",".join(map(str, v)) for v in table.values())}

    return {"format": INDEX_FORMAT, "version": version, "name": "\n".join(display),
            "region": "\n".join(region), "coords": " ".join(coords),
            "names": pack(names), "alts": pack(alts)}


def _unpack(raw: dict[str, Any]) -> dict[str, Any]:
    """On-disk form -> lookup tables. Rows stay as strings until a lookup needs them."""
    def table(t: dict[str, str]) -> dict[str, str]:
        keys, ids = t["keys"], t["ids"]
        return dict(zip(keys.split("\n"), ids.split(" "))) if keys else {}

    out = {"name": raw["name"].split("\n"), "region": raw["region"].split("\n"),
           "coords": raw["coords"].split(" "), "names": table(raw["names"]),
           "alts": table(raw["alts"])}
    if not (len(out["name"]) == len(out["region"]) == len(out["coords"])):
        raise ValueError("place index columns differ in length")
    return out


def _read_cache(path: Path, version: str | None) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or data.get("format") != INDEX_FORMAT:
            return None
        if version is not None and data.get("version") != version:
            return None
        return _unpack(data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _write_cache(path: Path, raw: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".places_index.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        log.warning("Could not save the place index to %s: %s", path, exc)


def load_index(cache_path: str | Path | None = None, force_rebuild: bool = False) -> dict[str, Any]:
    """Return the place index, building or loading it the first time.

    An empty dict means geodata isn't available; callers then fall back to
    plain text matching.
    """
    global _index
    if _index is not None and not force_rebuild:
        return _index
    with _lock:
        if _index is not None and not force_rebuild:
            return _index
        path = Path(cache_path) if cache_path else INDEX_PATH
        start = time.perf_counter()
        version = _geonames_version()
        index = None if force_rebuild else _read_cache(path, version)
        source = "cache"
        if index is None:
            try:
                version, cities = _load_geonames()
            except ImportError:
                log.info("geonamescache is not installed; using simple text matching "
                         "for job locations.")
                index, source = {}, "unavailable"
            except Exception as exc:  # broken data file etc.
                log.warning("Could not load place data: %s", exc)
                index, source = {}, "unavailable"
            else:
                raw = _build_index(version, cities)
                del cities
                _write_cache(path, raw)
                index, source = _unpack(raw), "built"
        seconds = time.perf_counter() - start
        _last_load.clear()
        _last_load.update(source=source, seconds=round(seconds, 3), path=str(path),
                          places=len(index.get("name", [])) if index else 0)
        if source != "unavailable":
            log.info("Place index %s in %.2fs (%d places)", source, seconds,
                     _last_load["places"])
        _index = index
        _lookup.cache_clear()
        geocode.cache_clear()
        return _index


def index_info() -> dict[str, Any]:
    """How the index was last obtained: source ('built'/'cache'/'unavailable'), seconds, path."""
    return dict(_last_load)


def set_index_path(path: str | Path | None) -> None:
    """Point the on-disk cache somewhere else (tests) and forget the loaded index."""
    global INDEX_PATH
    INDEX_PATH = Path(path) if path else _default_index_path()
    reset_index()


def reset_index() -> None:
    global _index
    with _lock:
        _index = None
        _last_load.clear()
    _lookup.cache_clear()
    geocode.cache_clear()


# --------------------------------------------------------------------------
# Geocoding
# --------------------------------------------------------------------------
_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_LEAD = re.compile(r"^\s*(?:remote|hybrid|on[- ]?site|in[- ]office)\s+(?:in|from|near|at)\s+", re.I)
_NOISE = re.compile(r"\b(?:remote|hybrid|on[- ]?site|in[- ]office|work from home|wfh|greater|"
                    r"metropolitan|metro|metroplex|area|region)\b", re.I)
_SEPARATORS = re.compile(r"[,;|•]|\s[-–—/]\s")
_DC = re.compile(r"\bd\.\s?c\.?(?=\W|$)", re.I)
_US = re.compile(r"\bu\.\s?s\.(?:\s?a\.)?(?=\W|$)", re.I)


def _parts(text: str) -> list[str]:
    s = _DC.sub("DC", text or "")
    s = _US.sub("US", s)
    s = _PAREN.sub(" ", s)
    s = _ZIP.sub(" ", s)
    s = _LEAD.sub("", s)
    s = _NOISE.sub(" ", s)
    return [p.strip(" .-") for p in _SEPARATORS.split(s) if p.strip(" .-")]


def _pick(rows: list[list[Any]], admin1: str | None, country: str | None,
          prefer_country: str) -> list[Any] | None:
    if not rows:
        return None
    if admin1:
        rows = [r for r in rows if r[2] == admin1 and r[1] == (country or "US")]
        if not rows:
            return None  # that name isn't in that state
    elif country:
        rows = [r for r in rows if r[1] == country]
        if not rows:
            return None
    else:
        preferred = [r for r in rows if r[1] == prefer_country]
        rows = preferred or rows
    return max(rows, key=lambda r: r[5])


@lru_cache(maxsize=8192)
def _lookup(name: str, admin1: str | None, country: str | None,
            prefer_country: str) -> Place | None:
    index = load_index()
    if not index or not name:
        return None
    for table in ("names", "alts"):
        ids = index[table].get(name)
        if ids:
            rows = []
            for i in map(int, ids.split(",")):
                cc, _, admin = index["region"][i].partition(" ")
                lat, lon, pop = index["coords"][i].split(",")
                rows.append([index["name"][i], cc, admin, float(lat), float(lon), int(pop)])
            row = _pick(rows, admin1, country, prefer_country)
            if row:
                return Place(name=row[0], admin1=row[2], country=row[1], lat=row[3],
                             lon=row[4], population=row[5])
    return None


def _split_trailing_state(name: str) -> tuple[str, str] | None:
    """'austin tx' -> ('austin', 'TX'); 'santa fe new mexico' -> ('santa fe', 'NM')."""
    words = name.split()
    for n in (3, 2, 1):
        if len(words) > n:
            tail = " ".join(words[-n:])
            code = _STATE_CODES.get(tail) if n == 1 and len(tail) == 2 else _STATE_BY_NAME.get(tail)
            if code:
                return " ".join(words[:-n]), code
    return None


def _find(city: str, admin1: str | None, country: str | None, prefer: str) -> Place | None:
    name = normalize(city)
    place = _lookup(name, admin1, country, prefer)
    if place:
        return place
    if not admin1:
        split = _split_trailing_state(name)
        if split:
            place = _lookup(split[0], split[1], "US", prefer)
            if place:
                return place
    # "Dallas-Fort Worth" / "Minneapolis/St. Paul": try each half.
    for piece in re.split(r"[-/–]", city):
        if piece.strip() and piece.strip() != city.strip():
            place = _lookup(normalize(piece), admin1, country, prefer)
            if place:
                return place
    return None


@lru_cache(maxsize=8192)
def geocode(text: str, prefer_country: str = "US") -> Place | None:
    """Turn a location like "Austin, TX" or "Hybrid in Marfa, Texas" into a Place.

    Returns None for remote-only, country-only, empty or unknown text.
    """
    parts = _parts(text or "")
    if not parts:
        return None
    city, admin1, country = parts[0], None, None
    rest = parts[1:]
    if len(parts) >= 3 and len(normalize(parts[-1])) == 2:
        # "Austin, TX, US" / "Toronto, ON, CA": a trailing two-letter code is the country.
        country = normalize(parts[-1]).upper()
        rest = parts[1:-1]
    for extra in rest:
        code = _state_code(extra)
        if code and not admin1 and country in (None, "US"):
            admin1, country = code, "US"
            continue
        cc = _country_code(extra)
        if cc and not country:
            country = cc
    city_key = normalize(city)
    if _country_code(city) and len(parts) == 1:
        return None  # just "United States"
    if not admin1 and city_key in _STATE_BY_NAME and city_key not in ("new york", "washington dc"):
        return None  # "Texas" or "Texas, United States": a whole state, not a town
    return _find(city, admin1, country, prefer_country or "US")


# --------------------------------------------------------------------------
# Distance and fit
# --------------------------------------------------------------------------
_EARTH_RADIUS_MILES = 3958.8


def distance_miles(a: Place, b: Place) -> float:
    """Great-circle distance in miles (haversine)."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * _EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(h)))


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 <= x0:
        return y1
    t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
    return y0 + (y1 - y0) * t


_REMOTE_WORD = re.compile(r"\b(remote|work from home|wfh|anywhere)\b", re.I)


def _home_town(home: str) -> str:
    parts = _parts(home)
    return parts[0] if parts else ""


def location_fit(home: str, job_location: str, is_remote: bool | None,
                 commute_miles: float = 25, remote_ok: bool = True) -> LocationFit:
    """Score 0-1 for how well the job's location suits someone living at ``home``."""
    job_location = job_location or ""
    commute = max(1.0, float(commute_miles or 25))
    job_place = geocode(job_location) if job_location.strip() else None
    says_remote = bool(_REMOTE_WORD.search(job_location)) and job_place is None
    if is_remote is True or says_remote:
        return LocationFit(0.7 if remote_ok else 0.15, None, "Remote", False)
    if not (home or "").strip():
        return LocationFit(0.5, None, "", False)

    home_place = geocode(home)
    if home_place is None or job_place is None:
        town = _home_town(home)
        if town and re.search(rf"(?<!\w){re.escape(town)}(?!\w)", job_location, re.I):
            return LocationFit(1.0, 0.0, "Same town", True)
        return LocationFit(0.35, None, "Location unknown", False)

    miles = 0.0 if home_place == job_place else distance_miles(home_place, job_place)
    if miles <= 5:
        return LocationFit(1.0, round(miles, 1), "Same town", True)
    if miles <= commute:
        score = _lerp(miles, 5, commute, 1.0, 0.75)
    elif miles <= 2 * commute:
        score = _lerp(miles, commute, 2 * commute, 0.75, 0.25)
    elif miles <= 4 * commute:
        score = _lerp(miles, 2 * commute, 4 * commute, 0.25, 0.05)
    else:
        score = 0.0
    return LocationFit(round(score, 3), round(miles, 1), f"{round(miles)} mi from home",
                       miles <= commute)


# --------------------------------------------------------------------------
# Employer size
# --------------------------------------------------------------------------
_NUMBER = re.compile(r"\d[\d,]*")


def employee_count(text: str | None) -> int | None:
    """Upper end of a company-size string: "11 to 50" -> 50, "10,000+" -> 10000."""
    if text is None:
        return None
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return int(text) if text == text and text > 0 else None
    s = str(text).strip().lower()
    if not s:
        return None
    if re.search(r"self[- ]?employed|sole proprietor", s):
        return 1
    nums = [int(n.replace(",", "")) for n in _NUMBER.findall(s) if n.replace(",", "")]
    nums = [n for n in nums if n > 0]
    return max(nums) if nums else None


def local_employer_score(job: Job, fit: LocationFit) -> tuple[float, str]:
    """Bonus (0-1, label) for a small employer within the person's commute."""
    if not fit.is_local:
        return 0.0, ""
    size = employee_count(getattr(job, "company_employees", None))
    if size is None:
        return 0.5, "Local employer"
    if size <= 200:
        return 1.0, "Small local employer"
    if size <= 1000:
        return 0.6, "Local employer"
    return 0.2, ""
