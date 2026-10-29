import json
import threading

import pytest

from core.models import Job
from matching import location as loc
from matching.location import (
    LocationFit,
    Place,
    distance_miles,
    employee_count,
    geocode,
    local_employer_score,
    location_fit,
)


def _job(employees=None, location="Austin, TX"):
    return Job(id="j1", title="Clerk", company="Acme", location=location,
               company_employees=employees)


@pytest.fixture(scope="module")
def geo(tmp_path_factory):
    """Build the real index once for this module, in a temporary cache file."""
    pytest.importorskip("geonamescache")
    path = tmp_path_factory.mktemp("geo") / "places_index.json"
    loc.set_index_path(path)
    loc.load_index()
    assert loc.index_info()["source"] in ("built", "cache")
    yield path
    loc.set_index_path(None)


@pytest.fixture
def no_geodata(monkeypatch, tmp_path):
    """Pretend geonamescache isn't installed (and no cache file exists)."""
    def missing():
        raise ImportError("no geonamescache")
    old_path = loc.INDEX_PATH
    monkeypatch.setattr(loc, "_load_geonames", missing)
    monkeypatch.setattr(loc, "_geonames_version", lambda: None)
    loc.set_index_path(tmp_path / "none.json")
    yield
    monkeypatch.undo()
    loc.set_index_path(old_path)


# ------------------------------------------------------------------ geocode
@pytest.mark.parametrize("text", [
    "Austin, TX", "Austin, Texas", "Austin, TX, US", "Austin, TX 78701", "Greater Austin Area",
    "Hybrid in Austin, TX", "Austin, TX (Hybrid)", "austin tx", "Austin, Texas, United States",
])
def test_geocode_austin_variants(geo, text):
    place = geocode(text)
    assert place is not None
    assert (place.name, place.admin1, place.country) == ("Austin", "TX", "US")


@pytest.mark.parametrize("text", ["Washington, D.C.", "Washington, DC", "Washington DC"])
def test_geocode_washington_dc(geo, text):
    place = geocode(text)
    assert place is not None and place.admin1 == "DC" and place.country == "US"


@pytest.mark.parametrize("text", ["Remote", "Remote - US", "United States", "", "   ",
                                  "Qwxzyville, ZZ", "Texas", "Texas, United States"])
def test_geocode_none(geo, text):
    assert geocode(text) is None


def test_geocode_small_town_and_abbreviations(geo):
    marfa = geocode("Marfa, TX")
    assert marfa is not None and marfa.admin1 == "TX" and marfa.population < 5000
    assert geocode("St. Louis, MO").admin1 == "MO"
    assert geocode("Ft. Worth, TX").name == "Fort Worth"
    assert geocode("Mt. Pleasant, SC").name == "Mount Pleasant"


def test_geocode_ambiguous_names(geo):
    il, mo = geocode("Springfield, IL"), geocode("Springfield, MO")
    assert il.admin1 == "IL" and mo.admin1 == "MO"
    assert distance_miles(il, mo) > 150
    # No state: prefer the US, then the biggest place.
    paris = geocode("Paris")
    assert paris.country == "US"
    assert geocode("Paris", prefer_country="FR").country == "FR"
    assert geocode("London, UK").country == "GB"
    # A name that doesn't exist in the given state is not guessed elsewhere.
    assert geocode("Marfa, ME") is None


def test_distances(geo):
    austin = geocode("Austin, TX")
    assert distance_miles(austin, geocode("Round Rock, TX")) == pytest.approx(17, rel=0.1)
    assert distance_miles(austin, geocode("Houston, TX")) == pytest.approx(146, rel=0.1)
    assert distance_miles(austin, austin) == 0


def test_distance_without_index():
    a = Place("A", "", "", 0.0, 0.0, 0)
    b = Place("B", "", "", 0.0, 1.0, 0)
    assert distance_miles(a, b) == pytest.approx(69.1, rel=0.01)


# ------------------------------------------------------------------ location_fit
def test_fit_same_town_and_near(geo):
    fit = location_fit("Austin, TX", "Austin, TX 78701", False)
    assert fit == LocationFit(1.0, 0.0, "Same town", True)
    near = location_fit("Austin, TX", "Round Rock, TX", False, commute_miles=25)
    assert near.is_local and 0.75 <= near.score < 1.0
    assert near.label.endswith("mi from home") and near.label.startswith("1")


def test_fit_score_bands(geo):
    far = location_fit("Austin, TX", "Houston, TX", None, commute_miles=25)
    assert far.score == 0.0 and not far.is_local and far.label == f"{round(far.miles)} mi from home"
    # Same pair with a long commute allowance lands in the 1x-2x band.
    mid = location_fit("Austin, TX", "Houston, TX", None, commute_miles=150)
    assert 0.75 <= mid.score <= 1.0 and mid.is_local
    band2 = location_fit("Austin, TX", "Houston, TX", None, commute_miles=80)
    assert 0.25 <= band2.score <= 0.75 and not band2.is_local
    band3 = location_fit("Austin, TX", "Houston, TX", None, commute_miles=50)
    assert 0.05 <= band3.score <= 0.25


def test_fit_remote_rules(geo):
    assert location_fit("Austin, TX", "Anywhere", True) == LocationFit(0.7, None, "Remote", False)
    assert location_fit("Austin, TX", "Remote", None, remote_ok=False).score == 0.15
    assert location_fit("Austin, TX", "Remote - US", None).label == "Remote"
    assert location_fit("", "Remote", None).label == "Remote"


def test_fit_unknown_and_empty_home(geo):
    assert location_fit("", "Austin, TX", False) == LocationFit(0.5, None, "", False)
    unknown = location_fit("Austin, TX", "Somewhere Nice", False)
    assert unknown == LocationFit(0.35, None, "Location unknown", False)
    assert location_fit("Austin, TX", "", None).label == "Location unknown"


def test_fallback_without_geonamescache(no_geodata):
    assert geocode("Austin, TX") is None
    assert loc.index_info()["source"] == "unavailable"
    fit = location_fit("Marfa, TX", "Downtown Marfa, Texas", False)
    assert fit == LocationFit(1.0, 0.0, "Same town", True)
    assert location_fit("Marfa, TX", "Marfaville, TX", False).label == "Location unknown"
    assert location_fit("Marfa, TX", "Remote", True).label == "Remote"
    assert local_employer_score(_job("1-10"), fit) == (1.0, "Small local employer")


# ------------------------------------------------------------------ employers
@pytest.mark.parametrize("text, n", [
    ("11 to 50", 50), ("1-10", 10), ("10,000+", 10000), ("501 - 1,000 employees", 1000),
    ("Self-employed", 1), ("", None), (None, None), ("lots", None), ("0", None), (250, 250),
])
def test_employee_count(text, n):
    assert employee_count(text) == n


def test_local_employer_score():
    local = LocationFit(1.0, 3.0, "Same town", True)
    far = LocationFit(0.0, 300.0, "300 mi from home", False)
    assert local_employer_score(_job("11 to 50"), far) == (0.0, "")
    assert local_employer_score(_job("11 to 50"), local) == (1.0, "Small local employer")
    assert local_employer_score(_job("201 to 500"), local) == (0.6, "Local employer")
    assert local_employer_score(_job("10,000+"), local) == (0.2, "")
    assert local_employer_score(_job(None), local) == (0.5, "Local employer")


# ------------------------------------------------------------------ cache
def test_cache_created_then_reused(monkeypatch, tmp_path):
    path = tmp_path / "idx.json"
    calls = []
    tiny = {"1": {"name": "Marfa", "countrycode": "US", "admin1code": "TX",
                  "latitude": 30.3097, "longitude": -104.0213, "population": 1733,
                  "alternatenames": []}}

    def fake():
        calls.append(1)
        return "test-ver", tiny

    monkeypatch.setattr(loc, "_load_geonames", fake)
    monkeypatch.setattr(loc, "_geonames_version", lambda: "test-ver")
    old_path = loc.INDEX_PATH
    try:
        loc.set_index_path(path)
        assert geocode("Marfa, TX").population == 1733
        assert loc.index_info()["source"] == "built" and path.exists() and calls == [1]
        assert json.loads(path.read_text())["version"] == "test-ver"

        loc.reset_index()
        assert geocode("Marfa, Texas").name == "Marfa"
        assert loc.index_info()["source"] == "cache" and calls == [1]

        # A new geonamescache version rebuilds.
        monkeypatch.setattr(loc, "_geonames_version", lambda: "newer")
        loc.reset_index()
        loc.load_index()
        assert loc.index_info()["source"] == "built" and calls == [1, 1]
    finally:
        loc.set_index_path(old_path)


def test_index_loads_once_across_threads(monkeypatch, tmp_path):
    calls = []

    def fake():
        calls.append(1)
        return "v", {}

    monkeypatch.setattr(loc, "_load_geonames", fake)
    monkeypatch.setattr(loc, "_geonames_version", lambda: "v")
    old_path = loc.INDEX_PATH
    try:
        loc.set_index_path(tmp_path / "idx.json")
        threads = [threading.Thread(target=loc.load_index) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert calls == [1]
    finally:
        loc.set_index_path(old_path)
