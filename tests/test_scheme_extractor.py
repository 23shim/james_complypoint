"""Tests for scheme name extraction (location identity)."""

import pytest

from classification.scheme_extractor import (
    extract_scheme,
    _strip_noise,
    _find_address_triggers,
    _find_place_names,
)
from classification.models import EntityDefinition


@pytest.fixture
def address_entity():
    """Minimal address entity definition for testing."""
    return EntityDefinition(
        name="address",
        tokens=[
            "road", "street", "lane", "avenue", "drive", "close",
            "court", "crescent", "way", "place", "terrace", "grove",
            "gardens", "park", "mews", "walk", "rise", "square",
            "hill", "view", "row", "green", "fields", "heights",
        ],
        abbreviations=[
            "rd", "st", "ln", "ave", "av", "dr", "cl", "ct",
            "cres", "gr", "pk", "sq", "mt", "pl", "gdns", "gdn", "tce",
        ],
        compiled_patterns=[],
        abbreviation_map={
            "rd": "road",
            "st": "street",
            "ln": "lane",
            "ave": "avenue",
            "av": "avenue",
            "dr": "drive",
            "cl": "close",
            "ct": "court",
            "cres": "crescent",
            "gr": "grove",
            "pk": "park",
            "sq": "square",
            "mt": "mount",
            "pl": "place",
            "gdns": "gardens",
            "gdn": "gardens",
            "tce": "terrace",
        },
    )


@pytest.fixture
def place_names():
    """Small set of place names for testing."""
    return {
        "dartford", "faversham", "medway", "ashford",
        "hoo", "lenham", "oare", "gravesend",
        "st albans", "kings lynn", "milton keynes",
    }


# -----------------------------------------------------------------------
# Noise stripping
# -----------------------------------------------------------------------

class TestStripNoise:

    def test_phase_stripped(self):
        assert _strip_noise("Phase 2 Liberty Park") == "Liberty Park"

    def test_phase_ordinal_stripped(self):
        assert _strip_noise("Phase One Oak Lane") == "Oak Lane"

    def test_phase_compound_stripped(self):
        assert _strip_noise("Phase 4 & 5 Hoo Road") == "Hoo Road"

    def test_date_range_stripped(self):
        assert _strip_noise("Oak Lane 2020-2021") == "Oak Lane"

    def test_standalone_year_stripped(self):
        assert _strip_noise("Cedar Heights 2023") == "Cedar Heights"

    def test_new_annotation_stripped(self):
        assert _strip_noise("New Dec 20 Hoo Road") == "Hoo Road"

    def test_new_scheme_stripped(self):
        assert _strip_noise("New scheme Oak Lane") == "Oak Lane"

    def test_combined_noise(self):
        result = _strip_noise("Phase 2 Liberty Park Hoo Road 2020-2021")
        assert result == "Liberty Park Hoo Road"

    def test_orphan_separator_cleaned(self):
        result = _strip_noise("Phase 2 - Liberty Park")
        assert result == "Liberty Park"

    def test_no_noise(self):
        assert _strip_noise("Liberty Park") == "Liberty Park"

    def test_empty_string(self):
        assert _strip_noise("") == ""

    def test_written_date_stripped(self):
        assert _strip_noise("Hoo Road 10 July 24") == "Hoo Road"


# -----------------------------------------------------------------------
# Full extraction pipeline
# -----------------------------------------------------------------------

class TestExtractScheme:

    def test_address_trigger_with_context(self, address_entity, place_names):
        """Phase stripped, address trigger + preceding words captured."""
        result = extract_scheme(
            "Phase 2 Liberty Park Hoo Road",
            address_entity, place_names,
        )
        assert "liberty" in result
        assert "park" in result
        assert "hoo" in result
        assert "road" in result
        assert "phase" not in result

    def test_place_name_only(self, address_entity, place_names):
        """Scheme with place name but no address trigger."""
        result = extract_scheme("Faversham Lakes", address_entity, place_names)
        assert "faversham" in result

    def test_address_and_place_combined(self, address_entity, place_names):
        """Both address trigger and place name included."""
        result = extract_scheme(
            "Cotton Lane Dartford", address_entity, place_names,
        )
        assert "cotton" in result
        assert "lane" in result
        assert "dartford" in result

    def test_abbreviation_expanded(self, address_entity, place_names):
        """Abbreviations like 'Rd' expanded to 'road'."""
        result = extract_scheme("Hoo Rd", address_entity, place_names)
        assert "road" in result
        assert "hoo" in result
        assert "rd" not in result

    def test_multiple_triggers(self, address_entity, place_names):
        """Multiple address triggers both captured."""
        result = extract_scheme("Park Road Medway", address_entity, place_names)
        assert "park" in result
        assert "road" in result
        assert "medway" in result

    def test_phase_and_date_stripped(self, address_entity, place_names):
        """Phase + date both stripped before extraction."""
        result = extract_scheme(
            "Phase 3 Oak Lane 2020-2021", address_entity, place_names,
        )
        assert "oak" in result
        assert "lane" in result
        assert "phase" not in result
        assert "2020" not in result

    def test_bigram_place_name(self, address_entity, place_names):
        """Multi-word place name matched via bigram."""
        result = extract_scheme(
            "Phase 1 St Albans Road", address_entity, place_names,
        )
        assert "st" in result.split()
        assert "albans" in result
        assert "road" in result

    def test_empty_input(self, address_entity, place_names):
        assert extract_scheme("", address_entity, place_names) == ""

    def test_whitespace_only(self, address_entity, place_names):
        assert extract_scheme("   ", address_entity, place_names) == ""

    def test_result_is_lowercase(self, address_entity, place_names):
        result = extract_scheme("Hoo Road", address_entity, place_names)
        assert result == result.lower()

    def test_punctuation_stripped(self, address_entity, place_names):
        result = extract_scheme(
            "Cotton Lane, Dartford", address_entity, place_names,
        )
        assert "," not in result

    def test_noise_only_returns_empty(self, address_entity, place_names):
        """If noise stripping removes everything, return empty."""
        result = extract_scheme("Phase 2 2023", address_entity, place_names)
        assert result == ""


# -----------------------------------------------------------------------
# Address trigger detection (unit tests)
# -----------------------------------------------------------------------

class TestFindAddressTriggers:

    def _run(self, text, address_entity):
        words = text.split()
        words_clean = [w.lower() for w in words]
        include = set()
        expansions = {}
        _find_address_triggers(
            words, words_clean, address_entity, include, expansions,
        )
        return include, expansions

    def test_single_trigger(self, address_entity):
        include, _ = self._run("Hoo Road", address_entity)
        assert 0 in include  # "Hoo" (preceding)
        assert 1 in include  # "Road" (trigger)

    def test_two_preceding_words(self, address_entity):
        include, _ = self._run("Liberty Park Hoo Road", address_entity)
        # "Road" trigger captures "Park" and "Hoo" preceding
        assert 1 in include  # "Park"
        assert 2 in include  # "Hoo"
        assert 3 in include  # "Road"

    def test_abbreviation_expansion(self, address_entity):
        _, expansions = self._run("Hoo Rd", address_entity)
        assert 1 in expansions
        assert expansions[1] == "road"

    def test_following_word_captured(self, address_entity):
        include, _ = self._run("Park Hoo Something", address_entity)
        # "Park" trigger: following word "Hoo"
        assert 1 in include  # "Hoo" (following of Park)

    def test_following_digit_blocked(self, address_entity):
        include, _ = self._run("Park 123", address_entity)
        # "123" starts with digit — not captured as following
        assert 1 not in include


# -----------------------------------------------------------------------
# Place name detection (unit tests)
# -----------------------------------------------------------------------

class TestFindPlaceNames:

    def test_single_token(self, place_names):
        words_clean = ["liberty", "park", "hoo", "road"]
        include = set()
        _find_place_names(words_clean, place_names, include)
        assert 2 in include  # "hoo"

    def test_bigram(self, place_names):
        words_clean = ["st", "albans", "road"]
        include = set()
        _find_place_names(words_clean, place_names, include)
        assert 0 in include  # "st"
        assert 1 in include  # "albans"

    def test_no_match(self, place_names):
        words_clean = ["liberty", "oak"]
        include = set()
        _find_place_names(words_clean, place_names, include)
        assert len(include) == 0

    def test_empty_place_names_set(self):
        words_clean = ["dartford"]
        include = set()
        _find_place_names(words_clean, set(), include)
        assert len(include) == 0
