"""Tests for address extraction from noisy folder/file name segments."""

import pytest

from classification.address_extractor import (
    extract_address,
    purge_dates,
    _anchor_trigger_extract,
    post_process,
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


# -----------------------------------------------------------------------
# Full extraction pipeline tests (spec examples)
# -----------------------------------------------------------------------

class TestExtractAddress:

    def test_number_range_with_street(self, address_entity):
        """'plots 67-70 Hawkes Way' -> '67-70 hawkes way'"""
        result = extract_address("plots 67-70 Hawkes Way", address_entity)
        assert result == "67-70 hawkes way"

    def test_number_street_with_trailing_city(self, address_entity):
        """'24 Warren Wood Road, Rochester' -> '24 warren wood road rochester'"""
        result = extract_address(
            "24 Warren Wood Road, Rochester", address_entity,
        )
        assert result == "24 warren wood road rochester"

    def test_parenthetical_noise_stripped(self, address_entity):
        """'Cotton Lane (Vent-Axia fan' -> 'cotton lane'"""
        result = extract_address("Cotton Lane (Vent-Axia fan", address_entity)
        assert result == "cotton lane"

    def test_trailing_date_stripped(self, address_entity):
        """'22-27 Hadlow Close 271115' -> '22-27 hadlow close'"""
        result = extract_address("22-27 Hadlow Close 271115", address_entity)
        assert result == "22-27 hadlow close"

    def test_plain_street_name(self, address_entity):
        """'Lenham Road' -> 'lenham road'"""
        result = extract_address("Lenham Road", address_entity)
        assert result == "lenham road"

    def test_abbreviation_expanded(self, address_entity):
        """'Manston Rd' -> 'manston road'"""
        result = extract_address("Manston Rd", address_entity)
        assert result == "manston road"

    def test_abbreviation_cres(self, address_entity):
        """'Mills Cres' -> 'mills crescent'"""
        result = extract_address("Mills Cres", address_entity)
        assert result == "mills crescent"

    def test_house_number_with_letter(self, address_entity):
        """'44B Magpie Hall Road' -> '44b magpie hall road'"""
        result = extract_address("44B Magpie Hall Road", address_entity)
        assert result == "44b magpie hall road"

    def test_postcode_only(self, address_entity):
        """'TN13 1AB' -> 'tn13 1ab' (postcode-only segment)"""
        result = extract_address("TN13 1AB", address_entity)
        assert result == "tn13 1ab"

    def test_street_with_postcode(self, address_entity):
        """'Lenham Road ME10 3ZZ' -> 'lenham road me10 3zz'"""
        result = extract_address("Lenham Road ME10 3ZZ", address_entity)
        assert result == "lenham road me10 3zz"

    def test_empty_string(self, address_entity):
        result = extract_address("", address_entity)
        assert result == ""

    def test_no_trigger_token(self, address_entity):
        """No street suffix -> empty result."""
        result = extract_address("Finance Department", address_entity)
        assert result == ""

    def test_ampersand_range(self, address_entity):
        """'3 & 5 Oak Lane' -> '3 & 5 oak lane'"""
        result = extract_address("3 & 5 Oak Lane", address_entity)
        assert result == "3 & 5 oak lane"

    def test_no_abbreviation_map(self):
        """Without abbreviation map, abbreviations stay as-is."""
        entity = EntityDefinition(
            name="address",
            tokens=["road"],
            abbreviations=["rd"],
            compiled_patterns=[],
            abbreviation_map={},
        )
        result = extract_address("Manston Rd", entity)
        assert result == "manston rd"

    def test_st_marys_not_expanded(self, address_entity):
        """'St Mary's Close' — 'st' in preceding words NOT expanded."""
        result = extract_address("St Mary's Close", address_entity)
        # "st" is only expanded when it IS the trigger token.
        # Here the trigger is "close", so "st" stays as-is.
        assert "saint" not in result
        assert "close" in result


# -----------------------------------------------------------------------
# Date purge tests
# -----------------------------------------------------------------------

class TestPurgeDates:

    def test_six_digit_date_code(self):
        assert purge_dates("Hawkes Way 160219") == "Hawkes Way"

    def test_dotted_date(self):
        assert purge_dates("Lenham Road 18.10.13") == "Lenham Road"

    def test_slashed_date(self):
        assert purge_dates("Lenham Road 20/05/22") == "Lenham Road"

    def test_written_date_day_month_year(self):
        assert purge_dates("Oak Lane 10 July 24") == "Oak Lane"

    def test_written_date_month_year(self):
        assert purge_dates("Elm Street Aug 2022") == "Elm Street"

    def test_parenthesised_year_range(self):
        assert purge_dates("Park Avenue (2015-2016)") == "Park Avenue"

    def test_bare_year_range(self):
        assert purge_dates("Park Avenue 2015-2016") == "Park Avenue"

    def test_no_dates(self):
        assert purge_dates("Hawkes Way") == "Hawkes Way"

    def test_multiple_dates(self):
        result = purge_dates("Elm Street 160219 18.10.13")
        assert result == "Elm Street"


# -----------------------------------------------------------------------
# Full pipeline date-purge integration
# -----------------------------------------------------------------------

class TestExtractAddressWithDates:

    def test_trailing_six_digit(self, address_entity):
        result = extract_address("Hawkes Way 160219", address_entity)
        assert result == "hawkes way"

    def test_trailing_dotted_date(self, address_entity):
        result = extract_address("Lenham Road 18.10.13", address_entity)
        assert result == "lenham road"

    def test_trailing_written_date(self, address_entity):
        result = extract_address("Oak Lane 10 July 24", address_entity)
        assert result == "oak lane"

    def test_trailing_month_year(self, address_entity):
        result = extract_address("Elm Street Aug 2022", address_entity)
        assert result == "elm street"

    def test_parenthesised_year_range(self, address_entity):
        result = extract_address("Park Avenue (2015-2016)", address_entity)
        assert result == "park avenue"


# -----------------------------------------------------------------------
# Anchor-Trigger edge cases
# -----------------------------------------------------------------------

class TestAnchorTriggerEdgeCases:

    def test_trigger_as_first_word(self, address_entity):
        """Trigger at start — no preceding words or anchor."""
        result = extract_address("Road Rochester", address_entity)
        assert result == "road rochester"

    def test_only_trigger_word(self, address_entity):
        """Just the trigger word alone."""
        result = extract_address("Lane", address_entity)
        assert result == "lane"

    def test_following_word_starts_with_paren(self, address_entity):
        """Rule D — don't capture word starting with '('."""
        result = extract_address("Oak Lane (phase 2)", address_entity)
        assert result == "oak lane"

    def test_following_word_is_numeric(self, address_entity):
        """Rule D — don't capture word starting with digit."""
        result = extract_address("Hadlow Close 271115", address_entity)
        # 271115 is also purged as a 6-digit date, so nothing follows.
        assert result == "hadlow close"

    def test_greedy_anchor_complex_range(self, address_entity):
        """Greedy anchor captures all contiguous numeric tokens."""
        result = extract_address("Phase 2 1-4 & 6 Oak Lane", address_entity)
        # "2" is numeric and contiguous with the range — greedy capture
        assert result == "2 1-4 & 6 oak lane"


# -----------------------------------------------------------------------
# Post-process unit tests
# -----------------------------------------------------------------------

class TestPostProcess:

    def test_lowercase(self):
        assert post_process("Lenham Road") == "lenham road"

    def test_strip_punctuation(self):
        assert post_process("Lenham Road,") == "lenham road"

    def test_collapse_whitespace(self):
        assert post_process("Lenham  Road") == "lenham road"

    def test_strip_leading_trailing(self):
        assert post_process("  , Lenham Road . ") == "lenham road"
