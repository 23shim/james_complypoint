"""Tests for plot extraction from noisy folder/file name segments."""

import re

import pytest

from classification.plot_extractor import (
    extract_plot,
    _build_trigger_set,
    _keyword_forward_extract,
    _apply_descending_heuristic,
    _apply_address_aware_trim,
    _pattern_fallback,
)
from classification.models import EntityDefinition


@pytest.fixture
def plot_entity():
    """Minimal plot entity definition for testing (mirrors housing.yaml)."""
    return EntityDefinition(
        name="plot",
        tokens=[
            "plot", "unit", "flat", "apartment", "dwelling",
            "bungalow", "maisonette", "penthouse", "studio",
            "annexe", "lot",
        ],
        abbreviations=["plt", "apt", "blk", "ph"],
        compiled_patterns=[
            re.compile(r"\bp[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bplot[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bunit[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bflat[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bblk[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bphase[-_ ]?\d{1,2}\b", re.IGNORECASE),
            re.compile(r"\bblock[-_ ]?[a-z0-9]{1,3}\b", re.IGNORECASE),
            re.compile(r"\bplt[-_ ]?\d{1,4}\b", re.IGNORECASE),
            re.compile(r"\blot[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bapt[-_ ]?\d{1,2}[.]\d{2,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bhouse[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\blodge[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
            re.compile(r"\bcottage[-_ ]?\d{1,4}[a-z]?\b", re.IGNORECASE),
        ],
        abbreviation_map={
            "plt": "plot",
            "apt": "apartment",
            "blk": "block",
            "ph": "phase",
        },
    )


# -----------------------------------------------------------------------
# Full extraction pipeline tests (spec examples)
# -----------------------------------------------------------------------

class TestExtractPlot:

    def test_simple_plot(self, plot_entity):
        """'Plot 34' -> 'plot 34'"""
        result = extract_plot("Plot 34", plot_entity)
        assert result == "plot 34"

    def test_plot_ampersand(self, plot_entity):
        """'Plot 1 & 2' -> 'plot 1 & 2'"""
        result = extract_plot("Plot 1 & 2", plot_entity)
        assert result == "plot 1 & 2"

    def test_plot_range_with_trailing_address(self, plot_entity):
        """'Plot 16-30 Colemans Close ashford,' -> 'plot 16-30'"""
        result = extract_plot("Plot 16-30 Colemans Close ashford,", plot_entity)
        assert result == "plot 16-30"

    def test_descending_heuristic_trims(self, plot_entity):
        """'Plot 66 - 42 Hawkes Way' -> 'plot 66' (66 > 42)"""
        result = extract_plot("Plot 66 - 42 Hawkes Way", plot_entity)
        assert result == "plot 66"

    def test_plot_to_range(self, plot_entity):
        """'Plot 77 to 79 Finberry & Song Thrush' -> 'plot 77 to 79'"""
        result = extract_plot("Plot 77 to 79 Finberry & Song Thrush", plot_entity)
        assert result == "plot 77 to 79"

    def test_ascending_range_kept(self, plot_entity):
        """'Plot 1 - 5' -> 'plot 1 - 5' (1 < 5, valid range)"""
        result = extract_plot("Plot 1 - 5", plot_entity)
        assert result == "plot 1 - 5"

    def test_stops_at_second_trigger(self, plot_entity):
        """'Plots 63-68 Block 2 Blossom Park Hoo' -> 'plots 63-68'"""
        result = extract_plot("Plots 63-68 Block 2 Blossom Park Hoo", plot_entity)
        assert result == "plots 63-68"

    def test_block_with_parenthetical(self, plot_entity):
        """'Block 25-33 (Filmer House)' -> 'block 25-33'"""
        result = extract_plot("Block 25-33 (Filmer House)", plot_entity)
        assert result == "block 25-33"

    def test_block_single_letter(self, plot_entity):
        """'Block E Flats Gas flue boxing' -> 'block e'"""
        result = extract_plot("Block E Flats Gas flue boxing", plot_entity)
        assert result == "block e"

    def test_block_stops_at_second_block(self, plot_entity):
        """'Block A & Block B (Architects)' -> 'block a'"""
        result = extract_plot("Block A & Block B (Architects)", plot_entity)
        assert result == "block a"

    def test_block_number_range(self, plot_entity):
        """'Block 1 & 2 (Sketch of Flat locations)' -> 'block 1 & 2'"""
        result = extract_plot("Block 1 & 2 (Sketch of Flat locations)", plot_entity)
        assert result == "block 1 & 2"

    def test_flat_with_letter_suffix(self, plot_entity):
        """'Flat 3A Fuggles Close' -> 'flat 3a'"""
        result = extract_plot("Flat 3A Fuggles Close", plot_entity)
        assert result == "flat 3a"

    def test_unit_comma_separated(self, plot_entity):
        """'Unit 8, 9' -> 'unit 8 9'"""
        result = extract_plot("Unit 8, 9", plot_entity)
        assert result == "unit 8 9"

    def test_pattern_fallback(self, plot_entity):
        """'P239-Lv0' -> 'p239' (pattern fallback)"""
        result = extract_plot("P239-Lv0", plot_entity)
        assert result == "p239"

    def test_address_with_plot(self, plot_entity):
        """'Walderslade Road site inspection photos & Plot 15' -> 'plot 15'"""
        result = extract_plot(
            "Walderslade Road site inspection photos & Plot 15", plot_entity,
        )
        assert result == "plot 15"

    def test_no_plot_trigger(self, plot_entity):
        """'Cotton Lane (Vent-Axia fan' -> '' (no plot trigger)"""
        result = extract_plot("Cotton Lane (Vent-Axia fan", plot_entity)
        assert result == ""

    def test_trigger_no_number(self, plot_entity):
        """'Apartment floor finishes' -> '' (no number after trigger)"""
        result = extract_plot("Apartment floor finishes", plot_entity)
        assert result == ""

    def test_no_plot_in_address(self, plot_entity):
        """'22-27 Hadlow Close 271115' -> '' (date purge + no plot trigger)"""
        result = extract_plot("22-27 Hadlow Close 271115", plot_entity)
        assert result == ""

    def test_abbreviation_expanded(self, plot_entity):
        """'Blk 2 Stonehaven' -> 'block 2'"""
        result = extract_plot("Blk 2 Stonehaven", plot_entity)
        assert result == "block 2"

    def test_phase_stops_at_dash_trigger(self, plot_entity):
        """'Phase 1 - Plots 174-180' -> 'phase 1'"""
        result = extract_plot("Phase 1 - Plots 174-180", plot_entity)
        assert result == "phase 1"

    def test_date_purge_with_block(self, plot_entity):
        """'Block G Gas meter boxes May 2012' -> 'block g'"""
        result = extract_plot("Block G Gas meter boxes May 2012", plot_entity)
        assert result == "block g"

    def test_empty_string(self, plot_entity):
        result = extract_plot("", plot_entity)
        assert result == ""

    def test_whitespace_only(self, plot_entity):
        result = extract_plot("   ", plot_entity)
        assert result == ""


# -----------------------------------------------------------------------
# Trigger set building
# -----------------------------------------------------------------------

class TestBuildTriggerSet:

    def test_includes_tokens(self, plot_entity):
        triggers = _build_trigger_set(plot_entity)
        assert "plot" in triggers
        assert "unit" in triggers
        assert "flat" in triggers

    def test_includes_abbreviations(self, plot_entity):
        triggers = _build_trigger_set(plot_entity)
        assert "plt" in triggers
        assert "blk" in triggers
        assert "ph" in triggers

    def test_includes_pattern_keywords(self, plot_entity):
        triggers = _build_trigger_set(plot_entity)
        # Derived from compiled patterns
        assert "block" in triggers
        assert "phase" in triggers
        assert "house" in triggers
        assert "lodge" in triggers
        assert "cottage" in triggers


# -----------------------------------------------------------------------
# Descending-number heuristic
# -----------------------------------------------------------------------

class TestDescendingHeuristic:

    def test_descending_trims(self):
        """66 > 42 → trim from dash onward."""
        captured = ["66", "-", "42"]
        result = _apply_descending_heuristic(captured)
        assert result == ["66"]

    def test_ascending_keeps(self):
        """1 < 5 → keep all."""
        captured = ["1", "-", "5"]
        result = _apply_descending_heuristic(captured)
        assert result == ["1", "-", "5"]

    def test_complex_descending(self):
        """76 & 77 - 28 → 77 > 28 → trim."""
        captured = ["76", "&", "77", "-", "28"]
        result = _apply_descending_heuristic(captured)
        assert result == ["76", "&", "77"]

    def test_no_dash(self):
        """No dash → unchanged."""
        captured = ["1", "&", "2"]
        result = _apply_descending_heuristic(captured)
        assert result == ["1", "&", "2"]

    def test_empty(self):
        result = _apply_descending_heuristic([])
        assert result == []

    def test_equal_numbers_kept(self):
        """5 - 5 → equal, not descending → keep."""
        captured = ["5", "-", "5"]
        result = _apply_descending_heuristic(captured)
        assert result == ["5", "-", "5"]


# -----------------------------------------------------------------------
# Pattern fallback
# -----------------------------------------------------------------------

class TestPatternFallback:

    def test_p_number(self, plot_entity):
        """P239 from 'P239-Lv0'."""
        result = _pattern_fallback("P239-Lv0", plot_entity)
        assert result == "p239"

    def test_no_match(self, plot_entity):
        """No pattern match."""
        result = _pattern_fallback("Finance Department", plot_entity)
        assert result == ""

    def test_blk_pattern(self, plot_entity):
        """'blk2' matched by pattern."""
        result = _pattern_fallback("blk2 something", plot_entity)
        assert result == "blk2"


# -----------------------------------------------------------------------
# Additional edge cases
# -----------------------------------------------------------------------

class TestPlotEdgeCases:

    def test_plot_with_six_digit_date(self, plot_entity):
        """'Plot 5 160219' — date stripped, just 'plot 5'."""
        result = extract_plot("Plot 5 160219", plot_entity)
        assert result == "plot 5"

    def test_plot_with_dotted_date(self, plot_entity):
        """'Unit 12 18.10.13' — date stripped."""
        result = extract_plot("Unit 12 18.10.13", plot_entity)
        assert result == "unit 12"

    def test_plural_plots(self, plot_entity):
        """'Plots 1-10' — plural trigger still matches."""
        result = extract_plot("Plots 1-10", plot_entity)
        assert result == "plots 1-10"

    def test_plural_blocks(self, plot_entity):
        """'Blocks 1 & 2' — plural trigger."""
        result = extract_plot("Blocks 1 & 2", plot_entity)
        assert result == "blocks 1 & 2"

    def test_dwelling_trigger(self, plot_entity):
        """'Dwelling 7 Elm Drive' -> 'dwelling 7'"""
        result = extract_plot("Dwelling 7 Elm Drive", plot_entity)
        assert result == "dwelling 7"

    def test_lot_trigger(self, plot_entity):
        """'Lot 42 Site Plan' -> 'lot 42'"""
        result = extract_plot("Lot 42 Site Plan", plot_entity)
        assert result == "lot 42"

    def test_no_abbreviation_map(self):
        """Without abbreviation map, abbreviations stay as-is."""
        entity = EntityDefinition(
            name="plot",
            tokens=["plot"],
            abbreviations=["plt"],
            compiled_patterns=[],
            abbreviation_map={},
        )
        result = extract_plot("Plt 5", entity)
        assert result == "plt 5"


# -----------------------------------------------------------------------
# Address entity fixture (for address-aware trimming tests)
# -----------------------------------------------------------------------

@pytest.fixture
def address_entity():
    """Minimal address entity definition for testing (mirrors housing.yaml)."""
    return EntityDefinition(
        name="address",
        tokens=[
            "road", "street", "lane", "avenue", "drive", "close",
            "court", "crescent", "way", "place", "terrace", "grove",
            "gardens", "park", "rise", "hill", "walk", "green",
            "meadow", "square", "mews", "row", "view", "mount",
            "heights", "vale", "chase", "wharf", "quay", "passage",
        ],
        abbreviations=[
            "rd", "st", "ln", "ave", "av", "dr", "cl", "ct",
            "cres", "gr", "pk", "sq", "mt", "pl", "gdns", "gdn", "tce",
        ],
        compiled_patterns=[
            re.compile(
                r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b",
                re.IGNORECASE,
            ),
        ],
        abbreviation_map={
            "rd": "road", "st": "street", "ln": "lane",
            "ave": "avenue", "av": "avenue", "dr": "drive",
            "cl": "close", "ct": "court", "cres": "crescent",
            "gr": "grove", "pk": "park", "sq": "square",
            "mt": "mount", "pl": "place", "gdns": "gardens",
            "gdn": "garden", "tce": "terrace",
        },
    )


# -----------------------------------------------------------------------
# Address-aware range trimming (full pipeline)
# -----------------------------------------------------------------------

class TestAddressAwareTrim:
    """Tests for address-aware false-range detection."""

    def test_standalone_dash_with_address(self, plot_entity, address_entity):
        """'Plot 10 - 12 Pippin Court' → 'plot 10' (12 is house number)."""
        result = extract_plot(
            "Plot 10 - 12 Pippin Court", plot_entity, address_entity,
        )
        assert result == "plot 10"

    def test_hyphenated_large_gap_with_address(
        self, plot_entity, address_entity,
    ):
        """'Plot 10-103 Manston Road' → 'plot 10' (gap 93 > 30)."""
        result = extract_plot(
            "Plot 10-103 Manston Road", plot_entity, address_entity,
        )
        assert result == "plot 10"

    def test_hyphenated_small_gap_with_address(
        self, plot_entity, address_entity,
    ):
        """'Plot 16-30 Colemans Close' → 'plot 16-30' (gap 14 ≤ 30)."""
        result = extract_plot(
            "Plot 16-30 Colemans Close ashford,", plot_entity, address_entity,
        )
        assert result == "plot 16-30"

    def test_standalone_dash_no_address(self, plot_entity, address_entity):
        """'Plot 10 - 12 Something' → 'plot 10 - 12' (no suffix)."""
        result = extract_plot(
            "Plot 10 - 12 Something", plot_entity, address_entity,
        )
        assert result == "plot 10 - 12"

    def test_standalone_dash_no_remaining(self, plot_entity, address_entity):
        """'Plot 1 - 5' → 'plot 1 - 5' (no remaining words)."""
        result = extract_plot("Plot 1 - 5", plot_entity, address_entity)
        assert result == "plot 1 - 5"

    def test_single_number_with_address(self, plot_entity, address_entity):
        """'Plot 1 West Road' → 'plot 1' (single number, no range)."""
        result = extract_plot(
            "Plot 1 West Road", plot_entity, address_entity,
        )
        assert result == "plot 1"

    def test_single_number_dash_address(self, plot_entity, address_entity):
        """'Plot 1 - West Road' → 'plot 1' (dash then address, no range)."""
        result = extract_plot(
            "Plot 1 - West Road", plot_entity, address_entity,
        )
        assert result == "plot 1"

    def test_edrm_manston(self, plot_entity, address_entity):
        """Real-world: EDRM- Plot 10-103 Manston Road."""
        result = extract_plot(
            "EDRM- Plot 10-103 Manston Road- Callie Rafferty & David Green",
            plot_entity,
            address_entity,
        )
        assert result == "plot 10"

    def test_edrm_pippin_court(self, plot_entity, address_entity):
        """Real-world: EDRM Plot 10 - 12 Pippin Court."""
        result = extract_plot(
            "EDRM Plot 10 - 12 Pippin Court - Emmanuel Nnaji NO SALES PROGRESSION CASE",
            plot_entity,
            address_entity,
        )
        assert result == "plot 10"

    def test_descending_still_works_with_address(
        self, plot_entity, address_entity,
    ):
        """'Plot 66 - 42 Hawkes Way' → 'plot 66' (descending wins first)."""
        result = extract_plot(
            "Plot 66 - 42 Hawkes Way", plot_entity, address_entity,
        )
        assert result == "plot 66"

    def test_no_address_entity_backwards_compat(self, plot_entity):
        """Without address entity, ascending range with address kept."""
        result = extract_plot("Plot 10 - 12 Pippin Court", plot_entity)
        assert result == "plot 10 - 12"


# -----------------------------------------------------------------------
# _apply_address_aware_trim unit tests
# -----------------------------------------------------------------------

class TestApplyAddressAwareTrim:

    @pytest.fixture
    def tokens(self):
        return {"road", "street", "court", "close", "way", "lane"}

    def test_standalone_dash_trim(self, tokens):
        captured = ["10", "-", "12"]
        remaining = ["Pippin", "Court"]
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["10"]

    def test_hyphenated_large_gap(self, tokens):
        captured = ["10-103"]
        remaining = ["Manston", "Road"]
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["10"]

    def test_hyphenated_small_gap_kept(self, tokens):
        captured = ["16-30"]
        remaining = ["Colemans", "Close"]
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["16-30"]

    def test_no_address_suffix(self, tokens):
        captured = ["10", "-", "12"]
        remaining = ["Something", "Else"]
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["10", "-", "12"]

    def test_no_remaining(self, tokens):
        captured = ["1", "-", "5"]
        remaining = []
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["1", "-", "5"]

    def test_single_number_no_change(self, tokens):
        """Single number with address — no dash to trim."""
        captured = ["1"]
        remaining = ["West", "Road"]
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["1"]

    def test_address_suffix_beyond_window(self, tokens):
        """Address suffix beyond 3-word window — no trim."""
        captured = ["10", "-", "12"]
        remaining = ["Alpha", "Beta", "Gamma", "Road"]
        result = _apply_address_aware_trim(captured, remaining, tokens)
        assert result == ["10", "-", "12"]
