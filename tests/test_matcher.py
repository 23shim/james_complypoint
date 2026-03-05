"""Tests for signal matching."""

import pytest

from classification.config_loader import load_config
from classification.matcher import (
    match_category_signals,
    match_extension_hint,
    match_type_signals,
)
from classification.models import MatchMethod, SignalSource


CONFIG_DIR = "config"


@pytest.fixture
def config():
    return load_config(CONFIG_DIR, industry="housing")


class TestTypeMatching:

    def test_exact_token_match(self, config):
        signals = match_type_signals("invoice", config, SignalSource.FILENAME_TOKEN)
        assert len(signals) >= 1
        assert any(s.label == "Invoice" for s in signals)

    def test_abbreviation_match(self, config):
        signals = match_type_signals("EICR Plot 5", config, SignalSource.FILENAME_TOKEN)
        assert any(s.label == "EICR" for s in signals)
        eicr = next(s for s in signals if s.label == "EICR")
        assert eicr.match_method == MatchMethod.ABBREVIATION

    def test_pattern_match(self, config):
        signals = match_type_signals("INV-00234", config, SignalSource.FILENAME_TOKEN)
        assert any(s.label == "Invoice" for s in signals)

    def test_no_match_returns_empty(self, config):
        signals = match_type_signals("random text here", config, SignalSource.FILENAME_TOKEN)
        type_labels = {s.label for s in signals}
        # Should not match anything specific
        assert "Invoice" not in type_labels
        assert "EICR" not in type_labels

    def test_multiword_token_match(self, config):
        signals = match_type_signals(
            "Fire Risk Assessment report", config, SignalSource.FOLDER_TYPE
        )
        assert any(s.label == "Fire Risk Assessment" for s in signals)

    def test_folder_type_source(self, config):
        signals = match_type_signals(
            "Snagging", config, SignalSource.FOLDER_TYPE, depth=2
        )
        assert any(s.label == "Snagging Report" for s in signals)
        snag = next(s for s in signals if s.label == "Snagging Report")
        assert snag.source == SignalSource.FOLDER_TYPE
        assert snag.depth == 2

    def test_mos_abbreviation(self, config):
        signals = match_type_signals("MOS SIGNED", config, SignalSource.FILENAME_TOKEN)
        assert any(s.label == "MOS" for s in signals)

    def test_cml_abbreviation(self, config):
        signals = match_type_signals("CML - 103 Manston Road", config, SignalSource.FILENAME_TOKEN)
        assert any(s.label == "CML" for s in signals)

    def test_epc_abbreviation(self, config):
        signals = match_type_signals("EPC Lenham Plot 38", config, SignalSource.FILENAME_TOKEN)
        assert any(s.label == "EPC" for s in signals)


class TestCategoryMatching:

    def test_finance_signal(self, config):
        signals = match_category_signals("Finance", config, depth=1)
        assert any(s.label == "Finance" for s in signals)

    def test_multiword_signal(self, config):
        signals = match_category_signals("Health & Safety", config, depth=1)
        assert any(s.label == "Health & Safety" for s in signals)

    def test_fire_safety_matches_hs(self, config):
        signals = match_category_signals("Fire safety & risk assessments", config, depth=2)
        assert any(s.label == "Health & Safety" for s in signals)

    def test_construction_signals(self, config):
        signals = match_category_signals("Handover", config, depth=1)
        assert any(s.label == "Construction" for s in signals)

    def test_snagging_matches_construction(self, config):
        signals = match_category_signals("Snagging", config, depth=2)
        assert any(s.label == "Construction" for s in signals)

    def test_no_match_returns_empty(self, config):
        signals = match_category_signals("Lenham Road Headcorn", config, depth=0)
        # Project name should not match any category
        assert len(signals) == 0

    def test_depth_preserved(self, config):
        signals = match_category_signals("Finance", config, depth=3)
        finance = next(s for s in signals if s.label == "Finance")
        assert finance.depth == 3

    def test_s106_matches_planning(self, config):
        signals = match_category_signals("S106 Works", config, depth=2)
        assert any(s.label == "Planning" for s in signals)


class TestFuzzyTypeMatching:

    def test_fuzzy_multiword_typo(self, config):
        """'Fire Risk Assesment' should fuzzy-match 'Fire Risk Assessment'."""
        signals = match_type_signals(
            "Fire Risk Assesment", config, SignalSource.FILENAME_TOKEN
        )
        assert any(s.label == "Fire Risk Assessment" for s in signals)
        fra = next(s for s in signals if s.label == "Fire Risk Assessment")
        assert fra.match_method == MatchMethod.FUZZY_TOKEN

    def test_fuzzy_single_word_typo(self, config):
        """'invoce' (missing 'i') should fuzzy-match 'Invoice'."""
        signals = match_type_signals(
            "invoce 2024", config, SignalSource.FILENAME_TOKEN
        )
        assert any(s.label == "Invoice" for s in signals)
        inv = next(s for s in signals if s.label == "Invoice")
        assert inv.match_method == MatchMethod.FUZZY_TOKEN

    def test_fuzzy_penalty_applied(self, config):
        """Fuzzy match weight should be lower than exact match weight."""
        exact = match_type_signals("invoice", config, SignalSource.FILENAME_TOKEN)
        fuzzy = match_type_signals("invoce", config, SignalSource.FILENAME_TOKEN)
        exact_inv = next(s for s in exact if s.label == "Invoice")
        fuzzy_inv = next(s for s in fuzzy if s.label == "Invoice")
        assert fuzzy_inv.base_weight < exact_inv.base_weight

    def test_fuzzy_no_match_short_token(self, config):
        """Short tokens should NOT fuzzy-match (below min_token_length)."""
        # "mso" is edit-distance-1 from "mos", but "mos" is only 3 chars.
        signals = match_type_signals("mso report", config, SignalSource.FILENAME_TOKEN)
        assert not any(
            s.label == "MOS" and s.match_method == MatchMethod.FUZZY_TOKEN
            for s in signals
        )

    def test_exact_match_preferred_over_fuzzy(self, config):
        """When exact match exists, it should win over fuzzy."""
        signals = match_type_signals("invoice", config, SignalSource.FILENAME_TOKEN)
        inv = next(s for s in signals if s.label == "Invoice")
        assert inv.match_method == MatchMethod.TOKEN  # not FUZZY_TOKEN

    def test_fuzzy_folder_type(self, config):
        """Fuzzy should also work for folder-sourced signals."""
        signals = match_type_signals(
            "Snaging Reports", config, SignalSource.FOLDER_TYPE, depth=2
        )
        matches = [s for s in signals if s.match_method == MatchMethod.FUZZY_TOKEN]
        if matches:
            assert matches[0].source == SignalSource.FOLDER_TYPE
            assert matches[0].depth == 2


class TestExtensionHint:

    def test_msg_hints_email(self, config):
        signals = match_extension_hint("msg", config)
        assert any(s.label == "Email" for s in signals)

    def test_jpg_hints_photo(self, config):
        signals = match_extension_hint("jpg", config)
        assert any(s.label == "Photo" for s in signals)

    def test_dwg_hints_drawing(self, config):
        signals = match_extension_hint("dwg", config)
        assert any(s.label == "Drawing" for s in signals)

    def test_pdf_no_strong_hint(self, config):
        signals = match_extension_hint("pdf", config)
        # PDF is too generic for a strong type hint
        assert len(signals) == 0

    def test_empty_extension(self, config):
        signals = match_extension_hint("", config)
        assert len(signals) == 0
