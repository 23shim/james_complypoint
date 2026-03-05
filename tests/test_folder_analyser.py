"""Tests for folder path analysis (Tier 1)."""

import pytest

from classification.config_loader import load_config
from classification.folder_analyser import analyse_folders
from classification.models import SignalSource


CONFIG_DIR = "config"


@pytest.fixture
def config():
    return load_config(CONFIG_DIR, industry="housing")


class TestFolderAnalysis:

    def test_clear_category(self, config):
        signals = analyse_folders(["Project A", "Finance", "Invoices"], config)
        cat_signals = [s for s in signals if s.source == SignalSource.FOLDER_CATEGORY]
        assert any(s.label == "Finance" for s in cat_signals)

    def test_clear_type(self, config):
        signals = analyse_folders(["Project A", "Finance", "Invoices"], config)
        type_signals = [s for s in signals if s.source == SignalSource.FOLDER_TYPE]
        assert any(s.label == "Invoice" for s in type_signals)

    def test_no_signals_for_project_name(self, config):
        signals = analyse_folders(["Lenham Road Headcorn"], config)
        type_cat = [
            s for s in signals
            if s.source in (SignalSource.FOLDER_TYPE, SignalSource.FOLDER_CATEGORY)
        ]
        assert len(type_cat) == 0

    def test_empty_segments(self, config):
        signals = analyse_folders([], config)
        assert len(signals) == 0

    def test_depth_preserved(self, config):
        signals = analyse_folders(["Root", "Finance"], config)
        finance = [s for s in signals if s.label == "Finance"]
        assert len(finance) == 1
        assert finance[0].depth == 1


class TestRealTechServePatterns:
    """Tests based on actual TechServe data patterns."""

    def test_handover_fire_safety(self, config):
        segments = ["Lenham Road Headcorn", "Handover", "Fire safety & risk assessments"]
        signals = analyse_folders(segments, config)

        # Should find Construction category from "Handover"
        cat_signals = [s for s in signals if s.source == SignalSource.FOLDER_CATEGORY]
        assert any(s.label == "Construction" for s in cat_signals)

        # Should find H&S from "Fire safety & risk assessments"
        assert any(s.label == "Health & Safety" for s in cat_signals)

        # H&S should be deeper than Construction
        construction = next(s for s in cat_signals if s.label == "Construction")
        hs = next(s for s in cat_signals if s.label == "Health & Safety")
        assert hs.depth > construction.depth

    def test_handover_snagging(self, config):
        segments = ["Lenham Road Headcorn", "Handover", "Snagging"]
        signals = analyse_folders(segments, config)

        type_signals = [s for s in signals if s.source == SignalSource.FOLDER_TYPE]
        assert any(s.label == "Snagging Report" for s in type_signals)

    def test_drawings_folder(self, config):
        segments = ["Lenham Road Headcorn", "Drawings"]
        signals = analyse_folders(segments, config)

        type_signals = [s for s in signals if s.source == SignalSource.FOLDER_TYPE]
        assert any(s.label == "Drawing" for s in type_signals)

    def test_handover_type_signal(self, config):
        segments = ["Project A", "Handover"]
        signals = analyse_folders(segments, config)

        type_signals = [s for s in signals if s.source == SignalSource.FOLDER_TYPE]
        assert any(s.label == "Handover Pack" for s in type_signals)


class TestRealHomeBuyPatterns:
    """Tests based on actual HomeBuy data patterns."""

    def test_resales_sales(self, config):
        segments = ["HOMEBUY", "RESALES"]
        signals = analyse_folders(segments, config)

        cat_signals = [s for s in signals if s.source == SignalSource.FOLDER_CATEGORY]
        assert any(s.label == "Sales" for s in cat_signals)
