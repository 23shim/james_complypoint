"""Tests for filename analysis (Tier 2)."""

import pytest

from classification.config_loader import load_config
from classification.filename_analyser import analyse_filename
from classification.models import SignalSource


CONFIG_DIR = "config"


@pytest.fixture
def config():
    return load_config(CONFIG_DIR, industry="housing")


class TestFilenameAnalysis:

    def test_clear_type_match(self, config):
        signals = analyse_filename("Invoice 2024-001", "pdf", config)
        type_signals = [s for s in signals if s.source != SignalSource.EXTENSION_HINT]
        assert any(s.label == "Invoice" for s in type_signals)

    def test_abbreviation_match(self, config):
        signals = analyse_filename("EICR Plot 5", "pdf", config)
        type_signals = [s for s in signals if s.source != SignalSource.EXTENSION_HINT]
        assert any(s.label == "EICR" for s in type_signals)

    def test_extension_hint_msg(self, config):
        signals = analyse_filename("some email subject", "msg", config)
        ext_signals = [s for s in signals if s.source == SignalSource.EXTENSION_HINT]
        assert any(s.label == "Email" for s in ext_signals)

    def test_extension_hint_dwg(self, config):
        signals = analyse_filename("floor plan", "dwg", config)
        ext_signals = [s for s in signals if s.source == SignalSource.EXTENSION_HINT]
        assert any(s.label == "Drawing" for s in ext_signals)

    def test_no_signals_generic_filename(self, config):
        signals = analyse_filename("document_001", "pdf", config)
        type_signals = [s for s in signals if s.source != SignalSource.EXTENSION_HINT]
        # Generic name should produce few or no type matches
        assert not any(s.label == "Invoice" for s in type_signals)
        assert not any(s.label == "EICR" for s in type_signals)

    def test_empty_filename(self, config):
        signals = analyse_filename("", "pdf", config)
        # Only extension hint possible
        type_signals = [s for s in signals if s.source != SignalSource.EXTENSION_HINT]
        assert len(type_signals) == 0


class TestRealFilenamePatterns:
    """Tests with real filename patterns from the data."""

    def test_mos_signed(self, config):
        signals = analyse_filename("MOS SIGNED 25% - 103 Manston Road", "pdf", config)
        assert any(s.label == "MOS" for s in signals)

    def test_cml_draft(self, config):
        signals = analyse_filename("CML Draft - 103 Manston Road", "pdf", config)
        assert any(s.label == "CML" for s in signals)

    def test_epc_certificate(self, config):
        signals = analyse_filename("3 Garden Drive_ EPC", "pdf", config)
        assert any(s.label == "EPC" for s in signals)

    def test_conveyance_plan(self, config):
        signals = analyse_filename("Individual Conveyance Plan Plot 151", "pdf", config)
        assert any(s.label == "Conveyance" for s in signals)

    def test_lease_document(self, config):
        signals = analyse_filename("Signed lease 103 Manston Gardens", "pdf", config)
        assert any(s.label == "Lease" for s in signals)

    def test_offer_letter(self, config):
        signals = analyse_filename("Offer Letter - David Callie", "pdf", config)
        assert any(s.label == "Offer" for s in signals)

    def test_valuation_report(self, config):
        signals = analyse_filename("Mortgage Valuation Report", "pdf", config)
        assert any(s.label == "Valuation" for s in signals)

    def test_fire_risk_assessment(self, config):
        signals = analyse_filename("FRA Block A 2023", "pdf", config)
        assert any(s.label == "Fire Risk Assessment" for s in signals)

    def test_nhbc(self, config):
        signals = analyse_filename("NHBC Buildmark Certificate", "pdf", config)
        assert any(s.label == "NHBC Certificate" for s in signals)

    def test_photo_by_extension(self, config):
        signals = analyse_filename("DSC00234", "jpg", config)
        ext_signals = [s for s in signals if s.source == SignalSource.EXTENSION_HINT]
        assert any(s.label == "Photo" for s in ext_signals)

    def test_email_by_extension(self, config):
        signals = analyse_filename("RE Meeting Notes", "msg", config)
        ext_signals = [s for s in signals if s.source == SignalSource.EXTENSION_HINT]
        assert any(s.label == "Email" for s in ext_signals)
