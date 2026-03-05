"""Tests for confidence scoring and conflict resolution."""

import pytest

from classification.config_loader import load_config
from classification.models import MatchMethod, Signal, SignalSource
from classification.scorer import score


CONFIG_DIR = "config"


@pytest.fixture
def weights():
    config = load_config(CONFIG_DIR, industry="housing")
    return config.weights


def _make_signal(
    source=SignalSource.FILENAME_TOKEN,
    label="Invoice",
    match_term="invoice",
    method=MatchMethod.TOKEN,
    weight=0.70,
    depth=-1,
    text="Invoice 001",
):
    return Signal(
        source=source,
        label=label,
        match_term=match_term,
        match_method=method,
        base_weight=weight,
        depth=depth,
        text=text,
    )


class TestTypeResolution:

    def test_filename_type_is_primary(self, weights):
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Drawing", "drawing", depth=1, weight=0.55, text="Drawings"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Invoice"

    def test_folder_type_when_no_filename(self, weights):
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Drawing", "drawing", depth=1, weight=0.50, text="Drawings"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Drawing"
        # Folder-only: base weight used directly (no secondary discount)
        assert result.type_confidence == 0.50

    def test_no_signals_returns_unknown(self, weights):
        result = score([], weights)
        assert result.inferred_type == "Unknown"
        assert result.type_confidence == 0.0
        assert result.overall_confidence == 0.0

    def test_extension_hint_only(self, weights):
        signals = [
            _make_signal(SignalSource.EXTENSION_HINT, "Email", ".msg", weight=0.15, text="msg"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Email"
        assert result.type_confidence == 0.15


class TestCategoryResolution:

    def test_deepest_category_wins(self, weights):
        signals = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "Construction", "handover", depth=1, weight=0.30, text="Handover"),
            _make_signal(SignalSource.FOLDER_CATEGORY, "Health & Safety", "fire safety", depth=2, weight=0.30, text="Fire safety"),
        ]
        result = score(signals, weights)
        assert result.inferred_category == "Health & Safety"

    def test_single_category(self, weights):
        signals = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "Finance", "finance", depth=1, weight=0.30, text="Finance"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.inferred_category == "Finance"

    def test_no_category_returns_unknown(self, weights):
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.inferred_category == "Unknown"


class TestReinforcement:

    def test_folder_filename_agreement_boosts(self, weights):
        # Both folder and filename say Invoice
        signals_agree = [
            _make_signal(SignalSource.FOLDER_TYPE, "Invoice", "invoices", depth=2, weight=0.55, text="Invoices"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        signals_no_folder = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result_agree = score(signals_agree, weights)
        result_no_folder = score(signals_no_folder, weights)
        # Agreement should produce higher confidence
        assert result_agree.type_confidence > result_no_folder.type_confidence

    def test_agreement_boost_in_reasoning(self, weights):
        # Folder and filename agree on type
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Invoice", "invoices", depth=1, weight=0.50, text="Invoices"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 001"),
        ]
        result = score(signals, weights)
        # Should include agreement boost in reasoning
        assert any("agree" in r.lower() for r in result.reasoning_trace)


class TestCategoryDomainReinforcement:

    def test_matching_domain_boosts(self, weights):
        """MOS (domain=Sales) in Sales folder should boost."""
        type_domains = {"MOS": "Sales"}
        signals = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "Sales", "sales", depth=1, weight=0.30, text="Sales"),
            _make_signal(SignalSource.FILENAME_ABBREVIATION, "MOS", "mos", weight=0.50, text="MOS SIGNED"),
        ]
        result_with = score(signals, weights, type_domains=type_domains)
        result_without = score(signals, weights, type_domains={})
        assert result_with.type_confidence > result_without.type_confidence

    def test_mismatching_domain_no_penalty(self, weights):
        """MOS (domain=Sales) in Construction folder — no penalty (mis-filed doc is still that doc)."""
        type_domains = {"MOS": "Sales"}
        signals = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "Construction", "construction", depth=1, weight=0.50, text="Construction"),
            _make_signal(SignalSource.FILENAME_ABBREVIATION, "MOS", "mos", weight=0.65, text="MOS SIGNED"),
        ]
        result_with = score(signals, weights, type_domains=type_domains)
        result_without = score(signals, weights, type_domains={})
        assert result_with.type_confidence == result_without.type_confidence

    def test_universal_domain_no_modifier(self, weights):
        """Invoice (domain=universal) in any category has no modifier."""
        type_domains = {"Invoice": "universal"}
        signals = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "HR", "hr", depth=1, weight=0.30, text="HR"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result_with = score(signals, weights, type_domains=type_domains)
        result_without = score(signals, weights, type_domains={})
        assert result_with.type_confidence == result_without.type_confidence


class TestConflict:

    def test_folder_filename_disagreement_penalises(self, weights):
        # Folder says Drawing, filename says Invoice — true conflict (not parent-child)
        signals_conflict = [
            _make_signal(SignalSource.FOLDER_TYPE, "Drawing", "drawings", depth=1, weight=0.55, text="Drawings"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        signals_clean = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result_conflict = score(signals_conflict, weights)
        result_clean = score(signals_clean, weights)
        # Conflict should produce lower or equal confidence
        assert result_conflict.type_confidence <= result_clean.type_confidence


class TestParentChild:

    def test_handover_pack_containing_certificate_boosts(self, weights):
        """Certificate inside Handover Pack folder should boost, not penalise."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Handover Pack", "handover", depth=1, weight=0.55, text="22 Handover to Defect Period"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Certificate", "certificate", weight=0.70, text="Gas Safety Certificate"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Certificate"
        # Should be boosted, not penalised — confidence must be above base
        assert result.type_confidence > 0.70
        assert any("child" in r for r in result.reasoning_trace)

    def test_drawing_folder_containing_plan_boosts(self, weights):
        """Plan inside Drawing folder should boost, not penalise."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Drawing", "drawings", depth=1, weight=0.55, text="16 Drawings"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Plan", "plan", weight=0.70, text="Site Layout Plan"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Plan"
        assert result.type_confidence > 0.70
        assert any("child" in r for r in result.reasoning_trace)

    def test_report_folder_containing_site_inspection_boosts(self, weights):
        """Site Inspection inside Report folder should boost."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Report", "reports", depth=1, weight=0.55, text="14 Meeting and Reports"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Site Inspection", "site inspection", weight=0.70, text="WKHA Site Inspection Report"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Site Inspection"
        assert result.type_confidence > 0.70

    def test_true_conflict_still_penalises(self, weights):
        """Non parent-child mismatch should still penalise."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Handover Pack", "handover", depth=1, weight=0.55, text="Handover"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Invoice"
        # Invoice is NOT a child of Handover Pack — should be penalised
        assert result.type_confidence < 0.70
        assert any("penalty" in r for r in result.reasoning_trace)

    def test_handover_containing_epc_boosts(self, weights):
        """EPC inside Handover Pack folder is a valid child."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Handover Pack", "handover", depth=1, weight=0.55, text="Handover"),
            _make_signal(SignalSource.FILENAME_ABBREVIATION, "EPC", "epc", weight=0.50, text="EPC Certificate"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "EPC"
        assert result.type_confidence > 0.50
        assert any("child" in r for r in result.reasoning_trace)

    def test_practical_completion_containing_certificate_boosts(self, weights):
        """Certificate inside Practical Completion folder is valid."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Practical Completion", "practical completion", depth=1, weight=0.55, text="Practical Completion"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Certificate", "cert", weight=0.70, text="PC Certificate"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Certificate"
        assert result.type_confidence > 0.70


class TestConfidenceBands:

    def test_high_band_is_ready(self, weights):
        """Strong signals → High band → Ready for Phase 2."""
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Invoice", "invoices", depth=2, weight=0.50, text="Invoices"),
            _make_signal(SignalSource.FOLDER_CATEGORY, "Finance", "finance", depth=1, weight=0.50, text="Finance"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 2024-001"),
        ]
        result = score(signals, weights)
        assert result.confidence_band == "High"
        assert result.readiness_status == "Ready"
        assert result.overall_confidence >= 0.60

    def test_filename_token_alone_is_ready(self, weights):
        """A filename token match alone should be High/Ready — it's strong evidence."""
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 2024-001"),
        ]
        result = score(signals, weights)
        assert result.confidence_band == "High"
        assert result.readiness_status == "Ready"
        assert result.overall_confidence >= 0.60

    def test_medium_band_review(self, weights):
        """Moderate signals → Medium band → Review."""
        # Folder type only (0.40 type), no filename confirmation
        # overall = 0.40 → Medium (0.35 ≤ 0.40 < 0.50)
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Drawing", "drawing", depth=1, weight=0.40, text="Drawings"),
            _make_signal(SignalSource.FOLDER_CATEGORY, "Construction", "construction", depth=0, weight=0.50, text="Construction"),
        ]
        result = score(signals, weights)
        assert result.confidence_band == "Medium"
        assert result.readiness_status == "Review"
        assert 0.35 <= result.overall_confidence < 0.50

    def test_low_band_not_ready(self, weights):
        """Weak signals → Low band → Not Ready."""
        signals = [
            _make_signal(SignalSource.EXTENSION_HINT, "Email", ".msg", weight=0.15, text="msg"),
        ]
        result = score(signals, weights)
        assert result.confidence_band == "Low"
        assert result.readiness_status == "Not Ready"

    def test_unknown_type_is_low_band(self, weights):
        """No signals → Unknown type → Low band."""
        result = score([], weights)
        assert result.confidence_band == "Low"
        assert result.readiness_status == "Not Ready"

    def test_band_in_reasoning_trace(self, weights):
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert any("band=" in r for r in result.reasoning_trace)


class TestReasoningTrace:

    def test_has_final_summary(self, weights):
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert any("final:" in r for r in result.reasoning_trace)

    def test_empty_signals_still_has_trace(self, weights):
        result = score([], weights)
        assert len(result.reasoning_trace) > 0

    def test_confidence_clamped(self, weights):
        # Very strong signals shouldn't exceed 1.0
        signals = [
            _make_signal(SignalSource.FOLDER_TYPE, "Invoice", "invoices", depth=1, weight=0.90, text="Invoices"),
            _make_signal(SignalSource.FOLDER_TYPE, "Invoice", "invoice", depth=2, weight=0.90, text="Invoice"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.90, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.type_confidence <= 1.0
        assert result.overall_confidence <= 1.0


class TestEntityResolution:
    """Entity resolution uses deepest-wins, same as category resolution."""

    def test_deepest_entity_wins(self, weights):
        """When two segments match the same entity type, deepest wins."""
        signals = [
            _make_signal(
                SignalSource.FOLDER_ENTITY, "plot", "plot", depth=1,
                weight=0.80, text="Phase 1",
            ),
            _make_signal(
                SignalSource.FOLDER_ENTITY, "plot", "plot", depth=2,
                weight=0.80, text="Plot 12",
            ),
        ]
        result = score(signals, weights)
        assert result.entities.get("plot") == "Plot 12"
        assert result.entity_depths.get("plot") == 2

    def test_multiple_entity_types(self, weights):
        """Different entity types resolved independently."""
        signals = [
            _make_signal(
                SignalSource.FOLDER_ENTITY, "plot", "plot", depth=2,
                weight=0.80, text="Plot 12",
            ),
            _make_signal(
                SignalSource.FOLDER_ENTITY, "address", "road", depth=1,
                weight=0.80, text="Lenham Road",
            ),
        ]
        result = score(signals, weights)
        assert result.entities.get("plot") == "Plot 12"
        assert result.entities.get("address") == "Lenham Road"
        assert result.entity_depths.get("plot") == 2
        assert result.entity_depths.get("address") == 1

    def test_no_entity_signals(self, weights):
        """No entity signals → empty entity dicts."""
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice",
                         weight=0.70, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.entities == {}
        assert result.entity_confidences == {}
        assert result.entity_depths == {}

    def test_entity_confidence_from_match_method(self, weights):
        """Entity confidence reflects the match method weight."""
        signals = [
            _make_signal(
                SignalSource.FOLDER_ENTITY, "plot", "plot", depth=1,
                weight=0.65, text="Phase 4",
                method=MatchMethod.PATTERN,
            ),
        ]
        result = score(signals, weights)
        assert result.entity_confidences.get("plot") == 0.65

    def test_entity_in_reasoning_trace(self, weights):
        """Entity resolution appears in reasoning trace."""
        signals = [
            _make_signal(
                SignalSource.FOLDER_ENTITY, "plot", "plot", depth=1,
                weight=0.80, text="Plot 12",
            ),
        ]
        result = score(signals, weights)
        assert any("entity:" in r for r in result.reasoning_trace)


class TestAmbiguityDetection:

    def test_close_secondary_type_flagged(self, weights):
        """When runner-up type is close to primary, flag ambiguity."""
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.70, text="Invoice 001"),
            _make_signal(SignalSource.FOLDER_TYPE, "Valuation", "valuation", depth=1, weight=0.65, text="Valuations"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Invoice"
        assert result.secondary_type == "Valuation"
        assert any("ambiguous" in r for r in result.reasoning_trace)

    def test_distant_secondary_type_not_flagged(self, weights):
        """When runner-up type is far from primary, no ambiguity flag."""
        signals = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 001"),
            _make_signal(SignalSource.EXTENSION_HINT, "Email", ".msg", weight=0.20, text="msg"),
        ]
        result = score(signals, weights)
        assert result.inferred_type == "Invoice"
        assert not any("ambiguous" in r for r in result.reasoning_trace)


class TestOverallConfidenceSimplified:

    def test_overall_equals_type_confidence(self, weights):
        """Overall confidence is now just type confidence (category is informational)."""
        signals = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "Finance", "finance", depth=1, weight=0.50, text="Finance"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 001"),
        ]
        result = score(signals, weights)
        assert result.overall_confidence == result.type_confidence

    def test_category_doesnt_affect_readiness(self, weights):
        """Category absence should not prevent a strong filename match from being Ready."""
        signals_no_cat = [
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 001"),
        ]
        signals_with_cat = [
            _make_signal(SignalSource.FOLDER_CATEGORY, "Finance", "finance", depth=1, weight=0.50, text="Finance"),
            _make_signal(SignalSource.FILENAME_TOKEN, "Invoice", "invoice", weight=0.85, text="Invoice 001"),
        ]
        result_no_cat = score(signals_no_cat, weights)
        result_with_cat = score(signals_with_cat, weights)
        # Both should be High/Ready — category doesn't gatekeep
        assert result_no_cat.readiness_status == "Ready"
        assert result_with_cat.readiness_status == "Ready"
        # Same overall confidence (category is informational only)
        assert result_no_cat.overall_confidence == result_with_cat.overall_confidence
