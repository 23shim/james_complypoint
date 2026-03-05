"""Tests for entity extraction from folder segments."""

import pytest

from classification.config_loader import load_config
from classification.entity_extractor import (
    extract_entities,
    extract_entities_with_confidence,
    find_all_entity_matches,
)


CONFIG_DIR = "config"


@pytest.fixture
def config():
    return load_config(CONFIG_DIR, industry="housing")


class TestEntityDefinitionsLoaded:

    def test_plot_entity_defined(self, config):
        assert "plot" in config.entities

    def test_address_entity_defined(self, config):
        assert "address" in config.entities

    def test_plot_tokens_lowercase(self, config):
        for token in config.entities["plot"].tokens:
            assert token == token.lower()

    def test_address_tokens_lowercase(self, config):
        for token in config.entities["address"].tokens:
            assert token == token.lower()

    def test_plot_patterns_compiled(self, config):
        assert len(config.entities["plot"].compiled_patterns) > 0

    def test_no_entities_for_nonexistent_industry(self):
        config = load_config(CONFIG_DIR, industry="nonexistent")
        assert len(config.entities) == 0


class TestPlotExtraction:

    def test_plot_token_match(self, config):
        segments = ["Lenham Road", "Plot 12", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Plot 12"

    def test_plot_pattern_P_number(self, config):
        segments = ["Development", "P10", "Invoices"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "P10"

    def test_plot_pattern_with_hyphen(self, config):
        segments = ["Development", "Plot-12", "Invoices"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Plot-12"

    def test_unit_match(self, config):
        segments = ["Woodberry", "Unit 5A", "Legal"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Unit 5A"

    def test_flat_match(self, config):
        segments = ["Building", "Flat 3", "Certificates"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Flat 3"

    def test_block_match(self, config):
        segments = ["Development", "Block A", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Block A"

    def test_phase_match(self, config):
        segments = ["Lenham Road", "Phase 2", "Construction"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Phase 2"

    def test_abbreviation_plt(self, config):
        segments = ["Development", "Plt 5", "Reports"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Plt 5"

    def test_abbreviation_apt(self, config):
        segments = ["Building", "Apt 12", "Certificates"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Apt 12"

    def test_deepest_plot_wins(self, config):
        segments = ["Phase 1", "Plot 5", "Invoices"]
        result = extract_entities(segments, config)
        # Plot 5 is deeper than Phase 1
        assert result.get("plot") == "Plot 5"

    def test_no_plot_in_unrelated_folders(self, config):
        segments = ["Finance", "Invoices", "2023"]
        result = extract_entities(segments, config)
        assert "plot" not in result


class TestAddressExtraction:

    def test_road_suffix(self, config):
        segments = ["Lenham Road", "Plot 12", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Lenham Road"

    def test_street_suffix(self, config):
        segments = ["High Street", "Unit 5", "Legal"]
        result = extract_entities(segments, config)
        assert result.get("address") == "High Street"

    def test_lane_suffix(self, config):
        segments = ["Green Lane", "Plot 1", "Certificates"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Green Lane"

    def test_avenue_suffix(self, config):
        segments = ["Park Avenue", "Block A", "Reports"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Park Avenue"

    def test_close_suffix(self, config):
        segments = ["Willow Close", "Plot 3", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Willow Close"

    def test_crescent_suffix(self, config):
        segments = ["Oak Crescent", "Plot 7", "Construction"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Oak Crescent"

    def test_drive_suffix(self, config):
        segments = ["Elmwood Drive", "Unit 2", "Legal"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Elmwood Drive"

    def test_court_suffix(self, config):
        segments = ["Windsor Court", "Flat 8", "Certificates"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Windsor Court"

    def test_mews_suffix(self, config):
        segments = ["Stable Mews", "Unit 1", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Stable Mews"

    def test_abbreviation_rd(self, config):
        segments = ["Manston Rd", "Plot 5", "Invoices"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Manston Rd"

    def test_abbreviation_ave(self, config):
        segments = ["Park Ave", "Block B", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Park Ave"

    def test_deepest_address_wins(self, config):
        segments = ["Green Lane", "Elm Road", "Invoices"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Elm Road"

    def test_no_address_in_unrelated_folders(self, config):
        segments = ["Finance", "Invoices", "2023"]
        result = extract_entities(segments, config)
        assert "address" not in result


class TestMultipleEntities:

    def test_both_plot_and_address_extracted(self, config):
        segments = ["Lenham Road", "Plot 12", "Finance", "Invoices"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Lenham Road"
        assert result.get("plot") == "Plot 12"

    def test_address_and_unit(self, config):
        segments = ["High Street", "Block A", "Unit 5", "Certificates"]
        result = extract_entities(segments, config)
        assert result.get("address") == "High Street"
        assert result.get("plot") == "Unit 5"


class TestTypeCategoryFiltering:
    """Segments matching document types or categories are skipped."""

    def test_site_inspection_skipped_for_address(self, config):
        # "site inspection" is a document type — should not be an address
        # even though the deeper segment contains "road"
        segments = [
            "Small Hythe Road Tenterden",
            "Site inspection photos",
            "Small Hythe Road site inspection photos 250319",
        ]
        result = extract_entities(segments, config)
        assert result.get("address") == "Small Hythe Road Tenterden"

    def test_snagging_folder_skipped(self, config):
        segments = ["Elm Road", "Snagging", "Plot 12"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Elm Road"

    def test_handover_folder_skipped(self, config):
        segments = ["Oak Lane", "Handover", "Unit 5"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Oak Lane"

    def test_finance_category_skipped(self, config):
        segments = ["Maple Drive", "Phase 2", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Maple Drive"
        assert result.get("plot") == "Phase 2"

    def test_construction_category_skipped(self, config):
        segments = ["Birch Close", "Block A", "Construction"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Birch Close"
        assert result.get("plot") == "Block A"

    def test_fire_risk_assessment_skipped(self, config):
        segments = ["Beech Avenue", "Fire Risk Assessment"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Beech Avenue"

    def test_photos_folder_skipped(self, config):
        segments = ["Willow Way", "Plot 3", "Photos"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Willow Way"
        assert result.get("plot") == "Plot 3"

    def test_deeper_non_type_address_still_wins(self, config):
        # "Snagging" is filtered, but "Elm Road" at depth 2 is a valid
        # non-type/category segment — it should win over "Green Lane"
        segments = ["Green Lane", "Snagging", "Elm Road"]
        result = extract_entities(segments, config)
        assert result.get("address") == "Elm Road"

    def test_all_segments_filtered_means_no_entity(self, config):
        segments = ["Finance", "Snagging", "Photos"]
        result = extract_entities(segments, config)
        assert "address" not in result
        assert "plot" not in result

    def test_confidence_returned_with_filtering(self, config):
        segments = [
            "Lenham Road",
            "Site inspection photos",
            "Lenham Road site inspection photos 200117",
        ]
        result = extract_entities_with_confidence(segments, config)
        match = result.get("address")
        assert match is not None
        assert match.value == "Lenham Road"
        assert match.depth == 0
        assert match.confidence > 0.0

    def test_construction_folder_no_bare_address(self, config):
        # "Construction Road" matches Construction category —
        # bare "road" token should be suppressed on organisational folders
        segments = ["Development", "Construction Road", "Plot 12"]
        result = extract_entities(segments, config)
        assert result.get("address") != "Construction Road"

    def test_safety_folder_no_bare_address(self, config):
        # "Safety Road signage" matches Health & Safety category —
        # bare "road" token should be suppressed
        segments = ["Development", "Safety Road signage"]
        result = extract_entities(segments, config)
        assert result.get("address") != "Safety Road signage"

    def test_snagging_preserves_plot_pattern(self, config):
        # "Snagging Plot 12" matches Snagging type — but "Plot 12"
        # still matches via pattern, so plot IS detected
        segments = ["Elm Road", "Snagging Plot 12"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Snagging Plot 12"

    def test_site_inspection_preserves_plot_pattern(self, config):
        # "Site inspection Plot 12" matches type — but pattern
        # match for Plot 12 still works
        segments = ["Elm Road", "Site inspection Plot 12"]
        result = extract_entities_with_confidence(segments, config)
        match = result.get("plot")
        assert match is not None
        assert match.confidence == 0.65


class TestAmbiguousPlotTokensRemoved:
    """Tokens like 'house', 'lodge', 'cottage', 'block' removed from
    plot token list to prevent false positives on building names.
    Number-qualified patterns still match genuine plot references."""

    def test_alpha_house_no_plot(self, config):
        segments = ["Development", "Alpha House", "Finance"]
        result = extract_entities(segments, config)
        assert "plot" not in result

    def test_tennyson_lodge_no_plot(self, config):
        segments = ["Development", "Tennyson Lodge", "Finance"]
        result = extract_entities(segments, config)
        assert "plot" not in result

    def test_energy_house_no_plot(self, config):
        segments = ["Development", "Energy House", "Finance"]
        result = extract_entities(segments, config)
        assert "plot" not in result

    def test_house_3_matches_via_pattern(self, config):
        segments = ["Development", "House 3", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "House 3"

    def test_house_3_confidence_is_pattern(self, config):
        segments = ["Development", "House 3", "Finance"]
        result = extract_entities_with_confidence(segments, config)
        assert result["plot"].confidence == 0.65

    def test_lodge_5_matches_via_pattern(self, config):
        segments = ["Development", "Lodge 5", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Lodge 5"

    def test_cottage_1_matches_via_pattern(self, config):
        segments = ["Development", "Cottage 1", "Finance"]
        result = extract_entities(segments, config)
        assert result.get("plot") == "Cottage 1"

    def test_block_a_still_matches_via_pattern(self, config):
        segments = ["Development", "Block A", "Finance"]
        result = extract_entities_with_confidence(segments, config)
        assert result["plot"].value == "Block A"
        assert result["plot"].confidence == 0.65

    def test_plot_12_still_matches_via_token(self, config):
        segments = ["Development", "Plot 12", "Finance"]
        result = extract_entities_with_confidence(segments, config)
        assert result["plot"].value == "Plot 12"
        assert result["plot"].confidence == 0.80

    def test_phase_2_still_matches_via_pattern(self, config):
        """'Phase 2' no longer matches via token (removed) but still via pattern at 0.65."""
        segments = ["Development", "Phase 2", "Finance"]
        result = extract_entities_with_confidence(segments, config)
        assert result["plot"].value == "Phase 2"
        assert result["plot"].confidence == 0.65


class TestEdgeCases:

    def test_empty_segments(self, config):
        result = extract_entities([], config)
        assert result == {}

    def test_no_entities_config(self):
        config = load_config(CONFIG_DIR, industry="nonexistent")
        result = extract_entities(["Lenham Road", "Plot 12"], config)
        assert result == {}

    def test_case_insensitive_matching(self, config):
        segments = ["LENHAM ROAD", "PLOT 12"]
        result = extract_entities(segments, config)
        assert result.get("address") == "LENHAM ROAD"
        assert result.get("plot") == "PLOT 12"

    def test_segment_value_preserves_original_case(self, config):
        segments = ["Lenham Road", "Plot 12"]
        result = extract_entities(segments, config)
        # Value should be the original segment text, not lowercased
        assert result.get("address") == "Lenham Road"
        assert result.get("plot") == "Plot 12"


class TestFindAllEntityMatches:
    """Tests for find_all_entity_matches (returns ALL matches, not just deepest)."""

    def test_returns_all_plot_matches(self, config):
        segments = ["Phase 1", "Plot 5", "Invoices"]
        matches = find_all_entity_matches(segments, "plot", config)
        values = {m.value for m in matches}
        assert "Phase 1" in values
        assert "Plot 5" in values
        assert len(matches) == 2

    def test_deepest_first_ordering(self, config):
        segments = ["Phase 1", "Plot 5", "Invoices"]
        matches = find_all_entity_matches(segments, "plot", config)
        assert len(matches) >= 2
        assert matches[0].depth > matches[1].depth

    def test_empty_segments(self, config):
        matches = find_all_entity_matches([], "plot", config)
        assert matches == []

    def test_unknown_entity_type(self, config):
        matches = find_all_entity_matches(["Plot 12"], "nonexistent", config)
        assert matches == []

    def test_no_matches(self, config):
        matches = find_all_entity_matches(["Finance", "Legal"], "plot", config)
        assert matches == []

    def test_organisational_folder_pattern_only(self, config):
        """Snagging is a category signal — only pattern matches pass."""
        segments = ["Snagging Plot 12"]
        matches = find_all_entity_matches(segments, "plot", config)
        assert len(matches) == 1
        assert matches[0].confidence == 0.65  # pattern, not token

    def test_single_match(self, config):
        segments = ["Finance", "Plot 12"]
        matches = find_all_entity_matches(segments, "plot", config)
        assert len(matches) == 1
        assert matches[0].value == "Plot 12"
        assert matches[0].depth == 1

    def test_address_entity_type(self, config):
        segments = ["Lenham Road", "Plot 12"]
        matches = find_all_entity_matches(segments, "address", config)
        assert len(matches) == 1
        assert matches[0].value == "Lenham Road"
