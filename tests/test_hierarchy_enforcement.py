"""Tests for entity hierarchy enforcement in the classification engine."""

import pandas as pd
import pytest

from classification.engine import enforce_entity_hierarchy


def _make_df(rows):
    """Build a DataFrame from row dicts with sensible defaults."""
    defaults = {
        "entity_scheme": "",
        "entity_scheme_confidence": 0.0,
        "entity_scheme_path": "",
        "entity_scheme_depth": -1,
        "entity_plot": "",
        "entity_plot_confidence": 0.0,
        "entity_plot_depth": -1,
        "entity_address": "",
        "entity_address_confidence": 0.0,
        "entity_address_depth": -1,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


# ---- Suppression cases ----


class TestSuppression:

    def test_plot_at_scheme_depth_suppressed(self):
        df = _make_df([{
            "entity_scheme": "Liberty Park",
            "entity_scheme_depth": 0,
            "entity_plot": "Liberty Park",
            "entity_plot_confidence": 0.80,
            "entity_plot_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == ""
        assert result.iloc[0]["entity_plot_confidence"] == 0.0
        assert result.iloc[0]["entity_plot_depth"] == -1

    def test_address_at_scheme_depth_suppressed(self):
        df = _make_df([{
            "entity_scheme": "Bridge House, Dover Road",
            "entity_scheme_depth": 0,
            "entity_address": "Bridge House, Dover Road",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_address"] == ""
        assert result.iloc[0]["entity_address_confidence"] == 0.0
        assert result.iloc[0]["entity_address_depth"] == -1

    def test_plot_above_scheme_suppressed(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road Dev",
            "entity_scheme_depth": 2,
            "entity_plot": "Phase 1",
            "entity_plot_confidence": 0.80,
            "entity_plot_depth": 1,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == ""

    def test_both_entities_at_scheme_level_suppressed(self):
        df = _make_df([{
            "entity_scheme": "Liberty Park Hoo Road",
            "entity_scheme_depth": 0,
            "entity_plot": "Liberty Park Hoo Road",
            "entity_plot_confidence": 0.80,
            "entity_plot_depth": 0,
            "entity_address": "Liberty Park Hoo Road",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == ""
        assert result.iloc[0]["entity_address"] == ""


# ---- Keep cases ----


class TestKept:

    def test_plot_deeper_than_scheme_kept(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road",
            "entity_scheme_depth": 0,
            "entity_plot": "Plot 12",
            "entity_plot_confidence": 0.80,
            "entity_plot_depth": 1,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == "Plot 12"
        assert result.iloc[0]["entity_plot_confidence"] == 0.80
        assert result.iloc[0]["entity_plot_depth"] == 1

    def test_address_deeper_than_scheme_kept(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road",
            "entity_scheme_depth": 0,
            "entity_address": "Fuggles Close",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 1,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_address"] == "Fuggles Close"
        assert result.iloc[0]["entity_address_confidence"] == 0.80


# ---- No-op cases ----


class TestNoop:

    def test_no_scheme_leaves_plot_unchanged(self):
        """No scheme → Rule 1 cannot fire → plot entity preserved."""
        df = _make_df([{
            "entity_plot": "Plot 12",
            "entity_plot_confidence": 0.80,
            "entity_plot_depth": 1,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == "Plot 12"

    def test_no_scheme_leaves_address_unchanged(self):
        """No scheme → Rule 1 cannot fire → address entity preserved."""
        df = _make_df([{
            "entity_address": "Lenham Road",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 1,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_address"] == "Lenham Road"

    def test_no_entities_with_scheme_is_noop(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road",
            "entity_scheme_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == ""
        assert result.iloc[0]["entity_address"] == ""

    def test_empty_dataframe(self):
        df = _make_df([])
        result = enforce_entity_hierarchy(df)
        assert len(result) == 0


# ---- Mixed rows ----


class TestSchemeNameMarker:
    """Change 6: Suppressed entities matching the scheme name get marked."""

    def test_address_matching_scheme_name_marked(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road",
            "entity_scheme_depth": 0,
            "entity_address": "Lenham Road",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_address"] == ""  # still suppressed
        assert result.iloc[0]["entity_address_is_scheme_name"] == True

    def test_plot_matching_scheme_name_marked(self):
        df = _make_df([{
            "entity_scheme": "Phase 2",
            "entity_scheme_depth": 0,
            "entity_plot": "Phase 2",
            "entity_plot_confidence": 0.80,
            "entity_plot_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == ""
        assert result.iloc[0]["entity_plot_is_scheme_name"] == True

    def test_entity_not_matching_scheme_name_not_marked(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road",
            "entity_scheme_depth": 1,
            "entity_address": "Oak Lane",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 0,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_address"] == ""  # suppressed (above scheme)
        assert result.iloc[0]["entity_address_is_scheme_name"] == False

    def test_deeper_entity_not_marked(self):
        df = _make_df([{
            "entity_scheme": "Lenham Road",
            "entity_scheme_depth": 0,
            "entity_address": "Fuggles Close",
            "entity_address_confidence": 0.80,
            "entity_address_depth": 1,
        }])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_address"] == "Fuggles Close"  # kept
        assert result.iloc[0]["entity_address_is_scheme_name"] == False


class TestMixedRows:

    def test_only_affected_rows_suppressed(self):
        df = _make_df([
            {  # Row 0: plot at scheme level -> suppress
                "entity_scheme": "Liberty Park",
                "entity_scheme_depth": 0,
                "entity_plot": "Liberty Park",
                "entity_plot_confidence": 0.80,
                "entity_plot_depth": 0,
            },
            {  # Row 1: plot below scheme -> keep
                "entity_scheme": "Liberty Park",
                "entity_scheme_depth": 0,
                "entity_plot": "Plot 12",
                "entity_plot_confidence": 0.80,
                "entity_plot_depth": 1,
            },
            {  # Row 2: no scheme -> keep
                "entity_plot": "Phase 2",
                "entity_plot_confidence": 0.80,
                "entity_plot_depth": 0,
            },
        ])
        result = enforce_entity_hierarchy(df)
        assert result.iloc[0]["entity_plot"] == ""
        assert result.iloc[1]["entity_plot"] == "Plot 12"
        assert result.iloc[2]["entity_plot"] == "Phase 2"
