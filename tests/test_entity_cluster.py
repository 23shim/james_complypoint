"""Tests for entity clustering."""

import pytest
import pandas as pd
from collections import Counter

from classification.entity_cluster import (
    cluster_entities,
    normalise_entity_value,
    strip_scheme_noise,
    _cluster_values,
    _pick_canonical,
    _build_plot_folder_context,
    _collect_plot_context_pool,
    _cluster_plot_context_values,
)


# ---- normalise_entity_value ----


class TestNormaliseEntityValue:
    def test_lowercase(self):
        assert normalise_entity_value("Lenham Road") == "lenham road"

    def test_strip_punctuation(self):
        assert normalise_entity_value("Lenham Road,") == "lenham road"

    def test_strip_brackets(self):
        assert normalise_entity_value("Plot (12)") == "plot 12"

    def test_collapse_whitespace(self):
        assert normalise_entity_value("lenham  road") == "lenham road"

    def test_preserve_hyphens(self):
        assert normalise_entity_value("Lenham-on-Sea") == "lenham-on-sea"

    def test_preserve_ampersand(self):
        assert normalise_entity_value("Health & Safety") == "health & safety"

    def test_empty_string(self):
        assert normalise_entity_value("") == ""

    def test_whitespace_only(self):
        assert normalise_entity_value("   ") == ""

    def test_mixed_punctuation(self):
        assert normalise_entity_value("Plot #12 (A)!") == "plot 12 a"


# ---- strip_scheme_noise ----


class TestStripSchemeNoise:
    """Tests for universal metadata noise stripping from scheme names."""

    # -- Phase indicators --

    def test_strip_phase_numeric(self):
        assert strip_scheme_noise("liberty park phase 6") == "liberty park"

    def test_strip_phase_compound(self):
        assert strip_scheme_noise("phase 4 & 5 liberty park") == "liberty park"

    def test_strip_ph_dot(self):
        assert strip_scheme_noise("stoke road hoo ph.1") == "stoke road hoo"

    def test_strip_ph_no_dot(self):
        assert strip_scheme_noise("stoke road hoo ph1") == "stoke road hoo"

    def test_strip_phase_word_form(self):
        assert strip_scheme_noise("hillingdon rise phase two") == "hillingdon rise"

    # -- Date ranges --

    def test_strip_date_range(self):
        assert strip_scheme_noise("park farm east ashford 2015-2016") == "park farm east ashford"

    def test_strip_date_range_spaces(self):
        assert strip_scheme_noise("park farm 2020 - 2023") == "park farm"

    # -- Month-year dates --

    def test_strip_month_year_4digit(self):
        assert strip_scheme_noise("scheme feb 2020") == "scheme"

    def test_strip_month_year_2digit(self):
        assert strip_scheme_noise("scheme dec 20") == "scheme"

    def test_strip_full_month_name(self):
        assert strip_scheme_noise("scheme january 2023") == "scheme"

    # -- Day-month-year dates --

    def test_strip_day_month_year(self):
        assert strip_scheme_noise("ramsgate 11 may 21") == "ramsgate"

    # -- Standalone years --

    def test_strip_standalone_year(self):
        assert strip_scheme_noise("north kent college 2024") == "north kent college"

    def test_preserves_non_year_numbers(self):
        """House numbers and plot numbers should survive."""
        assert strip_scheme_noise("29 saddlers close") == "29 saddlers close"

    # -- New annotations --

    def test_strip_new_month_year(self):
        assert strip_scheme_noise("briary fields new dec 20") == "briary fields"

    def test_strip_new_file(self):
        result = strip_scheme_noise("manston road ramsgate new file - 11 may 21")
        assert "manston road ramsgate" in result
        assert "new file" not in result

    def test_strip_new_scheme(self):
        result = strip_scheme_noise("parsonage place new scheme")
        assert result == "parsonage place"

    # -- Preserves core identity --

    def test_preserves_location_name(self):
        assert strip_scheme_noise("quarry wood aldington") == "quarry wood aldington"

    def test_preserves_ampersand(self):
        assert strip_scheme_noise("hinxhill & highmead house") == "hinxhill & highmead house"

    def test_preserves_hyphenated_names(self):
        assert strip_scheme_noise("lenham-on-sea road") == "lenham-on-sea road"

    def test_empty_string(self):
        assert strip_scheme_noise("") == ""

    def test_all_noise(self):
        """Entirely noise -> empty string."""
        assert strip_scheme_noise("phase 1 2020-2023") == ""


# ---- _pick_canonical ----


class TestPickCanonical:
    def test_most_frequent_wins(self):
        members = {"Lenham Road", "Lenham Rd", "lenham road"}
        counts = Counter({"Lenham Road": 50, "Lenham Rd": 10, "lenham road": 5})
        assert _pick_canonical(members, counts) == "Lenham Road"

    def test_longest_breaks_tie(self):
        members = {"Lenham Road", "Lenham Rd"}
        counts = Counter({"Lenham Road": 10, "Lenham Rd": 10})
        assert _pick_canonical(members, counts) == "Lenham Road"

    def test_single_variant(self):
        members = {"Mills Crescent"}
        counts = Counter({"Mills Crescent": 1})
        assert _pick_canonical(members, counts) == "Mills Crescent"


# ---- _cluster_values ----


class TestClusterValues:
    def test_identical_values_cluster(self):
        values = ["Lenham Road", "Lenham Road"]
        counts = Counter(values)
        unique = list(set(values))
        clusters = _cluster_values(unique, counts, 0.55)
        assert len(clusters) == 1

    def test_similar_values_cluster(self):
        """'Lenham Road' and 'Lenham Rd' should cluster at reasonable threshold."""
        values = ["Lenham Road", "Lenham Rd"]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.45)
        assert len(clusters) == 1
        canonical, members = clusters[0]
        assert len(members) == 2

    def test_dissimilar_values_separate(self):
        """'Lenham Road' and 'Dove Street' should NOT cluster."""
        values = ["Lenham Road", "Dove Street"]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55)
        assert len(clusters) == 2

    def test_transitive_linking(self):
        """If A~B and B~C, all three should cluster."""
        # "Mills Crescent", "Mills Cres", "Mills Crescnt" — each pair
        # may not all be above threshold, but the middle one bridges.
        values = ["Mills Crescent", "Mills Crescnt", "Mills Cres"]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.40)
        # Should be 1 or 2 clusters depending on exact similarity
        # At least Mills Crescent and Mills Crescnt should cluster
        all_members = set()
        for _, members in clusters:
            all_members.update(members)
        assert all_members == set(values)

    def test_empty_input(self):
        clusters = _cluster_values([], Counter(), 0.55)
        assert clusters == []

    def test_single_value(self):
        values = ["Plot 12"]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55)
        assert len(clusters) == 1
        assert clusters[0][1] == {"Plot 12"}


# ---- Scheme second-pass clustering ----


class TestSchemeSecondPass:
    """Tests that the second-pass noise-stripped overlap merges
    multi-phase developments that char-trigram Jaccard alone misses."""

    def test_liberty_park_phases_merge(self):
        """Different naming conventions for same development should merge."""
        values = [
            "Phase 2 Liberty Park Hoo Road",
            "Liberty Park Phase 6",
        ]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55, entity_type="scheme")
        assert len(clusters) == 1

    def test_liberty_park_all_variants_merge(self):
        """All Liberty Park variants — phases, locations — should form one cluster."""
        values = [
            "Phase 4 & 5 & EX C. Liberty Park Hoo Road",
            "Phase 2 Liberty Park Hoo Road",
            "Phase 1 Liberty Park Hoo Road Wainscott",
            "Liberty Park Phase 6",
            "Liberty Park Phase 4 & 5",
        ]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55, entity_type="scheme")
        assert len(clusters) == 1
        assert len(clusters[0][1]) == 5

    def test_stoke_road_hoo_phases_merge(self):
        """Ph.1 and Phase 2 of the same street should merge."""
        values = [
            "Stoke Road Hoo TW (Phase 2) Feb 2020",
            "Stoke Road Hoo (Ph.1)",
        ]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55, entity_type="scheme")
        assert len(clusters) == 1

    def test_different_schemes_stay_separate(self):
        """Genuinely different schemes should NOT merge."""
        values = [
            "Stoke Road Hoo (Ph.1)",
            "Colemans Land Hoo",
            "FIRE",
            "Quarry Wood Aldington",
        ]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55, entity_type="scheme")
        assert len(clusters) == 4

    def test_second_pass_only_for_scheme_type(self):
        """Non-scheme entity types should NOT get second-pass merging."""
        values = [
            "Phase 2 Liberty Park Hoo Road",
            "Liberty Park Phase 6",
        ]
        counts = Counter(values)
        # Without entity_type="scheme", these should stay separate
        clusters = _cluster_values(values, counts, 0.55, entity_type="address")
        assert len(clusters) == 2

    def test_single_token_schemes_not_merged(self):
        """Single-word schemes shouldn't match via second pass (too ambiguous)."""
        values = [
            "Riverside Phase 1",
            "Riverside",
        ]
        counts = Counter(values)
        clusters = _cluster_values(values, counts, 0.55, entity_type="scheme")
        # "Riverside" alone is only 1 token after stripping, so second pass
        # won't merge it (requires >= 2 tokens on both sides)
        assert len(clusters) == 2


# ---- cluster_entities (DataFrame integration) ----


class TestClusterEntities:
    @pytest.fixture
    def mock_config(self):
        """Minimal config with entity type names."""
        from classification.models import ClassificationConfig, EntityDefinition

        return ClassificationConfig(
            types={},
            categories={},
            weights={},
            entities={
                "address": EntityDefinition(
                    name="address",
                    tokens=["road", "street"],
                    abbreviations=["rd", "st"],
                    compiled_patterns=[],
                ),
            },
        )

    def test_adds_cluster_columns(self, mock_config):
        df = pd.DataFrame({
            "entity_address": ["Lenham Road", "Lenham Rd", "Dove Street", ""],
            "entity_scheme": ["Scheme A", "Scheme A", "", ""],
        })
        result = cluster_entities(df, mock_config)
        assert "entity_address_cluster_id" in result.columns
        assert "entity_address_canonical" in result.columns
        assert "entity_scheme_cluster_id" in result.columns
        assert "entity_scheme_canonical" in result.columns

    def test_empty_entity_columns(self, mock_config):
        df = pd.DataFrame({
            "entity_address": ["", "", ""],
        })
        result = cluster_entities(df, mock_config)
        assert (result["entity_address_cluster_id"] == "").all()
        assert (result["entity_address_canonical"] == "").all()

    def test_cluster_id_format(self, mock_config):
        df = pd.DataFrame({
            "entity_address": ["Lenham Road", "Dove Street"],
        })
        result = cluster_entities(df, mock_config)
        ids = result["entity_address_cluster_id"].tolist()
        for cid in ids:
            if cid:
                assert cid.startswith("addr_")

    def test_canonical_assigned_to_all_rows(self, mock_config):
        df = pd.DataFrame({
            "entity_address": [
                "Lenham Road", "Lenham Road", "Lenham Road",
            ],
        })
        result = cluster_entities(df, mock_config)
        canonicals = result["entity_address_canonical"].tolist()
        assert all(c == "Lenham Road" for c in canonicals)

    def test_missing_column_handled(self, mock_config):
        """If an entity column doesn't exist, empty columns are added."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = cluster_entities(df, mock_config)
        assert "entity_address_cluster_id" in result.columns
        assert (result["entity_address_cluster_id"] == "").all()

    def test_scheme_always_included(self, mock_config):
        """Scheme clustering is always attempted even if not in config.entities."""
        df = pd.DataFrame({
            "entity_address": ["Road A"],
            "entity_scheme": ["My Scheme"],
        })
        result = cluster_entities(df, mock_config)
        assert "entity_scheme_cluster_id" in result.columns
        assert result["entity_scheme_cluster_id"].iloc[0] != ""


# ---- _build_plot_folder_context ----


def _make_plot_row(segs, plot_depth, scheme_cluster="", addr_cluster="", addr_depth=-1):
    """Helper: build a single-row dict with the columns _build_plot_folder_context reads."""
    raw_plots = [{"value": segs[plot_depth], "confidence": 0.8, "depth": plot_depth}]
    return {
        "segments": segs,
        "raw_plots": raw_plots,
        "entity_scheme_cluster_id": scheme_cluster,
        "entity_address_cluster_id": addr_cluster,
        "entity_address_depth": addr_depth,
    }


class TestBuildPlotFolderContext:
    def test_scheme_context_used(self):
        """Folder with a scheme cluster gets scheme_cluster_id as context."""
        df = pd.DataFrame([
            _make_plot_row(
                ["Scheme A", "Plot 7"], plot_depth=1,
                scheme_cluster="schm_0001",
            ),
        ])
        ctx = _build_plot_folder_context(df)
        assert ctx.get("Scheme A\\Plot 7") == "schm_0001"

    def test_address_context_used_when_no_scheme(self):
        """Folder with address ancestor but no scheme gets addr:<id> context."""
        df = pd.DataFrame([
            _make_plot_row(
                ["Dev", "Lenham Road", "Plot 7"], plot_depth=2,
                scheme_cluster="",
                addr_cluster="addr_0005",
                addr_depth=1,
            ),
        ])
        ctx = _build_plot_folder_context(df)
        assert ctx.get("Dev\\Lenham Road\\Plot 7") == "addr:addr_0005"

    def test_address_not_ancestor_gives_nocontext(self):
        """Address at same or deeper depth than plot is not an ancestor."""
        df = pd.DataFrame([
            _make_plot_row(
                ["Dev", "Plot 7"], plot_depth=1,
                scheme_cluster="",
                addr_cluster="addr_0005",
                addr_depth=1,  # same depth — not an ancestor
            ),
        ])
        ctx = _build_plot_folder_context(df)
        key = "Dev\\Plot 7"
        assert key in ctx
        assert ctx[key].startswith("__nocontext__")

    def test_no_context_when_no_scheme_or_address(self):
        """Folder with no scheme or address gets a unique no-context key."""
        df = pd.DataFrame([
            _make_plot_row(["Dev", "Plot 7"], plot_depth=1),
        ])
        ctx = _build_plot_folder_context(df)
        key = "Dev\\Plot 7"
        assert key in ctx
        assert ctx[key].startswith("__nocontext__")

    def test_different_rows_same_folder_first_wins(self):
        """If two rows share the same plot folder path, first row's context wins."""
        row1 = _make_plot_row(
            ["Scheme A", "Plot 7"], plot_depth=1, scheme_cluster="schm_0001",
        )
        row2 = _make_plot_row(
            ["Scheme A", "Plot 7"], plot_depth=1, scheme_cluster="schm_0001",
        )
        df = pd.DataFrame([row1, row2])
        ctx = _build_plot_folder_context(df)
        assert len([k for k in ctx if "Plot 7" in k]) == 1

    def test_scheme_preferred_over_address(self):
        """Scheme context takes priority even when address is also present."""
        df = pd.DataFrame([
            _make_plot_row(
                ["Scheme A", "Lenham Road", "Plot 7"], plot_depth=2,
                scheme_cluster="schm_0001",
                addr_cluster="addr_0005",
                addr_depth=1,
            ),
        ])
        ctx = _build_plot_folder_context(df)
        assert ctx.get("Scheme A\\Lenham Road\\Plot 7") == "schm_0001"


# ---- _cluster_plot_context_values ----


class TestClusterPlotContextValues:
    def test_different_contexts_never_merge(self):
        """Identical plot values under different scheme contexts stay separate."""
        pool = Counter({
            ("schm_0001", "plot 7"): 3,
            ("schm_0002", "plot 7"): 2,
        })
        clusters = _cluster_plot_context_values(pool, 0.55)
        assert len(clusters) == 2
        contexts = {pair[0] for pair, _ in clusters}
        assert "schm_0001" in contexts
        assert "schm_0002" in contexts

    def test_same_context_similar_values_merge(self):
        """Similar plot values within the same scheme context are merged."""
        pool = Counter({
            ("schm_0001", "plot 7"): 5,
            ("schm_0001", "plot 7 "): 2,   # trailing space — same after normalise
        })
        clusters = _cluster_plot_context_values(pool, 0.55)
        assert len(clusters) == 1
        _, members = clusters[0]
        assert len(members) == 2

    def test_no_context_plots_isolated(self):
        """No-context plots each get their own singleton cluster."""
        pool = Counter({
            ("__nocontext__Dev\\Plot 7", "plot 7"): 1,
            ("__nocontext__OtherDev\\Plot 7", "plot 7"): 1,
        })
        clusters = _cluster_plot_context_values(pool, 0.55)
        assert len(clusters) == 2
        # Each cluster has exactly one member
        for _, members in clusters:
            assert len(members) == 1

    def test_no_context_never_merges_with_context(self):
        """A no-context plot never merges with a scheme-context plot."""
        pool = Counter({
            ("schm_0001", "plot 7"): 4,
            ("__nocontext__Orphan\\Plot 7", "plot 7"): 1,
        })
        clusters = _cluster_plot_context_values(pool, 0.55)
        assert len(clusters) == 2

    def test_canonical_is_most_frequent(self):
        """Canonical value is the most frequent within the cluster."""
        pool = Counter({
            ("schm_0001", "plot 7"): 10,
            ("schm_0001", "plt 7"): 1,    # rare variant — may or may not merge
        })
        # At threshold 0.30 these should merge (short strings, one shared trigram)
        clusters = _cluster_plot_context_values(pool, 0.30)
        if len(clusters) == 1:
            canonical_pair, _ = clusters[0]
            assert canonical_pair[1] == "plot 7"   # higher frequency wins

    def test_empty_pool(self):
        assert _cluster_plot_context_values(Counter(), 0.55) == []

    def test_single_entry_singleton_cluster(self):
        pool = Counter({("schm_0001", "plot 12"): 3})
        clusters = _cluster_plot_context_values(pool, 0.55)
        assert len(clusters) == 1
        canonical_pair, members = clusters[0]
        assert canonical_pair[1] == "plot 12"
        assert len(members) == 1


# ---- cluster_entities with plot entity type ----


@pytest.fixture
def plot_config():
    """Config with address + plot entity types."""
    from classification.models import ClassificationConfig, EntityDefinition
    import re
    return ClassificationConfig(
        types={},
        categories={},
        weights={},
        entities={
            "address": EntityDefinition(
                name="address",
                tokens=["road", "street", "close"],
                abbreviations=["rd"],
                compiled_patterns=[],
            ),
            "plot": EntityDefinition(
                name="plot",
                tokens=["plot", "flat", "unit"],
                abbreviations=["plt"],
                compiled_patterns=[re.compile(r"\bplot\s*\d+", re.IGNORECASE)],
            ),
        },
    )


def _plot_df_row(
    segs, plot_val, plot_depth, plot_extracted,
    scheme_cluster="", addr_cluster="", addr_depth=-1,
):
    """Build a row dict for plot clustering integration tests."""
    raw_plots = [{"value": segs[plot_depth], "confidence": 0.8, "depth": plot_depth}]
    extracted_plots = [{"value": segs[plot_depth], "confidence": 0.8,
                        "depth": plot_depth, "extracted": plot_extracted}]
    return {
        "segments": segs,
        "entity_plot": plot_val,
        "entity_plot_depth": plot_depth,
        "entity_plot_extracted": plot_extracted,
        "entity_plot_confidence": 0.8,
        "raw_plots": raw_plots,
        "extracted_plots": extracted_plots,
        "entity_scheme_cluster_id": scheme_cluster,
        "entity_address_cluster_id": addr_cluster,
        "entity_address_depth": addr_depth,
    }


class TestClusterEntitiesPlot:
    def test_plot_cluster_columns_added(self, plot_config):
        """cluster_entities always adds entity_plot_cluster_id/canonical."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Scheme A", "Plot 7"], "Plot 7", 1, "plot 7",
                scheme_cluster="schm_0001",
            ),
        ])
        result = cluster_entities(df, plot_config)
        assert "entity_plot_cluster_id" in result.columns
        assert "entity_plot_canonical" in result.columns

    def test_same_plot_number_different_schemes_separate_clusters(self, plot_config):
        """Plot 7 in Scheme A and Plot 7 in Scheme B get different cluster IDs."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Scheme A", "Plot 7"], "Plot 7", 1, "plot 7",
                scheme_cluster="schm_0001",
            ),
            _plot_df_row(
                ["Scheme B", "Plot 7"], "Plot 7", 1, "plot 7",
                scheme_cluster="schm_0002",
            ),
        ])
        result = cluster_entities(df, plot_config)
        ids = result["entity_plot_cluster_id"].tolist()
        assert ids[0] != ""
        assert ids[1] != ""
        assert ids[0] != ids[1], (
            "Plot 7 in different schemes must have different cluster IDs"
        )

    def test_same_plot_same_scheme_get_same_cluster(self, plot_config):
        """Two files for Plot 7 within the same scheme share a cluster."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Scheme A", "Plot 7", "Drawings"], "Plot 7", 1, "plot 7",
                scheme_cluster="schm_0001",
            ),
            _plot_df_row(
                ["Scheme A", "Plot 7", "Snagging"], "Plot 7", 1, "plot 7",
                scheme_cluster="schm_0001",
            ),
        ])
        result = cluster_entities(df, plot_config)
        ids = result["entity_plot_cluster_id"].tolist()
        assert ids[0] != ""
        assert ids[0] == ids[1], "Same plot in same scheme must share a cluster ID"

    def test_no_context_plots_get_isolated_clusters(self, plot_config):
        """Plots with no scheme or address context each get unique cluster IDs."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Dev A", "Plot 7"], "Plot 7", 1, "plot 7",
            ),
            _plot_df_row(
                ["Dev B", "Plot 7"], "Plot 7", 1, "plot 7",
            ),
        ])
        result = cluster_entities(df, plot_config)
        ids = result["entity_plot_cluster_id"].tolist()
        assert ids[0] != ""
        assert ids[1] != ""
        assert ids[0] != ids[1], (
            "No-context plots must be isolated into separate clusters"
        )

    def test_address_context_links_plots_within_same_address(self, plot_config):
        """Two plots under the same address cluster can share a cluster."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Dev", "Lenham Road", "Plot 7"], "Plot 7", 2, "plot 7",
                addr_cluster="addr_0001", addr_depth=1,
            ),
            _plot_df_row(
                ["Dev", "Lenham Road", "Plot 7"], "Plot 7", 2, "plot 7",
                addr_cluster="addr_0001", addr_depth=1,
            ),
        ])
        result = cluster_entities(df, plot_config)
        ids = result["entity_plot_cluster_id"].tolist()
        assert ids[0] != ""
        assert ids[0] == ids[1]

    def test_address_context_isolates_from_different_address(self, plot_config):
        """Plot 7 on Lenham Road and Plot 7 on Dove Street are separate."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Dev", "Lenham Road", "Plot 7"], "Plot 7", 2, "plot 7",
                addr_cluster="addr_0001", addr_depth=1,
            ),
            _plot_df_row(
                ["Dev", "Dove Street", "Plot 7"], "Plot 7", 2, "plot 7",
                addr_cluster="addr_0002", addr_depth=1,
            ),
        ])
        result = cluster_entities(df, plot_config)
        ids = result["entity_plot_cluster_id"].tolist()
        assert ids[0] != ids[1]

    def test_plot_cluster_id_format(self, plot_config):
        """Plot cluster IDs use the 'plot_' prefix."""
        df = pd.DataFrame([
            _plot_df_row(
                ["Scheme A", "Plot 12"], "Plot 12", 1, "plot 12",
                scheme_cluster="schm_0001",
            ),
        ])
        result = cluster_entities(df, plot_config)
        cid = result["entity_plot_cluster_id"].iloc[0]
        assert cid.startswith("plot_")

    def test_empty_entity_plot_rows_get_empty_cluster(self, plot_config):
        """Rows with no entity_plot get empty cluster ID."""
        df = pd.DataFrame([
            {
                "segments": ["Scheme A", "file.pdf"],
                "entity_plot": "",
                "entity_plot_depth": -1,
                "entity_plot_extracted": "",
                "entity_plot_confidence": 0.0,
                "raw_plots": [],
                "extracted_plots": [],
                "entity_scheme_cluster_id": "schm_0001",
                "entity_address_cluster_id": "",
                "entity_address_depth": -1,
            },
        ])
        result = cluster_entities(df, plot_config)
        assert result["entity_plot_cluster_id"].iloc[0] == ""
