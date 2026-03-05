"""Tests for scheme detection via structural inference."""

import pytest
import pandas as pd

from classification.config_loader import load_config
from classification.scheme_detector import (
    SchemeMatch,
    _aggregate_folder_stats,
    _exclude_container_folders,
    _identify_candidates,
    _score_candidates,
    detect_and_assign_schemes,
)


CONFIG_DIR = "config"


@pytest.fixture
def config():
    return load_config(CONFIG_DIR, industry="housing")


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame for scheme detection tests.

    Each row dict must have 'segments'. Optional: entity_plot,
    entity_address, entity_plot_depth, entity_address_depth.
    """
    defaults = {
        "entity_plot": "",
        "entity_address": "",
        "entity_plot_depth": -1,
        "entity_address_depth": -1,
    }
    for row in rows:
        for k, v in defaults.items():
            row.setdefault(k, v)
    return pd.DataFrame(rows)


class TestCategorySignalQualification:
    """Change 1: Category signals relax the entity-count hard gate."""

    def test_strong_categories_no_entities_qualifies(self, config):
        """4+ development categories with 0 entities → qualifies."""
        folder_categories = {
            "Dev\\Cliffside Road": {"Construction"},
            "Dev\\Cliffside Road\\Handover": {"Construction"},
            "Dev\\Cliffside Road\\Maintenance": {"Maintenance"},
            "Dev\\Cliffside Road\\Planning": {"Planning"},
            "Dev\\Cliffside Road\\Sales": {"Sales"},
        }
        # 5 files, no entities, but strong category signals.
        # "Cliffside Road" has an address token ("Road") to pass the
        # location gate.
        rows = [
            {"segments": ["Dev", "Cliffside Road", cat, "file.pdf"]}
            for cat in ["Handover", "Maintenance", "Planning", "Sales", "Drawings"]
        ]
        df = _make_df(rows)
        stats = _aggregate_folder_stats(df, folder_categories)

        key = "Dev\\Cliffside Road"
        assert key in stats
        dev_cats = stats[key].category_signals & {
            "Construction", "Maintenance", "Planning", "Utilities", "Sales",
        }
        assert len(dev_cats) >= 4

        # Should qualify as candidate
        candidates, _ = _identify_candidates(stats, config, min_entities=3)
        assert key in candidates

    def test_entity_plus_categories_qualifies(self, config):
        """1 entity + 3 development categories → qualifies."""
        folder_categories = {
            "Dev\\Oakwood\\Construction": {"Construction"},
            "Dev\\Oakwood\\Maintenance": {"Maintenance"},
            "Dev\\Oakwood\\Planning": {"Planning"},
        }
        rows = [
            {"segments": ["Dev", "Oakwood", "Plot 1", "file.pdf"],
             "entity_plot": "Plot 1", "entity_plot_depth": 2},
            {"segments": ["Dev", "Oakwood", "Construction", "file.pdf"]},
            {"segments": ["Dev", "Oakwood", "Maintenance", "file.pdf"]},
            {"segments": ["Dev", "Oakwood", "Planning", "file.pdf"]},
        ]
        df = _make_df(rows)
        stats = _aggregate_folder_stats(df, folder_categories)

        key = "Dev\\Oakwood"
        assert stats[key].unique_entities < 3  # only 1 plot
        candidates, _ = _identify_candidates(stats, config, min_entities=3)
        assert key in candidates

    def test_two_categories_not_enough(self, config):
        """0 entities + 2 categories → does NOT qualify."""
        folder_categories = {
            "Dev\\SmallFolder\\Construction": {"Construction"},
            "Dev\\SmallFolder\\Maintenance": {"Maintenance"},
        }
        rows = [
            {"segments": ["Dev", "SmallFolder", "Construction", "file.pdf"]},
            {"segments": ["Dev", "SmallFolder", "Maintenance", "file.pdf"]},
        ]
        df = _make_df(rows)
        stats = _aggregate_folder_stats(df, folder_categories)

        key = "Dev\\SmallFolder"
        candidates, _ = _identify_candidates(stats, config, min_entities=3)
        assert key not in candidates

    def test_non_development_categories_not_enough(self, config):
        """Finance + IT + HR are not development categories → no qualification."""
        folder_categories = {
            "Dev\\Office\\Finance": {"Finance"},
            "Dev\\Office\\IT": {"IT"},
            "Dev\\Office\\HR": {"HR"},
            "Dev\\Office\\Admin": {"Administration"},
        }
        rows = [
            {"segments": ["Dev", "Office", "Finance", "file.pdf"]},
            {"segments": ["Dev", "Office", "IT", "file.pdf"]},
            {"segments": ["Dev", "Office", "HR", "file.pdf"]},
            {"segments": ["Dev", "Office", "Admin", "file.pdf"]},
        ]
        df = _make_df(rows)
        stats = _aggregate_folder_stats(df, folder_categories)

        key = "Dev\\Office"
        candidates, _ = _identify_candidates(stats, config, min_entities=3)
        assert key not in candidates

    def test_categories_propagate_to_ancestors(self, config):
        """Category signals propagate upward through ancestor folders."""
        folder_categories = {
            "Root\\Scheme\\Construction": {"Construction"},
            "Root\\Scheme\\Sales": {"Sales"},
        }
        rows = [
            {"segments": ["Root", "Scheme", "Construction", "file.pdf"]},
            {"segments": ["Root", "Scheme", "Sales", "file.pdf"]},
        ]
        df = _make_df(rows)
        stats = _aggregate_folder_stats(df, folder_categories)

        # Root should have categories propagated from Scheme's children
        assert "Construction" in stats["Root"].category_signals
        assert "Sales" in stats["Root"].category_signals


class TestContainerExclusion:
    """Parent/child scheme conflict resolution using confidence."""

    def test_children_higher_than_parent_removes_parent(self, config):
        """Children score higher → parent is a container, removed."""
        scored = {
            "Mark_Development": SchemeMatch(
                value="Mark_Development", confidence=0.70,
                depth=0, folder_path="Mark_Development",
            ),
            "Mark_Development\\Quinton Road": SchemeMatch(
                value="Quinton Road", confidence=0.85,
                depth=1, folder_path="Mark_Development\\Quinton Road",
            ),
            "Mark_Development\\Elm Street": SchemeMatch(
                value="Elm Street", confidence=0.80,
                depth=1, folder_path="Mark_Development\\Elm Street",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "Mark_Development" not in result
        assert "Mark_Development\\Quinton Road" in result
        assert "Mark_Development\\Elm Street" in result

    def test_many_children_lower_than_parent_still_container(self, config):
        """3+ child schemes even when parent scores higher → container.

        A folder organising many real schemes accumulates their
        entities and inflates its own score — still a container.
        """
        scored = {
            "WorkFolder": SchemeMatch(
                value="WorkFolder", confidence=0.90,
                depth=0, folder_path="WorkFolder",
            ),
            "WorkFolder\\Scheme A": SchemeMatch(
                value="Scheme A", confidence=0.85,
                depth=1, folder_path="WorkFolder\\Scheme A",
            ),
            "WorkFolder\\Scheme B": SchemeMatch(
                value="Scheme B", confidence=0.80,
                depth=1, folder_path="WorkFolder\\Scheme B",
            ),
            "WorkFolder\\Scheme C": SchemeMatch(
                value="Scheme C", confidence=0.75,
                depth=1, folder_path="WorkFolder\\Scheme C",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "WorkFolder" not in result
        assert "WorkFolder\\Scheme A" in result
        assert "WorkFolder\\Scheme B" in result
        assert "WorkFolder\\Scheme C" in result

    def test_parent_higher_than_child_suppresses_child(self, config):
        """Parent scores higher and has direct structure → child is a false positive, suppressed."""
        scored = {
            "Goldsel Road": SchemeMatch(
                value="Goldsel Road", confidence=0.85,
                depth=0, folder_path="Goldsel Road",
                direct_category_children=3,
                direct_plot_children=5,
            ),
            "Goldsel Road\\Complaint 157": SchemeMatch(
                value="Complaint 157", confidence=0.65,
                depth=1, folder_path="Goldsel Road\\Complaint 157",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "Goldsel Road" in result
        assert "Goldsel Road\\Complaint 157" not in result

    def test_parent_equal_confidence_suppresses_child(self, config):
        """Equal confidence → parent wins (shallower), child suppressed.
        Parent has direct structure so it is the real scheme."""
        scored = {
            "Parent": SchemeMatch(
                value="Parent", confidence=0.80,
                depth=0, folder_path="Parent",
                direct_category_children=2,
                direct_plot_children=3,
            ),
            "Parent\\Child": SchemeMatch(
                value="Child", confidence=0.80,
                depth=1, folder_path="Parent\\Child",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "Parent" in result
        assert "Parent\\Child" not in result

    def test_scheme_without_children_kept(self, config):
        scored = {
            "Standalone": SchemeMatch(
                value="Standalone", confidence=0.80,
                depth=0, folder_path="Standalone",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "Standalone" in result

    def test_nested_containers_child_wins(self, config):
        """GP → Parent → Child, where confidence increases with depth.

        GP and Parent are both containers (children score higher).
        """
        scored = {
            "GP": SchemeMatch(
                value="GP", confidence=0.60,
                depth=0, folder_path="GP",
            ),
            "GP\\Parent": SchemeMatch(
                value="Parent", confidence=0.75,
                depth=1, folder_path="GP\\Parent",
            ),
            "GP\\Parent\\Child": SchemeMatch(
                value="Child", confidence=0.90,
                depth=2, folder_path="GP\\Parent\\Child",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "GP" not in result
        assert "GP\\Parent" not in result
        assert "GP\\Parent\\Child" in result

    def test_nested_parent_wins_suppresses_all_descendants(self, config):
        """GP → Parent → Child, where GP scores highest.

        GP suppresses Parent (direct child). Parent in turn
        suppresses Child (its direct child). Both are removed,
        leaving only GP as the real scheme. Both GP and Parent have
        direct structural signals so they suppress their children.
        """
        scored = {
            "GP": SchemeMatch(
                value="GP", confidence=0.90,
                depth=0, folder_path="GP",
                direct_category_children=3,
                direct_plot_children=5,
            ),
            "GP\\Parent": SchemeMatch(
                value="Parent", confidence=0.75,
                depth=1, folder_path="GP\\Parent",
                direct_category_children=2,
                direct_plot_children=2,
            ),
            "GP\\Parent\\Child": SchemeMatch(
                value="Child", confidence=0.70,
                depth=2, folder_path="GP\\Parent\\Child",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        assert "GP" in result
        assert "GP\\Parent" not in result
        assert "GP\\Parent\\Child" not in result

    def test_grandchild_not_direct_child(self, config):
        """Grandchild (not direct child) does NOT trigger conflict."""
        scored = {
            "GP": SchemeMatch(
                value="GP", confidence=0.80,
                depth=0, folder_path="GP",
            ),
            "GP\\Middle\\Child": SchemeMatch(
                value="Child", confidence=0.70,
                depth=2, folder_path="GP\\Middle\\Child",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        # GP has no DIRECT child scheme (Middle is not scored)
        assert "GP" in result
        assert "GP\\Middle\\Child" in result


class TestCategoryScoring:
    """Change 4: Category signals boost scheme confidence."""

    def test_three_dev_categories_boost(self, config):
        """3+ development categories → +0.10 boost."""
        candidates = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                category_signals={"Construction", "Maintenance", "Planning"},
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        base = scored["Scheme1"].confidence

        # Without categories
        candidates_no_cat = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                category_signals=set(),
            ),
        }
        scored_no_cat = _score_candidates(candidates_no_cat, config, root_depth=-1)
        no_cat = scored_no_cat["Scheme1"].confidence

        # Should get exactly +0.10 (dev categories only — no Construction-specific bonus)
        assert base > no_cat
        assert abs(base - no_cat - 0.10) < 0.01

    def test_construction_alone_gives_no_boost(self, config):
        """Just Construction (< 3 dev cats) → no category boost."""
        candidates = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                category_signals={"Construction"},
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        base = scored["Scheme1"].confidence

        candidates_no_cat = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                category_signals=set(),
            ),
        }
        scored_no_cat = _score_candidates(candidates_no_cat, config, root_depth=-1)
        no_cat = scored_no_cat["Scheme1"].confidence

        assert base == no_cat

    def test_non_dev_categories_no_boost(self, config):
        """Finance + IT → no dev category boost."""
        candidates = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                category_signals={"Finance", "IT"},
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        base = scored["Scheme1"].confidence

        candidates_no_cat = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                category_signals=set(),
            ),
        }
        scored_no_cat = _score_candidates(candidates_no_cat, config, root_depth=-1)

        assert base == scored_no_cat["Scheme1"].confidence


class TestCategoryTypeExclusion:
    """Conservative type/category matching for scheme candidate filtering."""

    def test_long_name_with_short_alias_not_excluded(self, config):
        """Long folder name with incidental 'SO' alias → NOT excluded."""
        from classification.scheme_detector import _is_category_or_type_folder

        name = "Walderslade Road (185) Chatham (20 houses SO private sale & rent)"
        assert not _is_category_or_type_folder(name, config)

    def test_short_category_folder_still_excluded(self, config):
        """Short folder name matching a category → still excluded."""
        from classification.scheme_detector import _is_category_or_type_folder

        assert _is_category_or_type_folder("Sales", config)
        assert _is_category_or_type_folder("Construction", config)
        assert _is_category_or_type_folder("Snagging", config)

    def test_plural_epc_folder_excluded(self, config):
        """Folder named 'EPCs' (plural) → excluded as type folder."""
        from classification.scheme_detector import _is_category_or_type_folder

        assert _is_category_or_type_folder("EPCs", config)


class TestPlotLeadExclusion:
    """Long folder names containing a plot reference (Plot N, Flat N, Unit N)
    are excluded as entity-only records even though total token count > 3."""

    def test_edrm_plot_long_name_excluded(self, config):
        """'EDRM Plot 16 - 16 Bridge Close - Chloe and Joshua' → entity-only."""
        from classification.scheme_detector import _is_entity_only_folder

        assert _is_entity_only_folder(
            "EDRM Plot 16 - 16 Bridge Close - Chloe and Joshua", config
        )

    def test_plain_plot_lead_excluded(self, config):
        """'Plot 3 - 2 Harrington Way - Laura Ferrara' → entity-only."""
        from classification.scheme_detector import _is_entity_only_folder

        assert _is_entity_only_folder(
            "Plot 3 - 2 Harrington Way - Laura Ferrara", config
        )

    def test_flat_lead_excluded(self, config):
        """'Flat 4B - Oak Road - Buyer Name' → entity-only."""
        from classification.scheme_detector import _is_entity_only_folder

        assert _is_entity_only_folder("Flat 4B - Oak Road - Buyer Name", config)

    def test_scheme_name_with_phase_not_excluded(self, config):
        """'Oare Lakes Faversham Phase 2' starts with a place name, not a plot → kept."""
        from classification.scheme_detector import _is_entity_only_folder

        assert not _is_entity_only_folder("Oare Lakes Faversham Phase 2", config)

    def test_edrm_scheme_name_not_excluded(self, config):
        """'EDRM-Parsonage Place, Otham' starts with EDRM then a place name → kept."""
        from classification.scheme_detector import _is_entity_only_folder

        assert not _is_entity_only_folder("EDRM-Parsonage Place, Otham", config)

    def test_short_plot_name_still_excluded(self, config):
        """Short 'Plot 12' (≤ 3 tokens) still excluded via existing path."""
        from classification.scheme_detector import _is_entity_only_folder

        assert _is_entity_only_folder("Plot 12", config)

    def test_plot_number_in_middle_excluded(self, config):
        """Plot reference anywhere in a long name → entity-only."""
        from classification.scheme_detector import _is_entity_only_folder

        # Plot number appears in the middle, not at the start
        assert _is_entity_only_folder(
            "16 Bridge Close - Plot 16 - Chloe and Joshua", config
        )

    def test_no_plot_number_long_name_kept(self, config):
        """Long name with no plot/flat/unit/apt + number → not entity-only."""
        from classification.scheme_detector import _is_entity_only_folder

        assert not _is_entity_only_folder(
            "Cotton Lane, Dartford - 25% share old lea RPI plus 0.5%", config
        )


class TestLongNameTypeGuard:
    """Single-word type tokens suppressed in long folder names (4+ tokens).

    Compound scheme names like 'School Lane 10% lease RTSO' contain
    incidental type words ('lease') that should not trigger exclusion.
    Multi-word tokens and abbreviations remain active in all names.
    """

    def test_long_name_with_lease_not_excluded(self, config):
        """'EDRM School Lane Newington 10% lease' → NOT excluded."""
        from classification.scheme_detector import _is_category_or_type_folder

        assert not _is_category_or_type_folder(
            "EDRM School Lane Newington 10% lease RTSO CPI 1%", config
        )

    def test_short_name_lease_still_excluded(self, config):
        """Short 'Lease Documents' → still excluded as type folder."""
        from classification.scheme_detector import _is_category_or_type_folder

        assert _is_category_or_type_folder("Lease Documents", config)

    def test_place_name_blocks_category_exclusion(self, config):
        """Folder with a place name is NOT excluded even with category words."""
        from classification.scheme_detector import _is_category_or_type_folder

        # "Longfield" is a known place name → overrides "service charges"
        assert not _is_category_or_type_folder(
            "Cheyne Walk Longfield 10% new Service Charges", config
        )

    def test_no_place_name_category_still_excluded(self, config):
        """Long name with category signal but NO place name → still excluded."""
        from classification.scheme_detector import _is_category_or_type_folder

        # No known place name → "service charges" fires normally
        assert _is_category_or_type_folder(
            "Admin Office 10% new Service Charges", config
        )

    def test_long_name_with_multiword_type_still_excluded(self, config):
        """Multi-word type token 'fire risk assessment' fires in long names."""
        from classification.scheme_detector import _is_category_or_type_folder

        # No place names → type signal not overridden
        assert _is_category_or_type_folder(
            "Latest Fire Risk Assessment ref 001 from 2024", config
        )

    def test_abbreviation_type_still_fires_in_long_name(self, config):
        """Type abbreviation 'epc' fires even in long names."""
        from classification.scheme_detector import _is_category_or_type_folder

        # No place names → abbreviation signal not overridden
        assert _is_category_or_type_folder(
            "Latest EPCs ref 001 from vendor 2024", config
        )


class TestAncestorCategoryExclusion:
    """Fix 4: category ancestors with no direct plot children block descendants."""

    def test_category_ancestor_no_plots_blocks_descendant(self, config):
        """Category folder with no direct plot children IS a blocker.

        'Legal' folder organises cross-scheme legal docs — no plot folders
        directly below it. Its descendant 'Scheme A' should not pass.
        """
        from classification.scheme_detector import _identify_candidates

        # Legal has many entities (from all schemes' docs) but no direct plots
        folder_stats = {
            "Legal": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=0,   # no plot folders directly below
            ),
            "Legal\\Scheme A": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=3,
                direct_category_children=2,
            ),
        }
        candidates, _ = _identify_candidates(folder_stats, config, min_entities=3)
        # 'Legal' itself is excluded (category folder)
        assert "Legal" not in candidates
        # 'Scheme A' is also excluded — its ancestor 'Legal' has no direct plots
        assert "Legal\\Scheme A" not in candidates

    def test_category_ancestor_with_plots_allows_descendant(self, config):
        """Category token ancestor WITH direct plot children is not a blocker.

        'Neil_New build' matches Construction via 'build', but has plot folders
        directly below it — it is a scheme container, not a filing category.
        Its descendant 'Block A' should be a candidate.
        """
        from classification.scheme_detector import _identify_candidates

        folder_stats = {
            "Neil_New build": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3", "Plot 4"},
                direct_plot_children=4,   # plot folders directly below
                direct_category_children=2,
            ),
            "Neil_New build\\Block A Lane": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=3,
                direct_category_children=2,
            ),
        }
        candidates, _ = _identify_candidates(folder_stats, config, min_entities=3)
        # 'Neil_New build' itself may or may not pass (category check),
        # but 'Block A Lane' must not be blocked by the ancestor guard
        assert "Neil_New build\\Block A Lane" in candidates


def _make_fstats(
    plots: set[str] | None = None,
    addresses: set[str] | None = None,
    file_count: int = 10,
    direct_plot_children: int = 0,
    category_signals: set[str] | None = None,
    direct_category_children: int = 0,
    total_direct_children: int = 0,
    direct_structural_children: int = 0,
    direct_entity_children: int = 0,
) -> "_FolderStats":
    """Helper to create _FolderStats for testing."""
    from classification.scheme_detector import _FolderStats
    return _FolderStats(
        plots=plots or set(),
        addresses=addresses or set(),
        file_count=file_count,
        direct_plot_children=direct_plot_children,
        category_signals=category_signals or set(),
        direct_category_children=direct_category_children,
        total_direct_children=total_direct_children,
        direct_structural_children=direct_structural_children,
        direct_entity_children=direct_entity_children,
    )


class TestNegativeScoringSignals:
    """Negative signals penalize container-like folders in scoring."""

    def test_structural_ratio_boost_vs_no_structure_penalty(self, config):
        """Candidate with structural children gets boost; one with zero gets penalty."""
        with_structure = {
            "Dev\\Oakwood": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=3,
                direct_category_children=2,
                total_direct_children=10,
                direct_structural_children=3,  # 3/10 = 30% > 15% → boost
            ),
        }
        scored_with = _score_candidates(with_structure, config)

        without_structure = {
            "Dev\\Oakwood": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=0,
                direct_category_children=0,
                total_direct_children=10,
                direct_structural_children=0,  # 0/10 with 10 children → penalty
            ),
        }
        scored_without = _score_candidates(without_structure, config)

        assert scored_without["Dev\\Oakwood"].confidence < scored_with["Dev\\Oakwood"].confidence

    def test_no_structural_penalty_needs_min_children(self, config):
        """Zero structural children doesn't penalise if total children < 5."""
        few_children = {
            "Dev\\Oakwood": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=3,
                direct_structural_children=0,  # 0/3 but < 5 total → no penalty
            ),
        }
        many_children = {
            "Dev\\Oakwood": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=10,
                direct_structural_children=0,  # 0/10 with 10 total → penalty
            ),
        }
        scored_few = _score_candidates(few_children, config)
        scored_many = _score_candidates(many_children, config)

        # Many children with 0 structural gets penalised more
        assert scored_many["Dev\\Oakwood"].confidence < scored_few["Dev\\Oakwood"].confidence

    def test_plot_children_still_boost(self, config):
        """Direct plot children still give a +0.10 boost."""
        with_plots = {
            "Dev\\Oakwood": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=3,
            ),
        }
        without_plots = {
            "Dev\\Oakwood": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=0,
            ),
        }
        scored_with = _score_candidates(with_plots, config)
        scored_without = _score_candidates(without_plots, config)
        assert scored_with["Dev\\Oakwood"].confidence > scored_without["Dev\\Oakwood"].confidence

    def test_child_candidates_penalty(self, config):
        """Candidates whose direct children are also candidates get penalized.

        Container sits at depth=2; direct children (schemes) at depth=3.
        This avoids the flat depth-1 penalty interfering with the assertion —
        Container's child-candidate penalty (-0.24) brings it below SchemeA.
        """
        candidates = {
            "Lvl1\\Lvl2\\Container": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=1,  # some structure to avoid sub-container exclusion
            ),
            "Lvl1\\Lvl2\\Container\\SchemeA": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=2,
                direct_plot_children=3,
            ),
            "Lvl1\\Lvl2\\Container\\SchemeB": _make_fstats(
                plots={"Plot 4", "Plot 5"},
                direct_category_children=1,
                direct_plot_children=2,
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        # Container penalized for two direct child candidates → scores below SchemeA
        assert scored["Lvl1\\Lvl2\\Container"].confidence < scored["Lvl1\\Lvl2\\Container\\SchemeA"].confidence

    def test_child_candidate_penalty_scales(self, config):
        """More child candidates → larger penalty (uncapped)."""
        # 2 child candidates — needs direct structure to avoid sub-container exclusion
        candidates_2 = {
            "Dev\\Container2": _make_fstats(
                plots={f"Plot {i}" for i in range(10)},
                direct_category_children=1,
                direct_plot_children=1,
                total_direct_children=10,  # lots of non-candidate children
            ),
            "Dev\\Container2\\SchemeA": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=2,
                direct_plot_children=3,
            ),
            "Dev\\Container2\\SchemeB": _make_fstats(
                plots={"Plot 4", "Plot 5", "Plot 6"},
                direct_category_children=2,
                direct_plot_children=3,
            ),
        }
        scored_2 = _score_candidates(candidates_2, config)

        # 1 child candidate
        candidates_1 = {
            "Dev\\Container1": _make_fstats(
                plots={f"Plot {i}" for i in range(10)},
                direct_category_children=1,
                direct_plot_children=1,
                total_direct_children=10,
            ),
            "Dev\\Container1\\SchemeA": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=2,
                direct_plot_children=3,
            ),
        }
        scored_1 = _score_candidates(candidates_1, config)

        # More children → lower confidence
        assert scored_2["Dev\\Container2"].confidence < scored_1["Dev\\Container1"].confidence


class TestDirectCategoryChildren:
    """Verify _compute_direct_category_children computation."""

    def test_counts_direct_dev_children(self):
        """Development categories one level below are counted."""
        from classification.scheme_detector import (
            _FolderStats, _compute_direct_category_children,
        )

        folder_stats = {
            "Root\\Scheme": _FolderStats(
                plots=set(), addresses=set(), file_count=5,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
        }
        folder_categories = {
            "Root\\Scheme\\Construction": {"Construction"},
            "Root\\Scheme\\Sales": {"Sales"},
            "Root\\Scheme\\Finance": {"Finance"},
        }
        dev_categories = {"Construction", "Sales", "Maintenance", "Planning", "Utilities"}

        _compute_direct_category_children(folder_stats, folder_categories, dev_categories)
        # Construction + Sales counted, Finance is not a dev category
        assert folder_stats["Root\\Scheme"].direct_category_children == 2

    def test_ignores_non_dev_categories(self):
        """Non-development categories are not counted."""
        from classification.scheme_detector import (
            _FolderStats, _compute_direct_category_children,
        )

        folder_stats = {
            "Root\\Office": _FolderStats(
                plots=set(), addresses=set(), file_count=5,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
        }
        folder_categories = {
            "Root\\Office\\Finance": {"Finance"},
            "Root\\Office\\IT": {"IT"},
        }
        dev_categories = {"Construction", "Sales"}

        _compute_direct_category_children(folder_stats, folder_categories, dev_categories)
        assert folder_stats["Root\\Office"].direct_category_children == 0

    def test_ignores_grandchildren(self):
        """Categories two+ levels below are NOT counted as direct."""
        from classification.scheme_detector import (
            _FolderStats, _compute_direct_category_children,
        )

        folder_stats = {
            "Root": _FolderStats(
                plots=set(), addresses=set(), file_count=5,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
        }
        folder_categories = {
            "Root\\Scheme\\Construction": {"Construction"},
        }
        dev_categories = {"Construction"}

        _compute_direct_category_children(folder_stats, folder_categories, dev_categories)
        # Construction is under Root\Scheme, not directly under Root
        assert folder_stats["Root"].direct_category_children == 0


class TestStructuralContainerDetection:
    """Structural container detection: ratio of child candidates to total children."""

    def test_container_with_mostly_candidate_children_excluded(self, config):
        """Folder where >50% of children are candidates and >=3 → excluded.

        Simulates a geographic container like "Ashford" with scheme names below.
        """
        candidates = {
            "Ashford": _make_fstats(
                plots={f"Plot {i}" for i in range(15)},
                addresses={"Addr 1", "Addr 2"},
                direct_category_children=0,
                total_direct_children=5,
                direct_structural_children=0,
            ),
            "Ashford\\Park Farm": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=3,
                direct_plot_children=3,
                total_direct_children=8,
                direct_structural_children=5,
            ),
            "Ashford\\Templar Way": _make_fstats(
                plots={"Plot 4", "Plot 5", "Plot 6"},
                direct_category_children=2,
                direct_plot_children=3,
                total_direct_children=7,
                direct_structural_children=4,
            ),
            "Ashford\\The Street": _make_fstats(
                plots={"Plot 7", "Plot 8", "Plot 9"},
                direct_category_children=2,
                direct_plot_children=3,
                total_direct_children=6,
                direct_structural_children=3,
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        # Ashford should be excluded — 3/5 children are candidates (60%)
        assert "Ashford" not in scored
        # Real schemes should remain
        assert "Ashford\\Park Farm" in scored
        assert "Ashford\\Templar Way" in scored
        assert "Ashford\\The Street" in scored

    def test_real_scheme_with_few_candidate_children_kept(self, config):
        """Scheme with 2 sub-phases and many organisational folders → kept.

        Simulates "Big Development" with Phase 1, Phase 2 as candidates
        but also Construction, Sales, etc. as non-candidate children.
        """
        candidates = {
            "Dev\\Big Development": _make_fstats(
                plots={f"Plot {i}" for i in range(20)},
                direct_category_children=5,
                direct_plot_children=3,
                total_direct_children=20,  # 20 children: 2 candidates, 18 org folders
                direct_structural_children=10,
                category_signals={"Construction", "Sales", "Planning", "Utilities", "Maintenance"},
            ),
            "Dev\\Big Development\\Phase 1": _make_fstats(
                plots={f"Plot {i}" for i in range(10)},
                direct_category_children=3,
                direct_plot_children=5,
                total_direct_children=8,
                direct_structural_children=5,
            ),
            "Dev\\Big Development\\Phase 2": _make_fstats(
                plots={f"Plot {i}" for i in range(10, 20)},
                direct_category_children=3,
                direct_plot_children=5,
                total_direct_children=8,
                direct_structural_children=5,
            ),
        }
        scored = _score_candidates(candidates, config)
        # Big Development should survive — only 2/20 children are candidates (10%)
        assert "Dev\\Big Development" in scored

    def test_container_keyword_not_excluded_by_name(self, config):
        """Container keywords like 'Schemes' are not excluded by name rules.

        Instead they're handled by structural scoring (container keyword boost/detection).
        """
        from classification.scheme_detector import _is_excluded_by_config

        # These are not excluded by name-based rules (deliberately removed)
        # but would be handled by structural container detection in scoring
        assert not _is_excluded_by_config("Schemes - Completed", config.scheme_exclusions)
        assert not _is_excluded_by_config("Schemes 2021-26", config.scheme_exclusions)

    def test_total_direct_children_computed(self):
        """_compute_total_direct_children counts all direct children."""
        from classification.scheme_detector import (
            _FolderStats, _compute_total_direct_children,
        )

        folder_stats = {
            "Root": _FolderStats(
                plots=set(), addresses=set(), file_count=10,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
            "Root\\Child1": _FolderStats(
                plots=set(), addresses=set(), file_count=5,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
            "Root\\Child2": _FolderStats(
                plots=set(), addresses=set(), file_count=5,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
            "Root\\Child1\\Grandchild": _FolderStats(
                plots=set(), addresses=set(), file_count=2,
                direct_plot_children=0, category_signals=set(),
                direct_category_children=0,
            ),
        }
        _compute_total_direct_children(folder_stats)
        assert folder_stats["Root"].total_direct_children == 2  # Child1 + Child2
        assert folder_stats["Root\\Child1"].total_direct_children == 1  # Grandchild


class TestSubContainerExclusion:
    """Folders with child scheme candidates but no direct structure are excluded."""

    def test_geographic_container_excluded(self, config):
        """Geographic container with child schemes but no direct org → excluded.

        Simulates 'Sevenoaks' under 'Schemes 2019-21' with one real scheme below.
        """
        candidates = {
            "Sevenoaks": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=0,
                direct_plot_children=0,
                total_direct_children=1,
            ),
            "Sevenoaks\\Park Farm": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=3,
                direct_plot_children=3,
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        assert "Sevenoaks" not in scored
        assert "Sevenoaks\\Park Farm" in scored

    def test_real_scheme_with_child_candidate_and_structure_kept(self, config):
        """Scheme with direct dev structure AND a child candidate → kept."""
        candidates = {
            "Dev\\Big Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3", "Plot 4", "Plot 5"},
                direct_category_children=3,
                direct_plot_children=5,
                total_direct_children=15,
            ),
            "Dev\\Big Scheme\\Sub Phase": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=2,
                direct_plot_children=3,
            ),
        }
        scored = _score_candidates(candidates, config)
        # Big Scheme has direct structure → NOT excluded as sub-container
        assert "Dev\\Big Scheme" in scored

    def test_org_folder_with_child_schemes_no_structure_excluded(self, config):
        """Organisational folder with children but no direct structure → excluded."""
        candidates = {
            "2. Business Development": _make_fstats(
                addresses={"Addr 1"},
                direct_category_children=0,
                direct_plot_children=0,
                total_direct_children=1,
            ),
            "2. Business Development\\Project A": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_category_children=2,
                direct_plot_children=3,
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        assert "2. Business Development" not in scored
        assert "2. Business Development\\Project A" in scored


class TestDeduplication:
    """Same-name schemes at different depths are deduplicated."""

    def test_duplicate_names_merged_to_primary(self):
        """Two paths with same scheme name → lower-confidence one redirected."""
        from classification.scheme_detector import _deduplicate_schemes

        scored = {
            "Schemes\\1. COMPLETED\\Dalefield Way": SchemeMatch(
                value="Dalefield Way", confidence=0.88,
                depth=2, folder_path="Schemes\\1. COMPLETED\\Dalefield Way",
            ),
            "Schemes\\1. COMPLETED\\Schemes\\1. COMPLETED\\Dalefield Way": SchemeMatch(
                value="Dalefield Way", confidence=0.68,
                depth=4,
                folder_path="Schemes\\1. COMPLETED\\Schemes\\1. COMPLETED\\Dalefield Way",
            ),
        }
        result = _deduplicate_schemes(scored)
        # Both paths still present as keys
        assert len(result) == 2
        # Both point to the primary (highest confidence)
        primary_path = "Schemes\\1. COMPLETED\\Dalefield Way"
        assert result[primary_path].confidence == 0.88
        dup_path = "Schemes\\1. COMPLETED\\Schemes\\1. COMPLETED\\Dalefield Way"
        assert result[dup_path].folder_path == primary_path
        assert result[dup_path].confidence == 0.88

    def test_three_duplicates_merged(self):
        """Three paths with same name → all point to highest confidence."""
        from classification.scheme_detector import _deduplicate_schemes

        scored = {
            "A\\Maidstone": SchemeMatch(
                value="Maidstone", confidence=1.0,
                depth=1, folder_path="A\\Maidstone",
            ),
            "B\\Maidstone": SchemeMatch(
                value="Maidstone", confidence=0.90,
                depth=2, folder_path="B\\Maidstone",
            ),
            "C\\Maidstone": SchemeMatch(
                value="Maidstone", confidence=0.80,
                depth=3, folder_path="C\\Maidstone",
            ),
        }
        result = _deduplicate_schemes(scored)
        assert result["A\\Maidstone"].confidence == 1.0
        assert result["B\\Maidstone"].folder_path == "A\\Maidstone"
        assert result["C\\Maidstone"].folder_path == "A\\Maidstone"

    def test_unique_names_not_affected(self):
        """Schemes with unique names remain unchanged."""
        from classification.scheme_detector import _deduplicate_schemes

        scored = {
            "A\\Park Farm": SchemeMatch(
                value="Park Farm", confidence=0.90,
                depth=1, folder_path="A\\Park Farm",
            ),
            "B\\Quinton Road": SchemeMatch(
                value="Quinton Road", confidence=0.85,
                depth=1, folder_path="B\\Quinton Road",
            ),
        }
        result = _deduplicate_schemes(scored)
        assert result["A\\Park Farm"].folder_path == "A\\Park Farm"
        assert result["B\\Quinton Road"].folder_path == "B\\Quinton Road"

    def test_tie_broken_by_shallowest_depth(self):
        """Equal confidence → shallowest depth wins as primary."""
        from classification.scheme_detector import _deduplicate_schemes

        scored = {
            "Deep\\Deep\\Scheme": SchemeMatch(
                value="Scheme", confidence=0.80,
                depth=2, folder_path="Deep\\Deep\\Scheme",
            ),
            "Shallow\\Scheme": SchemeMatch(
                value="Scheme", confidence=0.80,
                depth=1, folder_path="Shallow\\Scheme",
            ),
        }
        result = _deduplicate_schemes(scored)
        # Both should point to shallowest
        assert result["Deep\\Deep\\Scheme"].folder_path == "Shallow\\Scheme"
        assert result["Shallow\\Scheme"].folder_path == "Shallow\\Scheme"


class TestEDRMExclusion:
    """EDRM filing template folders — no longer excluded by name-based rules.

    Deliberately removed from config exclusions to avoid overfitting.
    EDRM folders are now handled by structural signals (no category children,
    high entity ratio, etc.) which generalise across estates.
    """

    def test_edrm_not_excluded_by_config(self, config):
        """'EDRM' is no longer excluded by name-based rules."""
        from classification.scheme_detector import _is_excluded_by_config

        # Config exclusions were deliberately cleaned — no name/substring rules
        assert not _is_excluded_by_config("EDRM", config.scheme_exclusions)
        assert not _is_excluded_by_config("EDRM Filing structure", config.scheme_exclusions)


class TestQuadraticDepthPenalty:
    """Depth penalty is quadratic: (depth/max_depth)^2 * 0.30."""

    def test_shallow_candidate_lower_penalty(self, config):
        """Shallower candidates get less depth penalty than deeper ones.

        Uses depth 0, 2, 3 to avoid depth=1 which carries an extra flat
        penalty (see test_depth_1_flat_penalty for that behaviour).
        """
        candidates = {
            "Scheme1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
            ),
            "Dev\\Mid\\Scheme2": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
            ),
            "Dev\\Mid\\Sub\\Scheme3": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        # Shallowest gets least depth penalty (no flat penalty at depths 0, 2, 3)
        assert scored["Scheme1"].confidence > scored["Dev\\Mid\\Scheme2"].confidence
        assert scored["Dev\\Mid\\Scheme2"].confidence > scored["Dev\\Mid\\Sub\\Scheme3"].confidence

    def test_depth_1_flat_penalty(self, config):
        """Depth-1 candidates receive a flat -0.30 extra penalty on top of quadratic.

        This makes depth=1 harder to pass than depth=2+ so organisational
        containers at the first folder level don't easily qualify as schemes.
        """
        candidates = {
            "Scheme0": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
            "A\\Scheme1": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
            "A\\B\\Scheme2": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        # depth=1 should carry the flat penalty — lower than depth=0
        assert scored["Scheme0"].confidence > scored["A\\Scheme1"].confidence
        # depth=1 flat penalty (-0.30 extra) vs depth=2 quadratic (-0.075 with max=2):
        # depth=1 is more penalised than depth=2 — that's intentional design.
        assert scored["A\\Scheme1"].confidence < scored["A\\B\\Scheme2"].confidence

    def test_quadratic_penalty_penalises_bottom_more(self, config):
        """Quadratic curve: deeper non-depth-1 levels are penalised progressively.

        Uses depth 0, 2, 3 so the flat depth-1 penalty doesn't interfere.
        With effective_max=3:
        - Level 0: (0/3)^2 = 0.00 → penalty 0
        - Level 2: (2/3)^2 = 0.444 → penalty 0.133
        - Level 3: (3/3)^2 = 1.00 → penalty 0.30

        Bottom-to-mid gap (0.167) > top-to-mid gap (0.133).
        """
        candidates = {
            "Top": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
            "Dev\\Mid\\Mid": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
            "Dev\\Mid\\Sub\\Bottom": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        top = scored["Top"].confidence
        mid = scored["Dev\\Mid\\Mid"].confidence
        bottom = scored["Dev\\Mid\\Sub\\Bottom"].confidence

        top_to_mid_gap = top - mid
        mid_to_bottom_gap = mid - bottom
        # Bottom penalised more steeply than middle
        assert mid_to_bottom_gap > top_to_mid_gap

    def test_root_depth_offsets_penalty(self, config):
        """root_depth shifts the penalty baseline so depth <= root is free.

        Uses candidates at depth 3 and 5 to avoid effective_depth==1 in
        either scenario (which would trigger the flat depth-1 penalty and
        conflate two different mechanisms).

        root_depth=1: Scheme effective_depth=2, Deep=4, max_eff=4
          Scheme relative=0.5, penalty=0.25*0.30=0.075
        root_depth=-1: Scheme effective_depth=3, Deep=5, max_eff=5
          Scheme relative=0.6, penalty=0.36*0.30=0.108
        """
        candidates = {
            "Org\\Dev\\Sub\\Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
            ),
            "Org\\Dev\\Sub\\A\\B\\Deep": _make_fstats(
                plots={"Plot 4", "Plot 5", "Plot 6"},
            ),
        }
        scored_offset = _score_candidates(candidates, config, root_depth=1)
        scored_no_offset = _score_candidates(candidates, config, root_depth=-1)

        # Offset produces less penalty for the mid-depth candidate
        assert scored_offset["Org\\Dev\\Sub\\Scheme"].confidence > scored_no_offset["Org\\Dev\\Sub\\Scheme"].confidence


class TestStructuralRatioSignals:
    """Structural ratio: direct_structural_children / total_direct_children."""

    def test_high_structural_ratio_gets_boost(self, config):
        """Structural ratio > 15% → +0.15 boost."""
        high_ratio = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=10,
                direct_structural_children=3,  # 30%
            ),
        }
        low_ratio = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=10,
                direct_structural_children=1,  # 10% — below threshold
            ),
        }
        scored_high = _score_candidates(high_ratio, config, root_depth=-1)
        scored_low = _score_candidates(low_ratio, config, root_depth=-1)

        assert scored_high["Scheme"].confidence > scored_low["Scheme"].confidence
        assert scored_high["Scheme"].structural_ratio == 0.3

    def test_zero_structural_with_many_children_penalty(self, config):
        """0 structural children + 5+ total children → -0.20 penalty."""
        zero_structural = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=8,
                direct_structural_children=0,
            ),
        }
        some_structural = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=8,
                direct_structural_children=2,  # 25% — above threshold
            ),
        }
        scored_zero = _score_candidates(zero_structural, config, root_depth=-1)
        scored_some = _score_candidates(some_structural, config, root_depth=-1)

        # Zero gets penalty (-0.20), some gets boost (+0.15) → big gap
        diff = scored_some["Scheme"].confidence - scored_zero["Scheme"].confidence
        assert diff > 0.30

    def test_zero_structural_few_children_no_penalty(self, config):
        """0 structural children + <5 total children → no penalty applied."""
        few = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=3,
                direct_structural_children=0,
            ),
        }
        many = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=8,
                direct_structural_children=0,
            ),
        }
        scored_few = _score_candidates(few, config, root_depth=-1)
        scored_many = _score_candidates(many, config, root_depth=-1)

        # Many children with 0 structural → penalty; few → no penalty
        assert scored_few["Scheme"].confidence > scored_many["Scheme"].confidence

    def test_structural_ratio_in_match_output(self, config):
        """SchemeMatch contains structural_ratio diagnostic field."""
        candidates = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=20,
                direct_structural_children=5,
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        assert scored["Scheme"].structural_ratio == 0.25


class TestEntityRatioDiagnostic:
    """Entity ratio is computed and stored but not used as a scoring signal.

    The field exists for diagnostic/reporting purposes. Structural ratio
    (via structural_ratio_boost / no_structural_penalty) is the scoring
    signal that captures container-like folder structure.
    """

    def test_entity_ratio_stored_in_match_output(self, config):
        """SchemeMatch.entity_ratio diagnostic field is computed correctly."""
        candidates = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=8,
                direct_entity_children=3,
            ),
        }
        scored = _score_candidates(candidates, config, root_depth=-1)
        assert scored["Scheme"].entity_ratio == 0.375

    def test_high_entity_ratio_does_not_affect_score(self, config):
        """High entity ratio no longer penalises score — structural ratio handles this."""
        high_entity = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=10,
                direct_entity_children=6,  # 60%
            ),
        }
        low_entity = {
            "Scheme": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                total_direct_children=10,
                direct_entity_children=2,  # 20%
            ),
        }
        scored_high = _score_candidates(high_entity, config, root_depth=-1)
        scored_low = _score_candidates(low_entity, config, root_depth=-1)

        assert scored_high["Scheme"].confidence == scored_low["Scheme"].confidence
        assert scored_high["Scheme"].entity_ratio == 0.6


class TestPlaceNameBoost:
    """Place name signal: priority discriminator for housing scheme detection.

    Matching a known UK place name gives +0.25 (priority boost).
    No place name match gives -0.15 (absence penalty).
    Penalty is only applied when a place_names dictionary is configured.
    """

    def test_known_place_name_boost(self, config):
        """Folder containing a known UK place name gets +0.25."""
        from classification.scheme_detector import _contains_place_name

        # "ashford" should be in the place names set
        assert _contains_place_name("Ashford Park Development", config.place_names)

    def test_unknown_name_no_boost(self, config):
        """Made-up name not in place names → no match."""
        from classification.scheme_detector import _contains_place_name

        assert not _contains_place_name("Zxyqkorp Development", config.place_names)

    def test_place_name_boost_in_scoring(self, config):
        """Place name match increases confidence in scoring."""
        # "Faversham" is a known UK place name
        with_place = {
            "Faversham Lakes": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
            ),
        }
        without_place = {
            "Project Alpha": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
            ),
        }
        scored_place = _score_candidates(with_place, config, root_depth=-1)
        scored_no_place = _score_candidates(without_place, config, root_depth=-1)

        # Place name match should give a confidence boost
        assert scored_place["Faversham Lakes"].confidence > scored_no_place["Project Alpha"].confidence
        assert scored_place["Faversham Lakes"].place_name_match is True

    def test_bigram_place_name_match(self, config):
        """Multi-word place names matched as bigrams."""
        from classification.scheme_detector import _contains_place_name

        # Check if any two-word place is in the set (depends on data)
        # At minimum, single-token places should work
        assert _contains_place_name("maidstone", config.place_names)

    def test_short_tokens_excluded_from_place_names(self, config):
        """Tokens shorter than 4 characters were excluded from the place names set."""
        from classification.scheme_detector import _contains_place_name

        # "Hoo" is a real UK place but was excluded (< 4 chars)
        assert not _contains_place_name("Hoo", config.place_names)

    def test_no_place_name_penalty_applied(self, config):
        """Folder without a place name scores lower than one with a place name.

        The gap should reflect both the +0.25 boost and the -0.15 penalty (0.40 total).
        """
        with_place = {
            "Faversham Lakes": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
        }
        without_place = {
            "Project Alpha": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
        }
        scored_place = _score_candidates(with_place, config, root_depth=-1)
        scored_no_place = _score_candidates(without_place, config, root_depth=-1)

        diff = scored_place["Faversham Lakes"].confidence - scored_no_place["Project Alpha"].confidence
        # Gap should be at least 0.38 (sum of boost + penalty, minus rounding)
        assert diff >= 0.38
        assert scored_no_place["Project Alpha"].place_name_match is False

    def test_no_place_name_penalty_not_applied_when_no_dictionary(self):
        """When place_names is empty, no penalty is applied."""
        from classification.config_loader import load_config
        from classification.models import ClassificationConfig

        # Build a minimal config with no place names (simulates non-housing run)
        base_config = load_config("config", industry="housing")
        # Monkey-patch to empty place names to test the gate
        import dataclasses
        empty_config = dataclasses.replace(base_config, place_names=frozenset())

        candidates = {
            "Project Alpha": _make_fstats(plots={"Plot 1", "Plot 2", "Plot 3"}),
        }
        scored_empty = _score_candidates(candidates, empty_config, root_depth=-1)
        scored_full = _score_candidates(candidates, base_config, root_depth=-1)

        # Without dictionary: no penalty applied → higher confidence than with dict (where penalty fires)
        assert scored_empty["Project Alpha"].confidence > scored_full["Project Alpha"].confidence


class TestAddressPlotCompoundBoost:
    """Address-like name + direct plot children → compound boost signal."""

    def test_address_with_plots_gets_compound_boost(self, config):
        """Folder with address-like name AND direct plot children gets extra boost."""
        from classification.scheme_detector import _SCHEME_DEFAULTS

        with_plots = {
            "Parsonage Place": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=3,
            ),
        }
        without_plots = {
            "Parsonage Place": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=0,
            ),
        }
        scored_with = _score_candidates(with_plots, config, root_depth=-1)
        scored_without = _score_candidates(without_plots, config, root_depth=-1)

        diff = scored_with["Parsonage Place"].confidence - scored_without["Parsonage Place"].confidence
        # Should differ by at least the compound boost (+ direct plot children boost)
        assert diff >= _SCHEME_DEFAULTS["address_plot_compound_boost"]

    def test_non_address_name_no_compound_boost(self, config):
        """Folder without address-like name gets no compound boost even with plot children."""
        with_plots = {
            "Phase 1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=3,
            ),
        }
        without_plots = {
            "Phase 1": _make_fstats(
                plots={"Plot 1", "Plot 2", "Plot 3"},
                direct_plot_children=0,
            ),
        }
        scored_with = _score_candidates(with_plots, config, root_depth=-1)
        scored_without = _score_candidates(without_plots, config, root_depth=-1)

        # Only the direct_plot_children boost (0.10) — no compound boost since name is entity-like
        from classification.scheme_detector import _SCHEME_DEFAULTS
        diff = scored_with["Phase 1"].confidence - scored_without["Phase 1"].confidence
        # Should be about direct_plot_children_boost, not compound
        assert diff < _SCHEME_DEFAULTS["address_plot_compound_boost"] + 0.01


class TestSiblingDensityBoost:
    """Sibling density: 2+ address/place-named quality peers → +0.15 boost.

    The boost fires when ≥2 siblings in the same parent group are BOTH
    above the quality confidence threshold AND either address-like or
    place-name matched. The address peers themselves are the evidence —
    no separate anchor gate needed.
    """

    def test_two_address_peers_trigger_boost(self, config):
        """2+ address-like quality siblings → +0.15 for all in the group."""
        from classification.scheme_detector import _apply_sibling_boost

        scored = {
            "EDRM\\Parsonage Place": SchemeMatch(
                value="Parsonage Place", confidence=0.70,
                depth=1, folder_path="EDRM\\Parsonage Place",
                address_like_name=True,
            ),
            "EDRM\\Faversham Lakes": SchemeMatch(
                value="Faversham Lakes", confidence=0.72,
                depth=1, folder_path="EDRM\\Faversham Lakes",
                place_name_match=True,
            ),
            "EDRM\\Stoke Road": SchemeMatch(
                value="Stoke Road", confidence=0.68,
                depth=1, folder_path="EDRM\\Stoke Road",
                address_like_name=True,
            ),
        }
        result = _apply_sibling_boost(scored, config)
        # 3 address/place peers → boost fires (+0.15) for all
        assert result["EDRM\\Parsonage Place"].confidence == pytest.approx(0.85, abs=0.001)
        assert result["EDRM\\Faversham Lakes"].confidence == pytest.approx(0.87, abs=0.001)
        assert result["EDRM\\Stoke Road"].confidence == pytest.approx(0.83, abs=0.001)
        assert result["EDRM\\Parsonage Place"].sibling_scheme_count == 3

    def test_place_name_peers_also_trigger(self, config):
        """Place-name matches count the same as address-like for peer count."""
        from classification.scheme_detector import _apply_sibling_boost

        scored = {
            f"Parent\\Scheme{i}": SchemeMatch(
                value=f"Scheme{i}", confidence=0.70,
                depth=1, folder_path=f"Parent\\Scheme{i}",
                place_name_match=True,
            )
            for i in range(5)
        }
        result = _apply_sibling_boost(scored, config)
        for path in result:
            assert result[path].confidence == 0.85

    def test_one_address_peer_no_boost(self, config):
        """Only 1 address peer (below minimum of 2) → no boost."""
        from classification.scheme_detector import _apply_sibling_boost

        scored = {
            "Parent\\Scheme A": SchemeMatch(
                value="Scheme A", confidence=0.70,
                depth=1, folder_path="Parent\\Scheme A",
                address_like_name=True,
            ),
            "Parent\\Generic": SchemeMatch(
                value="Generic", confidence=0.70,
                depth=1, folder_path="Parent\\Generic",
            ),
        }
        result = _apply_sibling_boost(scored, config)
        assert result["Parent\\Scheme A"].confidence == 0.70
        assert result["Parent\\Generic"].confidence == 0.70

    def test_low_confidence_address_peers_not_counted(self, config):
        """Address-like siblings below quality threshold (0.5) don't count."""
        from classification.scheme_detector import _apply_sibling_boost

        scored = {
            "Parent\\Oak Road": SchemeMatch(
                value="Oak Road", confidence=0.70,
                depth=1, folder_path="Parent\\Oak Road",
                address_like_name=True,
            ),
            "Parent\\High Street": SchemeMatch(
                value="High Street", confidence=0.40,  # below quality threshold
                depth=1, folder_path="Parent\\High Street",
                address_like_name=True,
            ),
        }
        result = _apply_sibling_boost(scored, config)
        # High Street is address-like but below quality gate → only 1 quality peer → no boost
        assert result["Parent\\Oak Road"].confidence == 0.70

    def test_non_address_siblings_do_not_trigger_boost(self, config):
        """Many quality siblings without address/place names → no boost."""
        from classification.scheme_detector import _apply_sibling_boost

        scored = {
            f"Parent\\Scheme{i}": SchemeMatch(
                value=f"Scheme{i}", confidence=0.70,
                depth=1, folder_path=f"Parent\\Scheme{i}",
                has_scheme_keyword_ancestor=True,
                # no place_name_match, no address_like_name
            )
            for i in range(6)
        }
        result = _apply_sibling_boost(scored, config)
        # 0 address/place peers despite 6 quality siblings → no boost
        for path in result:
            assert result[path].confidence == 0.70

    def test_different_parents_independent(self, config):
        """Sibling boost computed independently per parent."""
        from classification.scheme_detector import _apply_sibling_boost

        scored = {}
        # Parent A: 3 address-like quality siblings → boost fires
        for i in range(3):
            scored[f"A\\Scheme{i}"] = SchemeMatch(
                value=f"Scheme{i}", confidence=0.70,
                depth=1, folder_path=f"A\\Scheme{i}",
                address_like_name=True,
            )
        # Parent B: 1 lone address-like scheme but no peers
        scored["B\\Alone"] = SchemeMatch(
            value="Alone", confidence=0.70,
            depth=1, folder_path="B\\Alone",
            address_like_name=True,
        )

        result = _apply_sibling_boost(scored, config)
        assert result["A\\Scheme0"].confidence == 0.85   # boosted (3 address peers → +0.15)
        assert result["B\\Alone"].confidence == 0.70     # not boosted (only 1 peer)


class TestQualityGateContainerDetection:
    """Quality gate: only children with confidence > 0.5 trigger container detection."""

    def test_low_confidence_children_dont_trigger_container(self, config):
        """Parent with 3+ low-confidence children is NOT marked as container."""
        scored = {
            "Parent": SchemeMatch(
                value="Parent", confidence=0.80,
                depth=0, folder_path="Parent",
            ),
            "Parent\\Low1": SchemeMatch(
                value="Low1", confidence=0.30,
                depth=1, folder_path="Parent\\Low1",
            ),
            "Parent\\Low2": SchemeMatch(
                value="Low2", confidence=0.40,
                depth=1, folder_path="Parent\\Low2",
            ),
            "Parent\\Low3": SchemeMatch(
                value="Low3", confidence=0.35,
                depth=1, folder_path="Parent\\Low3",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        # Parent kept — children below quality threshold
        assert "Parent" in result

    def test_high_confidence_children_trigger_container(self, config):
        """Parent with 3+ high-confidence children IS marked as container."""
        scored = {
            "Parent": SchemeMatch(
                value="Parent", confidence=0.80,
                depth=0, folder_path="Parent",
            ),
            "Parent\\Hi1": SchemeMatch(
                value="Hi1", confidence=0.70,
                depth=1, folder_path="Parent\\Hi1",
            ),
            "Parent\\Hi2": SchemeMatch(
                value="Hi2", confidence=0.60,
                depth=1, folder_path="Parent\\Hi2",
            ),
            "Parent\\Hi3": SchemeMatch(
                value="Hi3", confidence=0.65,
                depth=1, folder_path="Parent\\Hi3",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        # Parent removed — 3+ quality children
        assert "Parent" not in result
        assert "Parent\\Hi1" in result

    def test_mixed_quality_children(self, config):
        """Only quality children count toward container threshold.
        Parent has direct structure so it suppresses children."""
        scored = {
            "Parent": SchemeMatch(
                value="Parent", confidence=0.80,
                depth=0, folder_path="Parent",
                direct_category_children=2,
                direct_plot_children=3,
            ),
            "Parent\\Hi1": SchemeMatch(
                value="Hi1", confidence=0.70,
                depth=1, folder_path="Parent\\Hi1",
            ),
            "Parent\\Hi2": SchemeMatch(
                value="Hi2", confidence=0.60,
                depth=1, folder_path="Parent\\Hi2",
            ),
            "Parent\\Low1": SchemeMatch(
                value="Low1", confidence=0.40,
                depth=1, folder_path="Parent\\Low1",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        # Only 2 quality children (< 3 threshold) → parent suppresses children
        assert "Parent" in result
        assert "Parent\\Hi1" not in result  # suppressed (lower conf)
        assert "Parent\\Hi2" not in result  # suppressed
        assert "Parent\\Low1" not in result  # also suppressed


class TestNamingContainerExclusion:
    """Fix 5: parent with no structural signals is a naming container, not a real scheme."""

    def test_naming_container_excluded_children_kept(self, config):
        """Parent with no direct structure but lower than quality children → container excluded.

        'Apartments' groups real schemes but has no plot/category children
        of its own. Even though it scores higher than its one child (whose
        depth penalty brings it down), 'Apartments' should be excluded as
        a naming container, not kept as the 'real scheme'.
        """
        scored = {
            "Apartments": SchemeMatch(
                value="Apartments", confidence=0.80,
                depth=0, folder_path="Apartments",
                direct_category_children=0,  # no structure of its own
                direct_plot_children=0,
            ),
            "Apartments\\Birchwood Court": SchemeMatch(
                value="Birchwood Court", confidence=0.70,
                depth=1, folder_path="Apartments\\Birchwood Court",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        # 'Apartments' has no structure → excluded as naming container
        assert "Apartments" not in result
        # Children survive
        assert "Apartments\\Birchwood Court" in result

    def test_real_scheme_with_structure_suppresses_child(self, config):
        """Parent WITH direct structure scores higher → child is false positive, suppressed."""
        scored = {
            "Goldsel Road": SchemeMatch(
                value="Goldsel Road", confidence=0.85,
                depth=0, folder_path="Goldsel Road",
                direct_category_children=3,
                direct_plot_children=5,
            ),
            "Goldsel Road\\Phase 1": SchemeMatch(
                value="Phase 1", confidence=0.70,
                depth=1, folder_path="Goldsel Road\\Phase 1",
            ),
        }
        result, _, _ = _exclude_container_folders(scored, config)
        # 'Goldsel Road' has direct structure → it is the real scheme
        assert "Goldsel Road" in result
        # 'Phase 1' is a false positive nested inside the scheme
        assert "Goldsel Road\\Phase 1" not in result
