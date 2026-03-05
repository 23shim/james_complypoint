"""Tests for fuzzy matching index and variant generation."""

import pytest

from classification.fuzzy import _edit_distance_1_variants, build_fuzzy_index
from classification.config_loader import load_config


CONFIG_DIR = "config"


@pytest.fixture
def config():
    return load_config(CONFIG_DIR, industry="housing")


# ---- _edit_distance_1_variants ----


class TestEditDistance1Variants:

    def test_known_deletion(self):
        variants = _edit_distance_1_variants("hello")
        # "helo" is hello with one 'l' removed
        assert "helo" in variants

    def test_known_insertion(self):
        variants = _edit_distance_1_variants("hello")
        # "helllo" is hello with extra 'l'
        assert "helllo" in variants

    def test_known_substitution(self):
        variants = _edit_distance_1_variants("hello")
        # "hallo" is hello with e→a
        assert "hallo" in variants

    def test_known_transposition(self):
        variants = _edit_distance_1_variants("hello")
        # "hlelo" is hello with e and l swapped
        assert "hlelo" in variants

    def test_original_excluded(self):
        variants = _edit_distance_1_variants("hello")
        assert "hello" not in variants

    def test_empty_string(self):
        variants = _edit_distance_1_variants("")
        # Only insertions of single chars
        assert len(variants) == 26

    def test_single_char(self):
        variants = _edit_distance_1_variants("a")
        # Deletions: "" (1), Insertions: 26*2=52, Substitutions: 25, Transpositions: 0
        # Minus original: won't be there
        assert "" in variants  # deletion
        assert "ba" in variants  # insertion
        assert "b" in variants  # substitution

    def test_typo_assesment(self):
        """'assesment' should be a variant of 'assessment' (missing 's')."""
        variants = _edit_distance_1_variants("assessment")
        assert "assesment" in variants

    def test_typo_certifcate(self):
        """'certifcate' should be a variant of 'certificate' (transposition)."""
        variants = _edit_distance_1_variants("certificate")
        assert "certifcate" in variants

    def test_typo_saftey(self):
        """'saftey' should be a variant of 'safety' (transposition)."""
        variants = _edit_distance_1_variants("safety")
        assert "saftey" in variants


# ---- build_fuzzy_index ----


class TestBuildFuzzyIndex:

    def test_index_not_empty(self, config):
        assert len(config.fuzzy_index) > 0

    def test_common_typo_maps_to_canonical(self, config):
        # "assesment" (missing 's') should map to "assessment"
        # if "assessment" is a dictionary word >= 5 chars
        assert config.fuzzy_index.get("assesment") == "assessment"

    def test_short_words_excluded(self, config):
        # Words < 5 chars should NOT generate fuzzy variants.
        # "sale" (4 chars) is in MOS tokens ("memorandum of sale").
        # "sal" should not map to "sale" via the fuzzy index.
        assert "sal" not in config.fuzzy_index

    def test_canonical_words_not_remapped(self, config):
        # "invoice" is a canonical dictionary word — it should NOT
        # appear as a variant pointing to some other word.
        assert "invoice" not in config.fuzzy_index

    def test_assessment_is_canonical(self, config):
        # "assessment" itself should not be in the index (it's canonical).
        assert "assessment" not in config.fuzzy_index

    def test_saftey_maps_to_safety(self, config):
        # "safety" is in Health & Safety category signals, so it's
        # a canonical word. "saftey" (transposition) should map to it.
        assert config.fuzzy_index.get("saftey") == "safety"

    def test_index_values_are_strings(self, config):
        for variant, canonical in config.fuzzy_index.items():
            assert isinstance(variant, str)
            assert isinstance(canonical, str)
            assert len(canonical) >= 5
