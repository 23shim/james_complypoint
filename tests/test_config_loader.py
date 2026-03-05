"""Tests for classification config loading and merging."""

import pytest

from classification.config_loader import load_config


CONFIG_DIR = "config"


class TestLoadConfig:

    def test_loads_base_and_housing(self):
        config = load_config(CONFIG_DIR, industry="housing")
        # Should have base types + housing types
        assert "Invoice" in config.types
        assert "MOS" in config.types
        assert "EICR" in config.types

    def test_base_categories_present(self):
        config = load_config(CONFIG_DIR, industry="housing")
        assert "Finance" in config.categories
        assert "Legal" in config.categories
        assert "HR" in config.categories

    def test_industry_categories_added(self):
        config = load_config(CONFIG_DIR, industry="housing")
        assert "Construction" in config.categories
        assert "Sales" in config.categories
        assert "Planning" in config.categories

    def test_category_signals_are_lowercase(self):
        config = load_config(CONFIG_DIR, industry="housing")
        for cat in config.categories.values():
            for signal in cat.signals:
                assert signal == signal.lower(), f"Signal not lowercase: {signal}"

    def test_type_tokens_are_lowercase(self):
        config = load_config(CONFIG_DIR, industry="housing")
        for typ in config.types.values():
            for token in typ.tokens:
                assert token == token.lower(), f"Token not lowercase: {token}"

    def test_patterns_compiled(self):
        config = load_config(CONFIG_DIR, industry="housing")
        invoice = config.types["Invoice"]
        assert len(invoice.compiled_patterns) > 0
        # Pattern should match "INV-001"
        assert invoice.compiled_patterns[0].search("INV-001")

    def test_weights_loaded(self):
        config = load_config(CONFIG_DIR, industry="housing")
        assert "signal_weights" in config.weights
        assert "reinforcement" in config.weights
        assert "thresholds" in config.weights

    def test_missing_industry_handled_gracefully(self):
        # Non-existent industry should not crash — just loads base
        config = load_config(CONFIG_DIR, industry="nonexistent")
        assert "Invoice" in config.types
        # Housing-specific types should NOT be present
        assert "MOS" not in config.types

    def test_type_belongs_to_set(self):
        config = load_config(CONFIG_DIR, industry="housing")
        assert config.types["Invoice"].belongs_to == "universal"
        assert config.types["MOS"].belongs_to == "Sales"
        assert config.types["EICR"].belongs_to == "Health & Safety"

    def test_extension_hints_loaded(self):
        config = load_config(CONFIG_DIR, industry="housing")
        assert "msg" in config.types["Email"].extensions
        assert "jpg" in config.types["Photo"].extensions

    def test_new_housing_types_loaded(self):
        config = load_config(CONFIG_DIR, industry="housing")
        assert "Party Wall Award" in config.types
        assert "Appraisal" in config.types
        assert "Insurance" in config.types
        assert "Tender" in config.types
        assert "Planning Application" in config.types
        assert "Land Registry" in config.types
        assert "Deed" in config.types
        assert "NDA" in config.types
        assert "Topographic Survey" in config.types
        assert "Utilities Search" in config.types
        assert "Quotation" in config.types
        assert "Programme" in config.types
        assert "Planning Order" in config.types
        assert "Licence" in config.types

    def test_parent_child_relationships_loaded(self):
        config = load_config(CONFIG_DIR, industry="housing")
        parent_child = config.weights.get("parent_child", {})
        assert "Handover Pack" in parent_child
        assert "Certificate" in parent_child["Handover Pack"]
        assert "EPC" in parent_child["Handover Pack"]
        assert "Drawing" in parent_child
        assert "Plan" in parent_child["Drawing"]
        assert "Report" in parent_child
        assert "Site Inspection" in parent_child["Report"]
