"""Tests for schema mapping and validation."""

import pandas as pd
import pytest

from ingestion.schema import map_columns, validate


class TestMapColumns:

    def test_renames_columns(self):
        df = pd.DataFrame({"Path": ["/a/b"], "Size": [100]})
        result = map_columns(df, {"Path": "full_path", "Size": "size"})
        assert list(result.columns) == ["full_path", "size"]

    def test_drops_unmapped_columns(self):
        df = pd.DataFrame({"Path": ["/a"], "Extra": ["x"]})
        result = map_columns(df, {"Path": "full_path"})
        assert "Extra" not in result.columns
        assert "full_path" in result.columns

    def test_handles_missing_source_columns(self):
        """Missing source columns are skipped, not errored."""
        df = pd.DataFrame({"Path": ["/a"]})
        result = map_columns(df, {"Path": "full_path", "Owner": "owner"})
        assert "full_path" in result.columns
        assert "owner" not in result.columns


class TestValidate:

    def test_passes_with_required_columns(self):
        df = pd.DataFrame({"full_path": ["/a"]})
        validate(df, ["full_path"])  # should not raise

    def test_fails_on_missing_required(self):
        df = pd.DataFrame({"size": [100]})
        with pytest.raises(ValueError, match="Required columns missing"):
            validate(df, ["full_path"])
