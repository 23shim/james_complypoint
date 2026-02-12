"""Tests for path decomposition."""

import pandas as pd
import pytest

from ingestion.path_parser import decompose_paths, _split_filename


class TestSplitFilename:
    """Unit tests for filename splitting."""

    def test_normal_file(self):
        assert _split_filename("report.pdf") == ("report", "pdf")

    def test_no_extension(self):
        assert _split_filename("README") == ("README", "")

    def test_multiple_dots(self):
        assert _split_filename("report.v2.pdf") == ("report.v2", "pdf")

    def test_hidden_file(self):
        assert _split_filename(".gitignore") == (".gitignore", "")

    def test_uppercase_extension(self):
        stem, ext = _split_filename("PHOTO.JPG")
        assert stem == "PHOTO"
        assert ext == "jpg"

    def test_empty_string(self):
        assert _split_filename("") == ("", "")


class TestDecomposePaths:
    """Integration tests for full path decomposition."""

    ROOT = "\\\\wkh_apps1\\Techserv\\Data\\Development"

    def _make_df(self, paths: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"full_path": paths})

    def test_standard_path(self):
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\Project A\\Handover\\report.pdf"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["segments"] == ["Project A", "Handover"]
        assert result.iloc[0]["filename_stem"] == "report"
        assert result.iloc[0]["extension"] == "pdf"
        assert result.iloc[0]["depth"] == 2

    def test_root_level_file(self):
        """File directly in the root — zero segments."""
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\notes.txt"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["segments"] == []
        assert result.iloc[0]["filename_stem"] == "notes"
        assert result.iloc[0]["extension"] == "txt"
        assert result.iloc[0]["depth"] == 0

    def test_deep_nesting(self):
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\A\\B\\C\\D\\E\\file.docx"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["segments"] == ["A", "B", "C", "D", "E"]
        assert result.iloc[0]["depth"] == 5

    def test_no_extension(self):
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\Folder\\README"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["filename_stem"] == "README"
        assert result.iloc[0]["extension"] == ""

    def test_case_insensitive_root(self):
        """Root matching should be case-insensitive (Windows paths)."""
        df = self._make_df([
            "\\\\WKH_APPS1\\TECHSERV\\DATA\\DEVELOPMENT\\Project\\file.pdf"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["segments"] == ["Project"]
        assert result.iloc[0]["filename_stem"] == "file"

    def test_different_root_prefix(self):
        """Verify it works with HomeBuy-style paths too."""
        root = "\\\\wkh_apps1\\WIN95\\DATA\\Development_Recovered2\\HOMEBUY"
        df = self._make_df([
            "\\\\wkh_apps1\\WIN95\\DATA\\Development_Recovered2\\HOMEBUY\\LEASEHOLD\\2024\\lease.pdf"
        ])
        result = decompose_paths(df, root)

        assert result.iloc[0]["segments"] == ["LEASEHOLD", "2024"]
        assert result.iloc[0]["filename_stem"] == "lease"

    def test_special_characters_in_path(self):
        """Filenames with ampersands, brackets, etc. should survive."""
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\Fire safety & risk\\Plot (1)\\cert_v2.pdf"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["segments"] == ["Fire safety & risk", "Plot (1)"]
        assert result.iloc[0]["filename_stem"] == "cert_v2"

    def test_relative_path_preserved(self):
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\A\\B\\file.txt"
        ])
        result = decompose_paths(df, self.ROOT)

        assert result.iloc[0]["relative_path"] == "A\\B\\file.txt"

    def test_multiple_rows(self):
        """Verify batch processing works correctly."""
        df = self._make_df([
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\P1\\a.pdf",
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\P2\\Sub\\b.docx",
            "\\\\wkh_apps1\\Techserv\\Data\\Development\\c.xlsx",
        ])
        result = decompose_paths(df, self.ROOT)

        assert len(result) == 3
        assert result.iloc[0]["depth"] == 1
        assert result.iloc[1]["depth"] == 2
        assert result.iloc[2]["depth"] == 0
