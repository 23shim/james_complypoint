"""Tests for row filtering."""

import pandas as pd
import pytest

from ingestion.filters import apply_filters


EXCLUSION_CONFIG = {
    "filenames": ["Thumbs.db", "desktop.ini"],
    "prefixes": ["~$"],
    "extensions": ["tmp", "bak"],
}


def _make_df(paths: list[str], folder_counts: list[int] | None = None) -> pd.DataFrame:
    """Helper to create test DataFrames."""
    data = {"full_path": paths}
    if folder_counts is not None:
        data["folder_count"] = folder_counts
    return pd.DataFrame(data)


class TestNullPathRemoval:

    def test_removes_none_paths(self):
        df = _make_df(["\\\\server\\file.pdf", None])
        result, summary = apply_filters(df, {})
        assert len(result) == 1
        assert summary.null_paths == 1

    def test_removes_empty_paths(self):
        df = _make_df(["\\\\server\\file.pdf", "", "   "])
        result, summary = apply_filters(df, {})
        assert len(result) == 1
        assert summary.null_paths == 2

    def test_removes_numeric_paths(self):
        """TreeSize exports can corrupt paths to 0 beyond row limits."""
        df = pd.DataFrame({"full_path": ["\\\\server\\file.pdf", 0, 0, 0]})
        result, summary = apply_filters(df, {})
        assert len(result) == 1
        assert summary.null_paths == 3

    def test_removes_paths_without_separators(self):
        """A value like 'abc' is not a valid file path."""
        df = _make_df(["\\\\server\\file.pdf", "abc"])
        result, summary = apply_filters(df, {})
        assert len(result) == 1


class TestFolderRowRemoval:

    def test_removes_paths_ending_with_backslash(self):
        df = _make_df([
            "\\\\server\\folder\\",
            "\\\\server\\file.pdf",
        ])
        result, summary = apply_filters(df, {})
        assert len(result) == 1
        assert summary.folder_rows == 1

    def test_removes_rows_with_folder_count(self):
        df = _make_df(
            ["\\\\server\\dir", "\\\\server\\file.pdf"],
            folder_counts=[3, 0],
        )
        result, summary = apply_filters(df, {})
        assert len(result) == 1
        assert summary.folder_rows == 1

    def test_keeps_files_with_zero_folder_count(self):
        df = _make_df(
            ["\\\\server\\file.pdf"],
            folder_counts=[0],
        )
        result, _ = apply_filters(df, {})
        assert len(result) == 1


class TestSystemFileRemoval:

    def test_removes_thumbs_db(self):
        df = _make_df([
            "\\\\server\\folder\\Thumbs.db",
            "\\\\server\\folder\\file.pdf",
        ])
        result, summary = apply_filters(df, EXCLUSION_CONFIG)
        assert len(result) == 1
        assert summary.system_files == 1

    def test_case_insensitive_filename_match(self):
        df = _make_df(["\\\\server\\THUMBS.DB"])
        result, _ = apply_filters(df, EXCLUSION_CONFIG)
        assert len(result) == 0

    def test_removes_office_temp_files(self):
        df = _make_df([
            "\\\\server\\~$document.docx",
            "\\\\server\\real_document.docx",
        ])
        result, summary = apply_filters(df, EXCLUSION_CONFIG)
        assert len(result) == 1
        assert result.iloc[0]["full_path"].endswith("real_document.docx")

    def test_removes_by_extension(self):
        df = _make_df([
            "\\\\server\\backup.bak",
            "\\\\server\\temp.tmp",
            "\\\\server\\report.pdf",
        ])
        result, summary = apply_filters(df, EXCLUSION_CONFIG)
        assert len(result) == 1

    def test_no_exclusion_config(self):
        """Empty config should not filter any files."""
        df = _make_df(["\\\\server\\Thumbs.db", "\\\\server\\file.pdf"])
        result, _ = apply_filters(df, {})
        assert len(result) == 2


class TestFilterSummary:

    def test_summary_counts(self):
        df = _make_df([
            "\\\\server\\folder\\",           # folder row
            None,                              # null path
            "\\\\server\\Thumbs.db",           # system file
            "\\\\server\\~$temp.docx",         # temp file
            "\\\\server\\real.pdf",            # keeper
            "\\\\server\\also_real.docx",      # keeper
        ])
        _, summary = apply_filters(df, EXCLUSION_CONFIG)

        assert summary.initial_count == 6
        assert summary.final_count == 2
        assert summary.total_removed == 4
