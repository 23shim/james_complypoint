"""
Merge HomeBuy 'Cleaned Data' and 'Unclassified' sheets into a single
CSV that the ComplyPoint pipeline can ingest.

Reads both sheets from context/Cleaned_Data_HomeBuy.xlsx, keeps only
the original File Path column (renamed to 'Path' for the csv format),
deduplicates on exact path, and writes the result.

Usage:
    python scripts/merge_homebuy.py
    python scripts/merge_homebuy.py --input context/Cleaned_Data_HomeBuy.xlsx --output context/homebuy_merged.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def read_cleaned_data(xlsx_path: str) -> pd.DataFrame:
    """Read the 'Cleaned Data' sheet (header at row 1)."""
    df = pd.read_excel(xlsx_path, sheet_name="Cleaned Data", header=0)
    print(f"  Cleaned Data: {len(df):,} rows, columns: {list(df.columns[:5])}...")
    return df


def read_unclassified(xlsx_path: str) -> pd.DataFrame:
    """Read the 'Unclassified' sheet (title row + blank row, header at row 3)."""
    df = pd.read_excel(xlsx_path, sheet_name="Unclassified", header=2)
    print(f"  Unclassified: {len(df):,} rows, columns: {list(df.columns[:5])}...")
    return df


def merge_sheets(xlsx_path: str, output_path: str) -> None:
    print(f"Reading: {xlsx_path}")

    df_clean = read_cleaned_data(xlsx_path)
    df_unclass = read_unclassified(xlsx_path)

    # Both sheets have the same column structure — use File Path as the
    # primary column.  Rename to 'Path' to match schema.yaml csv format.
    if "File Path" not in df_clean.columns:
        print(f"ERROR: 'File Path' column not found in Cleaned Data. "
              f"Available: {list(df_clean.columns)}")
        sys.exit(1)
    if "File Path" not in df_unclass.columns:
        print(f"ERROR: 'File Path' column not found in Unclassified. "
              f"Available: {list(df_unclass.columns)}")
        sys.exit(1)

    # Concatenate both sheets
    combined = pd.concat([df_clean, df_unclass], ignore_index=True)
    total_before = len(combined)
    print(f"\n  Combined: {total_before:,} rows")

    # Drop rows with empty/null File Path
    combined = combined.dropna(subset=["File Path"])
    combined = combined[combined["File Path"].astype(str).str.strip().str.len() > 0]
    after_nulls = len(combined)
    if after_nulls < total_before:
        print(f"  Dropped {total_before - after_nulls:,} rows with empty paths")

    # Deduplicate on exact File Path
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["File Path"], keep="first")
    after_dedup = len(combined)
    dupes = before_dedup - after_dedup
    print(f"  Duplicates removed: {dupes:,}")
    print(f"  Final: {after_dedup:,} unique rows")

    # Keep only the Path column for the pipeline (rename File Path -> Path)
    output_df = combined[["File Path"]].rename(columns={"File Path": "Path"})

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n  Output: {output_path} ({after_dedup:,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge HomeBuy sheets into pipeline-ready CSV",
    )
    parser.add_argument(
        "--input", default="context/Cleaned_Data_HomeBuy.xlsx",
        help="Path to HomeBuy XLSX file",
    )
    parser.add_argument(
        "--output", default="context/homebuy_merged.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    merge_sheets(args.input, args.output)


if __name__ == "__main__":
    main()
