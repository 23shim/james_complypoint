"""
Convert classified JSONL to CSV with columns aligned to the
Document Estate Index schema.

Groups:
  1. Core Identity
  2. Classification
  3. Entity Layer — Plot, Address, Project Ref
  4. Entity Layer — Scheme
  5. Clustering
  6. Retention
  7. Metadata / Diagnostics

Usage:
    python scripts/jsonl_to_csv.py <input.jsonl> [output.csv]
"""

import csv
import json
import sys
from pathlib import Path

# ── Column spec: (csv_header, jsonl_key, default) ──────────────────────
COLUMN_SPEC = [
    # ── Core Identity ──
    ("File ID",            "file_id",             ""),
    ("Full Path",          "full_path",           ""),
    ("Relative Path",      "relative_path",       ""),
    ("Source System",      "source_system",       ""),
    ("File Name",          "filename",            ""),
    ("Filename Stem",      "filename_stem",       ""),
    ("Extension",          "extension",           ""),
    ("Modified Date",      "last_modified",       ""),
    ("Accessed Date",      "last_accessed",       ""),
    ("Size (KB)",          "size",                ""),
    ("Allocated (KB)",     "allocated",           ""),
    ("File Count",         "file_count",          ""),
    ("Folder Count",       "folder_count",        ""),
    ("Pct Parent",         "pct_parent",          ""),
    ("Owner",              "owner",               ""),
    ("Depth",              "depth",               ""),
    # ── Classification ──
    ("Primary Document Type",      "inferred_type",              ""),
    ("Type Confidence",            "type_confidence",            ""),
    ("Secondary Type",             "secondary_type",             ""),
    ("Secondary Type Confidence",  "secondary_type_confidence",  ""),
    ("Category",                   "inferred_category",          ""),
    ("Category Confidence",        "category_confidence",        ""),
    ("Overall Confidence",         "overall_confidence",         ""),
    ("Confidence Band",            "confidence_band",            ""),
    ("Readiness Status",           "readiness_status",           ""),
    # ── Entity Layer — Plot ──
    ("Plot / Unit",              "entity_plot",                  ""),
    ("Plot Confidence",          "entity_plot_confidence",       ""),
    ("Plot Depth",               "entity_plot_depth",            ""),
    ("Plot Is Scheme Name",      "entity_plot_is_scheme_name",   ""),
    ("Plot Cluster ID",          "entity_plot_cluster_id",       ""),
    ("Plot Canonical",           "entity_plot_canonical",        ""),
    ("Raw Plots",                "raw_plots",                    ""),
    # ── Entity Layer — Address ──
    ("Address",                  "entity_address",               ""),
    ("Address Confidence",       "entity_address_confidence",    ""),
    ("Address Depth",            "entity_address_depth",         ""),
    ("Address Is Scheme Name",   "entity_address_is_scheme_name", ""),
    ("Address Cluster ID",       "entity_address_cluster_id",    ""),
    ("Address Canonical",        "entity_address_canonical",     ""),
    ("Raw Addresses",            "raw_addresss",                 ""),
    # ── Entity Layer — Project Ref ──
    ("Project Ref",                  "entity_project_ref",               ""),
    ("Project Ref Confidence",       "entity_project_ref_confidence",    ""),
    ("Project Ref Depth",            "entity_project_ref_depth",         ""),
    ("Project Ref Cluster ID",       "entity_project_ref_cluster_id",    ""),
    ("Project Ref Canonical",        "entity_project_ref_canonical",     ""),
    ("Raw Project Refs",             "raw_project_refs",                 ""),
    # ── Entity Layer — Scheme ──
    ("Scheme",                   "entity_scheme",                ""),
    ("Scheme Confidence",        "entity_scheme_confidence",     ""),
    ("Scheme Path",              "entity_scheme_path",           ""),
    ("Scheme Depth",             "entity_scheme_depth",          ""),
    ("Scheme Place Name",        "entity_scheme_place_name_match", ""),
    ("Scheme Address Match",     "entity_scheme_address_like_name", ""),
    ("Scheme Cluster ID",        "entity_scheme_cluster_id",     ""),
    ("Scheme Canonical",         "entity_scheme_canonical",      ""),
    # ── Retention ──
    ("Retention Category",     "retention_category",          ""),
    ("Retention Years",        "retention_years",             ""),
    ("Calculated Expiry Date", "calculated_expiry_date",      ""),
    ("Retention Status",       "retention_status",            ""),
    ("Retention Basis",        "retention_basis",             ""),
    ("Sensitivity Level",      "sensitivity_level",           ""),
    # ── Clustering — Entity ──
    ("Scheme Cluster Size",    "entity_scheme_cluster_size",  ""),
    ("Address Cluster Size",   "entity_address_cluster_size", ""),
    ("Plot Cluster Size",      "entity_plot_cluster_size",    ""),
    # ── Clustering — Duplicates ──
    ("Duplicate Group ID",     "dup_group_id",            ""),
    ("Duplicate Group Size",   "dup_group_size",          ""),
    ("Is Latest in Group",     "dup_is_latest",           ""),
    ("Potential Duplicate",    "potential_duplicate",      ""),
    ("Cross-Estate Duplicate", "dup_is_cross_estate",     ""),
    ("Cross-Folder Duplicate", "dup_is_cross_folder",     ""),
    ("Copy Path Suspect",      "dup_copy_path_suspect",   ""),
    ("Duplicate Confidence",   "dup_confidence",          ""),
    # ── Diagnostics ──
    ("Path Segments",          "segments",                    ""),
    ("Reasoning Trace",        "reasoning_trace",             ""),
    ("Batch Run ID",           "batch_run_id",                ""),
    ("Run Timestamp",          "run_timestamp",               ""),
]

HEADERS = [col[0] for col in COLUMN_SPEC]
KEYS    = [col[1] for col in COLUMN_SPEC]
DEFAULTS = {col[1]: col[2] for col in COLUMN_SPEC}


def format_value(key, val):
    """Convert a value to a CSV-safe string."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, list):
        if val and isinstance(val[0], dict):
            # e.g. raw_plots: [{"value": "...", "confidence": 0.8, "depth": 2}]
            parts = []
            for item in val:
                v = item.get("value", "")
                c = item.get("confidence", "")
                parts.append(f"{v} ({c})")
            return " | ".join(parts)
        return " | ".join(str(v) for v in val)
    return val


def convert_row(raw: dict) -> dict:
    """Map a raw JSONL record to the agreed CSV columns."""
    return {
        header: format_value(key, raw.get(key, DEFAULTS[key]))
        for header, key in zip(HEADERS, KEYS)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python jsonl_to_csv.py <input.jsonl> [output.csv]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".csv")

    rows_written = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=HEADERS)
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            writer.writerow(convert_row(raw))
            rows_written += 1
            if rows_written % 100_000 == 0:
                print(f"  {rows_written:,} rows written...")

    print(f"Done -- {rows_written:,} rows -> {output_path}")


if __name__ == "__main__":
    main()
