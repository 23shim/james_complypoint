"""
Convert classified JSONL to a Dataverse-ready CSV matching the
Document Estate Index schema.

Groups:
  1. Core Identity
  2. Classification
  3. Entity Layer
  4. Retention
  5. Clustering

Usage:
    python scripts/jsonl_to_csv_slim.py <input.jsonl> [output.csv]
"""

import csv
import json
import sys
from pathlib import Path

COLUMN_SPEC = [
    # ── Core Identity ──
    ("File ID",            "file_id",             ""),
    ("Full Path",          "full_path",           ""),
    ("Source System",      "source_system",       ""),
    ("File Name",          "filename",            ""),
    ("Extension",          "extension",           ""),
    ("Modified Date",      "last_modified",       ""),
    ("Size (KB)",          "size",                ""),
    # ── Classification ──
    ("Primary Document Type",  "inferred_type",           ""),
    ("Secondary Type",         "secondary_type",          ""),
    ("Confidence Score",       "overall_confidence",      ""),
    ("Confidence Band",        "confidence_band",         ""),
    # ── Entity Layer ──
    ("Scheme / Project",       "entity_scheme",              ""),
    ("Scheme Cluster ID",      "entity_scheme_cluster_id",   ""),
    ("Scheme Cluster Name",    "entity_scheme_canonical",    ""),
    ("Address / Street",       "entity_address",             ""),
    ("Address Cluster ID",     "entity_address_cluster_id",  ""),
    ("Address Cluster Name",   "entity_address_canonical",   ""),
    ("Plot / Unit",            "entity_plot",                ""),
    ("Plot Cluster ID",        "entity_plot_cluster_id",     ""),
    ("Plot Cluster Name",      "entity_plot_canonical",      ""),
    ("Scheme Confidence",      "entity_scheme_confidence",   ""),
    ("Address Confidence",     "entity_address_confidence",  ""),
    ("Plot Confidence",        "entity_plot_confidence",     ""),
    # ── Retention ──
    ("Retention Category",     "retention_category",      ""),
    ("Retention Years",        "retention_years",         ""),
    ("Calculated Expiry Date", "calculated_expiry_date",  ""),
    ("Retention Status",       "retention_status",        ""),
    ("Sensitivity Level",      "sensitivity_level",       ""),
    ("Retention Risk Score",   "retention_risk_score",    ""),
    # ── Clustering — Entity ──
    ("Scheme Cluster Size",    "entity_scheme_cluster_size",  ""),
    ("Address Cluster Size",   "entity_address_cluster_size", ""),
    ("Plot Cluster Size",      "entity_plot_cluster_size",    ""),
    # ── Clustering — Duplicates ──
    ("Duplicate Group ID",     "dup_group_id",            ""),
    ("Duplicate Group Size",   "dup_group_size",          ""),
    ("Is Latest in Group",     "dup_is_latest",           ""),
    ("Potential Duplicate",    "potential_duplicate",      ""),
]

HEADERS  = [col[0] for col in COLUMN_SPEC]
KEYS     = [col[1] for col in COLUMN_SPEC]
DEFAULTS = {col[1]: col[2] for col in COLUMN_SPEC}


def format_value(key, val):
    """Convert a value to a CSV-safe string."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, list):
        if val and isinstance(val[0], dict):
            parts = []
            for item in val:
                v = item.get("value", "")
                c = item.get("confidence", "")
                parts.append(f"{v} ({c})")
            return " | ".join(parts)
        return " | ".join(str(v) for v in val)
    return val


def convert_row(raw: dict) -> dict:
    """Map a raw JSONL record to the Dataverse CSV columns."""
    return {
        header: format_value(key, raw.get(key, DEFAULTS[key]))
        for header, key in zip(HEADERS, KEYS)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python jsonl_to_csv_slim.py <input.jsonl> [output.csv]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    default_out = input_path.with_name(input_path.stem + "_dataverse.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else default_out

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
