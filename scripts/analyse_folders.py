"""Folder structure analysis.

Loads a classified JSONL file and prints the unique folder segment
values found at each depth level, useful for understanding the
organisational hierarchy of an estate.

Usage:
    python scripts/analyse_folders.py output/combined_classified.jsonl
"""

import json
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/analyse_folders.py <classified.jsonl>")

    input_path = sys.argv[1]

    print("Loading JSONL...")
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows")

    CONSTRUCTION_CATS = [
        "handover", "snagging", "sales", "construction", "planning", "drawings",
        "epcs", "epc", "site inspection", "health & safety", "health and safety",
        "h&s", "warranties", "legal", "finance", "contracts", "tender",
        "correspondence", "defects", "completion", "reports", "design",
        "structural", "drainage", "landscaping", "architect", "surveys",
        "demolition", "minutes", "meetings", "valuations", "accounts",
        "insurance", "nhbc", "photos", "photographs", "images", "quotations",
        "specifications", "spec", "build", "section", "road", "sewer",
        "employer", "practical", "clerk of works", "cladding", "mechanical",
        "electrical", "foundations", "party wall", "building control",
        "certificate", "warranty", "site", "progress", "costing", "payment",
        "legals", "solicitor", "appraisal"
    ]
    print("Script loaded OK")


if __name__ == "__main__":
    main()
