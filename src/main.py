"""
ComplyPoint Phase 1 — Metadata Discovery & Readiness Engine

CLI entry point and pipeline orchestration.

Usage:
    python src/main.py --config config/run_config.yaml --input data/export.xlsx

The pipeline runs in a strict linear sequence:
    Read → Map Columns → Validate → Filter → Normalise → Decompose Paths → Write
"""

import logging
import sys
import time
from pathlib import Path

# Ensure src/ is on the Python path so ingestion.* imports resolve
sys.path.insert(0, str(Path(__file__).parent))

import click
import yaml

from ingestion.reader import read_file
from ingestion.schema import map_columns, validate
from ingestion.filters import apply_filters
from ingestion.normaliser import normalise
from ingestion.path_parser import decompose_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("complypoint")


def load_configs(config_path: Path) -> dict:
    """Load run config and merge with schema config."""
    with open(config_path) as f:
        run_config = yaml.safe_load(f)

    schema_path = config_path.parent / "schema.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema config not found at {schema_path}. "
            f"Expected it alongside the run config."
        )

    with open(schema_path) as f:
        schema_config = yaml.safe_load(f)

    run_config["schema"] = schema_config
    return run_config


@click.command()
@click.option(
    "--config", "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g. config/run_config.yaml)",
)
@click.option(
    "--input", "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to input file (XLSX or CSV)",
)
@click.option(
    "--output", "output_path",
    default=None,
    type=click.Path(),
    help="Output path (default: output/<input_name>_ingested.jsonl)",
)
def main(config_path: str, input_path: str, output_path: str | None):
    """Run the Phase 1 ingestion pipeline."""
    start = time.time()
    config_path = Path(config_path)
    input_path = Path(input_path)

    logger.info(f"ComplyPoint Phase 1 — Ingestion Pipeline")
    logger.info(f"Input: {input_path.name}")

    # ----------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------
    config = load_configs(config_path)
    format_name = config["source"]["format"]
    schema_config = config["schema"]
    format_config = schema_config["formats"].get(format_name)

    if not format_config:
        available = list(schema_config["formats"].keys())
        raise click.UsageError(
            f"Format '{format_name}' not found in schema.yaml. Available: {available}"
        )

    # ----------------------------------------------------------
    # STEP 1: Read raw file
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Reading input file")
    df = read_file(input_path, format_name, format_config)

    # ----------------------------------------------------------
    # STEP 2: Map columns to internal schema
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Mapping columns to internal schema")
    column_map = format_config["column_map"]
    df = map_columns(df, column_map)

    required = schema_config["internal_schema"]["required_columns"]
    validate(df, required)

    # ----------------------------------------------------------
    # STEP 3: Filter rows
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Filtering rows")
    exclusion_config = schema_config.get("exclusion_patterns", {})
    df, filter_summary = apply_filters(df, exclusion_config)

    # ----------------------------------------------------------
    # STEP 4: Normalise data
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Normalising data")
    df = normalise(df)

    # ----------------------------------------------------------
    # STEP 5: Decompose paths
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Decomposing paths")
    root_prefix = config["source"]["root_prefix"]
    df = decompose_paths(df, root_prefix)

    # ----------------------------------------------------------
    # STEP 6: Write output
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Writing output")

    if output_path is None:
        output_dir = Path(config.get("output", {}).get("directory", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_ingested.jsonl"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_json(output_path, orient="records", lines=True, date_format="iso")

    elapsed = time.time() - start
    logger.info(f"Output: {output_path} ({len(df):,} records)")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    click.echo(f"\n{'=' * 60}")
    click.echo("Pipeline Summary")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Input:     {input_path.name}")
    click.echo(f"  Format:    {format_name}")
    click.echo(f"  Records:   {filter_summary.initial_count:,} -> {filter_summary.final_count:,}")
    click.echo(f"  Removed:   {filter_summary.total_removed:,}")
    click.echo(f"  Output:    {output_path}")
    click.echo(f"  Time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
