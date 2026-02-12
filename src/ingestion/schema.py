"""
Schema mapping and validation.

The anti-corruption layer between source formats and the internal
pipeline. After this step, no downstream code ever sees source-specific
column names.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def map_columns(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename source columns to internal schema names.

    Only renames columns that exist in the DataFrame.
    Columns not in the map are dropped — the internal pipeline
    should never depend on source-specific columns.

    Args:
        df: Raw DataFrame from reader.
        column_map: {source_name: internal_name} from schema.yaml.

    Returns:
        DataFrame with internal column names only.
    """
    present = {
        source: internal
        for source, internal in column_map.items()
        if source in df.columns
    }

    missing = set(column_map.keys()) - set(present.keys())
    if missing:
        logger.warning(f"Source columns not found in data: {missing}")

    df = df.rename(columns=present)
    internal_cols = list(present.values())
    df = df[internal_cols]

    logger.info(f"Mapped {len(present)} columns → {internal_cols}")
    return df


def validate(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Check that required columns exist after mapping.

    Raises ValueError with a clear message if any are missing.
    This is a hard stop — the pipeline cannot continue without
    required columns.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing after mapping: {missing}. "
            f"Available: {list(df.columns)}. "
            f"Check column_map in schema.yaml matches the source file headers."
        )
    logger.info("Schema validation passed")
