"""
Data normalisation.

Format-independent cleaning applied after column mapping and filtering.
Handles Unicode normalisation (Arabic readiness), date parsing,
and owner field cleanup.
"""

import logging
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalisations. Returns a new DataFrame."""
    df = df.copy()

    df = _normalise_unicode(df)
    df = _normalise_dates(df)
    df = _normalise_owner(df)

    logger.info("Normalisation complete")
    return df


def _normalise_unicode(df: pd.DataFrame) -> pd.DataFrame:
    """NFC-normalise the full_path column.

    NFC normalisation ensures that characters like Arabic
    letters with diacritics are stored in their composed form,
    preventing matching failures from invisible encoding differences.
    """
    df["full_path"] = df["full_path"].apply(
        lambda x: unicodedata.normalize("NFC", str(x)) if pd.notna(x) else x
    )
    return df


def _normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns into proper datetime objects.

    Uses errors='coerce' so unparseable dates become NaT
    rather than crashing the pipeline.
    """
    for col in ["last_modified", "last_accessed"]:
        if col not in df.columns:
            continue

        # Skip if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning(f"  {col}: {null_count:,} null values")
            continue

        df[col] = pd.to_datetime(df[col], errors="coerce")
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.warning(f"  {col}: {null_count:,} values could not be parsed as dates")

    return df


def _normalise_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Clean owner field: strip whitespace, flag deleted AD accounts."""
    if "owner" not in df.columns:
        return df

    df["owner"] = df["owner"].astype(str).str.strip()

    # Windows SIDs indicate deleted Active Directory accounts
    sid_mask = df["owner"].str.startswith("S-1-5-", na=False)
    sid_count = sid_mask.sum()
    if sid_count > 0:
        logger.info(f"  {sid_count:,} entries with SID-style owners (deleted AD accounts)")

    # Handle 'nan' strings from astype(str) on actual nulls
    df.loc[df["owner"] == "nan", "owner"] = ""

    return df
