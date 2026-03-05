"""
Retention rule loading and application.

Loads industry-specific retention rules from YAML and maps
inferred document types to retention periods and status.
Runs AFTER classification — does not affect scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
import pandas as pd

logger = logging.getLogger(__name__)


def load_retention_rules(config_dir: str | Path, industry: str) -> dict:
    """Load retention rules for the given industry.

    Args:
        config_dir: Root config directory (contains retention/).
        industry: Industry name — loads retention/{industry}.yaml.

    Returns:
        Dict with keys: default_retention_years, rules.
    """
    path = Path(config_dir) / "retention" / f"{industry}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Retention rules not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rule_count = len(data.get("rules", {}))
    logger.info(f"Retention rules loaded: {rule_count} rules ({industry})")
    return data


_SENSITIVITY_MULTIPLIER = {"high": 1.25, "medium": 1.0, "low": 0.7}


def _compute_risk_score(
    status: str | None,
    sensitivity: str | None,
    expiry_iso: str | None,
    now: datetime,
) -> int:
    """Compute a 0–100 retention risk score for a single row.

    Scoring logic:
      - Unknown status/sensitivity → 0 (not enough signal to score)
      - Long-term (permanent) → 5 (properly governed, minimal risk)
      - Expired: 60–95 depending on how long overdue
      - Active: 5–50 depending on proximity to expiry

    Sensitivity multiplier (high=1.25, medium=1.0, low=0.7) scales the
    base score.  Result is capped at 100.
    """
    if not status or status == "Unknown":
        return 0

    sens_mult = _SENSITIVITY_MULTIPLIER.get(sensitivity or "", 1.0)

    if status == "Long-term":
        return int(min(5 * sens_mult, 100))

    # Parse expiry date to compute time delta
    if not expiry_iso:
        return 0
    try:
        expiry_dt = pd.Timestamp(expiry_iso)
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.tz_localize("UTC")
    except Exception:
        return 0

    years_until = (expiry_dt - pd.Timestamp(now)).days / 365.25

    if status == "Expired":
        years_overdue = abs(years_until)
        if years_overdue > 5:
            base = 95
        elif years_overdue > 2:
            base = 80
        elif years_overdue > 1:
            base = 70
        else:
            base = 60
    else:  # Active
        if years_until < 1:
            base = 50
        elif years_until < 3:
            base = 30
        elif years_until < 5:
            base = 15
        else:
            base = 5

    return int(min(base * sens_mult, 100))


def apply_retention(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Map inferred_type to retention rule and compute status.

    Adds columns:
        retention_years         — years to retain (int or "permanent")
        retention_basis         — legal/regulatory basis
        retention_status        — Active / Expired / Long-term / Unknown
        retention_category      — inferred category used for retention context
        sensitivity_level       — high / medium / low
        calculated_expiry_date  — last_modified + retention_years (ISO string)

    Args:
        df: DataFrame with inferred_type, inferred_category, and
            last_modified columns.
        rules: Dict from load_retention_rules().

    Returns:
        DataFrame with retention columns added.
    """
    rule_map = rules.get("rules", {})
    default_years = rules.get("default_retention_years", 6)
    default_sensitivity = rules.get("default_sensitivity", "medium")
    now = datetime.now(timezone.utc)

    years_list = []
    basis_list = []
    status_list = []
    sensitivity_list = []
    expiry_list = []
    category_list = []

    for _, row in df.iterrows():
        inferred_type = row.get("inferred_type", "Unknown")
        inferred_category = row.get("inferred_category", "Unknown")

        # Unknown type -> Unknown retention
        if inferred_type == "Unknown":
            years_list.append(None)
            basis_list.append(None)
            status_list.append("Unknown")
            sensitivity_list.append(None)
            expiry_list.append(None)
            category_list.append(inferred_category)
            continue

        # Look up rule for this type
        rule = rule_map.get(inferred_type)
        if rule:
            years = rule["years"]
            basis = rule["basis"]
            sensitivity = rule.get("sensitivity", default_sensitivity)
        else:
            years = default_years
            basis = f"Default retention ({default_years} years)"
            sensitivity = default_sensitivity

        years_list.append(years)
        basis_list.append(basis)
        sensitivity_list.append(sensitivity)
        category_list.append(inferred_category)

        # Compute status and expiry date
        if years == "permanent":
            status_list.append("Long-term")
            expiry_list.append(None)
            continue

        last_mod = row.get("last_modified")
        if pd.isna(last_mod):
            status_list.append("Unknown")
            expiry_list.append(None)
            continue

        # Ensure last_mod is a datetime
        if not isinstance(last_mod, datetime):
            last_mod = pd.Timestamp(last_mod)

        # Make timezone-aware for comparison
        if last_mod.tzinfo is None:
            last_mod = last_mod.replace(tzinfo=timezone.utc)

        try:
            expiry = last_mod.replace(year=last_mod.year + years)
        except ValueError:
            # Leap year edge case: Feb 29 -> Feb 28 in non-leap target year
            expiry = last_mod.replace(month=3, day=1, year=last_mod.year + years)

        expiry_list.append(expiry.isoformat())

        if expiry < now:
            status_list.append("Expired")
        else:
            status_list.append("Active")

    df["retention_years"] = years_list
    df["retention_basis"] = basis_list
    df["retention_status"] = status_list
    df["retention_category"] = category_list
    df["sensitivity_level"] = sensitivity_list
    df["calculated_expiry_date"] = expiry_list

    # Compute retention_risk_score (0–100).
    # Combines retention status, sensitivity weighting, and time
    # to/past expiry into a single numeric score for Dataverse
    # filtering and dashboard use.
    df["retention_risk_score"] = df.apply(
        lambda r: _compute_risk_score(
            r.get("retention_status"),
            r.get("sensitivity_level"),
            r.get("calculated_expiry_date"),
            now,
        ),
        axis=1,
    )

    # Log distribution
    status_counts = df["retention_status"].value_counts()
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count:,}")

    sensitivity_counts = df["sensitivity_level"].value_counts()
    for level, count in sensitivity_counts.items():
        if level:
            logger.info(f"  sensitivity {level}: {count:,}")

    high_risk = (df["retention_risk_score"] >= 60).sum()
    if high_risk:
        logger.info(f"  retention risk >= 60: {high_risk:,}")

    return df
