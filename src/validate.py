"""Data validation module for JoSAA cutoff records.
Enforces strict schema, null, monotonicity, uniqueness, and sanity constraints.
Fails loudly if data corruption or unexpected schemas are detected.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


class DataValidationError(ValueError):
    """Raised when cutoff dataset fails integrity or schema checks."""
    pass


REQUIRED_COLUMNS: List[str] = [
    "year",
    "institute",
    "program",
    "quota",
    "seat_type",
    "gender",
    "opening_rank",
    "closing_rank",
]


def validate_cutoff_dataset(
    df: pd.DataFrame,
    strict_monotonicity: bool = False,
) -> Dict[str, Any]:
    """Validate dataset schema, null values, uniqueness, and rank monotonicity.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset of cutoff records to validate.
    strict_monotonicity : bool, default False
        If True, raises DataValidationError when opening_rank > closing_rank.
        If False, counts violations and reports them in summary.

    Returns
    -------
    dict
        Validation report metrics (row counts, violation counts).

    Raises
    ------
    DataValidationError
        If schema is missing required columns, contains nulls in mandatory fields,
        or contains duplicate offering rows.
    """
    if df.empty:
        raise DataValidationError("Cutoff dataset is empty (0 rows).")

    # 1. Schema check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Dataset schema violation: missing required columns: {missing_cols}"
        )

    # 2. Mandatory null checks
    critical_cols = ["year", "institute", "program", "quota", "seat_type", "gender", "closing_rank"]
    null_counts = df[critical_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise DataValidationError(
            f"Dataset integrity error: unexpected nulls in critical columns:\n{cols_with_nulls.to_dict()}"
        )

    # 3. Duplicate seat offering check per year
    dup_cols = ["year", "institute", "program", "quota", "seat_type", "gender"]
    dups = df.duplicated(subset=dup_cols, keep=False)
    dup_count = int(dups.sum())
    if dup_count > 0:
        sample_dups = df[dups][dup_cols].head(4).to_dict(orient="records")
        raise DataValidationError(
            f"Dataset integrity error: found {dup_count} duplicate seat offerings per cycle. "
            f"Sample duplicates: {sample_dups}"
        )

    # 4. Monotonicity check (opening_rank <= closing_rank)
    if "opening_rank" in df.columns:
        valid_ranks = df[df["opening_rank"].notna() & df["closing_rank"].notna()]
        non_monotonic = valid_ranks[valid_ranks["opening_rank"] > valid_ranks["closing_rank"]]
        non_monotonic_count = len(non_monotonic)
        if non_monotonic_count > 0 and strict_monotonicity:
            raise DataValidationError(
                f"Monotonicity error: {non_monotonic_count} rows have opening_rank > closing_rank."
            )
    else:
        non_monotonic_count = 0

    # 5. Strictly positive ranks
    invalid_closing = (df["closing_rank"] <= 0).sum()
    if invalid_closing > 0:
        raise DataValidationError(
            f"Dataset integrity error: {invalid_closing} rows have closing_rank <= 0."
        )

    return {
        "total_rows": len(df),
        "unique_institutes": int(df["institute"].nunique()),
        "unique_programs": int(df["program"].nunique()),
        "cycles": sorted([int(y) for y in df["year"].unique()]),
        "non_monotonic_count": non_monotonic_count,
    }
