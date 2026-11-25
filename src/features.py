"""Feature engineering for JoSAA cutoff records.

Adds branch groupings and time-lag features per key (institute, program, quota, seat_type, gender):
- prev_closing_1, prev_closing_2, prev_closing_3
- mean_last_2, delta_1yr, trend_slope
- years_of_history, branch_group, log_closing_rank
Saves result to data/processed/cutoffs.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def assign_branch_group(program_name: str) -> str:
    """Categorize academic program into high-level branch discipline."""
    p = program_name.lower()

    # Check computer science & IT disciplines
    if any(k in p for k in [
        "computer", "data science", "artificial intelligence", " ai", "machine learning",
        "information technology", "software", "cyber", "computing"
    ]):
        return "CSE"
    # Check Electronics & Communication
    if any(k in p for k in [
        "electronics", "communication", "telecommunication", "vlsi", "avionics"
    ]):
        return "ECE"
    # Check Electrical
    if any(k in p for k in ["electrical", "power engineering", "instrumentation"]):
        return "EE"
    # Check Mechanical & Aerospace
    if any(k in p for k in [
        "mechanical", "production", "manufacturing", "industrial",
        "automobile", "aerospace", "aeronautical", "mechatronics"
    ]):
        return "MECH"
    # Check Civil & Infrastructure
    if any(k in p for k in ["civil", "structural", "environmental", "transportation"]):
        return "CIVIL"
    # Check Chemical & Petroleum
    if any(k in p for k in ["chemical", "petroleum", "petrochemical", "polymer", "plastic"]):
        return "CHEM"
    # Check Metallurgical, Materials & Mining
    if any(k in p for k in ["metallurg", "materials", "mining", "mineral", "ceramic"]):
        return "META"
    # Check Architecture & Planning
    if any(k in p for k in ["architect", "planning"]):
        return "ARCH"
    # Check Bio, Biotech & Food
    if any(k in p for k in ["bio", "biomedical", "biotechnology", "biological", "food"]):
        return "BIO"

    return "OTHER"


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag features and historical metrics per seat key."""
    # Ensure branch_group is present
    df = df.copy()
    if "branch_group" not in df.columns:
        df["branch_group"] = df["program"].apply(assign_branch_group)

    df["log_closing_rank"] = np.log(df["closing_rank"].astype(float))

    key_cols = ["institute", "program", "quota", "seat_type", "gender"]
    df = df.sort_values(by=key_cols + ["year"]).reset_index(drop=True)

    grouped = df.groupby(key_cols)

    # Historical lag ranks
    df["prev_closing_1"] = grouped["closing_rank"].shift(1)
    df["prev_closing_2"] = grouped["closing_rank"].shift(2)
    df["prev_closing_3"] = grouped["closing_rank"].shift(3)

    # Years of history prior to this year
    df["years_of_history"] = grouped.cumcount()

    # Mean of last 2 available closing ranks
    df["mean_last_2"] = np.where(
        df["prev_closing_1"].notna() & df["prev_closing_2"].notna(),
        (df["prev_closing_1"] + df["prev_closing_2"]) / 2.0,
        df["prev_closing_1"],
    )

    # Delta of last 1 year
    df["delta_1yr"] = df["prev_closing_1"] - df["prev_closing_2"]

    # Trend slope in log space: (ln(prev_1) - ln(prev_2)) or average over 3 years
    log_prev_1 = np.log(df["prev_closing_1"])
    log_prev_2 = np.log(df["prev_closing_2"])
    log_prev_3 = np.log(df["prev_closing_3"])

    slope_2pt = log_prev_1 - log_prev_2
    slope_3pt = (log_prev_1 - log_prev_3) / 2.0

    df["trend_slope"] = np.where(
        log_prev_3.notna(),
        slope_3pt,
        np.where(log_prev_2.notna(), slope_2pt, 0.0),
    )

    return df


def generate_and_save_cutoffs_parquet(
    cleaned_df: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    """Generate features on cleaned dataset and save to Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features = compute_lag_features(cleaned_df)
    df_features.to_parquet(output_path, index=False)
    print(f"Saved cutoffs parquet to {output_path} with {len(df_features)} rows.")
    return df_features


def main() -> None:
    from src.clean import clean_all_raw_files

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    cleaned_df = clean_all_raw_files(raw_dir, processed_dir)
    parquet_path = processed_dir / "cutoffs.parquet"
    generate_and_save_cutoffs_parquet(cleaned_df, parquet_path)


if __name__ == "__main__":
    main()

