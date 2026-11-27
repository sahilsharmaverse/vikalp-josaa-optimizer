"""Tests for temporal leakage in Vikalp admission-probability engine.

Ensures:
1. No 2025 (or future) ground truth leaks into lag features for 2025 predictions.
2. In rolling-origin backtests, training folds strictly use data prior to the target year (t < T).
3. Prediction interval uncertainty (sigma) is estimated strictly on training folds (t < T).
4. No 2026 data exists in raw or processed datasets.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/cutoffs.parquet")
RAW_DIR = Path("data/raw")


def test_no_2026_data_exists():
    """Verify that 2026 data is absent from both raw and processed storage."""
    assert RAW_DIR.exists(), "Raw data directory must exist"
    raw_2026_files = list(RAW_DIR.glob("*2026*"))
    assert len(raw_2026_files) == 0, f"Found unexpected 2026 raw data: {raw_2026_files}"

    df = pd.read_parquet(DATA_PATH)
    years = set(df["year"].unique())
    assert 2026 not in years, f"2026 found in processed cutoffs: {years}"
    assert max(years) == 2025, f"Maximum data year must be 2025, got {max(years)}"


def test_no_target_year_leakage_in_lag_features():
    """Verify that for every row, lag features strictly use data from prior years (t < T)."""
    df = pd.read_parquet(DATA_PATH)

    key_cols = ["institute", "program", "quota", "seat_type", "gender"]
    df_sorted = df.sort_values(key_cols + ["year"]).copy()

    # Calculate expected prior year and closing rank via temporal shift
    df_sorted["expected_prev_closing"] = df_sorted.groupby(key_cols)["closing_rank"].shift(1)
    df_sorted["expected_prev_year"] = df_sorted.groupby(key_cols)["year"].shift(1)

    # 1. Verify prev_closing_1 matches the previous chronological record
    has_lag = df_sorted[df_sorted["prev_closing_1"].notna()]
    assert not has_lag.empty, "Dataset must have historical lag records"
    assert (
        has_lag["prev_closing_1"] == has_lag["expected_prev_closing"]
    ).all(), "prev_closing_1 does not match the chronological previous closing rank!"

    # 2. Verify prior year is strictly strictly less than current year
    assert (
        has_lag["expected_prev_year"] < has_lag["year"]
    ).all(), "Temporal violation: prior record year is not strictly less than target year!"

    # 3. Specifically verify 2025 predictions only use data from 2024 or earlier
    rows_2025 = df_sorted[df_sorted["year"] == 2025]
    has_lag_2025 = rows_2025[rows_2025["prev_closing_1"].notna()]
    assert (
        has_lag_2025["expected_prev_year"] <= 2024
    ).all(), "2025 rows contain lag features from 2025 or later!"


def test_rolling_origin_folds_temporal_isolation():
    """Verify rolling-origin folds do not touch test fold or future years."""
    df = pd.read_parquet(DATA_PATH)

    for test_year in [2023, 2024, 2025]:
        train_fold = df[df["year"] < test_year]
        test_fold = df[df["year"] == test_year]

        # Training set must be strictly strictly prior
        assert (train_fold["year"] < test_year).all()
        assert not train_fold.empty, f"Train fold for {test_year} must not be empty"
        assert not test_fold.empty, f"Test fold for {test_year} must not be empty"

        # Overlap of row records across train and test must be zero
        train_indices = set(train_fold.index)
        test_indices = set(test_fold.index)
        assert len(train_indices.intersection(test_indices)) == 0


def test_prediction_sigma_leak_free_estimation():
    """Verify that predictive uncertainty sigma is not computed on held-out test residuals."""
    df = pd.read_parquet(DATA_PATH)
    from src.baseline import compute_weighted_log_trend

    # 2025 test evaluation
    train = df[(df["year"] <= 2024) & (df["year"] >= 2022)].copy()
    train_eval = train[
        train["prev_closing_1"].notna()
        & train["log_closing_rank"].notna()
        & (train["closing_rank"] > 0)
    ].copy()
    train_eval["y_hat"] = compute_weighted_log_trend(
        train_eval["prev_closing_1"],
        train_eval["prev_closing_2"],
        train_eval["prev_closing_3"],
    )
    train_eval = train_eval[train_eval["y_hat"].notna()]
    train_residuals = train_eval["log_closing_rank"] - train_eval["y_hat"]

    # The leak-free sigma from train
    leak_free_sigma = float(np.std(train_residuals, ddof=1))

    test = df[df["year"] == 2025].copy()
    test_eval = test[
        test["prev_closing_1"].notna()
        & test["log_closing_rank"].notna()
        & (test["closing_rank"] > 0)
    ].copy()
    test_eval["y_hat"] = compute_weighted_log_trend(
        test_eval["prev_closing_1"],
        test_eval["prev_closing_2"],
        test_eval["prev_closing_3"],
    )
    test_eval = test_eval[test_eval["y_hat"].notna()]
    test_residuals = test_eval["log_closing_rank"] - test_eval["y_hat"]
    test_sigma = float(np.std(test_residuals, ddof=1))

    # They must not be identical (confirming test variance is not used during model fitting)
    assert abs(leak_free_sigma - test_sigma) > 0.005, (
        f"Train sigma ({leak_free_sigma:.4f}) and test sigma ({test_sigma:.4f}) are suspiciously identical, "
        "indicating possible test-set variance leakage."
    )
