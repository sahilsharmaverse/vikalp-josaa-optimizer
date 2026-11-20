"""Baseline prediction model for JoSAA closing ranks.

Calculates weighted average of the last 2-3 years of log closing ranks:
- 3 prior years: weights = (0.5, 0.3, 0.2)
- 2 prior years: weights = (0.5/0.8, 0.3/0.8) = (0.625, 0.375)
- 1 prior year: weight = 1.0 (fallback)
Measures residual sigma and 80% interval coverage on a held-out year.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

Z_80 = 1.28155  # 80% central prediction interval critical value for N(0, 1)


@dataclass
class BaselineModel:
    """Holds fitted baseline parameters and evaluation metrics."""
    weights: Tuple[float, float, float]
    residual_sigma: float
    mae_baseline: float
    mae_naive: float
    interval_coverage_80: float
    held_out_year: int
    num_evaluated_rows: int


def compute_weighted_log_trend(
    p1: Union[float, np.ndarray, pd.Series],
    p2: Union[float, np.ndarray, pd.Series],
    p3: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """Compute weighted log closing rank using available history (weights: 0.5, 0.3, 0.2)."""
    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    p3_arr = np.asarray(p3, dtype=float)

    # Safe log transform
    with np.errstate(invalid="ignore", divide="ignore"):
        lp1 = np.where(p1_arr > 0, np.log(p1_arr), np.nan)
        lp2 = np.where(p2_arr > 0, np.log(p2_arr), np.nan)
        lp3 = np.where(p3_arr > 0, np.log(p3_arr), np.nan)

    has_1 = ~np.isnan(lp1)
    has_2 = has_1 & ~np.isnan(lp2)
    has_3 = has_2 & ~np.isnan(lp3)

    y_hat_3 = 0.5 * lp1 + 0.3 * lp2 + 0.2 * lp3
    y_hat_2 = (0.5 * lp1 + 0.3 * lp2) / 0.8
    y_hat_1 = lp1

    y_hat = np.where(has_3, y_hat_3, np.where(has_2, y_hat_2, np.where(has_1, y_hat_1, np.nan)))

    if isinstance(p1, (int, float)) and not isinstance(p1, np.ndarray):
        return float(y_hat.item()) if not np.isnan(y_hat.item()) else np.nan

    return y_hat


def predict_baseline_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add baseline log rank prediction (y_hat_baseline) and predicted closing rank."""
    out = df.copy()
    out["y_hat_baseline"] = compute_weighted_log_trend(
        out["prev_closing_1"],
        out["prev_closing_2"],
        out["prev_closing_3"],
    )
    out["pred_closing_rank_baseline"] = np.round(np.exp(out["y_hat_baseline"]))
    return out


def evaluate_baseline_on_held_out(
    df: pd.DataFrame, held_out_year: int = 2025
) -> BaselineModel:
    """Evaluate weighted trend baseline against naive last-year baseline on a held-out year."""
    test_data = df[df["year"] == held_out_year].copy()
    if test_data.empty:
        raise ValueError(f"No records found for held-out year {held_out_year}")

    scored = predict_baseline_for_df(test_data)

    # Require at least 2 years of history for robust baseline evaluation
    valid = scored[
        scored["y_hat_baseline"].notna()
        & scored["log_closing_rank"].notna()
        & scored["prev_closing_1"].notna()
        & scored["prev_closing_2"].notna()
    ].copy()

    if valid.empty:
        raise RuntimeError("No valid rows with >=2 years history to evaluate baseline.")

    y_true = valid["log_closing_rank"].values
    y_hat_base = valid["y_hat_baseline"].values
    y_hat_naive = np.log(valid["prev_closing_1"].values)

    res_base = y_true - y_hat_base
    res_naive = y_true - y_hat_naive

    mae_base = float(np.mean(np.abs(res_base)))
    mae_naive = float(np.mean(np.abs(res_naive)))
    sigma = float(np.std(res_base, ddof=1))

    # 80% coverage check: y_true in [y_hat - 1.28*sigma, y_hat + 1.28*sigma]
    lower_bound = y_hat_base - Z_80 * sigma
    upper_bound = y_hat_base + Z_80 * sigma
    coverage_80 = float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))

    return BaselineModel(
        weights=(0.5, 0.3, 0.2),
        residual_sigma=sigma,
        mae_baseline=mae_base,
        mae_naive=mae_naive,
        interval_coverage_80=coverage_80,
        held_out_year=held_out_year,
        num_evaluated_rows=len(valid),
    )


def main() -> None:
    parquet_path = Path("data/processed/cutoffs.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} does not exist. Run Phase 2 first.")

    df = pd.read_parquet(parquet_path)
    model = evaluate_baseline_on_held_out(df, held_out_year=2025)

    print("=" * 60)
    print(f"BASELINE EVALUATION ON HELD-OUT YEAR {model.held_out_year}")
    print("=" * 60)
    print(f"Evaluated Rows (>= 2 yrs history): {model.num_evaluated_rows:,}")
    print(f"Weighted Trend MAE (log rank)    : {model.mae_baseline:.4f}")
    print(f"Naive Last-Year MAE (log rank)   : {model.mae_naive:.4f}")
    print(f"Improvement over Naive           : {(model.mae_naive - model.mae_baseline):.4f} log points")
    print(f"Residual Sigma (sigma)           : {model.residual_sigma:.4f}")
    print(f"80% Interval Coverage            : {model.interval_coverage_80 * 100:.2f}% (Target: 70-90%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
