"""LightGBM regressor training and model validation.

Trains a LightGBM model to predict log closing rank using historical lag features:
- n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=42
- Time-based split: train on years <= 2024, held-out validation on 2025
- Compares MAE and 80% interval coverage against naive and weighted-trend baselines
- Saves model artifact to models/lgbm.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baseline import compute_weighted_log_trend

Z_80 = 1.28155

CATEGORICAL_FEATURES = ["institute_type", "branch_group", "quota", "seat_type"]
NUMERIC_FEATURES = [
    "prev_closing_1",
    "prev_closing_2",
    "prev_closing_3",
    "mean_last_2",
    "trend_slope",
    "delta_1yr",
    "years_of_history",
    "log_prev_1",
    "log_prev_2",
    "log_prev_3",
    "log_mean_2",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare log transformations and categorical dtypes for LightGBM."""
    df_feat = df.copy()

    with np.errstate(invalid="ignore", divide="ignore"):
        df_feat["log_prev_1"] = np.where(
            df_feat["prev_closing_1"].notna() & (df_feat["prev_closing_1"] > 0),
            np.log(df_feat["prev_closing_1"]),
            np.nan,
        )
        df_feat["log_prev_2"] = np.where(
            df_feat["prev_closing_2"].notna() & (df_feat["prev_closing_2"] > 0),
            np.log(df_feat["prev_closing_2"]),
            np.nan,
        )
        df_feat["log_prev_3"] = np.where(
            df_feat["prev_closing_3"].notna() & (df_feat["prev_closing_3"] > 0),
            np.log(df_feat["prev_closing_3"]),
            np.nan,
        )
        df_feat["log_mean_2"] = np.where(
            df_feat["mean_last_2"].notna() & (df_feat["mean_last_2"] > 0),
            np.log(df_feat["mean_last_2"]),
            np.nan,
        )

    for col in CATEGORICAL_FEATURES:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("category")

    return df_feat


def train_and_validate(
    data_path: Path,
    model_save_path: Optional[Path] = None,
    split_year: int = 2024,
    val_year: int = 2025,
) -> Tuple[lgb.LGBMRegressor, Dict[str, Dict[str, float]]]:
    """Train LightGBM on train set and evaluate against baselines on validation set."""
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run Phase 2 first.")

    raw_df = pd.read_parquet(data_path)

    # Filter keys with >= 2 years of history as per rules.md
    df = raw_df[
        raw_df["prev_closing_1"].notna() & raw_df["prev_closing_2"].notna()
    ].copy()
    df = prepare_features(df)

    train_mask = df["year"] <= split_year
    val_mask = df["year"] == val_year

    train_data = df[train_mask]
    val_data = df[val_mask]

    if train_data.empty or val_data.empty:
        raise ValueError(f"Insufficient data for split train <= {split_year}, val == {val_year}")

    X_train = train_data[ALL_FEATURES]
    y_train = train_data["log_closing_rank"].values

    X_val = val_data[ALL_FEATURES]
    y_val = val_data["log_closing_rank"].values

    # Train LightGBM regressor per specifications
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 1. LightGBM Predictions
    preds_lgb = model.predict(X_val)
    res_lgb = y_val - preds_lgb
    mae_lgb = float(np.mean(np.abs(res_lgb)))
    sigma_lgb = float(np.std(res_lgb, ddof=1))
    cov_80_lgb = float(np.mean((res_lgb >= -Z_80 * sigma_lgb) & (res_lgb <= Z_80 * sigma_lgb)))

    # 2. Naive Baseline (Last Year's Closing Rank)
    y_naive = X_val["log_prev_1"].values
    res_naive = y_val - y_naive
    mae_naive = float(np.mean(np.abs(res_naive)))
    sigma_naive = float(np.std(res_naive, ddof=1))
    cov_80_naive = float(np.mean((res_naive >= -Z_80 * sigma_naive) & (res_naive <= Z_80 * sigma_naive)))

    # 3. Weighted-Trend Baseline (0.5, 0.3, 0.2)
    y_weighted = compute_weighted_log_trend(
        val_data["prev_closing_1"].values,
        val_data["prev_closing_2"].values,
        val_data["prev_closing_3"].values,
    )
    res_weighted = y_val - y_weighted
    mae_weighted = float(np.mean(np.abs(res_weighted)))
    sigma_weighted = float(np.std(res_weighted, ddof=1))
    cov_80_weighted = float(np.mean((res_weighted >= -Z_80 * sigma_weighted) & (res_weighted <= Z_80 * sigma_weighted)))

    metrics = {
        "Baseline (last year)": {
            "mae": mae_naive,
            "sigma": sigma_naive,
            "coverage_80": cov_80_naive,
        },
        "Weighted trend": {
            "mae": mae_weighted,
            "sigma": sigma_weighted,
            "coverage_80": cov_80_weighted,
        },
        "LightGBM": {
            "mae": mae_lgb,
            "sigma": sigma_lgb,
            "coverage_80": cov_80_lgb,
        },
    }

    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"Saved LightGBM model to {model_save_path}")

    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM cutoff predictor.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/cutoffs.parquet"),
        help="Path to cutoffs parquet file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/lgbm.joblib"),
        help="Path to save trained model",
    )
    args = parser.parse_args()

    model, metrics = train_and_validate(
        data_path=args.data_path,
        model_save_path=args.model_path,
        split_year=2024,
        val_year=2025,
    )

    print("\n" + "=" * 65)
    print("MODEL VALIDATION COMPARISON (HELD-OUT YEAR 2025)")
    print("=" * 65)
    print(f"{'Model':<25} | {'MAE (log rank)':<15} | {'80% Coverage':<12} | {'Sigma':<8}")
    print("-" * 65)
    for name, m in metrics.items():
        print(f"{name:<25} | {m['mae']:<15.4f} | {m['coverage_80']*100:<11.2f}% | {m['sigma']:<8.4f}")
    print("=" * 65)

    lgb_mae = metrics["LightGBM"]["mae"]
    base_mae = metrics["Weighted trend"]["mae"]
    naive_mae = metrics["Baseline (last year)"]["mae"]

    print("\nKey Takeaways:")
    if lgb_mae < naive_mae:
        print(f"  + LightGBM beats the naive last-year baseline ({lgb_mae:.4f} vs {naive_mae:.4f}).")
    else:
        print(f"  - LightGBM did not beat the naive last-year baseline ({lgb_mae:.4f} vs {naive_mae:.4f}).")

    if lgb_mae <= base_mae:
        print(f"  + LightGBM beats the weighted-trend baseline ({lgb_mae:.4f} vs {base_mae:.4f}).")
    else:
        print(f"  * Weighted-trend baseline ({base_mae:.4f}) outperforms LightGBM ({lgb_mae:.4f}).")
        print("    Per rules.md and phases.md: Not tuning further; preserving weighted-trend baseline as primary predictor.")


if __name__ == "__main__":
    main()
