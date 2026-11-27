"""Clustered bootstrap evaluation for Vikalp (held-out 2025 backtest).

Resamples clusters at the (institute, program) level (N=837 clusters, 9,336 total rows)
with B=1,000 bootstrap iterations to account for within-series correlation.

Computes:
- Point estimates and 95% CIs for MAE (log rank)
- Point estimates and 95% CIs for Mean Relative Error
- Point estimates and 95% CIs for Empirical 80% Coverage (using leak-free train sigma)
- Point estimates and 95% CIs for Canonical Integrated Brier Score
- Difference statistics (Naive Baseline vs Weighted Trend)
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.baseline import compute_weighted_log_trend
from src.predict import Z_80

OUTPUT_DIR = Path("outputs/backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/processed/cutoffs.parquet")

def run_clustered_bootstrap(b_iterations: int = 1000, seed: int = 42):
    np.random.seed(seed)
    df = pd.read_parquet(DATA_PATH)

    # 1. Leak-free training split (t <= 2024)
    train_df = df[(df["year"] <= 2024) & (df["year"] >= 2022)].copy()
    train_eval = train_df[
        train_df["prev_closing_1"].notna()
        & train_df["log_closing_rank"].notna()
        & (train_df["closing_rank"] > 0)
    ].copy()
    train_eval["y_hat"] = compute_weighted_log_trend(
        train_eval["prev_closing_1"],
        train_eval["prev_closing_2"],
        train_eval["prev_closing_3"],
    )
    train_eval["y_naive"] = np.log(train_eval["prev_closing_1"])
    train_eval = train_eval[train_eval["y_hat"].notna()].copy()

    sigma_train_trend = float(np.std(train_eval["log_closing_rank"] - train_eval["y_hat"], ddof=1))
    sigma_train_naive = float(np.std(train_eval["log_closing_rank"] - train_eval["y_naive"], ddof=1))

    # 2. Test split (2025)
    test_df = df[df["year"] == 2025].copy()
    test_eval = test_df[
        test_df["prev_closing_1"].notna()
        & test_df["log_closing_rank"].notna()
        & (test_df["closing_rank"] > 0)
    ].copy()
    test_eval["y_hat"] = compute_weighted_log_trend(
        test_eval["prev_closing_1"],
        test_eval["prev_closing_2"],
        test_eval["prev_closing_3"],
    )
    test_eval["y_naive"] = np.log(test_eval["prev_closing_1"])
    test_eval = test_eval[test_eval["y_hat"].notna()].copy()

    test_eval["cluster_id"] = test_eval["institute"] + " | " + test_eval["program"]
    unique_clusters = test_eval["cluster_id"].unique()
    num_clusters = len(unique_clusters)

    print(f"Loaded test set: {len(test_eval):,} rows across {num_clusters} (institute, program) clusters.")
    print(f"Leak-free Train Sigmas -> Trend: {sigma_train_trend:.4f}, Naive: {sigma_train_naive:.4f}")

    # Group test rows by cluster index for fast bootstrap indexing
    cluster_groups = test_eval.groupby("cluster_id")
    cluster_arrays = {}
    for cid, g in cluster_groups:
        cluster_arrays[cid] = {
            "y_true": g["log_closing_rank"].values,
            "c_true": g["closing_rank"].values,
            "y_hat": g["y_hat"].values,
            "y_naive": g["y_naive"].values,
        }

    # Standardized z-grid for Canonical Integrated Brier Score
    z_grid = np.linspace(-2.5, 2.5, 21)

    def calc_metrics_for_arrays(y_t, c_t, y_h, y_n):
        res_t = y_t - y_h
        res_n = y_t - y_n

        mae_t = np.mean(np.abs(res_t))
        mae_n = np.mean(np.abs(res_n))
        mae_diff = mae_n - mae_t  # positive means trend is better

        rel_err_t = np.mean(np.abs(c_t - np.exp(y_h)) / c_t)
        rel_err_n = np.mean(np.abs(c_t - np.exp(y_n)) / c_t)

        cov_t = np.mean(np.abs(res_t) <= Z_80 * sigma_train_trend)
        cov_n = np.mean(np.abs(res_n) <= Z_80 * sigma_train_naive)

        # Integrated Brier
        all_p, all_a_t, all_a_n = [], [], []
        for z in z_grid:
            p = norm.cdf(-z)
            ln_r_t = y_h + z * sigma_train_trend
            ln_r_n = y_n + z * sigma_train_naive
            all_p.extend([p] * len(y_h))
            all_a_t.extend((ln_r_t <= y_t).astype(float))
            all_a_n.extend((ln_r_n <= y_t).astype(float))

        arr_p = np.array(all_p)
        brier_t = np.mean((arr_p - np.array(all_a_t))**2)
        brier_n = np.mean((arr_p - np.array(all_a_n))**2)

        return {
            "mae_trend": mae_t,
            "mae_naive": mae_n,
            "mae_diff": mae_diff,
            "rel_err_trend": rel_err_t,
            "rel_err_naive": rel_err_n,
            "cov80_trend": cov_t,
            "cov80_naive": cov_n,
            "brier_trend": brier_t,
            "brier_naive": brier_n,
        }

    # Point estimates on full 2025 dataset
    point_est = calc_metrics_for_arrays(
        test_eval["log_closing_rank"].values,
        test_eval["closing_rank"].values,
        test_eval["y_hat"].values,
        test_eval["y_naive"].values,
    )

    print("\nStarting Clustered Bootstrap (B=1,000 resamples)...")
    boot_records = []
    cluster_keys = np.array(list(cluster_arrays.keys()))

    for b in range(b_iterations):
        if (b + 1) % 200 == 0:
            print(f"  Iteration {b + 1}/{b_iterations}...")

        # Sample clusters with replacement
        sample_keys = np.random.choice(cluster_keys, size=num_clusters, replace=True)

        # Aggregate sampled cluster arrays
        b_y_true = np.concatenate([cluster_arrays[k]["y_true"] for k in sample_keys])
        b_c_true = np.concatenate([cluster_arrays[k]["c_true"] for k in sample_keys])
        b_y_hat = np.concatenate([cluster_arrays[k]["y_hat"] for k in sample_keys])
        b_y_naive = np.concatenate([cluster_arrays[k]["y_naive"] for k in sample_keys])

        b_metrics = calc_metrics_for_arrays(b_y_true, b_c_true, b_y_hat, b_y_naive)
        boot_records.append(b_metrics)

    boot_df = pd.DataFrame(boot_records)

    summary = {}
    print("\n" + "=" * 75)
    print("CANONICAL EVALUATION METRICS (2025 HELD-OUT BACKTEST) WITH 95% CLUSTERED CIs")
    print("=" * 75)
    print(f"{'Metric':<32} | {'Point Estimate':<15} | {'95% CI (Bootstrap)':<22}")
    print("-" * 75)

    for col in boot_df.columns:
        pt = point_est[col]
        ci_lower = float(np.percentile(boot_df[col], 2.5))
        ci_upper = float(np.percentile(boot_df[col], 97.5))
        summary[col] = {
            "point_estimate": float(pt),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
        }
        print(f"{col:<32} | {pt:<15.4f} | [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("=" * 75)

    # Save summary
    results_path = OUTPUT_DIR / "bootstrap_cis.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "num_eval_rows": len(test_eval),
                    "num_clusters": num_clusters,
                    "bootstrap_iterations": b_iterations,
                    "sigma_train_trend": sigma_train_trend,
                    "sigma_train_naive": sigma_train_naive,
                    "z_80": Z_80,
                },
                "metrics": summary,
            },
            f,
            indent=2,
        )
    print(f"\nSaved bootstrap summary to {results_path}")
    return summary

if __name__ == "__main__":
    run_clustered_bootstrap()
