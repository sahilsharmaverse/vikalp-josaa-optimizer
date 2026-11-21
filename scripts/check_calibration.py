"""Calibration analysis script for held-out years.

Evaluates coverage of 50%, 80%, 90% prediction intervals, Brier score,
and PIT values u = Phi((ln(actual_closing) - y_hat) / sigma).
Generates docs/calibration.png.
"""

from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from src.baseline import compute_weighted_log_trend
from src.simulation import get_model_metrics

def evaluate_calibration():
    df = pd.read_parquet("data/processed/cutoffs.parquet")
    metrics = get_model_metrics()
    current_sigma = metrics.get("sigma", 0.2285)

    z_50 = norm.ppf(0.75)   # ~0.67449
    z_80 = norm.ppf(0.90)   # ~1.28155
    z_90 = norm.ppf(0.95)   # ~1.64485

    results = []

    # Evaluable held-out years
    eval_years = [2023, 2024, 2025]

    for yr in eval_years:
        sub = df[(df["year"] == yr) & df["prev_closing_1"].notna() & df["prev_closing_2"].notna()].copy()
        if sub.empty:
            continue

        y_true = sub["log_closing_rank"].values
        y_pred = compute_weighted_log_trend(
            sub["prev_closing_1"].values,
            sub["prev_closing_2"].values,
            sub["prev_closing_3"].values,
        )

        valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
        y_t = y_true[valid_mask]
        y_p = y_pred[valid_mask]

        residuals = y_t - y_p
        raw_sigma_yr = float(np.std(residuals, ddof=1))
        mae_yr = float(np.mean(np.abs(residuals)))

        # Coverage using current_sigma = 0.2285
        cov_50 = np.mean(np.abs(residuals) <= z_50 * current_sigma) * 100.0
        cov_80 = np.mean(np.abs(residuals) <= z_80 * current_sigma) * 100.0
        cov_90 = np.mean(np.abs(residuals) <= z_90 * current_sigma) * 100.0

        # PIT values: u = Phi((ln(actual) - y_hat) / sigma)
        pit_values = norm.cdf(residuals / current_sigma)

        # Integrated Brier score over probability grid p in [0.05, 0.95]
        prob_grid = np.linspace(0.05, 0.95, 19)
        brier_scores = []
        for p in prob_grid:
            # model predicts admit prob p at rank threshold r = exp(y_hat + sigma * Phi^{-1}(1-p))
            # outcome is 1 if actual_closing >= r <=> ln(actual_closing) - y_hat >= sigma * norm.ppf(1-p)
            threshold = current_sigma * norm.ppf(1.0 - p)
            outcome = (residuals >= threshold).astype(float)
            brier_scores.append(np.mean((p - outcome) ** 2))
        avg_brier = float(np.mean(brier_scores))

        results.append({
            "year": yr,
            "n_rows": len(y_t),
            "mae": mae_yr,
            "raw_sigma": raw_sigma_yr,
            "cov_50": cov_50,
            "cov_80": cov_80,
            "cov_90": cov_90,
            "brier": avg_brier,
            "residuals": residuals,
            "pit": pit_values,
        })

    print("=" * 70)
    print("CALIBRATION METRICS WITH CURRENT SIGMA =", current_sigma)
    print("=" * 70)
    for r in results:
        print(f"Year {r['year']} (N={r['n_rows']:,}):")
        print(f"  MAE: {r['mae']:.4f} | Raw Year Std: {r['raw_sigma']:.4f}")
        print(f"  50% Interval Coverage: {r['cov_50']:.2f}% (Nominal: 50.0%)")
        print(f"  80% Interval Coverage: {r['cov_80']:.2f}% (Nominal: 80.0%)")
        print(f"  90% Interval Coverage: {r['cov_90']:.2f}% (Nominal: 90.0%)")
        print(f"  Brier Score: {r['brier']:.4f}")
        print("-" * 70)

    # If 80% coverage > 87% in most years, find scale factor on validation year (2025)
    val_res = next(r for r in results if r["year"] == 2025)
    val_residuals = val_res["residuals"]

    # Target: 80% coverage on 2025: fraction(|res| <= z_80 * (scale * current_sigma)) == 0.80
    # quantile of |res| at 0.80
    q80 = float(np.quantile(np.abs(val_residuals), 0.80))
    scale_factor = q80 / (z_80 * current_sigma)
    new_sigma = scale_factor * current_sigma

    print(f"\nPROPOSED SIGMA RESCALING:")
    print(f"Empirical 80th percentile of |residual| in 2025: {q80:.4f}")
    print(f"Theoretical 80th percentile with sigma={current_sigma:.4f}: {z_80 * current_sigma:.4f}")
    print(f"Proposed Scale Factor (fitted on 2025): {scale_factor:.4f}")
    print(f"Scaled Sigma: {new_sigma:.4f}")

    print("\nCHECK ON OTHER YEARS (BEFORE vs AFTER):")
    for r in results:
        res = r["residuals"]
        cov_80_before = r["cov_80"]
        cov_80_after = np.mean(np.abs(res) <= z_80 * new_sigma) * 100.0
        cov_50_after = np.mean(np.abs(res) <= z_50 * new_sigma) * 100.0
        cov_90_after = np.mean(np.abs(res) <= z_90 * new_sigma) * 100.0

        prob_grid = np.linspace(0.05, 0.95, 19)
        brier_scores_after = []
        for p in prob_grid:
            threshold = new_sigma * norm.ppf(1.0 - p)
            outcome = (res >= threshold).astype(float)
            brier_scores_after.append(np.mean((p - outcome) ** 2))
        avg_brier_after = float(np.mean(brier_scores_after))

        print(f"Year {r['year']}:")
        print(f"  80% Coverage: {cov_80_before:.2f}% -> {cov_80_after:.2f}%")
        print(f"  50% Coverage: {r['cov_50']:.2f}% -> {cov_50_after:.2f}%")
        print(f"  90% Coverage: {r['cov_90']:.2f}% -> {cov_90_after:.2f}%")
        print(f"  Brier Score:  {r['brier']:.4f} -> {avg_brier_after:.4f}")

    # Generate docs/calibration.png
    out_dir = Path("docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "calibration.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Reliability Diagram for 2024 and 2025
    nominal_levels = np.linspace(0.05, 0.95, 25)
    nominal_pct = nominal_levels * 100.0

    ax1.plot([0, 100], [0, 100], "k--", alpha=0.7, label="Perfect Calibration (Ideal)")

    colors_map = {2023: "#94A3B8", 2024: "#F59E0B", 2025: "#0EA5E9"}
    for r in results:
        res = r["residuals"]
        emp_cov = [np.mean(np.abs(res) <= norm.ppf(0.5 + p / 2.0) * current_sigma) * 100.0 for p in nominal_levels]
        ax1.plot(nominal_pct, emp_cov, "o-", color=colors_map[r["year"]], lw=2, ms=5, label=f"Year {r['year']} (cov_80={r['cov_80']:.1f}%)")

    ax1.axvline(80, color="#EF4444", linestyle=":", alpha=0.8)
    ax1.axhline(80, color="#EF4444", linestyle=":", alpha=0.8)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Nominal Prediction Interval (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Coverage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Prediction Interval Calibration (Held-Out Years)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: PIT Histogram u = Phi((ln(actual) - y_hat) / sigma) for 2025
    pit_2025 = val_res["pit"]
    ax2.hist(pit_2025, bins=20, density=True, color="#0EA5E9", alpha=0.7, edgecolor="#0284C7", label=f"2025 PIT (N={len(pit_2025):,})")
    ax2.axhline(1.0, color="#EF4444", linestyle="--", lw=2, label="Ideal Uniform U(0,1)")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("PIT Value u = Φ((ln(actual) − ŷ) / σ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title("Probability Integral Transform (PIT) Histogram", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper center", frameon=True)

    plt.tight_layout()
    fig.savefig(out_img, dpi=300)
    plt.close(fig)
    print(f"\nSaved calibration plot to {out_img}")

if __name__ == "__main__":
    evaluate_calibration()
