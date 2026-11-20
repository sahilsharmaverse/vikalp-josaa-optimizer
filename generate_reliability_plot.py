"""Generate reliability and calibration plots for held-out year validation.
Saves to docs/reliability.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.baseline import compute_weighted_log_trend
from src.optimize import DEFAULT_SIGMA

def generate_reliability_plot():
    data_path = Path("data/processed/cutoffs.parquet")
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / "reliability.png"

    df = pd.read_parquet(data_path)
    # Filter validation year 2025 with >= 2 years of history
    val_df = df[
        (df["year"] == 2025) &
        df["prev_closing_1"].notna() &
        df["prev_closing_2"].notna()
    ].copy()

    y_actual = val_df["log_closing_rank"].values
    y_pred = compute_weighted_log_trend(
        val_df["prev_closing_1"].values,
        val_df["prev_closing_2"].values,
        val_df["prev_closing_3"].values,
    )

    residuals = y_actual - y_pred
    sigma = float(np.std(residuals, ddof=1))
    mae = float(np.mean(np.abs(residuals)))

    # Nominal 80% interval
    z_80 = 1.28155
    in_80 = (residuals >= -z_80 * sigma) & (residuals <= z_80 * sigma)
    cov_80 = np.mean(in_80) * 100.0

    print(f"Validation year: 2025 (N = {len(val_df)} rows)")
    print(f"Residual MAE (log rank): {mae:.4f}")
    print(f"Residual sigma: {sigma:.4f} (Model DEFAULT_SIGMA: {DEFAULT_SIGMA:.4f})")
    print(f"80% Interval Coverage on 2025: {cov_80:.2f}%")

    # Evaluate multiple nominal coverage levels for calibration curve
    nominal_levels = np.linspace(0.10, 0.98, 25)
    empirical_coverages = []
    for p in nominal_levels:
        z = norm.ppf(0.5 + p / 2.0)
        emp_cov = np.mean((residuals >= -z * sigma) & (residuals <= z * sigma))
        empirical_coverages.append(emp_cov)

    empirical_coverages = np.array(empirical_coverages) * 100.0
    nominal_pct = nominal_levels * 100.0

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel 1: Reliability / Calibration Diagram
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.6, label="Perfect Calibration (Ideal)")
    ax1.plot(nominal_pct, empirical_coverages, "o-", color="#1F2A44", lw=2.2, ms=6, label="Empirical Coverage (2025)")

    # Highlight 80% point
    ax1.axvline(80, color="#DC2626", linestyle=":", alpha=0.7)
    ax1.axhline(cov_80, color="#DC2626", linestyle=":", alpha=0.7)
    ax1.plot(80, cov_80, "r*", ms=14, label=f"Nominal 80% -> Empirical {cov_80:.1f}%")

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Nominal Credible Interval Coverage (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Coverage on Test Set (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Prediction Interval Calibration (Held-Out 2025)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Panel 2: Log-Residual Density vs Normal Distribution
    count, bins, _ = ax2.hist(
        residuals, bins=60, density=True, alpha=0.65, color="#2563EB", edgecolor="#1D4ED8", label=f"Empirical Residuals (N={len(residuals)})"
    )
    x_axis = np.linspace(bins[0], bins[-1], 200)
    normal_density = norm.pdf(x_axis, loc=0, scale=sigma)
    ax2.plot(x_axis, normal_density, "r-", lw=2.5, label=f"Normal N(0, {sigma:.3f}²)")

    # Mark 80% bounds
    ax2.axvline(-z_80 * sigma, color="#DC2626", linestyle="--", label=f"±1.28σ (80% bounds: {cov_80:.1f}% covered)")
    ax2.axvline(z_80 * sigma, color="#DC2626", linestyle="--")

    ax2.set_xlabel("Log-Scale Residual: ln(CR_actual) - ln(CR_pred)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.set_title("Residual Error Distribution vs Gaussian Assumption", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Reliability plot saved successfully to {out_path}")

if __name__ == "__main__":
    generate_reliability_plot()
