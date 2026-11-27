"""Generates and verifies notebooks/01_eda.ipynb and notebooks/02_modeling_and_validation.ipynb."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    }


def create_markdown_cell(md: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in md.strip().split("\n")],
    }


def build_notebook(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.16",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate_eda_notebook():
    cells = [
        create_markdown_cell("""# 01 - Exploratory Data Analysis (EDA)
### JoSAA Counselling Cutoffs (2021 - 2025)
**Platform:** Vikalp Admission-Probability Engine
**Objective:** Characterize cutoff distributions, missingness patterns, cutoff drift dynamics, and category quota scaling."""),
        create_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Ensure root is in path
root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

DATA_PATH = root_dir / "data" / "processed" / "cutoffs.parquet"
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} total cutoff records.")"""),
        create_markdown_cell("""## 1. Dataset Overview & Annual Offerings
Examine the record count by year and verify that JoSAA cycles 2021 through 2025 are present."""),
        create_code_cell("""yearly_counts = df["year"].value_counts().sort_index()
print("Record count by JoSAA cycle:")
print(yearly_counts)

institutes_per_year = df.groupby("year")["institute"].nunique()
programs_per_year = df.groupby("year")["program"].nunique()

summary_by_year = pd.DataFrame({
    "Records": yearly_counts,
    "Institutes": institutes_per_year,
    "Programs": programs_per_year,
})
summary_by_year"""),
        create_markdown_cell("""## 2. Missingness Analysis & Sparse History
Quantify how many offerings have full 3-year history vs. sparse history (< 3 years of lags)."""),
        create_code_cell("""df_2025 = df[df["year"] == 2025].copy()
total_2025 = len(df_2025)

has_lag1 = df_2025["prev_closing_1"].notna().sum()
has_lag2 = df_2025["prev_closing_2"].notna().sum()
has_lag3 = df_2025["prev_closing_3"].notna().sum()

print(f"2025 Total Offerings        : {total_2025:,}")
print(f"Offerings with >= 1 Yr Lag : {has_lag1:,} ({has_lag1 / total_2025:.1%})")
print(f"Offerings with >= 2 Yrs Lag: {has_lag2:,} ({has_lag2 / total_2025:.1%})")
print(f"Offerings with >= 3 Yrs Lag: {has_lag3:,} ({has_lag3 / total_2025:.1%})")
print(f"Sparse History (< 3 Yrs)   : {total_2025 - has_lag3:,} ({(total_2025 - has_lag3) / total_2025:.1%})")"""),
        create_markdown_cell(r"""## 3. Cutoff Drift Dynamics (Year-over-Year Volatility)
Analyze how closing ranks shift year-over-year in log-space ($\Delta = \ln(C_t) - \ln(C_{t-1})$)."""),
        create_code_cell("""drift_eval = df[df["prev_closing_1"].notna() & (df["prev_closing_1"] > 0) & (df["closing_rank"] > 0)].copy()
drift_eval["log_drift"] = np.log(drift_eval["closing_rank"]) - np.log(drift_eval["prev_closing_1"])

drift_by_tier = drift_eval.groupby("institute_type")["log_drift"].agg(["count", "mean", "std", "median"])
drift_by_tier["std_rank_percent"] = (np.exp(drift_by_tier["std"]) - 1.0) * 100
print("Annual Log-Cutoff Drift by Institute Tier:")
drift_by_tier"""),
        create_markdown_cell("""## 4. Category Scaling (CRL vs Category Ranks)
Examine closing rank scales across different quota seat pools (OPEN vs OBC-NCL vs SC vs ST)."""),
        create_code_cell("""cat_summary = df_2025.groupby(["institute_type", "seat_type"])["closing_rank"].agg(["count", "median", "mean"]).reset_index()
cat_summary = cat_summary[cat_summary["count"] > 50].sort_values(["institute_type", "median"])
cat_summary.head(15)"""),
        create_markdown_cell("""## 5. Rank Distance to Admission Probability Mapping
Illustrate the normal cumulative distribution function mapping $P(\text{admit}) = 1 - \\Phi\\left(\frac{\\ln(R) - \\hat{y}}{\\sigma}\right)$."""),
        create_code_cell("""z_vals = np.linspace(-3, 3, 100)
p_admit = 1.0 - norm.cdf(z_vals)

fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
ax.plot(z_vals, p_admit * 100, color="#2563EB", linewidth=2.5, label="P(Admit)")
ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="z = 0 (Expected Cutoff, P=50%)")
ax.axhline(80, color="#166534", linestyle=":", label="Safe Zone (P >= 80%)")
ax.axhline(30, color="#D97706", linestyle=":", label="Target Zone (30% <= P < 80%)")
ax.set_title("Mapping Standardized Rank Distance z to Admission Probability", fontweight="bold")
ax.set_xlabel("Normalized Distance z = (ln(Rank) - y_hat) / sigma")
ax.set_ylabel("Estimated Admission Odds (%)")
ax.grid(alpha=0.4)
ax.legend()
plt.tight_layout()
plt.close()"""),
    ]
    nb = build_notebook(cells)
    out_path = NOTEBOOKS_DIR / "01_eda.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_path}")


def generate_modeling_notebook():
    cells = [
        create_markdown_cell("""# 02 - Modeling, Rolling-Origin Backtesting & Calibration
### Evaluating Predictive Accuracy, Uncertainty, and Subgroup Reliability
**Platform:** Vikalp Admission-Probability Engine
**Objective:** Compare candidate models across rolling folds, validate conformal prediction intervals, and audit probability calibration."""),
        create_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

root_dir = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.baseline import compute_weighted_log_trend
from src.predict import Z_80

DATA_PATH = root_dir / "data" / "processed" / "cutoffs.parquet"
df = pd.read_parquet(DATA_PATH)"""),
        create_markdown_cell("""## 1. Rolling-Origin Cross-Validation Results
Review the 3-year rolling origin benchmark (2023, 2024, 2025) comparing Naive, Weighted Trend, Ridge, and LightGBM."""),
        create_code_cell("""csv_path = root_dir / "outputs" / "model_comparison" / "model_comparison_results.csv"
if csv_path.exists():
    res_df = pd.read_csv(csv_path)
    summary_table = res_df.groupby("model")[["mae", "mean_rel_error", "coverage_80", "brier_score", "ece"]].mean().reset_index()
    print("3-Year Rolling Origin Summary:")
    print(summary_table.sort_values("mae").to_string(index=False))
else:
    print("Run scripts/run_model_comparison.py first to populate benchmark results.")"""),
        create_markdown_cell("""## 2. Held-Out 2025 Primary Evaluation with Clustered CIs
Evaluate the primary production model (Weighted Trend) on held-out 2025 round 5 cutoffs."""),
        create_code_cell("""ci_path = root_dir / "outputs" / "backtest" / "bootstrap_cis.json"
if ci_path.exists():
    import json
    with open(ci_path) as f:
        cis = json.load(f)
    print("Audited Bootstrap 95% Confidence Intervals (2025):")
    for k, v in cis["metrics"].items():
        print(f"  {k:<20}: {v['point_estimate']:.4f} [95% CI: {v['ci_95_lower']:.4f}, {v['ci_95_upper']:.4f}]")"""),
        create_markdown_cell("""## 3. Conformal Prediction vs Normal-Sigma Intervals
Compare non-parametric conformal prediction intervals against parametric Gaussian-sigma intervals."""),
        create_code_cell("""conf_path = root_dir / "outputs" / "calibration" / "conformal_vs_gaussian.csv"
if conf_path.exists():
    conf_df = pd.read_csv(conf_path)
    print("Conformal vs Gaussian Coverage Comparison:")
    print(conf_df.to_string(index=False))"""),
        create_markdown_cell("""## 4. Probability Calibration (Raw vs Platt vs Isotonic)
Examine reliability curves and Expected Calibration Error (ECE) across post-hoc calibration methods."""),
        create_code_cell("""calib_path = root_dir / "outputs" / "calibration" / "calibration_models_comparison.csv"
if calib_path.exists():
    cal_df = pd.read_csv(calib_path)
    print("Probability Calibration Models:")
    print(cal_df.to_string(index=False))"""),
        create_markdown_cell("""## 5. Subgroup Calibration Diagnoses
Examine calibration error and model bias across institute tiers, history lengths, and candidate rank ranges."""),
        create_code_cell("""sub_path = root_dir / "outputs" / "calibration" / "subgroup_calibration.csv"
if sub_path.exists():
    sub_df = pd.read_csv(sub_path)
    print("Subgroup Calibration Diagnoses:")
    print(sub_df[["category", "subgroup", "sample_count", "mae_log", "brier_score", "ece", "diagnosis"]].to_string(index=False))"""),
    ]
    nb = build_notebook(cells)
    out_path = NOTEBOOKS_DIR / "02_modeling_and_validation.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_eda_notebook()
    generate_modeling_notebook()
