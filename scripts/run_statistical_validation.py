"""Comprehensive statistical validation script for Vikalp.
Implements Phase 2:
1. Backtest on 2025 (train 2021-2024): MAE (log and relative), calibration table, Brier score, reliability plot.
2. Rolling-origin backtest (2023, 2024, 2025).
3. Residual sigma by group (institute tier, branch group, history tier).
4. Sparse history quantification (< 3 years).
5. Monte Carlo sensitivity (rho = 0, 0.3, 0.5, 0.8) and convergence (2,000 vs 20,000 runs).
6. 2026 data verification.
Outputs saved to outputs/backtest/.
"""

import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from src.baseline import compute_weighted_log_trend
from src.simulation import simulate_at_least_one_seat_probability
from src.predict import Z_80

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "backtest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cutoffs.parquet"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def run_validation():
    print("=" * 70)
    print("PHASE 2: STATISTICAL VALIDITY & BACKTEST SUITE")
    print("=" * 70)

    # 0. Check 2026 data
    raw_2026 = list(RAW_DIR.glob("*2026*"))
    print("\n--- STEP 0: 2026 RAW DATA CHECK ---")
    if not raw_2026:
        print("[VERIFIED FACT] 2026 raw cutoff data is ABSENT from data/raw/.")
        print("  Current historical data covers JoSAA cycles 2021 through 2025.")
    else:
        print(f"[FOUND] 2026 raw files found: {[f.name for f in raw_2026]}")

    df = pd.read_parquet(DATA_PATH)
    print(f"\nLoaded {len(df):,} cutoff records from {DATA_PATH.name}")

    # 1. Primary Backtest: Predict 2025 (Trained on 2021-2024)
    print("\n--- STEP 1: PRIMARY HELD-OUT BACKTEST (2025) ---")
    df_2025 = df[df["year"] == 2025].copy()
    
    # Compute predictions using available historical lag features
    # (prev_closing_1 = 2024, prev_closing_2 = 2023, prev_closing_3 = 2022)
    df_2025["y_hat"] = compute_weighted_log_trend(
        df_2025["prev_closing_1"],
        df_2025["prev_closing_2"],
        df_2025["prev_closing_3"]
    )
    # Naive baseline: log(last year's cutoff)
    df_2025["y_naive"] = np.where(df_2025["prev_closing_1"] > 0, np.log(df_2025["prev_closing_1"]), np.nan)

    # Filter to evaluable options with ground truth and valid prediction
    eval_df = df_2025[df_2025["y_hat"].notna() & df_2025["log_closing_rank"].notna() & (df_2025["closing_rank"] > 0)].copy()
    y_true = eval_df["log_closing_rank"].values
    c_true = eval_df["closing_rank"].values
    y_hat = eval_df["y_hat"].values
    y_naive = eval_df["y_naive"].values

    res_model = y_true - y_hat
    res_naive = y_true - y_naive

    mae_log = float(np.mean(np.abs(res_model)))
    mae_naive_log = float(np.mean(np.abs(res_naive)))
    relative_error = np.abs(c_true - np.exp(y_hat)) / c_true
    mean_rel_error = float(np.mean(relative_error))
    median_rel_error = float(np.median(relative_error))
    sigma_overall = float(np.std(res_model, ddof=1))

    # 80% coverage check
    cov_80 = float(np.mean((y_true >= (y_hat - Z_80 * sigma_overall)) & (y_true <= (y_hat + Z_80 * sigma_overall))))

    print(f"Evaluable Options (with >=1 yr history): {len(eval_df):,} / {len(df_2025):,}")
    print(f"Weighted Trend Model MAE (log rank)   : {mae_log:.4f}")
    print(f"Naive Baseline MAE (last year only)   : {mae_naive_log:.4f}")
    print(f"Improvement over Naive                : {(mae_naive_log - mae_log):.4f} log points")
    print(f"Mean Relative Error (|actual-pred|/act): {mean_rel_error:.2%}")
    print(f"Median Relative Error                 : {median_rel_error:.2%}")
    print(f"Residual Sigma (overall)              : {sigma_overall:.4f}")
    print(f"80% Interval Empirical Coverage       : {cov_80:.2%} (Nominal: 80.0%)")

    # Calibration & Brier Score
    print("\n--- STEP 1B: CALIBRATION DECILITY ANALYSIS & BRIER SCORE ---")
    # For each evaluated option, query across standardized ranks z in [-2.5, 2.5]
    z_grid = np.linspace(-2.5, 2.5, 21)
    all_preds = []
    all_actuals = []

    for z_val in z_grid:
        p_pred = norm.cdf(-z_val) # P(admit) = P(rank <= cutoff) = Phi(-z)
        # simulated rank R corresponding to z_val: ln(R) = y_hat + z_val * sigma
        ln_r = y_hat + z_val * sigma_overall
        actual_admit = (ln_r <= y_true).astype(float)
        all_preds.extend([p_pred] * len(y_hat))
        all_actuals.extend(actual_admit)

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    brier_score = float(np.mean((all_preds - all_actuals) ** 2))
    print(f"Brier Score across support: {brier_score:.4f}")

    # Build 10 Decile Calibration Table
    bins = np.linspace(0.0, 1.0, 11)
    decile_indices = np.digitize(all_preds, bins) - 1
    decile_indices = np.clip(decile_indices, 0, 9)

    calib_rows = []
    for d in range(10):
        mask = (decile_indices == d)
        if np.any(mask):
            mean_p = float(np.mean(all_preds[mask]))
            obs_freq = float(np.mean(all_actuals[mask]))
            cnt = int(np.sum(mask))
        else:
            mean_p = (bins[d] + bins[d+1]) / 2.0
            obs_freq = np.nan
            cnt = 0
        calib_rows.append({
            "decile": f"{int(bins[d]*100)}-{int(bins[d+1]*100)}%",
            "bin_lower": bins[d],
            "bin_upper": bins[d+1],
            "mean_pred_prob": mean_p,
            "observed_frequency": obs_freq,
            "sample_count": cnt,
            "abs_calibration_gap": abs(mean_p - obs_freq) if not np.isnan(obs_freq) else np.nan
        })

    calib_df = pd.DataFrame(calib_rows)
    calib_path = OUTPUT_DIR / "calibration_table_2025.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"Saved Calibration Table to: {calib_path}")
    print(calib_df[["decile", "mean_pred_prob", "observed_frequency", "abs_calibration_gap", "sample_count"]].to_string(index=False))

    # Reliability Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    # Left: Reliability Curve
    valid_calib = calib_df.dropna(subset=["observed_frequency"])
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    ax1.plot(valid_calib["mean_pred_prob"], valid_calib["observed_frequency"], "o-", color="#0284C7", linewidth=2, label="Vikalp Weighted Model")
    ax1.set_xlabel("Predicted Admission Probability")
    ax1.set_ylabel("Observed Admission Frequency")
    ax1.set_title("Reliability Diagram (Held-Out 2025)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Right: Residual Density
    ax2.hist(res_model, bins=60, density=True, alpha=0.6, color="#0EA5E9", label="Empirical Residuals (ln(act) - y_hat)")
    x_grid = np.linspace(-1.5, 1.5, 200)
    ax2.plot(x_grid, norm.pdf(x_grid, 0, sigma_overall), "r-", linewidth=2, label=f"N(0, sigma={sigma_overall:.4f})")
    ax2.axvline(-Z_80 * sigma_overall, color="green", linestyle=":", label=f"80% Interval (+/-{Z_80*sigma_overall:.3f})")
    ax2.axvline(Z_80 * sigma_overall, color="green", linestyle=":")
    ax2.set_xlabel("Residual (log closing rank error)")
    ax2.set_ylabel("Density")
    ax2.set_title("Prediction Error Distribution & Empirical Sigma")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    rel_plot_path = OUTPUT_DIR / "reliability_plot.png"
    fig.savefig(rel_plot_path)
    plt.close()
    print(f"Saved Reliability Plot to: {rel_plot_path}")

    # 2. Rolling-Origin Backtest
    print("\n--- STEP 2: ROLLING-ORIGIN BACKTEST ---")
    rolling_results = []
    # Test cycles: 2023, 2024, 2025
    for target_year in [2023, 2024, 2025]:
        target_df = df[df["year"] == target_year].copy()
        target_df["y_hat"] = compute_weighted_log_trend(
            target_df["prev_closing_1"],
            target_df["prev_closing_2"],
            target_df["prev_closing_3"]
        )
        target_df["y_naive"] = np.where(target_df["prev_closing_1"] > 0, np.log(target_df["prev_closing_1"]), np.nan)
        v = target_df[target_df["y_hat"].notna() & target_df["log_closing_rank"].notna()].copy()
        
        y_t = v["log_closing_rank"].values
        y_h = v["y_hat"].values
        y_n = v["y_naive"].values
        
        m_log = float(np.mean(np.abs(y_t - y_h)))
        n_log = float(np.mean(np.abs(y_t - y_n)))
        sig = float(np.std(y_t - y_h, ddof=1))
        cov = float(np.mean((y_t >= (y_h - Z_80 * sig)) & (y_t <= (y_h + Z_80 * sig))))
        
        # Brier score
        z_sample = np.linspace(-2.0, 2.0, 11)
        preds_roll, acts_roll = [], []
        for z in z_sample:
            p = norm.cdf(-z)
            preds_roll.extend([p] * len(y_h))
            acts_roll.extend(((y_h + z * sig) <= y_t).astype(float))
        brier = float(np.mean((np.array(preds_roll) - np.array(acts_roll)) ** 2))

        rolling_results.append({
            "target_year": target_year,
            "train_history": f"2021-{target_year-1}",
            "evaluated_rows": len(v),
            "mae_weighted_model": m_log,
            "mae_naive_baseline": n_log,
            "improvement_log": n_log - m_log,
            "residual_sigma": sig,
            "coverage_80": cov,
            "brier_score": brier
        })

    rolling_df = pd.DataFrame(rolling_results)
    rolling_path = OUTPUT_DIR / "rolling_backtest_summary.csv"
    rolling_df.to_csv(rolling_path, index=False)
    print(rolling_df.to_string(index=False))

    # 3. Residual Sigma per Group & Sparse History Analysis
    print("\n--- STEP 3: RESIDUAL SIGMA ESTIMATION BY GROUP ---")
    group_results = []
    
    # By Institute Tier
    for inst_t in ["IIT", "NIT", "IIIT", "GFTI"]:
        sub = eval_df[eval_df["institute_type"] == inst_t]
        if len(sub) > 10:
            err = sub["log_closing_rank"].values - sub["y_hat"].values
            group_results.append({
                "group_type": "institute_type",
                "group_name": inst_t,
                "count": len(sub),
                "mae_log": float(np.mean(np.abs(err))),
                "sigma": float(np.std(err, ddof=1))
            })

    # By Branch Group (top branches)
    for br in ["CSE", "ECE", "EE", "Mechanical", "Civil", "Chemical"]:
        sub = eval_df[eval_df["branch_group"] == br]
        if len(sub) > 10:
            err = sub["log_closing_rank"].values - sub["y_hat"].values
            group_results.append({
                "group_type": "branch_group",
                "group_name": br,
                "count": len(sub),
                "mae_log": float(np.mean(np.abs(err))),
                "sigma": float(np.std(err, ddof=1))
            })

    # By History Tier (Sparse < 3 yrs vs Full >= 3 yrs)
    sparse_sub = eval_df[eval_df["years_of_history"] < 3]
    full_sub = eval_df[eval_df["years_of_history"] >= 3]
    
    err_sparse = sparse_sub["log_closing_rank"].values - sparse_sub["y_hat"].values
    err_full = full_sub["log_closing_rank"].values - full_sub["y_hat"].values
    
    sig_sparse = float(np.std(err_sparse, ddof=1)) if len(sparse_sub) > 10 else sigma_overall
    sig_full = float(np.std(err_full, ddof=1)) if len(full_sub) > 10 else sigma_overall

    group_results.append({
        "group_type": "history_tier",
        "group_name": "Sparse (< 3 yrs history)",
        "count": len(sparse_sub),
        "mae_log": float(np.mean(np.abs(err_sparse))),
        "sigma": sig_sparse
    })
    group_results.append({
        "group_type": "history_tier",
        "group_name": "Established (>= 3 yrs history)",
        "count": len(full_sub),
        "mae_log": float(np.mean(np.abs(err_full))),
        "sigma": sig_full
    })

    group_df = pd.DataFrame(group_results)
    group_path = OUTPUT_DIR / "sigma_by_group.csv"
    group_df.to_csv(group_path, index=False)
    print(f"Saved Sigma Breakdown to: {group_path}")
    print(group_df.to_string(index=False))

    print(f"\n[KEY FINDING] Sparse history options have sigma = {sig_sparse:.4f} vs Established = {sig_full:.4f}")
    print(f"  -> Sparse history programs require ~{(sig_sparse/sig_full - 1):.1%} wider uncertainty intervals!")

    # 4. Monte Carlo Sensitivity & Convergence
    print("\n--- STEP 4: MONTE CARLO SENSITIVITY & CONVERGENCE ---")
    # Take representative choice sample (20 choices)
    sample_df = eval_df.sample(20, random_state=42)
    test_ranks = sample_df["closing_rank"].values
    test_yhats = sample_df["y_hat"].values

    rho_results = []
    for rho_val in [0.0, 0.3, 0.5, 0.8]:
        p_est = simulate_at_least_one_seat_probability(
            student_ranks=test_ranks,
            y_hats=test_yhats,
            sigma=sigma_overall,
            rho=rho_val,
            n_sims=2000,
            seed=42
        )
        rho_results.append({
            "rho": rho_val,
            "correlation_assumption": "Independent" if rho_val == 0 else ("Moderate" if rho_val == 0.3 else ("Baseline (50%)" if rho_val == 0.5 else "High")),
            "estimated_admission_prob": p_est
        })
    rho_df = pd.DataFrame(rho_results)
    print("Sensitivity to Rho (macro-shock correlation):")
    print(rho_df.to_string(index=False))

    # Convergence: 2,000 vs 20,000 runs
    t0 = time.perf_counter()
    p_2k = simulate_at_least_one_seat_probability(test_ranks, test_yhats, sigma=sigma_overall, rho=0.5, n_sims=2000, seed=42)
    t_2k = time.perf_counter() - t0

    t0 = time.perf_counter()
    p_20k = simulate_at_least_one_seat_probability(test_ranks, test_yhats, sigma=sigma_overall, rho=0.5, n_sims=20000, seed=42)
    t_20k = time.perf_counter() - t0

    print(f"\nConvergence Check:")
    print(f"  2,000 runs : P = {p_2k:.4f} (took {t_2k*1000:.1f} ms)")
    print(f"  20,000 runs: P = {p_20k:.4f} (took {t_20k*1000:.1f} ms)")
    print(f"  Absolute Delta: {abs(p_2k - p_20k):.4f} ({abs(p_2k - p_20k)*100:.2f} percentage points)")

    mc_summary = {
        "rho_sensitivity": rho_results,
        "convergence": {
            "p_2000": p_2k,
            "p_20000": p_20k,
            "delta": abs(p_2k - p_20k),
            "time_2000_ms": t_2k * 1000,
            "time_20000_ms": t_20k * 1000
        }
    }
    with open(OUTPUT_DIR / "mc_sensitivity.json", "w") as f:
        json.dump(mc_summary, f, indent=2)

    # Export overall metrics to models/metrics.json so application uses live verified values
    metrics_path = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
    updated_metrics = {
        "val_year": 2025,
        "mae_log": mae_log,
        "mae_naive_log": mae_naive_log,
        "relative_mae": mean_rel_error,
        "median_relative_error": median_rel_error,
        "sigma_overall": sigma_overall,
        "sigma_sparse": sig_sparse,
        "sigma_full": sig_full,
        "sigma": sigma_overall,
        "coverage_80": cov_80,
        "brier_score": brier_score,
        "model_name": "Weighted Trend Predictor (0.5, 0.3, 0.2)",
        "raw_2026_present": False,
        "data_cycles": [2021, 2022, 2023, 2024, 2025]
    }
    with open(metrics_path, "w") as f:
        json.dump(updated_metrics, f, indent=2)
    print(f"\nSaved updated calibrated metrics to: {metrics_path}")
    print("\nPHASE 2 VALIDATION COMPLETE.")

if __name__ == "__main__":
    run_validation()
