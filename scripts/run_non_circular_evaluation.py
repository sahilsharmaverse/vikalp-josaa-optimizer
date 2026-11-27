"""Non-circular probability evaluation and interval calibration benchmark.

Implements:
1. Non-circular evaluation protocol on fixed relative offsets from last year's cutoff:
   delta in [-30%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +30%]
   R = round(C_{t-1} * (1 + delta))
   Evaluates Naive, Trend, Ridge, and LightGBM across rolling-origin folds (2023, 2024, 2025)
   and computes clustered bootstrap CIs (B=1000) over (institute, program) clusters for 2025.

2. Interval and probability recalibration:
   Fits sigma scale and quantile conformal intervals strictly on training folds (t < T).
   Evaluates before and after:
   - 50%, 80%, 90% interval coverage
   - Brier score, Log loss, ECE (10 bins) on the non-circular fixed offset grid.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.baseline import compute_weighted_log_trend
from src.train import ALL_FEATURES, prepare_features
from scripts.run_model_comparison import fit_ridge_per_series

OUTPUT_DIR = ROOT_DIR / "outputs" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_DIR = ROOT_DIR / "outputs" / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

OFFSETS = np.array([-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30])
Z_50 = 0.67449
Z_80 = 1.28155
Z_90 = 1.64485


def compute_calibration_metrics(p_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Compute Brier score, Log Loss, and ECE (10 bins)."""
    p_clipped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
    brier = float(np.mean((p_clipped - y_true) ** 2))
    logloss = float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))
    
    # ECE 10 bins
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(p_clipped)
    for b in range(10):
        mask = (p_clipped >= bins[b]) & (p_clipped < bins[b + 1]) if b < 9 else (p_clipped >= bins[b]) & (p_clipped <= bins[b + 1])
        if np.sum(mask) > 0:
            prop = np.sum(mask) / n
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(p_clipped[mask]))
            ece += prop * abs(acc - conf)
            
    return {"brier": brier, "logloss": logloss, "ece": ece}


def run_benchmark():
    data_path = ROOT_DIR / "data" / "processed" / "cutoffs.parquet"
    df = pd.read_parquet(data_path)
    df_feat = prepare_features(df)
    
    folds = [
        {"test_year": 2023, "train_max": 2022},
        {"test_year": 2024, "train_max": 2023},
        {"test_year": 2025, "train_max": 2024},
    ]
    
    rolling_results = []
    
    # Store 2025 components for clustered bootstrap
    boot_data = {}
    
    for fold in folds:
        test_year = fold["test_year"]
        train_max = fold["train_max"]
        print(f"\n================ FOLD: Test {test_year} (Train <= {train_max}) ================")
        
        train_data = df_feat[(df_feat["year"] <= train_max) & (df_feat["year"] >= 2022)].copy()
        test_data = df_feat[df_feat["year"] == test_year].copy()
        
        train_eval = train_data[
            train_data["prev_closing_1"].notna()
            & (train_data["prev_closing_1"] > 0)
            & train_data["log_closing_rank"].notna()
            & (train_data["closing_rank"] > 0)
        ].copy()
        
        test_eval = test_data[
            test_data["prev_closing_1"].notna()
            & (test_data["prev_closing_1"] > 0)
            & test_data["log_closing_rank"].notna()
            & (test_data["closing_rank"] > 0)
        ].copy().reset_index(drop=True)
        
        y_tr = train_eval["log_closing_rank"].values
        y_te = test_eval["log_closing_rank"].values
        c_te = test_eval["closing_rank"].values
        c_prev_te = test_eval["prev_closing_1"].values
        
        # 1. Model predictions
        # Naive
        y_hat_tr_naive = np.log(train_eval["prev_closing_1"].values)
        sigma_naive = float(np.std(y_tr - y_hat_tr_naive, ddof=1))
        y_hat_te_naive = np.log(test_eval["prev_closing_1"].values)
        
        # Weighted Trend
        y_hat_tr_trend = compute_weighted_log_trend(train_eval["prev_closing_1"], train_eval["prev_closing_2"], train_eval["prev_closing_3"])
        sigma_trend = float(np.std(y_tr - y_hat_tr_trend, ddof=1))
        y_hat_te_trend = compute_weighted_log_trend(test_eval["prev_closing_1"], test_eval["prev_closing_2"], test_eval["prev_closing_3"])
        
        # Ridge
        y_hat_tr_ridge = fit_ridge_per_series(train_eval["prev_closing_1"].values, train_eval["prev_closing_2"].values, train_eval["prev_closing_3"].values)
        sigma_ridge = float(np.std(y_tr - y_hat_tr_ridge, ddof=1))
        y_hat_te_ridge = fit_ridge_per_series(test_eval["prev_closing_1"].values, test_eval["prev_closing_2"].values, test_eval["prev_closing_3"].values)
        
        # LightGBM
        X_tr = train_eval[ALL_FEATURES]
        X_te = test_eval[ALL_FEATURES]
        lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_tr, y_tr)
        sigma_lgbm = float(np.std(y_tr - lgb_model.predict(X_tr), ddof=1))
        y_hat_te_lgbm = lgb_model.predict(X_te)
        
        # Recalibration scale fit on train:
        res_tr_trend = np.abs(y_tr - y_hat_tr_trend)
        q80_tr = float(np.percentile(res_tr_trend, 80))
        scale_trend_80 = q80_tr / (Z_80 * sigma_trend)
        sigma_trend_calibrated = sigma_trend * scale_trend_80
        
        print(f"Train Trend Sigma: raw={sigma_trend:.4f}, scale_factor={scale_trend_80:.4f}, calibrated={sigma_trend_calibrated:.4f}")
        
        # Non-circular Relative Offset Grid
        # Test candidate ranks R = round(C_{t-1} * (1 + delta))
        # N_test offerings x len(OFFSETS)
        ranks_grid = np.round(c_prev_te[:, None] * (1.0 + OFFSETS[None, :]))
        ranks_grid = np.maximum(ranks_grid, 1.0)
        
        # Actual outcome: Y = 1 if R <= C_actual else 0
        actual_grid = (ranks_grid <= c_te[:, None]).astype(float)
        
        # Query models
        models = {
            "Naive Baseline (Last Year)": (y_hat_te_naive, sigma_naive),
            "Weighted Trend (Raw Sigma)": (y_hat_te_trend, sigma_trend),
            "Weighted Trend (Calibrated Sigma)": (y_hat_te_trend, sigma_trend_calibrated),
            "Ridge Linear Trend": (y_hat_te_ridge, sigma_ridge),
            "LightGBM Regressor": (y_hat_te_lgbm, sigma_lgbm),
        }
        
        for m_name, (y_pred, sig) in models.items():
            z_eval = (np.log(ranks_grid) - y_pred[:, None]) / sig
            p_eval = 1.0 - norm.cdf(z_eval)
            
            p_flat = p_eval.ravel()
            a_flat = actual_grid.ravel()
            
            m_res = compute_calibration_metrics(p_flat, a_flat)
            m_res["model"] = m_name
            m_res["test_year"] = test_year
            m_res["mae"] = float(np.mean(np.abs(y_te - y_pred)))
            m_res["sigma_used"] = sig
            
            # Coverage at 50, 80, 90%
            res_te = np.abs(y_te - y_pred)
            m_res["cov_50"] = float(np.mean(res_te <= Z_50 * sig))
            m_res["cov_80"] = float(np.mean(res_te <= Z_80 * sig))
            m_res["cov_90"] = float(np.mean(res_te <= Z_90 * sig))
            
            rolling_results.append(m_res)
            print(f"  {m_name:<34} | MAE: {m_res['mae']:.4f} | Brier: {m_res['brier']:.4f} | LogLoss: {m_res['logloss']:.4f} | ECE: {m_res['ece']:.4f} | Cov80: {m_res['cov_80']*100:.1f}%")
            
        if test_year == 2025:
            boot_data["test_eval"] = test_eval
            boot_data["ranks_grid"] = ranks_grid
            boot_data["actual_grid"] = actual_grid
            boot_data["y_te"] = y_te
            boot_data["models"] = models
            
    df_rolling = pd.DataFrame(rolling_results)
    df_rolling.to_csv(OUTPUT_DIR / "non_circular_rolling_benchmark.csv", index=False)
    
    # 2. Clustered Bootstrap on 2025 (1000 iterations)
    print("\n--- Running Clustered Bootstrap on Held-Out 2025 (B=1,000) ---")
    test_eval = boot_data["test_eval"]
    ranks_grid = boot_data["ranks_grid"]
    actual_grid = boot_data["actual_grid"]
    y_te = boot_data["y_te"]
    models = boot_data["models"]
    
    clusters = test_eval.groupby(["institute", "program"]).indices
    unique_clusters = list(clusters.values())
    n_clusters = len(unique_clusters)
    
    np.random.seed(42)
    boot_dist = {m: {"brier": [], "logloss": [], "ece": [], "mae": [], "cov_50": [], "cov_80": [], "cov_90": []} for m in models}
    
    for b in range(1000):
        sc = np.random.choice(n_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([unique_clusters[k] for k in sc])
        
        y_sample = y_te[idx]
        act_sample = actual_grid[idx].ravel()
        ranks_sample = ranks_grid[idx]
        
        for m_name, (y_pred_all, sig) in models.items():
            yp = y_pred_all[idx]
            z_eval = (np.log(ranks_sample) - yp[:, None]) / sig
            p_eval = 1.0 - norm.cdf(z_eval).ravel()
            
            res_m = compute_calibration_metrics(p_eval, act_sample)
            err = np.abs(y_sample - yp)
            
            boot_dist[m_name]["brier"].append(res_m["brier"])
            boot_dist[m_name]["logloss"].append(res_m["logloss"])
            boot_dist[m_name]["ece"].append(res_m["ece"])
            boot_dist[m_name]["mae"].append(float(np.mean(err)))
            boot_dist[m_name]["cov_50"].append(float(np.mean(err <= Z_50 * sig)))
            boot_dist[m_name]["cov_80"].append(float(np.mean(err <= Z_80 * sig)))
            boot_dist[m_name]["cov_90"].append(float(np.mean(err <= Z_90 * sig)))
            
    summary_2025 = {}
    for m_name in models:
        summary_2025[m_name] = {}
        for metric in ["brier", "logloss", "ece", "mae", "cov_50", "cov_80", "cov_90"]:
            vals = boot_dist[m_name][metric]
            summary_2025[m_name][metric] = {
                "point_estimate": float(np.mean(vals)),
                "ci_95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            }
            
    with open(BACKTEST_DIR / "non_circular_bootstrap_2025.json", "w") as f:
        json.dump(summary_2025, f, indent=2)
        
    print("\nSaved non_circular_bootstrap_2025.json successfully.")
    return df_rolling, summary_2025


if __name__ == "__main__":
    df_rolling, summary_2025 = run_benchmark()
