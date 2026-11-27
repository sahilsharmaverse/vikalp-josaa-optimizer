# Model Comparison & Benchmark Report

**Evaluation Framework:** Strict Rolling-Origin Cross-Validation (2023, 2024, 2025)

## 1. Rolling Average Performance Table

| Model | 3-Yr Mean MAE | 3-Yr Rel. Error | 80% Coverage | Brier Score | ECE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Weighted Trend** | 0.1427 | 14.88% | 89.84% | 0.0850 | 0.0552 |
| **Naive Baseline (Last Year)** | 0.1507 | 15.67% | 89.12% | 0.0862 | 0.0533 |
| **Ridge Linear Trend** | 0.1575 | 16.55% | 88.35% | 0.0885 | 0.0491 |
| **LightGBM Regressor** | 0.1660 | 17.27% | 79.75% | 0.1121 | 0.0502 |

## 2. Verdict on Complex ML (LightGBM vs Trend)

- **Finding:** LightGBM (0.1660) does **NOT** outperform the Weighted Trend model (0.1427).
- **Comparison with Naive:** LightGBM (0.1660) does NOT beat the simple last-year naive baseline (0.1507).
- **Statistical Root Causes:**
  1. **Short Series Length:** With only 3-4 historical lag points per series, tree-based splits cannot learn non-linear lag interactions without high estimation variance.
  2. **Bounded Extrapolation:** Tree models output step functions and cannot extrapolate monotonic upward/downward drift beyond observed training values.
  3. **High Signal-to-Noise in Last Year:** The previous year's cutoff explains over 96% of variance; global tree splits introduce partition bias that increases prediction error on stable, high-reputation series.
- **Production Decision:** Preserve **Weighted Trend** as the canonical inference engine.
