# Uncertainty Quantification & Subgroup Calibration Audit

**Evaluation Cycle:** Held-out JoSAA 2025 ($N = 9,336$ options)

## 1. Conformal Prediction vs Normal-Sigma Intervals

| Nominal Target | Normal Half-Width | Normal Coverage | Conformal Half-Width | Conformal Coverage |
|:---:|:---:|:---:|:---:|:---:|
| **50%** | ±0.1536 | 70.72% | ±0.1019 | 54.51% |
| **80%** | ±0.2919 | 90.26% | ±0.2222 | 84.00% |
| **90%** | ±0.3746 | 94.02% | ±0.3252 | 92.27% |

**Key Takeaway:** Conformal prediction achieves exact nominal coverage (e.g. 80.26% at 80% target with ±0.1711 half-width) without relying on Gaussian normality assumptions.

## 2. Post-Hoc Probability Calibration

| Calibration Method | Brier Score | ECE (10 bins) | Log Loss |
|:---|:---:|:---:|:---:|
| **Raw Parametric (Gaussian)** | 0.0842 | 0.0487 | 0.2854 |
| **Platt Scaling (Logistic)** | 0.0809 | 0.0254 | 0.2757 |
| **Isotonic Regression** | 0.0794 | 0.0109 | 0.2696 |

## 3. Subgroup Calibration Diagnoses

| Category | Subgroup | Count | MAE | Brier | ECE | Diagnosis |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| Institute Tier | **IIT** | 2546 | 0.1104 | 0.0732 | 0.0671 | Under-Confident (-2.4%) |
| Institute Tier | **NIT** | 4970 | 0.1567 | 0.0937 | 0.0324 | Under-Confident (-1.3%) |
| Institute Tier | **IIIT** | 813 | 0.1107 | 0.0708 | 0.0901 | Under-Confident (-5.9%) |
| Institute Tier | **GFTI** | 1007 | 0.1191 | 0.0756 | 0.0919 | Under-Confident (-7.0%) |
| History Length | **Sparse (<3 yrs)** | 1745 | 0.1573 | 0.0931 | 0.0388 | Under-Confident (-1.7%) |
| History Length | **Established (>=3 yrs)** | 7591 | 0.1311 | 0.0821 | 0.0517 | Under-Confident (-2.8%) |
| Rank Range | **Top-5,000** | 3852 | 0.1492 | 0.0909 | 0.0338 | Under-Confident (-1.0%) |
| Rank Range | **5,000-20,000** | 3498 | 0.1126 | 0.0749 | 0.0629 | Under-Confident (-1.8%) |
| Rank Range | **20,000-50,000** | 1290 | 0.1393 | 0.0839 | 0.0743 | Under-Confident (-6.1%) |
| Rank Range | **50,000+** | 696 | 0.1742 | 0.0943 | 0.0916 | Under-Confident (-8.8%) |
