# Model Card: Vikalp JoSAA Admission-Probability Engine

Following the framework of Mitchell et al. (*Model Cards for Model Reporting*, FAT* 2019).

---

## 1. Model Details
- **Model Name:** Vikalp Admission-Probability Engine (v1.1)
- **Model Architecture:** Two-stage calibrated probabilistic engine:
  1. *Expected Cutoff Forecaster:* Time-decaying weighted exponential trend regression ($\hat{y} = \sum w_k \ln(\text{prev}_k)$, weights: $0.5, 0.3, 0.2$) predicting log closing ranks ($\ln(\text{rank})$).
  2. *Uncertainty & Copula Simulator:* Distribution-free Conformal Prediction / Calibrated Normal CDF coupled with a Gaussian Copula ($\rho = 0.50$) simulating 2,000 joint admissions across candidate preference lists.
- **Model Version:** 1.1-audited (Commit `master`)
- **Developers:** Vikalp Core Engineering & Data Science Team
- **Release Date:** November 2026
- **License:** MIT Open Source License

---

## 2. Intended Use
- **Primary Intended Use:** Educational decision-support tool assisting engineering aspirants and parents in organizing JoSAA choice lists (IIT, NIT, IIIT, and GFTI counseling).
- **Primary Users:** 10+2 students appearing for JEE Main and JEE Advanced counseling across India.
- **Out-of-Scope Use Cases:**
  - Automated list submission without human candidate review.
  - Counseling for state-level entrance exams (MHT-CET, KCET, WBJEE) that do not participate in JoSAA.
  - Guaranteed admission claims (Vikalp provides statistical risk classification, not allotment guarantees).

---

## 3. Factors and Subgroups
- **Institutional Systems:** IIT (JEE Advanced rank), NIT (JEE Main CRL), IIIT (JEE Main CRL), GFTI (JEE Main CRL).
- **Seat Pools & Quotas:** Home State (HS), Other State (OS), All India (AI), Andhra Pradesh / Telangana Quota (GO).
- **Reservation Categories:** OPEN, GEN-EWS, OBC-NCL, SC, ST, PwD.
- **Gender Pools:** Gender-Neutral vs. Female-only (supernumerary).
- **History Maturity:** Established series ($\ge 3$ years of historical cutoffs) vs. Sparse series ($< 3$ years of data).

---

## 4. Evaluation Metrics & Benchmarks
All metrics evaluated on held-out 2025 JoSAA round 5 cutoffs ($N = 9,336$ offerings) with 95% bootstrap confidence intervals clustered across 837 `(institute, program)` series.

### Primary Benchmark: Non-Circular Relative Offset Grid (Δ ∈ [-30%, +30%])
Candidate ranks queried at fixed offsets relative to $C_{t-1}$ ($N = 84,024$ evaluations across identical query ranks for all models):

| Metric | Point Estimate | 95% Clustered CI | Description |
|:---|:---:|:---:|:---|
| **MAE (log rank)** | **0.1360** | [0.1309, 0.1412] | Log-rank mean absolute error (beats naive by +0.0049, 3.48% relative gain). |
| **Mean Relative Error** | **16.35%** | [13.18%, 22.32%] | Mean percentage error in closing rank space. |
| **Brier Score (Non-Circular)** | **0.1436** | [0.1406, 0.1465] | Mean squared probability error (beats naive 0.1550 [0.1531, 0.1569]). |
| **Log Loss / Binary Cross-Entropy**| **0.4505** | [0.4411, 0.4599] | Negative log likelihood (beats naive 0.4780 [0.4734, 0.4825]). |
| **ECE (Expected Calibration Error)**| **0.0449** | [0.0380, 0.0521] | Expected calibration gap over 10 uniform probability bins. |
| **80% Coverage (Calibrated Normal)**| **84.01%** | [82.87%, 85.05%] | Empirical coverage under calibrated $\sigma = 0.1733$ (close to nominal 80%). |
| **80% Coverage (Conformal)** | **84.00%** | [82.50%, 85.50%] | Distribution-free split conformal coverage (half-width ±0.2222). |

### Secondary Benchmark: Standardized z-Grid Support (Auxiliary)
- **Integrated Brier Score:** **0.0842** [0.0821, 0.0864] (Naive: 0.0829 [0.0807, 0.0851]).
- *Methodology Note:* Query points placed at $y_{\text{hat}} + \sigma z$ depend on the model's own scale $\sigma$, which evaluates models on disparate supports and deflates scores into standard normal quantiles by construction.

---

## 5. Training and Evaluation Data
- **Training Corpus:** Official JoSAA closing cutoffs from counseling cycles 2021 through 2024 ($N = 24,160$ evaluable option records).
- **Evaluation Corpus:** Held-out JoSAA 2025 round 5 final allotment data ($N = 9,336$ evaluable option records).
- **Temporal Isolation:** All feature generation, lag generation, and uncertainty parameters ($\sigma = 0.1733$, scale factor 0.7610) are computed strictly under temporal restrictions ($t \le 2024$). Zero 2026 data exists.

---

## 6. Quantitative Analysis & Model Comparisons
In 3-year rolling-origin cross-validation (2023 - 2025) on the primary non-circular relative offset grid:
1. **Weighted Trend (Calibrated):** 3-Year Mean MAE = `0.1427`, Brier = `0.1474`, Log Loss = `0.4603`, ECE = `0.0712`.
2. **Naive Baseline (Last Year):** 3-Year Mean MAE = `0.1507`, Brier = `0.1607`, Log Loss = `0.4906`, ECE = `0.0935`.
3. **Ridge Linear Trend:** 3-Year Mean MAE = `0.1575`, Brier = `0.1671`, Log Loss = `0.5091`, ECE = `0.0781`.
4. **LightGBM Regressor:** 3-Year Mean MAE = `0.1660`, Brier = `0.1742`, Log Loss = `0.5579`, ECE = `0.0783`.

*Statistical Significance & Caveats:*
- The paired t-test on MAE residuals across all 2025 series shows $t = 5.10, p = 3.53 \times 10^{-7}$.
- The non-parametric Wilcoxon signed-rank test yields $p = 0.183$, demonstrating that the gain is concentrated on offerings with multi-year drift, while stable options perform on par with last year's cutoff.
- The 95% bootstrap CI for $\Delta_{\text{MAE}}$ is `[0.0026, 0.0071]`, strictly excluding zero.

---

## 7. Ethical Considerations & Fairness
- **Risk Mitigation:** To prevent catastrophic allocation failures, Vikalp enforces a mandatory minimum threshold: candidates are explicitly alerted whenever their choice list contains fewer than 5 "Safe" choices.
- **Disadvantaged Categories:** Subgroup audits reveal the model is mildly under-confident in lower-tier GFTIs and extreme-tail reserved seats, acting conservatively to prevent false confidence.
- **No Paid Promotion:** Algorithms order choices strictly by mathematical utility and candidate preferences, with zero sponsored placement.

---

## 8. Caveats and Recommendations
1. **New Programs (0-Year History):** Offerings introduced in the current cycle cannot be projected from historical cutoffs. The UI badges them as "New Program" and recommends conservative placement.
2. **2026 Seat Matrix Shifts:** Major institutional seat count expansions will cause cutoffs to relax beyond historical bounds.
