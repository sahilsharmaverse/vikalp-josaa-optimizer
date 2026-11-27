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
All metrics evaluated on held-out 2025 JoSAA round 5 cutoffs ($N = 9,336$ offerings) with 95% bootstrap confidence intervals clustered across 837 `(institute, program)` series:

| Metric | Point Estimate | 95% Clustered CI | Description |
|:---|:---:|:---:|:---|
| **MAE (log rank)** | **0.1360** | [0.1309, 0.1412] | Log-rank mean absolute error (beats naive by +0.0049, $p < 0.001$). |
| **Mean Relative Error** | **16.35%** | [13.18%, 22.32%] | Mean percentage error in closing rank space. |
| **80% Coverage (Conformal)** | **84.00%** | [82.50%, 85.50%] | Distribution-free split conformal interval coverage (half-width ±0.2222). |
| **80% Coverage (Gaussian)** | **90.26%** | [89.37%, 91.10%] | Parametric Gaussian interval coverage using leak-free train $\sigma = 0.2278$. |
| **Integrated Brier Score** | **0.0842** | [0.0821, 0.0864] | Evaluated across standardized rank grid $z \in [-2.5, +2.5]$ ($N = 196,056$). |
| **ECE (Expected Calibration Error)** | **0.0487** | [0.0410, 0.0560] | Expected calibration gap over 10 uniform probability bins. |
| **Post-Hoc Isotonic ECE** | **0.0109** | [0.0080, 0.0140] | ECE achieved following isotonic calibration on validation fold. |

---

## 5. Training and Evaluation Data
- **Training Corpus:** Official JoSAA closing cutoffs from counseling cycles 2021 through 2024 ($N = 24,160$ evaluable option records).
- **Evaluation Corpus:** Held-out JoSAA 2025 round 5 final allotment data ($N = 9,336$ evaluable option records).
- **Temporal Isolation:** All feature generation, lag generation, and uncertainty parameters ($\sigma$) are computed strictly under temporal restrictions ($t < T$). Zero 2026 data exists.

---

## 6. Quantitative Analysis & Model Comparisons
In 3-year rolling-origin cross-validation (2023 - 2025), Vikalp's Weighted Trend model was benchmarked against three alternative paradigms:
1. **Weighted Trend:** 3-Year Mean MAE = `0.1427`, Brier = `0.0850`, ECE = `0.0552` (Top Performer).
2. **Naive Baseline (Last Year):** 3-Year Mean MAE = `0.1507`, Brier = `0.0862`, ECE = `0.0533`.
3. **Ridge Linear Trend:** 3-Year Mean MAE = `0.1575`, Brier = `0.0885`, ECE = `0.0491`.
4. **LightGBM Regressor:** 3-Year Mean MAE = `0.1660`, Brier = `0.1121`, ECE = `0.0502`.
*Finding:* Tree-based gradient boosting underperforms exponential smoothing on this short-series panel due to high partition variance and bounded extrapolation constraints.

---

## 7. Ethical Considerations & Fairness
- **Risk Mitigation:** To prevent catastrophic allocation failures, Vikalp enforces a mandatory minimum threshold: candidates are explicitly alerted whenever their choice list contains fewer than 5 "Safe" choices.
- **Disadvantaged Categories:** Subgroup audits reveal the model is mildly under-confident in lower-tier GFTIs and extreme-tail reserved seats, acting conservatively to prevent false confidence.
- **No Paid Promotion:** Algorithms order choices strictly by mathematical utility and candidate preferences, with zero sponsored placement.

---

## 8. Caveats and Recommendations
1. **New Programs (0-Year History):** Offerings introduced in the current cycle cannot be projected from historical cutoffs. The UI badges them as "New Program" and recommends conservative placement.
2. **2026 Seat Matrix Shifts:** Major institutional seat count expansions will cause cutoffs to relax beyond historical bounds.
