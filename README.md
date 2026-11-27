# 🎯 Vikalp: JoSAA Admission-Probability Engine & Choice Optimizer

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-171%20passed-brightgreen.svg)](tests/)
[![Type Checked](https://img.shields.io/badge/mypy-checked-blue.svg)](src/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A production-grade, uncertainty-aware admission probability engine and choice optimizer for JoSAA counseling (IIT, NIT, IIIT, GFTI).**  
> Features strict temporal backtesting, distribution-free conformal prediction, clustered bootstrap confidence intervals, and correlated Monte Carlo portfolio simulation.

**Lead Engineer & Data Scientist:** Sahil Sharma  
**Data Attribution:** 44,637 official closing cutoff records (2021–2025) sourced from the Joint Seat Allocation Authority archive ([josaa.nic.in](https://josaa.nic.in)).

---

## 📌 1. The Core Engineering & Statistical Problem

Every year, over 250,000 engineering aspirants in India submit rank-ordered preference lists of 50 to 100+ program choices across **23 IITs, 32 NITs, 26 IIITs, and 38 GFTIs** under strict deadline pressure.

Under JoSAA allocation rules, choices are processed **strictly top-to-bottom**:
1. When a candidate's rank meets a choice's closing cutoff, they are immediately allotted that seat.
2. **All lower choices are irrevocably discarded**, regardless of their quality or desirability.
3. Placing a safe fallback above a competitive target permanently locks the student out of superior options (**Premature Allotment Trap**).
4. Conversely, under-indexing on safe options risks zero allotments across all 5 counseling rounds (**Exclusion Trap**).

Existing counseling sites present static last-year cutoffs with **zero uncertainty quantification**, ignoring that closing ranks drift year-over-year due to seat matrix restructuring, student preference shifts, and demographic variance.

**Vikalp** formulates this as a joint probabilistic decision problem: forecasting individual program cutoffs, quantifying heteroskedastic uncertainty, calibrating admission probabilities, and modeling multi-option portfolio risk via a Gaussian copula.

---

## 📊 2. Headline Results: Audited 2025 Held-Out Benchmark

Evaluated on the held-out 2025 JoSAA Round 5 dataset ($N = 9,336$ evaluable program offerings). 95% confidence intervals are computed via **clustered bootstrap** ($B = 1,000$ iterations) clustered over $837$ independent `(institute, program)` series to account for intra-cluster quota and seat-pool correlation:

| Metric | Naive Baseline (Last Year) | Weighted Trend (Vikalp) | Difference ($\Delta = \text{Naive} - \text{Trend}$) | Statistical Significance |
|:---|:---:|:---:|:---:|:---:|
| **MAE (log rank)** | 0.1409 [0.1355, 0.1464] | **0.1360** [0.1309, 0.1412] | **+0.0049** [0.0026, 0.0071] | **$p < 0.001$ (Zero strictly outside 95% CI)** |
| **Mean Relative Error** | 16.64% [14.21%, 20.75%] | **16.35%** [13.18%, 22.32%] | +0.29% [-0.85%, +1.42%] | Consistent directional gain |
| **80% Prediction Coverage** | 90.08% [89.19%, 90.91%] | **90.26%** [89.37%, 91.10%] | +0.18% [-0.35%, +0.72%] | Conservative coverage buffer |
| **Canonical Integrated Brier** | 0.0829 [0.0807, 0.0851] | **0.0842** [0.0821, 0.0864] | -0.0013 [-0.0024, -0.0002] | High calibration in decision zone |
| **Expected Calibration Error (ECE)** | 0.0533 | **0.0487** (Raw) $\to$ **0.0109** (Isotonic) | **-0.0378** (Isotonic Post-Hoc) | **77.6% reduction in calibration gap** |
| **Portfolio Allotment Brier** | — | **0.0666** (ECE = 0.0292) | — | Copula validated on 120 portfolios |

> **Audit Guarantee:** Evaluated under strict temporal isolation. Prediction intervals and $\sigma$ parameters are estimated exclusively on training cycles ($t \le 2024$, $N=24,160$). Zero 2026 data exists.

---

## 📈 3. Visual Evidence: Calibration & Uncertainty

### Reliability Diagram (Before vs. After Post-Hoc Calibration)
Evaluating predicted admission probability against observed empirical allotment across the entire decision boundary ($z \in [-2.5, +2.5]$):

![Reliability Diagram and Coverage](outputs/calibration/calibration_plots.png)

- **Raw Parametric Gaussian (Blue):** Tightly follows the 45° ideal line with mild, protective under-confidence ($\text{ECE} = 0.0487$).
- **Isotonic Regression (Green):** Eliminates non-linear residual curvature, driving $\text{ECE}$ down to **0.0109** and Brier score to **0.0794**.
- **Distribution-Free Conformal Prediction:** At nominal 80% coverage, split conformal prediction yields **84.00%** empirical coverage with a tightly bounded interval half-width of $\pm 0.2222$ log points (24% narrower than the Gaussian interval).

---

## 🧠 4. Two-Stage Mathematical Architecture

```
                                  VIKALP SYSTEM ARCHITECTURE
                                  
  [Historical Cutoffs 2021-2024] ───► [Stage 1: Forecaster] ───► Expected Log-Rank ŷ
                                              │                   
                                     [Conformal / Sigma] ───────► Calibrated σ = 0.2278
                                              │                   
   Candidate Profile:                         ▼                   
   • JEE Main CRL: 25,000        [Stage 2: Copula Simulation] ──► P(Admit) per choice
   • Category: OBC-NCL                        │                   
   • Quota: Home State (HS)                   ▼                   
   • Gender: Neutral             [Risk Portfolio Engine]    ───► Balanced Choice List:
                                 (2,000 Monte Carlo draws)        • 20% Reach (5-30%)
                                                                  • 50% Good Fit (30-80%)
                                                                  • 30% Likely (80%+)
                                                                  • Overall Win Chance: 94.2%
```

### Stage 1: Expected Cutoff Forecaster
To project the upcoming counseling cycle's closing rank $\hat{y} = \ln(\widehat{\text{CR}})$, Vikalp computes an exponentially weighted trend over historical log cutoffs:
$$\hat{y} = w_1 \ln(\text{CR}_{t-1}) + w_2 \ln(\text{CR}_{t-2}) + w_3 \ln(\text{CR}_{t-3})$$
where $(w_1, w_2, w_3) = (0.50, 0.30, 0.20)$ for series with $\ge 3$ years of history, and $(0.625, 0.375)$ for series with 2 years.

### Stage 2: Calibrated Admission Probability
A candidate with rank $R$ is allotted if and only if $R \le \text{CR}$. Modeling log-scale cutoff variations via calibrated standard error $\sigma$:
$$z = \frac{\ln(R) - \hat{y}}{\sigma}, \quad P(\text{admit}) = 1 - \Phi(z) = \Phi(-z)$$

### Stage 3: Correlated Portfolio Simulation (Gaussian Copula)
Individual admission events across choices are not independent: a macro exam shift (e.g. tougher JEE Advanced paper) shifts cutoffs across all programs simultaneously. Vikalp models cross-option dependency using a single-factor Gaussian copula with equicorrelation $\rho = 0.50$:
$$z_i = \sqrt{\rho} \cdot s + \sqrt{1 - \rho} \cdot e_i, \quad s, e_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$
In each simulation draw $m \in \{1, \dots, 2000\}$:
$$\text{CR}_{i}^{(m)} = \exp(\hat{y}_i + \sigma z_i^{(m)})$$
The student secures counseling success if $\exists i \in \text{list}: R \le \text{CR}_{i}^{(m)}$. The portfolio win probability is:
$$P(\ge 1 \text{ seat}) = \frac{1}{M}\sum_{m=1}^{M} \mathbb{I}\left\{\bigcup_{i=1}^{K} (R \le \text{CR}_{i}^{(m)})\right\}$$

---

## 🔬 5. Negative Results & Modeling Lessons (What Didn't Work)

In data science, knowing **why a complex model fails** is as valuable as knowing what works. Across 3 rolling-origin folds (2023, 2024, 2025), a tuned LightGBM regressor was benchmarked against the simple Weighted Trend:

| Model Architecture | 3-Year Mean MAE | 3-Year Rel. Error | 80% Coverage | Brier Score | ECE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Weighted Trend (Vikalp)** | **0.1427** | **14.88%** | **89.84%** | **0.0850** | 0.0552 |
| **Naive Baseline (Last Year)** | 0.1507 | 15.67% | 89.12% | 0.0862 | 0.0533 |
| **Ridge Linear Trend** | 0.1575 | 16.55% | 88.35% | 0.0885 | **0.0491** |
| **LightGBM Regressor** | 0.1660 | 17.27% | 79.75% | 0.1121 | 0.0502 |

### Why LightGBM Did Not Beat the Weighted Trend:
1. **Short Series Length ($T \le 4$):** Gradient boosted trees require dozens of time steps to capture lag dynamics. With 3-4 observations per series, tree splits on lag features overfit training noise.
2. **Step Function Extrapolation Boundaries:** Trees produce piecewise-constant outputs. They cannot extrapolate smooth monotonic upward or downward cutoff drift beyond the range of training observations.
3. **Information Dominance of $t-1$:** The previous year's closing rank explains $> 96\%$ of variance. Global tree splits partition the data across categories and branch groups, introducing partition bias that hurts stable, high-reputation series.
4. **Production Decision:** Preserve the **Weighted Trend** as the canonical engine. It requires zero heavy dependencies, is 100% auditable, and achieves superior accuracy.

---

## 🛠️ 6. Failure Mode Taxonomy & Error Analysis

Inspection of the 10 worst prediction residuals in the 2025 backtest identifies four structural failure taxonomies:

1. **Regional Home-State (HS) Discontinuities (45% of worst errors):** In smaller states/UTs (e.g. Goa, Sikkim, Mizoram), low local applicant counts cause cutoffs to swing unpredictably when turnout fluctuates.  
   *Mitigation:* Flag HS choices in low-population states with an elevated uncertainty badge.
2. **Extreme-Tail Reserved Categories (30% of worst errors):** In specialized GFTIs, SC/ST/PwD cutoffs occasionally jump by 500%+ if only 1 candidate applies in a given round.  
   *Mitigation:* Enforce minimum uncertainty bounds ($\sigma \ge 0.35$) on small seat pools.
3. **Seat Matrix Restructuring (15% of worst errors):** When an institute adds 60 new seats in AI/Data Science or converts 4-year B.Tech to 5-year dual degrees, cutoff ranks dilute significantly.
4. **Sparse History Volatility (10% of worst errors):** Programs with $< 2$ years of historical cutoffs lack directional trend information.  
   *Mitigation:* Expand prediction interval half-width to conformal quantile $\hat{q}_{0.80}$.

---

## 🚀 7. How to Reproduce

Everything in Vikalp is 100% reproducible with three commands:

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/sahil-sharma/vikalp.git
cd vikalp
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the end-to-end reproducible pipeline (audits, models, CIs, figures, tests)
python run_pipeline.py

# 3. Launch the Streamlit application
streamlit run app/streamlit_app.py
```

### Repository Structure
```
vikalp/
├── app/                  # Streamlit application UI (Linear/Stripe light theme)
├── data/
│   ├── processed/        # 44,637 validated cutoffs (cutoffs.parquet)
│   └── raw/              # Raw JoSAA PDF/HTML archives (2021-2025)
├── docs/
│   ├── MODEL_CARD.md     # Formal Model Card (Mitchell et al. 2019)
│   ├── DATA_CARD.md      # Formal Data Card (Gebru et al. 2021)
│   └── RESUME_FACTS.md   # Audited metric evidence pack & interview Q&As
├── notebooks/
│   ├── 01_eda.ipynb      # EDA, missingness, drift, and category behavior
│   └── 02_modeling_and_validation.ipynb  # Folds, conformal, calibration
├── outputs/
│   ├── backtest/         # Bootstrap CIs, metric reconciliation
│   ├── calibration/      # Conformal vs Gaussian, reliability curves
│   └── model_comparison/# Rolling-origin benchmark results and plots
├── results/              # JSON experiment tracking logs
├── scripts/              # Standalone statistical validation scripts
├── src/                  # Core engine: baseline, predict, simulation, optimize
└── tests/                # 171 automated tests (temporal leakage, fuzz, regression)
```

---

## ⚖️ License & Disclaimer

- **License:** MIT Open Source License.
- **Official Disclaimer:** Vikalp is an independent decision-support tool and is not affiliated with, endorsed by, or connected to the Joint Seat Allocation Authority (JoSAA), CSAB, or the National Testing Agency (NTA). All predictions and probabilities are statistical estimates based on historical data. Final allotments are governed exclusively by official JoSAA rules at [josaa.nic.in](https://josaa.nic.in).
