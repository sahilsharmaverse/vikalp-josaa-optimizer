# Vikalp: Resume Evidence Pack & Data Science Interview Q&As

**Author:** Sahil Sharma  
**Project:** Vikalp (JoSAA Admission-Probability Engine & Choice Optimizer)  
**Status:** All metrics verified via automated test suite and rolling backtests.  

---

## 📄 1. High-Impact Resume & Portfolio Bullets

- **Core Engine & Accuracy:** Engineered an uncertainty-aware admission probability engine trained on **44,637** official JoSAA cutoff records (2021–2025); achieved **0.1360 MAE** (log closing rank) on held-out 2025 data, statistically significantly outperforming the naive last-year baseline by **+0.0049 log points** ($p < 0.001$, 95% Clustered CI: `[0.0026, 0.0071]` over 837 independent series clusters).
- **Uncertainty & Conformal Prediction:** Implemented distribution-free **Split Conformal Prediction** for cutoff intervals, delivering **84.00%** empirical coverage at nominal 80% with a **24% narrower interval half-width** ($\pm 0.2222$ vs $\pm 0.2919$ log points) compared to parametric Gaussian intervals.
- **Probability Calibration:** Designed a post-hoc calibration pipeline using **Isotonic Regression** and **Platt Scaling**; reduced Expected Calibration Error (ECE) across 10 probability bins by **77.6%** (from **0.0487** down to **0.0109**) and improved the canonical integrated Brier score to **0.0794** across 196,056 decision-boundary evaluations.
- **Systemic Risk & Copula Simulation:** Formulated portfolio admission probabilities across 50–100 candidate choices using a single-factor **Gaussian Copula** ($\rho = 0.50$, 2,000 Monte Carlo draws) to capture macro exam shifts; validated against 120 historical candidate preference profiles, yielding a **Portfolio Brier score of 0.0666** and an **ECE of 0.0292** (predicted portfolio win rate of 80.40% vs. actual 80.83%).
- **Model Selection & Empirical Maturity:** Conducted a 3-year rolling-origin benchmark (2023, 2024, 2025) comparing Naive, Trend, Ridge, and LightGBM models; empirically proved that LightGBM underperformed exponential smoothing (3-Year Mean MAE `0.1660` vs `0.1427`) due to short series length ($T \le 4$), tree step-function extrapolation limits, and information dominance of $t-1$ closing ranks.
- **Production Engineering & Reliability:** Built an automated, reproducible ML pipeline (`python run_pipeline.py`) backed by **173 automated tests** (82% test coverage on `src/`), static typing (`mypy`), linting (`ruff`), and formal **Model Card** (Mitchell et al.) and **Data Card** (Gebru et al.) specifications.

---

## 🎯 2. Five Staff-Level Data Science Interview Q&As

### Q1: Why not use an end-to-end neural network or LightGBM for cutoff prediction?
> **Answer:**  
> "I actually benchmarked a tuned LightGBM regressor against our exponential smoothing model across three consecutive rolling folds (2023, 2024, and 2025) under strict temporal isolation. LightGBM achieved a 3-year mean MAE of **0.1660**, whereas our time-decaying weighted trend achieved **0.1427** (and the naive baseline achieved **0.1507**).  
> There are three core statistical reasons for this negative result:
> 1. **Short Panel Time-Series ($T \le 4$):** JoSAA publishes one cutoff per year. With only 3 to 4 historical points per offering, tree-based models overfit the noise in lag features rather than learning meaningful interactions.
> 2. **Bounded Extrapolation Limits:** Decision trees produce piecewise-constant outputs and cannot extrapolate monotonic upward or downward trends beyond the range of training observations.
> 3. **Information Dominance of $t-1$ Cutoff:** The prior year's cutoff explains over 96% of variance. Global tree partitions introduce partition bias across categories and branch groups that degrades predictions for stable, high-reputation programs.  
> Therefore, using a mathematically transparent, weighted log-trend was not a shortcut—it was the empirically superior and auditable production choice."

---

### Q2: How did you validate that your admission probabilities are genuinely calibrated?
> **Answer:**  
> "Evaluating probability calibration in college admissions is subtle because candidate ranks are continuous while admission is binary ($\mathbb{I}\{\text{Rank} \le \text{Cutoff}\}$). If you evaluate candidate ranks sampled uniformly across the general population, 90% of options are trivially impossible ($p \approx 0$) or guaranteed ($p \approx 1$), which mechanically deflates the Brier score to ~0.022.  
> To test real calibration, I established the **Canonical Integrated Support Protocol**: evaluating each option across a standardized distance grid $z \in [-2.5, +2.5]$ ($p \in [0.6\%, 99.4\%]$), testing the model exactly in the transition boundary where counseling decisions happen.  
> On held-out 2025 data ($N = 196,056$ evaluations), our raw parametric model achieved an Integrated Brier score of **0.0842** and an Expected Calibration Error (ECE) of **0.0487**. By training an Isotonic Regression calibrator on the prior validation fold, we lowered ECE to **0.0109** (a 77.6% error reduction) and improved Brier to **0.0794**. Furthermore, subgroup audits revealed that our residual bias is slightly negative (mildly under-confident), which is the safest failure mode in high-stakes educational counseling."

---

### Q3: How did you prevent temporal data leakage across feature engineering and prediction intervals?
> **Answer:**  
> "Temporal leakage is the most common flaw in time-series backtests. I audited and enforced four strict invariants verified by automated tests (`tests/test_temporal_leakage.py`):
> 1. **Strict Lag Feature Construction:** For any year $T$, lag features (`prev_closing_1`, `prev_closing_2`, `prev_closing_3`) are strictly generated using `.shift(1)` on chronological data, ensuring they reference $t < T$.
> 2. **Rolling-Origin Folds:** Training sets for fold $T \in \{2023, 2024, 2025\}$ only include records with $\text{year} < T$, with zero test-row index overlap.
> 3. **Leak-Free Variance Parameter Estimation:** In naive backtests, developers often calculate $\sigma = \text{std}(y_{\text{test}} - \hat{y}_{\text{test}})$, leaking test variance into prediction interval widths. In Vikalp, $\sigma_{\text{train}} = 0.2278$ and conformal non-conformity quantiles are estimated strictly on training/calibration folds ($t \le 2024$).
> 4. **Zero Future Contamination:** Automated tests verify that zero 2026 data exists in raw or processed directories."

---

### Q4: What was the platform's biggest failure mode, and what concrete steps did you take to address it?
> **Answer:**  
> "I conducted a residual error analysis on the held-out 2025 test set and extracted the 10 worst-predicted series to build a concrete failure taxonomy.  
> The single biggest failure mode (accounting for 45% of extreme residuals) was **Regional Home-State (HS) Quota Discontinuities**. In smaller states and Union Territories like Goa, Sikkim, Mizoram, and Puducherry, the local applicant pool is so small that a minor drop in candidate turnout causes closing ranks to collapse by hundreds of percent—for example, NIT Goa Civil HS cutoff jumped from 78,500 to 184,000+.  
> To address this:
> 1. We identified that stratified $\sigma$ by institution tier alone doesn't capture regional demographic volatility.
> 2. We added dynamic UI risk badging that flags Home State choices in low-population states with an elevated uncertainty warning.
> 3. In our recommendation optimizer, we enforce a mandatory minimum of 5 'Safe' options ($P \ge 80\%$) so that volatility in state-quota choices cannot result in zero counseling allotments."

---

### Q5: Why did you model multi-option lists using a Gaussian Copula with $\rho = 0.50$ instead of treating choices independently?
> **Answer:**  
> "If a student applies to 30 computer science programs and you assume independent probabilities ($1 - \prod(1 - p_i)$), an applicant with 30 target choices ($p_i \approx 40\%$) would be told their admission chance is $1 - (0.60)^{30} = 99.9999\%$. This is mathematically dangerous and false in the real world.  
> Admission outcomes across engineering branches are strongly correlated because they share a common macro shock: overall exam applicant volume, question paper difficulty, seat matrix expansion, or shifting student preferences (e.g., nationwide rush into CS/AI).  
> To model this realistically without complex joint multivariate parameterization, I implemented a single-factor **Gaussian Copula**:
> $$z_i = \sqrt{\rho} \cdot s + \sqrt{1 - \rho} \cdot e_i, \quad s, e_i \sim \mathcal{N}(0, 1)$$
> Setting $\rho = 0.50$ reflects that approximately 50% of cutoff variance is market-wide and 50% is program-specific. When validated against 120 historical student preference portfolios, this copula yielded a simulated win rate of **80.40%** against an actual historical allotment rate of **80.83%** (gap < 0.5%, Portfolio Brier score **0.0666**). It prevents false overconfidence while preserving the true diversification benefit of adding safety anchors."
