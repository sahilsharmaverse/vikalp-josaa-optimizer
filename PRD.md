# PRD: Vikalp (Counselling Choice-List Optimizer for JoSAA)

**Author:** Sahil Sharma

## What to build
A web app that tells a JEE student which IIT / NIT / IIIT / GFTI seats they can realistically get, shows an **admission probability** for each option, and generates an ordered **JoSAA choice list** balanced across Dream / Target / Safe.

## Problem
- Students fill 50-100+ choices under stress using raw cutoff PDFs or predictors that show last year's cutoff with no uncertainty.
- A bad list wastes seats (safe options placed too high) or leaves no fallback (all dream options).
- Rank types confuse people: CRL vs category rank, AI / HS / OS quotas, gender-neutral vs female-only seats.

## Targeted users
- **Primary:** JEE Main / Advanced students (12th pass-outs and droppers) entering JoSAA, especially from Tier-2/3 cities with no paid counsellor.
- **Secondary:** parents, school and coaching counsellors.

## Scope
**In (v1):** JoSAA only. IIT, NIT, IIIT, GFTI. Final-round closing ranks for all years available (target 2021-2026). Predict the 2027 cycle.
**Out (v1):** KCET / MHT-CET / NEET, Docker, FastAPI, login, Hindi UI, mobile app, paid tiers.

## User inputs
- JEE Main CRL rank (for NIT / IIIT / GFTI)
- JEE Advanced rank (optional, for IITs)
- Category (OPEN / EWS / OBC-NCL / SC / ST) and the **category rank** if not OPEN
- Gender (female students are eligible for gender-neutral and female-only seats)
- Home state
- Preferred branches (multi-select), optional preferred states, institute types
- Choice list size (default 50)
- Priority mode: **College-first** or **Branch-first**

## Features
**Must-have (MVP)**
1. Admission probability for every eligible option (quota-aware)
2. Dream / Target / Safe label
3. Auto-built ordered choice list with a target risk mix (20% / 50% / 30%)
4. Filters: branch, state, institute type
5. Download choice list as CSV
6. Visible disclaimer: "Estimate, not official. Verify at josaa.nic.in."

**Should-have**
- Mini chart of the option's last 3-5 years of closing ranks plus the predicted range
- Warning for options placed after a near-certain option (unreachable)

**Won't have (v1):** anything in the Out list above.

## v1.1 Features (Interactive & Visual Expansion)
In v1.1, Vikalp expands from list optimization into interactive decision support, risk analysis, and shareable reports:

1. **Chance of Getting at Least One Seat (Monte Carlo Portfolio Estimate):**
   - Estimates the overall probability that a student gets admitted to at least one choice on their submitted list using a 2,000-run Monte Carlo simulation:
     - For each simulation $k \in \{1, \dots, 2000\}$, draw a shared macroeconomic/exam-level shock $s \sim \mathcal{N}(0, 1)$ and an option-specific shock $e_i \sim \mathcal{N}(0, 1)$ for each choice $i$.
     - Correlated shock: $z_i = \sqrt{\rho} s + \sqrt{1 - \rho} e_i$, with correlation default $\rho = 0.5$.
     - Simulated cutoff: $\text{closing}_{k, i} = \exp(\hat{y}_i + \sigma z_i)$, where $\sigma$ is the residual standard error read directly from the calibrated model metrics.
     - Student is admitted to option $i$ in simulation $k$ if $\text{student\_rank}_i \le \text{closing}_{k, i}$.
     - Overall probability is the share of simulations where at least one choice resulted in admission.
     - Display is capped at 99% and explicitly labeled as an "estimate", explaining in plain words: *"Monte Carlo estimate (2,000 simulations) assuming correlated exam/counselling cutoff shocks ($\rho = 0.5$)."*
   - Verification test: Under $\rho = 0$ (zero cross-option correlation), the Monte Carlo estimate must converge within statistical tolerance to the analytical independence formula $1 - \prod_{i=1}^N (1 - P_i)$.

2. **What-If Rank Sensitivity Slider:**
   - Real-time rank perturbation slider in the UI allowing students to test rank fluctuations (e.g. $\pm 10\%$, $\pm 20\%$, or manual rank shifts).
   - Recomputes probabilities, tier counts, and cumulative chances dynamically without altering the base student profile.
   - Helps students test border scenarios and stress-test their list against unexpected exam cutoff shifts.

3. **Compare 2-3 Options Side by Side:**
   - Interactive comparator allowing students to select 2 or 3 choices from their list.
   - Side-by-side cards comparing:
     - Predicted closing rank and 80% prediction interval
     - Admission probability circular gauge and tier tag
     - Historical 5-year closing rank trajectory
     - Quota type (AI/HS/OS), category pool, and institutional prestige score.

4. **"Why This Probability?" Deep-Dive Panel:**
   - Diagnostic panel for any selected choice that demystifies the underlying statistical model:
     - 5-year historical closing rank trajectory plot.
     - **Trend Badge:**
       - **Getting harder** (red) - Closing rank decreasing → program cutoffs tightening / competition rising
       - **Getting easier** (green) - Closing rank increasing → program cutoffs relaxing / easier to get
       - **Volatile** (grey) - Cutoffs fluctuate significantly across years
     - Step-by-step formula breakdown: student rank vs predicted median closing rank $\hat{y}$ with 80% prediction interval and model uncertainty $\sigma$ read from saved model metrics.

5. **PDF Export of Choice List:**
   - Formatted, printable/shareable PDF document of the optimized choice list generated via `reportlab`.
   - Contains student profile header, summary metrics (cumulative admission probability, risk mix counts), formatted table of choices with tiers, disclaimer, and official JoSAA verification note.

## Success criteria
- Model beats the naive baseline ("closing rank = last year's closing rank") on MAE of log(closing rank) on a held-out year
- Calibration: the 80% prediction interval contains 70-90% of true closing ranks on the held-out year
- App responds in under 3 seconds
- Deployed public link and README by end of today

## Risks
| Risk | Mitigation |
|---|---|
| Scraping the JoSAA archive is slow or blocked | Final round only, 3+ years minimum, fall back to copy-paste plus `pd.read_html` |
| Yearly shocks (JEE difficulty, new seats, policy) | Show probability ranges, never promise outcomes |
| Users treat output as a guarantee | Disclaimer on every results screen |
