# memory.md: Project progress (update after every phase)

> Not needed at the very start. Fill it in as you go so a new AI session can continue without re-explaining.

## Project
Vikalp: Counselling Choice-List Optimizer (JoSAA). Author: Sahil Sharma. Deadline: today (2026-11-19).

## Decisions made
- JoSAA only (IIT / NIT / IIIT / GFTI), KCET later
- Target: final-round closing rank per year; model predicts log(closing rank)
- Probability = 1 - Phi((ln(rank) - y_hat) / sigma)
- Labels: Dream < 0.30, Target 0.30-0.80, Safe >= 0.80; list mix 20 / 50 / 30
- Stack: Python, pandas, LightGBM, Streamlit. No Docker or FastAPI today.
- v1.1 Style: Bold dark theme with glass cards, navy-to-saffron gradient hero, and Electric Cyan (`#0EA5E9`) interactive accent.
- v1.1 Portfolio Admit Metric: 2,000-run Monte Carlo simulation with correlated shocks ($\rho = 0.5$) read using calibrated model error $\sigma$.
- v1.1 What-If Slider: Real-time rank sensitivity (-20% to +20%, 5% steps) with in-place rescoring and "Rebuild list at this rank" option.
- v1.1 "Why this probability?" Panel: Interactive `st.dialog` deep dive and inline panel, 5-year closing rank history, Plotly log-scale chart with shaded 80% prediction interval cone, student rank reference line, 5-year trend badges computed via log deltas $d_t = \ln(C_t) - \ln(C_{t-1})$ with named constants, plain-words mathematical explanation, model provenance, and official JoSAA verification disclaimer.

## Completed
- [x] PRD.md, Architecture.md, rules.md, phases.md, design.md written
- [x] Phase 0: Setup
- [x] Phase 1: Data collection
- [x] Phase 2: Clean and features
- [x] Phase 3: Baseline and probability
- [x] Phase 4: LightGBM and validation
- [x] Phase 5: Optimizer
- [x] Phase 6: Streamlit app
- [x] Phase 7: Ship
- [x] QA Pass (Fresh-clone, Sanity, Calibration, Edge cases, Deploy readiness)

## v1.1 Expansion Progress
- [x] Step A: Update docs (`vikalp/PRD.md` & `vikalp/design.md`) with user feedback (Monte Carlo $\rho=0.5$, trend badge names, prediction intervals, cyan accent, dark theme, ReportLab)
- [x] Feature 1: Chance of getting at least one seat (Monte Carlo engine `src/simulation.py`, 5 passing unit tests in `tests/test_simulation.py`, UI glass cards, 390px mobile & desktop screenshots verified)
- [x] Feature 2: What-if slider (Rescore engine `src/whatif.py`, 5 passing unit tests in `tests/test_whatif.py`, sub-second `@st.cache_data` updates, 6 screenshots at -20%, 0%, +20% for desktop & 390px mobile)
- [x] Quick Fixes (a) & (b): Seat admission chance with one decimal and 99.9% cap with fixed seed 42; comparison metrics count format (e.g. "10 to 25, +15").
- [x] Feature 3: Why-this-probability panel with 5-year trend badge (`src/trend.py`, `tests/test_trend.py`, `@st.dialog` + inline panel in `app/streamlit_app.py`, 6 screenshots captured across Getting harder / Getting easier / Volatile for desktop and 390px mobile).
- [ ] Feature 4: Compare 2-3 options side by side
- [ ] Feature 5: PDF export

## Currently working on
- Feature 3 completed and verified across all requested viewports and trend categories. Awaiting user OK to proceed to Feature 4 (Compare 2-3 options side by side).

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% prediction interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## QA & v1.1 Feature Verification
- **Feature 1 (Monte Carlo Simulation):** Implemented in `src/simulation.py`. In `tests/test_simulation.py`, verified that at $\rho=0$, the MC estimate matches $1 - \prod (1 - P_i)$ within 0.02.
- **Feature 2 (What-If Rank Slider):** Implemented in `src/whatif.py` with `@st.cache_data`. Tests in `tests/test_whatif.py` verify rank 1 clamp, monotonic probability changes, zero-shift identity, and strict row order preservation without reordering.
- **Quick Fixes (a) & (b):** Seat Admission Chance formatted to 1 decimal place with 99.9% display cap and fixed simulation seed (`seed=42`). Tier count comparisons formatted as `"X to Y, +Z"`.
- **Feature 3 ("Why this probability?" Panel & Trend Badge):**
  - Core logic implemented in `src/trend.py` with named constants: `MIN_HISTORY_YEARS = 3`, `VOLATILE_STD_THRESHOLD = 0.20`, `EASIER_MEAN_THRESHOLD = 0.05`, `HARDER_MEAN_THRESHOLD = -0.05`.
  - Trend badges: "Getting harder" (red), "Getting easier" (green), "Volatile" (grey), "Stable" (grey), "Not enough history" (grey) with label text next to color pill.
  - Interactive deep-dive modal implemented via `@st.dialog("Why this probability? · Deep Dive", width="large")` and inline panel.
  - Active rank callout dynamically reflects What-If shifts (e.g. `"Using your simulated rank: 16,500 · (Shifted by +10% from 15,000)"`).
  - Plotly dark-theme chart features actual closing ranks (up to 5 years), predicted closing rank star marker, shaded 80% prediction interval band, student rank horizontal reference line, log-scale y-axis with thousands separators, and all colors explicitly labeled.
  - Step-by-step plain words breakdown reveals $z$-score, 80% interval, log-space delta, formula $P = 1 - \Phi((\ln(\text{rank}) - \hat{y}) / \sigma)$, model provenance (Weighted Trend Predictor), measured error (MAE = 0.1393, 80% coverage = 90.06%), quota, seat type, and mandatory JoSAA verification reminder.
  - 7 unit tests in `tests/test_trend.py` verify synthetic rising, falling, volatile, <3 points, and stable series, plus probability numerical consistency against choice list rows.
- **Test Suite Status:** 34/34 tests passing.
