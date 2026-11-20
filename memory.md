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
- [ ] Feature 2: What-if slider
- [ ] Feature 3: Why-this-probability panel (5-year trend badges)
- [ ] Feature 4: Compare 2-3 options side by side
- [ ] Feature 5: PDF export

## Currently working on
- Feature 1 completed and verified. Awaiting user OK to proceed to Feature 2 (What-if rank slider).

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% prediction interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## QA & v1.1 Feature 1 Verification
- **Monte Carlo Simulation:** Implemented in `src/simulation.py`. In `tests/test_simulation.py`, verified that at $\rho=0$, the MC estimate matches $1 - \prod (1 - P_i)$ within 0.02.
- **UI & Layout:** Bold dark glass aesthetic configured with `.streamlit/config.toml` (`#0EA5E9` accent, `#0B0F19` background) and `app/styles.css`.
- **Responsive Display:** Screenshots captured at 1280px (desktop) and 390px (mobile iPhone viewport). Zero crashes; test suite has 22 passing tests.
