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

## Completed
- [x] PRD.md, Architecture.md, rules.md, phases.md, design.md written
- [x] Phase 0: Setup
- [x] Phase 1: Data collection
- [x] Phase 2: Clean and features
- [x] Phase 3: Baseline and probability
- [x] Phase 4: LightGBM and validation
- [ ] Phase 5: Optimizer
- [ ] Phase 6: Streamlit app
- [ ] Phase 7: Ship

## Currently working on
- Phase: 4 completed. Ready for Phase 5 (Optimizer).
- File: src/eligibility.py, src/optimize.py (Phase 5)

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## Open issues / notes
- Phase 4 completed:
  - `src/train.py`: Trained LightGBM regressor (`n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=42`) on log closing ranks with historical lag and categorical features.
  - Time-based validation on held-out year 2025:
    - LightGBM achieves MAE = 0.1471 and 80% interval coverage = 89.93%, beating the naive last-year baseline (MAE = 0.1485).
    - The weighted-trend baseline achieves MAE = 0.1393 and 80% coverage = 90.06%, outperforming LightGBM.
    - As explicitly instructed in `rules.md` and `phases.md` ("If LightGBM does not beat the baseline by the hard stop, keep the baseline and say so honestly in the README"), we will not tune further and will retain the weighted-trend predictor.
  - Trained model artifact saved to `models/lgbm.joblib`.
