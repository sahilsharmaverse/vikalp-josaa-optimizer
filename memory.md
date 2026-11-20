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
- [ ] Phase 4: LightGBM and validation
- [ ] Phase 5: Optimizer
- [ ] Phase 6: Streamlit app
- [ ] Phase 7: Ship

## Currently working on
- Phase: 3 completed. Ready for Phase 4 (LightGBM and validation).
- File: src/train.py (Phase 4)

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | | |

## Open issues / notes
- Phase 3 completed:
  - `src/baseline.py`: Weighted trend predictor (weights 0.5, 0.3, 0.2 for last 3 years; normalized for 2 years).
  - Evaluated on held-out year 2025: MAE = 0.1393 log points (beats naive last-year baseline 0.1485), residual sigma = 0.2285, 80% prediction interval coverage = 90.06%.
  - `src/predict.py`: Admission probability function `admit_probability(student_rank, y_hat, sigma) = 1 - Phi((ln(rank) - y_hat) / sigma)`, clipped to [0.01, 0.99], with risk tier labeling (`Dream` < 0.30, `Target` 0.30-0.80, `Safe` >= 0.80).
  - `tests/test_probability.py`: 8 unit tests covering rank equal to prediction, monotonicity, clipping bounds, invalid input errors, vectorized array inputs, and interval bounds. All 8 tests passing.
