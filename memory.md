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
- [ ] Phase 3: Baseline and probability
- [ ] Phase 4: LightGBM and validation
- [ ] Phase 5: Optimizer
- [ ] Phase 6: Streamlit app
- [ ] Phase 7: Ship

## Currently working on
- Phase: 2 completed. Ready for Phase 3 (Baseline and probability).
- File: src/baseline.py, src/predict.py (Phase 3)

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | | |
| Weighted trend | | |
| LightGBM | | |

## Open issues / notes
- Phase 1 & 2 completed:
  - 5 consecutive years of final-round cutoffs scraped into `data/raw/` (2021-2025), totaling 52,505 raw rows.
  - `src/clean.py` implemented: drops PwD and preparatory ranks ('P' suffix), filters to AI/HS/OS quotas, standardizes institute and program names, maps institute types (IIT, NIT, IIIT, GFTI) and states.
  - `data/processed/institutes.csv` created containing 129 normalized institutes.
  - `src/features.py` implemented: assigns branch groups (`CSE`, `ECE`, `EE`, `MECH`, `CIVIL`, `CHEM`, `META`, `ARCH`, `BIO`, `OTHER`) and computes time-lag features (`prev_closing_1`, `prev_closing_2`, `prev_closing_3`, `mean_last_2`, `delta_1yr`, `trend_slope`, `years_of_history`, `log_closing_rank`).
  - Saved `data/processed/cutoffs.parquet` (45,947 rows).
  - All 10 sanity checks passed cleanly (including IIT Bombay & Delhi CSE ranks, NIT Trichy & Surathkal CSE, opening <= closing monotonicity, Quota/Seat type validity, and lag alignment).
