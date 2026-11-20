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
- [ ] Phase 1: Data collection
- [ ] Phase 2: Clean and features
- [ ] Phase 3: Baseline and probability
- [ ] Phase 4: LightGBM and validation
- [ ] Phase 5: Optimizer
- [ ] Phase 6: Streamlit app
- [ ] Phase 7: Ship

## Currently working on
- Phase: 0 completed. Ready for Phase 1 (Data collection).
- File: src/scrape.py (Phase 1)

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | | |
| Weighted trend | | |
| LightGBM | | |

## Open issues / notes
- Phase 0 completed:
  - Repository structure initialized under `vikalp/`.
  - Python 3.11 virtual environment created and all required packages installed (`pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `plotly`, `streamlit`, `playwright`, `pyarrow`, `joblib`, `pytest`).
  - `.streamlit/config.toml` created with theme from `design.md`.
  - Streamlit verified.
