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
- [x] Phase 5: Optimizer
- [ ] Phase 6: Streamlit app
- [ ] Phase 7: Ship

## Currently working on
- Phase: 5 completed. Ready for Phase 6 (Streamlit app).
- File: app/streamlit_app.py (Phase 6)

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## Open issues / notes
- Phase 5 completed:
  - `src/eligibility.py`: Quota matching (AI always eligible; HS if institute state matches home state; OS otherwise), exam ranks separation (IIT requires JEE Advanced, others require JEE Main), category seats matching (OPEN uses CRL; reserved seats use category ranks), and gender rules (females eligible for Gender-Neutral and Female-only).
  - `src/optimize.py`: Constructs balanced choice list with 20% Dream / 50% Target / 30% Safe mix, supports College-first (prestige proxy) and Branch-first scoring, orders choices strictly Dream -> Target -> Safe, and flags choices placed after any option with P >= 0.95 as unreachable.
  - Implemented `build_list(student) -> DataFrame` and verified across 3 distinct student personas.
  - Added unit test suite in `tests/test_optimizer.py`. All 12 unit tests passing in pytest.
