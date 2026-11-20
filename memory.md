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
- [x] Phase 6: Streamlit app
- [ ] Phase 7: Ship

## Currently working on
- Phase: 6 completed. Ready for Phase 7 (Ship).
- File: README.md (Phase 7)

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## Open issues / notes
- Phase 6 completed:
  - `app/streamlit_app.py`: Implemented single-page Streamlit application per `design.md`.
  - Sidebar inputs: JEE Main CRL rank, optional JEE Advanced rank, Category & Category rank, Gender, Home State, Priority mode (College-first vs Branch-first), Preferred branches, Institute types, and Choice list size slider (10-100).
  - Main area displays:
    - 3 metric cards with exact design theme colors: Dream (Red #DC2626), Target (Amber #D97706), and Safe (Green #16A34A).
    - Results table with `st.column_config.ProgressColumn` for Admit Chance, risk tier pills, and 80% prediction intervals.
    - Warning alert banner for choices placed after near-certain options (P >= 95%).
    - Interactive 5-year closing rank trajectory Plotly line chart with student rank baseline and 2026 projected error ranges.
    - One-click CSV choice-list download button.
    - Mandatory disclaimer and "Built by Sahil Sharma" footer.
  - Successfully verified in browser at `http://localhost:8501`.
