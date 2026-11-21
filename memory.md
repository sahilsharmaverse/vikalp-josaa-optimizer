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
- [x] Feature 4: Compare 2-3 options side by side (`src/compare.py`, `tests/test_compare.py`, side-by-side deck, Plotly multi-series chart, aligned comparison table, JoSAA unreachable order warning, deterministic quick facts, 6 screenshots verified)
- [x] Quick Fixes (a) & (b): Set `pythonpath = .` in `pytest.ini` and `tests/conftest.py` so `pytest tests/` runs cleanly without `python -m`; safe query parameter parsing in `src/compare.py` (`parse_compare_query_params`) to ignore invalid/out-of-range tokens without crashing.
- [x] Feature 5: PDF export of the choice list (`src/pdf_report.py`, `tests/test_pdf.py`, in-memory `BytesIO` generation via ReportLab, print-friendly styling, repeated headers, `NumberedCanvas` "Page x of y", unreachable dagger footnote, methodology & disclaimer last page, Page 1 and Last Page rendered to PNG).

## Currently working on
- v1.1 Ship complete. All QA checks passed, B.Arch/B.Plan excluded, 5% floor enforced, CI pipeline established, and 46/46 unit tests passing.

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% prediction interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## QA & v1.1 Feature Verification
- **Feature 1 (Monte Carlo Simulation):** Implemented in `src/simulation.py`. In `tests/test_simulation.py`, verified that at $\rho=0$, the MC estimate matches $1 - \prod (1 - P_i)$ within 0.02.
- **Feature 2 (What-If Rank Slider):** Implemented in `src/whatif.py` with `@st.cache_data`. Tests in `tests/test_whatif.py` verify rank 1 clamp, monotonic probability changes, zero-shift identity, and strict row order preservation without reordering.
- **Quick Fixes (a) & (b) [What-if & Trend]:** Seat Admission Chance formatted to 1 decimal place with 99.9% display cap and fixed simulation seed (`seed=42`). Tier count comparisons formatted as `"X to Y, +Z"`.
- **Feature 3 ("Why this probability?" Panel & Trend Badge):**
  - Core logic implemented in `src/trend.py` with named constants: `MIN_HISTORY_YEARS = 3`, `VOLATILE_STD_THRESHOLD = 0.20`, `EASIER_MEAN_THRESHOLD = 0.05`, `HARDER_MEAN_THRESHOLD = -0.05`.
  - Trend badges: "Getting harder" (red), "Getting easier" (green), "Volatile" (grey), "Stable" (grey), "Not enough history" (grey) with label text next to color pill.
  - Interactive deep-dive modal implemented via `@st.dialog("Why this probability? · Deep Dive", width="large")` and inline panel.
  - Active rank callout dynamically reflects What-If shifts (e.g. `"Using your simulated rank: 16,500 · (Shifted by +10% from 15,000)"`).
  - Plotly dark-theme chart features actual closing ranks (up to 5 years), predicted closing rank star marker, shaded 80% prediction interval band, student rank horizontal reference line, log-scale y-axis with thousands separators, and all colors explicitly labeled.
  - Step-by-step plain words breakdown reveals $z$-score, 80% interval, log-space delta, formula $P = 1 - \Phi((\ln(\text{rank}) - \hat{y}) / \sigma)$, model provenance (Weighted Trend Predictor), measured error (MAE = 0.1393, 80% coverage = 90.06%), quota, seat type, and mandatory JoSAA verification reminder.
  - 7 unit tests in `tests/test_trend.py` verify synthetic rising, falling, volatile, <3 points, and stable series, plus probability numerical consistency against choice list rows.
- **Feature 4 (Side-by-Side Option Comparison):**
  - Core logic in `src/compare.py`: `validate_selection_count`, `check_order_unreachable`, `generate_quick_facts`, `render_svg_circular_gauge`, `render_svg_sparkline`, and `create_comparison_chart`.
  - Multiselect component enforces $[2, 3]$ selections with helpful status cues when $<2$ or $>3$ are selected.
  - Supports query parameter pre-selection (`?compare=1,2,3`).
  - Responsive layout: 2-3 columns on desktop, stacked on mobile (390 px) with SVG circular gauge, tier tag, predicted cutoff + 80% interval, trend badge, choice number, and cutoff sparkline.
  - Combined Plotly chart on log y-axis with colorblind-friendly markers (`circle`, `square`, `triangle-up`) and line styles (`solid`, `dash`, `dot`) plus active student rank reference line.
  - Aligned comparison table with identical row labels across all compared options.
  - JoSAA allocation order check: alerts user if an option sits after an earlier choice with admission probability $\ge 95\%$, marking unreachable options with clear warning badges.
  - Deterministic rule-based quick facts generator concluding with mandatory advisory: *"This compares admission chances only. Branch, placement, location and fees also matter."*
  - 6 unit tests in `tests/test_compare.py` verify selection bounds, probability numerical consistency across What-If shifts, sparse history handling, unreachable order detection, quick-facts formatting, and safe query parameter parsing.
- **Quick Fixes (a) & (b) [Pytest & Compare URL Params]:**
  - Pytest runs cleanly from project root (`pytest tests/`) via `pythonpath = .` in `pytest.ini` and `tests/conftest.py`.
  - Added `parse_compare_query_params` to `src/compare.py` and unit test `test_parse_compare_query_params` in `tests/test_compare.py` validating that malformed, negative, float, out-of-range, and duplicate tokens are safely ignored.
- **Feature 5 (PDF Choice List Report):**
  - Core logic in `src/pdf_report.py`: `build_pdf_report`, `NumberedCanvas`, `safe_escape`.
  - Pure in-memory generation via ReportLab (`io.BytesIO`) without persisting student data to disk.
  - Print-friendly layout: clean white background, dark `#0F172A` text, light color fills on tier cells (`#FEE2E2` Dream, `#FEF3C7` Target, `#DCFCE7` Safe) with explicit tier labels.
  - Page 1: Title, author attribution "Built by Sahil Sharma", IST generation timestamp, complete entered student profile, active rank callout (e.g. `"Rank used: 16,500 (simulated +10%)"` or `"Rank used: 15,000"`), Seat Admission Chance (1 decimal with plain-words shared-shock assumption), and Dream/Target/Safe count distribution.
  - Choice Table: Multi-page support up to 100 choices with `repeatRows=1` header repetition, XML escaping via `xml.sax.saxutils.escape`, calibrated 80% prediction intervals, 5-year trend badges, and two-pass `NumberedCanvas` generating "Page X of Y" and running footer on all pages.
  - Dagger (`†`) unreachable alert: Automatically detects choices ordered below an option with admission probability $\ge 95\%$ and adds an explanatory footnote.
  - Last Page: How to read Dream / Target / Safe risk tiers, 3-line plain-words probability calculation breakdown, model provenance & typical error metrics read from `models/metrics.json` (MAE = 0.1393, 80% coverage = 90.06%), JoSAA sequential allocation rule explanation, full legal disclaimer, and pinned "Always verify at josaa.nic.in" verification banner.
  - UI integration: Added "Download PDF Report" button right next to the CSV download button in `app/streamlit_app.py`.
  - Dev dependencies: Isolated `pypdf` and `pymupdf` in `requirements-dev.txt`.
  - 4 unit tests in `tests/test_pdf.py` verifying `%PDF` header, non-empty generation (<3 seconds) across 1, 50, and 100 choices, special characters escaping (`&`, `<`, quotes, long dash), extracted text validation (disclaimer, author, dagger footnote), and active what-if rank callouts.
- **Final Pass & Ship Prep:**
  - **$\sigma$ Decision:** Maintained Option C ($\sigma = 0.2285$) based on held-out Brier score evaluations across 2024, 2025, and 2026.
  - **B.Arch & B.Planning Exclusion:** Filtered out 1,305 Architecture and Planning options from `data/processed/cutoffs.parquet` and `src/clean.py` since they use JEE Main Paper 2 ranks. Added explanatory callouts in Streamlit UI and PDF report.
  - **Minimum Probability Floor (`MIN_LIST_PROB = 0.05`):** Enforced strictly in `src/optimize.py` and `src/predict.py`. Redefined Dream tier to $0.05 \le P < 0.30$. If fewer than $N$ options qualify, the optimizer returns a shorter list with a clear UI message instead of padding with $<5\%$ choices.
  - **CI Pipeline:** Added `.github/workflows/tests.yml` executing tests on Ubuntu Linux with Python 3.11.
  - **Full Documentation:** Authored comprehensive `vikalp/README.md` and `vikalp/QA_REPORT.md`.
- **Test Suite Status:** 46/46 tests passing.
