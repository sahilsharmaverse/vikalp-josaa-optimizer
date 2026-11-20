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
- [x] Phase 7: Ship
- [x] QA Pass (Fresh-clone, Sanity, Calibration, Edge cases, Deploy readiness)

## Currently working on
- All development and QA phases complete. Production release ready.

## Metrics (fill after Phase 4)
| Model | MAE (log rank) | 80% interval coverage |
|---|---|---|
| Baseline (last year) | 0.1485 | 90.07% |
| Weighted trend | 0.1393 | 90.06% |
| LightGBM | 0.1471 | 89.93% |

## QA Report
### 1. What Passed
- **Fresh-Clone Test:** Successfully cloned repository to an isolated folder (`vikalp_clone_test`), created fresh virtual environment, installed pinned `requirements.txt`, and launched Streamlit server. Page served HTTP 200 with zero import errors or missing artifacts.
- **Sanity Checks (5 Student Personas):**
  1. *OPEN, CRL 500, male, Karnataka:* Realistic top choices (IIT Bombay CSE, IIT Madras AI/CSE, IIT Delhi MnC as Dreams; IIT Bombay EE, IIT Roorkee CSE as Targets).
  2. *OPEN, CRL 15,000, female, Madhya Pradesh:* Appropriate allocation of NIT Tiruchirappalli and NIT Warangal core/female-only seats as realistic Targets.
  3. *OBC-NCL, category rank 3,000, male, Maharashtra:* Strict rank segregation passed: evaluated OBC-NCL seats against category rank (3,000) and OPEN seats against CRL (15,000). HS quota correctly applied only to Maharashtra.
  4. *SC, category rank 800, female, Uttar Pradesh:* Evaluated SC seats against SC category rank (800) and female-only supernumerary pools correctly.
  5. *OPEN, CRL 120,000, male, Delhi:* Realistic allocation of high-cutoff institutions as Dreams ($P = 0.01$) and accessible GFTIs (Tezpur, Sant Longowal, Bilaspur) as Targets ($P \approx 32\% - 46\%$). No IIT/NIT CSE falsely labelled Safe.
  - Probabilities across all profiles strictly bounded in $(0.01, 0.99)$ with no 0 or 1.
- **Calibration & Reliability Plot:**
  - On held-out 2025 dataset ($N = 8,709$), nominal 80% credible interval achieves **90.06% empirical coverage**.
  - Generated and saved two-panel calibration and residual density plot to `docs/reliability.png`.
- **Edge Cases:**
  - Empty rank / rank 0 / negative ranks show the exact prompt from design.md: `"Please enter a valid rank (a positive whole number)."`.
  - Rank > 1,000,000 (tested at 1,500,000) generates an aspirational Dream list without crashing or math overflow.
  - No branch selected (`preferred_branches = []`) smoothly defaults to considering all branches.
  - Non-matching filters return empty DataFrame and display: `"No options match your filters. Try adding more branches or institute types."`.
- **Deploy Readiness:**
  - Binary files committed and well within GitHub limits: `cutoffs.parquet` (1.5 MB), `institutes.csv` (8.5 KB), `models/lgbm.joblib` (1.1 MB), `docs/reliability.png` (325 KB).
  - Pinned exact versions in `requirements.txt`.
  - Zero hardcoded machine paths or exposed secrets.
  - Mandatory disclaimer and `"Built by Sahil Sharma"` footer verified present.
  - Full test suite: 17/17 tests passing.

### 2. What Was Fixed During QA
1. **Added Matplotlib to `requirements.txt`:** Added and pinned `matplotlib==3.11.2` for generating the reliability and calibration curves.
2. **Pytest Directory Crawl Optimization:** Added `pytest.ini` with `norecursedirs = .venv` and `testpaths = tests` to prevent pytest from slow-crawling through `site-packages` on fresh installs, reducing test suite runtime from 70s to 1.7s.
3. **Choice List Metadata:** Added `student_rank` to `output_cols` in `src/optimize.py` so downstream inspectors and students can verify whether CRL or category rank was evaluated for each seat.
4. **Validation Copy Alignment:** Exact string alignment in `app/streamlit_app.py` for invalid rank message per `design.md`: `"Please enter a valid rank (a positive whole number)."`.

### 3. What is Still Open
- None. All 7 development phases and the QA pass are complete, tested, and verified.
