# phases.md: Build in phases (target: finish today)

Project: Vikalp | Author: Sahil Sharma

Work top to bottom. Each phase has a time box. If you hit the time box, ship what you have and move on.

## Phase 0: Setup (30 min)
- Create repo and virtual env, install pandas, numpy, scikit-learn, lightgbm, plotly, streamlit, playwright
- Add the 6 `.md` files to the repo root and create the folder structure from `Architecture.md`
- **Done when:** `streamlit hello` runs and the folders exist

## Phase 1: Data collection (2 h, hard stop)
- `src/scrape.py` downloads the final-round OR-CR table for each year available (2021-2026 if possible, minimum 3 years)
- Save raw files as `data/raw/<year>_round<N>.csv`
- **Done when:** at least 3 years of final-round data are saved in `data/raw/`

## Phase 2: Clean and features (1 h)
- `src/clean.py`: normalize names, ranks, quotas, seat types, genders; apply the rules in `rules.md`
- Build `institutes.csv` (type, state)
- `src/features.py`: lag features per key
- Quick EDA notebook: row counts per year, rank distributions, a few sanity checks (IIT Bombay CSE closing rank should be very small, and so on)
- **Done when:** `data/processed/cutoffs.parquet` exists with the schema in `Architecture.md`

## Phase 3: Baseline and probability (1 h)
- `src/baseline.py`: weighted trend of the last 2-3 years, residual sigma on a held-out year
- `src/predict.py`: `admit_probability(student_rank, y_hat, sigma)`
- **Done when:** for a test rank, you get a list of options with probabilities that look sensible, and `tests/test_probability.py` passes

## Phase 4: LightGBM and validation (75 min, hard stop)
- `src/train.py`: LightGBM on log closing rank, time-based split (train <=2024, validate 2025, test 2026)
- Compare MAE against the baseline, check 80% interval coverage, save `models/lgbm.joblib`
- **Done when:** a metrics table (baseline vs LightGBM) is saved. If LightGBM does not beat the baseline by the hard stop, keep the baseline and say so honestly in the README.

## Phase 5: Optimizer (45 min)
- `src/eligibility.py`: quota logic (AI always; HS if institute state == home state; OS otherwise), gender, category rank
- `src/optimize.py`: Dream / Target / Safe labels, 20/50/30 mix, preference score (College-first or Branch-first), unreachable-option warning
- **Done when:** a function returns an ordered list of N options for a sample student

## Phase 6: Streamlit app (75 min)
- Sidebar inputs, results table with color-coded labels, filters, CSV download, disclaimer
- Follow `design.md`
- **Done when:** the whole flow works locally from rank input to CSV download

## Phase 7: Ship (45 min)
- README with problem, data source, method, metrics, screenshots, limitations
- Push to GitHub, deploy on Streamlit Community Cloud, test the public link
- **Done when:** public URL works and the README is complete

## Cut list if time runs out (cut in this order)
1. Chart of past closing ranks
2. LightGBM (ship the baseline)
3. Branch-first / College-first toggle (ship College-first only)
4. Female-only seat handling (ship gender-neutral only and say so)
