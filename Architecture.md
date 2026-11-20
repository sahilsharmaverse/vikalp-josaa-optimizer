# Architecture: Vikalp (Counselling Choice-List Optimizer)

**Author:** Sahil Sharma

## App flow
```
Student inputs (rank, category, gender, home state, prefs)
        |
        v
Eligibility filter  -> which (institute, program, quota, seat_type, gender) rows apply to this student
        |
        v
Predictor           -> predicted log(closing rank) for the 2027 cycle for each eligible row
        |
        v
Probability         -> P(admit) = 1 - Phi( (ln(student_rank) - y_hat) / sigma )
        |
        v
Optimizer           -> score by preference, label Dream/Target/Safe, enforce 20/50/30 mix, order list
        |
        v
Streamlit UI        -> table, filters, chart, CSV download
```

**Meaning of the formula:** a student is admitted if their rank <= closing rank. `y_hat` is the predicted log closing rank, `sigma` is the residual standard deviation measured on held-out years, `Phi` is the standard normal CDF.

## Tech stack
| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Scraping | Playwright (or `requests` + `pandas.read_html` if the form allows) |
| Data | pandas, numpy, parquet/CSV |
| Model | LightGBM (baseline: weighted trend), scikit-learn for metrics |
| App | Streamlit |
| Deploy | GitHub + Streamlit Community Cloud |
| Charts | Plotly |

## Folder and file structure
```
vikalp/
├── PRD.md
├── Architecture.md
├── rules.md
├── phases.md
├── design.md
├── memory.md
├── README.md
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                  # downloaded JoSAA tables, one file per year-round
│   └── processed/
│       ├── cutoffs.parquet   # cleaned long table
│       └── institutes.csv    # institute -> type, state, prestige proxy
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── scrape.py             # download raw OR-CR tables
│   ├── clean.py              # normalize names, ranks, quotas
│   ├── features.py           # lag features per key
│   ├── baseline.py           # weighted-trend predictor + residual sigma
│   ├── train.py              # LightGBM training, time-based validation
│   ├── predict.py            # predict_closing_rank(), admit_probability()
│   ├── eligibility.py        # quota / gender / category logic
│   └── optimize.py           # labels, risk mix, ordering
├── models/
│   └── lgbm.joblib
├── app/
│   └── streamlit_app.py
└── tests/
    └── test_probability.py
```

## Core data schema (`cutoffs.parquet`)
| Column | Example |
|---|---|
| year | 2025 |
| round | 6 (use the final round of each year) |
| institute | NIT Bhopal (cleaned name) |
| institute_type | IIT / NIT / IIIT / GFTI |
| state | Madhya Pradesh |
| program | Computer Science and Engineering (cleaned) |
| quota | AI / HS / OS |
| seat_type | OPEN / EWS / OBC-NCL / SC / ST |
| gender | Gender-Neutral / Female-only |
| opening_rank | 6025 |
| closing_rank | 10403 |

**Key** = `(institute, program, quota, seat_type, gender)`. One row per key per year.

## Features for the model
`prev_closing_1`, `prev_closing_2`, `prev_closing_3`, `mean_last_2`, `trend_slope`, `delta_1yr`, `years_of_history`, `institute_type`, `branch_group` (CSE / ECE / Mech / Civil / ...), `quota`, `seat_type`. Target: `log(closing_rank)` of the next year.

## Validation plan
- Train on years up to 2024, validate on 2025, final test on 2026
- Compare against the baseline (last year's closing rank)
- Retrain on all years including 2026 for the app (predicts 2027)
- Report: MAE on log rank, coverage of the 80% interval, a reliability plot
