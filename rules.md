# rules.md: Boundaries for the AI assistant and for me

Project: Vikalp | Author: Sahil Sharma

## Working rules
- Read `PRD.md`, `Architecture.md`, `phases.md` and `memory.md` before writing code.
- Work on **one phase at a time**. Do not start the next phase until "Done when" in `phases.md` is met.
- After each phase, update `memory.md` (completed, current file, open issues).
- Keep functions small and typed. One job per file as listed in `Architecture.md`.
- Do not add files or folders that are not in `Architecture.md` without asking.

## What to use
- Python 3.11, pandas, numpy, scikit-learn, LightGBM, Plotly, Streamlit, Playwright
- `pathlib` for paths, `joblib` for saving the model, `pytest` for tests
- Fixed random seed (`42`) everywhere
- `@st.cache_data` / `@st.cache_resource` in the app for data and model loading
- Time-based splits only (train on earlier years, test on later years)

## What to avoid
- No random train/test split (it leaks future years into training)
- No deep learning, no extra services, no database, no Docker, no FastAPI today
- No hard-coded absolute paths, no secrets in the repo
- No scraping loops without delays (`sleep` 1-2 s between requests) and no hammering the JoSAA site
- No claiming guarantees in the UI or README. Always show the disclaimer.
- Do not delete raw data files. Cleaning happens in code, writing to `data/processed/`.

## Data rules (the traps)
- Use the **final round** closing rank of each year as the target.
- Some ranks may carry a `P` suffix (preparatory seats). Drop those rows.
- OPEN seats use CRL; EWS / OBC-NCL / SC / ST seats use **category ranks**. Never compare a CRL against a category rank.
- IIT seats use JEE Advanced ranks. NIT / IIIT / GFTI seats use JEE Main ranks. Keep these separate in the UI.
- PwD seat types and quotas other than AI / HS / OS (GO, JK, LA) are out of scope for v1. Filter them out in `clean.py`.
- B.Arch and B.Planning programs use JEE Main Paper 2 ranks, not Paper 1 CRL. Exclude programs whose name contains "Architecture" or "Planning" (except "Naval Architecture") from data and choice lists.
- Normalize program names (strip text like "(4 Years, Bachelor of Technology)") and institute names consistently across years. Keep a rename map in `clean.py`.
- Drop keys with fewer than 2 years of history from the model; for the app, show them as "insufficient history".

## Probability and list rules
- Minimum list probability floor: `MIN_LIST_PROB = 0.05`. Options below 5% must never appear in the choice list.
- Labels: **Dream** 0.05 <= P < 0.30, **Target** 0.30 <= P < 0.80, **Safe** P >= 0.80.
- Target list mix: 20% Dream, 50% Target, 30% Safe.
- If fewer than N options qualify (P >= 0.05), return a shorter list with a clear message instead of padding it with options below the 5% floor.
- JoSAA allots the first option in your list that you qualify for, so anything placed after an option with P >= 0.95 is effectively unreachable. Warn or trim.
- Never show a probability as exactly 0% or 100%. Clip to 1%-99%.

## Error handling
- If a rank is missing or not a positive integer, show an inline message and stop; do not crash.
- If no options match the filters, say so and suggest loosening the filters.
- Wrap data and model loading in try/except with a clear message.
