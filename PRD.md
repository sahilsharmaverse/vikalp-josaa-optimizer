# PRD: Vikalp (Counselling Choice-List Optimizer for JoSAA)

**Author:** Sahil Sharma

## What to build
A web app that tells a JEE student which IIT / NIT / IIIT / GFTI seats they can realistically get, shows an **admission probability** for each option, and generates an ordered **JoSAA choice list** balanced across Dream / Target / Safe.

## Problem
- Students fill 50-100+ choices under stress using raw cutoff PDFs or predictors that show last year's cutoff with no uncertainty.
- A bad list wastes seats (safe options placed too high) or leaves no fallback (all dream options).
- Rank types confuse people: CRL vs category rank, AI / HS / OS quotas, gender-neutral vs female-only seats.

## Targeted users
- **Primary:** JEE Main / Advanced students (12th pass-outs and droppers) entering JoSAA, especially from Tier-2/3 cities with no paid counsellor.
- **Secondary:** parents, school and coaching counsellors.

## Scope
**In (v1):** JoSAA only. IIT, NIT, IIIT, GFTI. Final-round closing ranks for all years available (target 2021-2026). Predict the 2027 cycle.
**Out (v1):** KCET / MHT-CET / NEET, Docker, FastAPI, login, Hindi UI, mobile app, paid tiers.

## User inputs
- JEE Main CRL rank (for NIT / IIIT / GFTI)
- JEE Advanced rank (optional, for IITs)
- Category (OPEN / EWS / OBC-NCL / SC / ST) and the **category rank** if not OPEN
- Gender (female students are eligible for gender-neutral and female-only seats)
- Home state
- Preferred branches (multi-select), optional preferred states, institute types
- Choice list size (default 50)
- Priority mode: **College-first** or **Branch-first**

## Features
**Must-have (MVP)**
1. Admission probability for every eligible option (quota-aware)
2. Dream / Target / Safe label
3. Auto-built ordered choice list with a target risk mix (20% / 50% / 30%)
4. Filters: branch, state, institute type
5. Download choice list as CSV
6. Visible disclaimer: "Estimate, not official. Verify at josaa.nic.in."

**Should-have**
- Mini chart of the option's last 3-5 years of closing ranks plus the predicted range
- Warning for options placed after a near-certain option (unreachable)

**Won't have (v1):** anything in the Out list above.

## Success criteria
- Model beats the naive baseline ("closing rank = last year's closing rank") on MAE of log(closing rank) on a held-out year
- Calibration: the 80% prediction interval contains 70-90% of true closing ranks on the held-out year
- App responds in under 3 seconds
- Deployed public link and README by end of today

## Risks
| Risk | Mitigation |
|---|---|
| Scraping the JoSAA archive is slow or blocked | Final round only, 3+ years minimum, fall back to copy-paste plus `pd.read_html` |
| Yearly shocks (JEE difficulty, new seats, policy) | Show probability ranges, never promise outcomes |
| Users treat output as a guarantee | Disclaimer on every results screen |
