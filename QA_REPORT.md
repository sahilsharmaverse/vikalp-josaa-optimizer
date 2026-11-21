# QA Report: Vikalp v1.1 Production Release

**Author:** Sahil Sharma  
**Date:** 2026-11-21  
**Project:** Vikalp (JoSAA Choice-List Optimizer)  
**Test Suite Status:** **46 / 46 Passed** (`pytest tests/ -v`)

---

## 1. Executive Summary

This QA report details the comprehensive verification, calibration analysis, sanity testing across student candidate archetypes, edge-case testing, and ship readiness checks performed for the **Vikalp v1.1** release.

All automated unit tests pass without errors or warnings. File paths have been strictly configured using `Path(__file__)` to ensure cross-platform portability on both Windows and Linux CI environments.

---

## 2. Issues Fixed During Final Pass

| Issue Identified | Root Cause | Fix Implemented | Verification Result |
|---|---|---|---|
| **B.Arch & B.Planning Seats in Engineering Lists** | JoSAA lists Architecture and Planning seats alongside engineering, but B.Arch/B.Plan uses JEE Main Paper 2 scores rather than Paper 1 CRL. | Excluded all programs containing "Architecture" or "Planning" (except "Naval Architecture") from data and choice catalog in `src/clean.py` and `src/optimize.py`. Added user-facing notes in Streamlit UI and PDF. | **PASSED:** 1,305 non-Paper 1 rows removed. No Architecture/Planning options appear in engineering lists. |
| **Noise Below Minimum Probability Floor** | Unviable options near 1.0% were diluting the Dream pool and padding candidate lists. | Enforced named constant `MIN_LIST_PROB = 0.05` in `src/optimize.py`. Dream tier updated to $0.05 \le P < 0.30$. Shorter lists returned if fewer than $N$ options meet 5% threshold without padding. | **PASSED:** All options in choice lists strictly satisfy $P \ge 5.0\%$. Tested via `test_min_list_prob_floor_and_tier_constants`. |
| **Brier Score Degradation Under $\sigma$ Scaling** | Scaling down $\sigma$ from 0.2285 to 0.1540 to match nominal 80% coverage on 2025 caused Brier scores to degrade in every year due to heavy-tailed residuals. | Preserved $\sigma = 0.2285$ in `models/metrics.json` and code. Conservative 87–90% coverage protects students from understating admissions risk. Documented rationale in `README.md`. | **APPROVED & PASSED:** Option C confirmed by user. |
| **Terminal Pytest Import Resolution** | Running `pytest tests/` from a fresh terminal failed because current working directory was not in `sys.path`. | Added `pythonpath = .` to `pytest.ini` and created `tests/conftest.py`. | **PASSED:** `pytest tests/` runs cleanly without `python -m`. |
| **URL Parameter `?compare=X,Y` Fragility** | Malformed query parameter strings (e.g. `?compare=abc,-5,999`) could cause index errors. | Added `parse_compare_query_params` in `src/compare.py` with type coercion, range validation, and duplicate stripping. | **PASSED:** Tested via `test_parse_compare_query_params`. |

---

## 3. Calibration & PIT Validation Results

Evaluated across all held-out years using only prior cutoff cycles:

| Held-Out Cycle | Evaluated Rows | Log-Rank MAE | Residual Std ($\sigma_{\text{year}}$) | 50% Prediction Interval | 80% Prediction Interval | 90% Prediction Interval | Integrated Brier Score |
|---|---|---|---|---|---|---|---|
| **2023** | 7,481 | 0.1587 | 0.2272 | 64.38% | **87.23%** | 92.25% | 0.1535 |
| **2024** | 8,003 | 0.1486 | 0.2273 | 68.09% | **87.42%** | 92.15% | 0.1483 |
| **2025** | 8,709 | 0.1393 | 0.2285 | 70.34% | **90.06%** | 93.77% | 0.1430 |

### Findings & Observations
1. **Conservative Safety:** Coverage for the nominal 80% interval spans 87.2% to 90.1% due to leptokurtic error distribution. For college choice optimization, conservative coverage provides a vital safety buffer against volatile cutoffs.
2. **Reliability Plot & PIT Histogram:** Saved to `docs/calibration.png`. Confirms well-centered PIT distribution with no structural bias.

---

## 4. Sanity Test Verification (Candidate Archetypes)

All 6 test cases evaluated with `build_list` and 2,000-run Monte Carlo portfolio simulation ($\rho = 0.5$):

### Case (a): OPEN CRL 500, Male, Karnataka (Advanced Rank BLANK)
- **Top 10 Choices:** NIT Tiruchirappalli (Chem, Civil, CSE, EEE, ECE, ICE, Mech, MME, Production), NIT Warangal (Biotech). All $P = 99.0\%$ [Safe].
- **IIT Check:** Zero IIT options appeared (confirming Advanced rank gate).
- **Unreachable Alert:** Choice #1 has $P \ge 95\%$; 49 options sit after it.

### Case (b): OPEN CRL 500, Male, Karnataka (Advanced Rank 500)
- **Top 10 Choices:** 
  1. IIT Madras — AI and Data Analytics ($P = 5.1\%$) [Dream]
  2. IIT Delhi — Mathematics and Computing (Dual Degree) ($P = 13.1\%$) [Dream]
  3. IIT Kharagpur — Computer Science and Engineering ($P = 18.2\%$) [Dream]
  4. IIT Bombay — Electrical Engineering ($P = 36.0\%$) [Target]
  5. IIT Roorkee — Computer Science and Engineering ($P = 55.9\%$) [Target]
  6–10. IIT Bombay — Aerospace, Applied Geophysics, Mathematics, Chemical, Chemistry ($P = 99.0\%$) [Safe]
- **Summary:** Total: 50 | Dream: 3 | Target: 2 | Safe: 45 | Seat Chance: 100.0%.
- **Unreachable Alert:** Choice #6 has $P \ge 95\%$; 44 options sit after it.

### Case (c): OPEN CRL 15,000, Female, Madhya Pradesh (Adv Rank BLANK)
- **Top 10 Choices:** NIT Tiruchirappalli (Chemical $P = 29.1\%$, ICE $P = 13.1\%$, EEE Female-only $P = 5.2\%$, Mech $P = 1.4\% \to$ filtered by 5% floor to viable options). All options strictly $\ge 5.0\%$.
- **Summary:** Total: 50 | Dream: 10 | Target: 25 | Safe: 15 | Seat Chance: 100.0%.
- **Supernumerary Check:** Female-only seats properly included alongside Gender-Neutral pools.
- **Unreachable Alert:** Choice #38 has $P \ge 95\%$; 12 options sit after it.

### Case (d): OBC-NCL, CRL 18,000, Category Rank 3,000, Male, Maharashtra
- **Top 10 Choices:** IIIT Gwalior (IT, IT-Business), MNIT Jaipur (EE, AI), NIT Surathkal (EEE, Chem, Mech).
- **Summary:** Total: 50 | Dream: 10 | Target: 25 | Safe: 15 | Seat Chance: 100.0%.
- **Category Check:** Category seats evaluated strictly using category rank 3,000; OPEN seats evaluated using CRL.
- **Home State Check:** HS quota strictly restricted to institutes in Maharashtra (VNIT Nagpur).
- **Unreachable Alert:** Choice #37 has $P \ge 95\%$; 13 options sit after it.

### Case (e): SC, CRL 35,000, Category Rank 800, Female, Uttar Pradesh
- **Top 10 Choices:** NIT Tiruchirappalli (ECE SC $P = 15.3\%$, Civil OPEN $P = 14.5\%$), NIT Warangal (Mathematics $P = 28.7\%$, Biotech $P = 27.8\%$, Meta $P = 16.9\%$).
- **Summary:** Total: 50 | Dream: 10 | Target: 25 | Safe: 15 | Seat Chance: 100.0%.
- **Category Check:** SC category seats strictly evaluated using category rank 800.
- **Unreachable Alert:** Choice #36 has $P \ge 95\%$; 14 options sit after it.

### Case (f): OPEN CRL 120,000, Male, Delhi (Adv Rank BLANK)
- **Top 10 Choices:** Puducherry Technological University (Civil $P = 5.6\%$), NIAMT Ranchi (Production $P = 5.1\%$), NERIST Nirjuli (Agricultural $P = 23.6\%$), CIT Kokrajhar (Food $P = 8.9\%$), Tezpur University (Civil $P = 19.0\%$, B.Des $P = 12.1\%$, Mech $P = 7.9\%$).
- **Summary:** Total: 50 | Dream: 40 | Target: 10 | Safe: 0 | Seat Chance: 86.4%.
- **Unreachable Alert:** None. Zero options in list have $P \ge 95\%$.
- **Floor Check:** Minimum probability in list is $5.1\%$ (strictly $\ge 5\%$).

---

## 5. Edge Case Test Matrix

| Edge Case | Test Location | Handled Behavior | Result |
|---|---|---|---|
| **Empty or 0 Rank** | `test_empty_and_zero_rank` | UI halts with error; `filter_eligible_options` returns empty DataFrame | **PASSED** |
| **Negative Rank** | `test_negative_rank_handling` | `admit_probability` raises `ValueError`; UI halts with error | **PASSED** |
| **Rank > 1,000,000** | `test_rank_above_one_million` | Returns empty DataFrame safely under $P \ge 5\%$ floor without overflow | **PASSED** |
| **No Preferred Branch Selected** | `test_no_branch_selected` | Considers all available engineering branches across institutes | **PASSED** |
| **Filters Matching Nothing** | `test_filters_match_nothing` | Returns empty DataFrame; UI displays helpful filter guidance | **PASSED** |
| **Malformed `?compare` URL Query** | `test_parse_compare_query_params` | Ignores invalid tokens (`abc`, `-5`, `9999`) without crashing | **PASSED** |

---

## 6. Still Open & Known Limitations (Roadmap)

1. **Student-$t$ / Heavy-Tailed Distribution:** Gaussian CDF assumptions overstate interval widths in the middle while understating extreme outliers. Future versions will implement Student-$t$ or empirical quantile lookups.
2. **Automated Unreachable Pruning:** Currently, JoSAA top-to-bottom allocation rules flag options placed after a $\ge 95\%$ choice with a dagger (`†`) and warning banners. Future versions will offer a 1-click prune-and-backfill action.
3. **Specialized Non-Engineering Programs:** `Bachelor of Design` (B.Des at Tezpur University) is currently indexed under GFTI entries. Further manual review is scheduled to determine if UCEED rank requirements should warrant its own separate category.
4. **State-Level Counselling (KCET / MHT-CET):** Scoped for future v2 releases.

---

## 7. Sign-off

- **QA Engineer & Author:** Sahil Sharma
- **Release Status:** **READY TO SHIP**
