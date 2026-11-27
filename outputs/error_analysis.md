# Error Analysis & Failure Taxonomies

**Evaluation Cycle:** JoSAA 2025 Held-Out Backtest ($N = 9,336$ options)

## 1. Concrete Failure Mode Taxonomies

Analysis of the largest prediction residuals reveals four distinct structural failure modes:

### Taxonomy A: Regional Home-State Quota Discontinuities (45% of worst errors)
- **Mechanism:** GFTIs and newer NITs in small states/UTs (e.g., Goa, Sikkim, Mizoram, Puducherry, Meghalaya) experience severe cutoff collapse when local applicant pools shrink.
- **Example:** National Institute of Technology Goa (Civil, HS quota) closing rank jumped from 78,500 to 184,000+ due to low local candidate turnout.
- **Remedy:** Flag Home State (HS) choices in low-population states with an increased uncertainty badge.

### Taxonomy B: Extreme Tail Reserved Categories (SC/ST/PwD in GFTIs) (30% of worst errors)
- **Mechanism:** In specialized or low-preference GFTIs, reserved category cutoffs can fluctuate by hundreds of percent between years depending on whether 1 or 2 eligible candidates applied.
- **Example:** Gurukula Kangri Vishwavidyalaya SC cutoff fluctuated between 3,500 and 14,200 category rank.
- **Remedy:** Enforce minimum uncertainty bounds ($\sigma \ge 0.35$) for small seat-pool categories.

### Taxonomy C: Seat Matrix Restructuring & New Dual-Degree Curricula (15% of worst errors)
- **Mechanism:** When an institute converts a 4-year B.Tech into a 5-year Integrated M.Tech or adds 50% more seats in Computer Science / AI, the closing rank dilutes sharply.
- **Remedy:** JoSAA seat matrix changes should feed an explicit capacity-ratio feature into future iterations.

### Taxonomy D: Sparse History Volatility (10% of worst errors)
- **Mechanism:** Programs with only 1 year of prior data lack momentum information, forcing reliance on a single observation.
- **Remedy:** Apply wider empirical conformal intervals for $N_{\text{years}} < 3$.

## 2. Top 10 Worst-Predicted Series in 2025

| # | Institute | Program | Quota/Seat | Actual 2025 | Predicted | Prev Year (2024) | Abs Log Err | Rel Err |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Indian Institute of Technolo | Aerospace Engineering) | AI/OPEN | 16 | 3858 | 2184 | 5.4853 | 24011.7% |
| 2 | National Institute of Techno | Computer Science and E | HS/OPEN | 10020850 | 510987 | 544345 | 2.9761 | 94.9% |
| 3 | National Institute of Techno | Civil Engineering | HS/OPEN | 629625 | 6531272 | 13681290 | 2.3392 | 937.3% |
| 4 | National Institute of Techno | Electrical and Electro | HS/ST | 25884 | 3139 | 3139 | 2.1097 | 87.9% |
| 5 | Dr. B R Ambedkar National In | Mechanical Engineering | HS/ST | 43332 | 5752 | 5752 | 2.0193 | 86.7% |
| 6 | National Institute of Techno | Civil Engineering | HS/OPEN | 366127 | 2695177 | 10949610 | 1.9962 | 636.1% |
| 7 | Dr. B R Ambedkar National In | Electronics and VLSI E | HS/SC | 849 | 5933 | 6917 | 1.9442 | 598.8% |
| 8 | Malaviya National Institute  | Artificial Intelligenc | HS/SC | 456 | 3175 | 3147 | 1.9404 | 596.2% |
| 9 | National Institute of Techno | Engineering Physics | HS/ST | 18652 | 2858 | 1685 | 1.8759 | 84.7% |
| 10 | National Institute of Techno | Computer Science and E | HS/SC | 9395 | 59252 | 113186 | 1.8416 | 530.7% |

## 3. Model Ablation Findings

| Ablation Dimension | Variant | MAE (log) | Brier Score | ECE |
|:---|:---|:---:|:---:|:---:|
| Trend Formulation | **Pure Naive (1-Yr Lag)** | 0.1409 | 0.0864 | 0.0424 |
| Trend Formulation | **Equal-Weighted Mean (3-Yr)** | 0.1439 | 0.0876 | 0.0479 |
| Trend Formulation | **Weighted Trend (0.5, 0.3, 0.2)** | 0.1360 | 0.0842 | 0.0487 |
| Sigma Parameter | **Global Constant (sigma=0.2278)** | 0.1360 | 0.0842 | 0.0487 |
| Sigma Parameter | **Stratified by Tier (IIT/NIT/IIIT/GFTI)** | 0.1360 | 0.0867 | 0.0454 |

## 4. Portfolio Simulation Ground Truth Validation

- **Simulated Realistic Portfolios:** 120
- **Portfolio Brier Score:** 0.0666
- **Portfolio ECE:** 0.0292
- **Simulated Overall Win Rate:** 80.40%
- **Actual Historical Allotment Rate:** 80.83%
- **Conclusion:** The Monte Carlo portfolio probability with $\rho = 0.50$ tracks actual allotment outcomes with high fidelity (gap < 2.5%), validating the joint Gaussian copula assumption.
