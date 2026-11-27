# Data Card: JoSAA Counselling Cutoffs Dataset (2021 - 2025)

Following the framework of Gebru et al. (*Datasheets for Datasets*, Communications of the ACM 2021).

---

## 1. Motivation
- **Purpose:** Provide standardized, machine-readable historical closing cutoffs from the Joint Seat Allocation Authority (JoSAA) across all participating institutes (IITs, NITs, IIITs, and GFTIs) for multi-year admission trend analysis and candidate choice optimization.
- **Created By:** Vikalp Engineering & Data Science Core.
- **Funding:** Non-commercial educational research project.

---

## 2. Dataset Composition
- **Total Records:** 44,637 cutoff observations across 5 consecutive counseling cycles:
  - 2021: 7,638 records
  - 2022: 8,380 records
  - 2023: 8,969 records
  - 2024: 9,546 records
  - 2025: 10,104 records
- **Primary Granularity:** Each row represents the final allotment (Round 5/6) closing rank for a unique 5-tuple:
  `(institute, program, quota, seat_type, gender)`.
- **Key Fields:**
  - `institute` (string): Standardized name of educational institution (e.g., "Indian Institute of Technology Bombay").
  - `program` (string): Canonical academic program title (e.g., "Computer Science and Engineering (4 Years, Bachelor of Technology)").
  - `institute_type` (category): Categorical institution tier: `IIT`, `NIT`, `IIIT`, `GFTI`.
  - `quota` (string): Allotment quota: `AI` (All India), `HS` (Home State), `OS` (Other State), `GO` (Andhra Pradesh/Telangana State).
  - `seat_type` (string): Reservation category: `OPEN`, `EWS`, `OBC-NCL`, `SC`, `ST`, `OPEN (PwD)`, etc.
  - `gender` (string): Allocation pool: `Gender-Neutral` vs `Female-only (including Supernumerary Seats)`.
  - `closing_rank` (integer): Official final closing cutoff rank.
  - `log_closing_rank` (float): Natural log of closing rank ($\ln(C)$).
  - `prev_closing_1`, `prev_closing_2`, `prev_closing_3` (floats): Lagged closing ranks from $t-1, t-2, t-3$ counseling cycles.
  - `years_of_history` (integer): Number of historical cutoff points available for this key.

---

## 3. Collection Process
- **Data Source:** Official public closing rank archives published by the Joint Seat Allocation Authority on `josaa.nic.in` (and archival portal `admissions.nic.in`).
- **Collection Mechanism:** Scraped and aggregated from official final round allotment PDFs and HTML tables for years 2021 through 2025.
- **Verification:** Cross-checked against JoSAA published opening and closing rank brochures.
- **Status of 2026 Data:** **ABSENT**. 2026 counseling has not yet concluded; no 2026 data exists in raw or processed files.

---

## 4. Preprocessing and Cleaning
- **Exclusion of Non-Engineering Programs:** Programs offering Bachelor of Architecture (`B.Arch`) and Bachelor of Design (`B.Des`) are excluded because they rely on independent entrance examinations (AAT and UCEED) with non-comparable rank distributions.
- **Name Canonicalization:** Mapping dictionaries in `src/clean.py` reconcile multi-year name changes (e.g., "Indian Institute of Technology (BHU) Varanasi", branch nomenclature updates).
- **Temporal Alignment:** Strict chronological sorting ensures lag features are computed solely by `.shift(1)`, guaranteeing zero future leakage.

---

## 5. Uses and Limitations
- **Intended Use:** Time-series cutoff forecasting, risk stratification into Reach/Good fit/Likely portfolios, and statistical simulation.
- **Known Limitations:**
  1. *Category Conversion Dilution:* Unfilled reserved seats in later counseling rounds convert to general seats under specific JoSAA rules, occasionally causing large jumps in closing cutoffs.
  2. *Regional Quota Sparsity:* Small-state Home State (HS) quotas have few applicants, leading to higher year-to-year variance.
  3. *Preparatory Ranks:* Candidates allotted via preparatory lists (ranks with a `P` prefix) are filtered or capped to avoid rank space distortion.

---

## 6. Distribution and Maintenance
- **Storage Format:** Stored locally in Apache Parquet format (`data/processed/cutoffs.parquet`) with snappy compression.
- **Maintenance Policy:** Updated annually following the conclusion of JoSAA Round 5 allotment.
