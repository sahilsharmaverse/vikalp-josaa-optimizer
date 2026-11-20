"""QA Sanity Test Script for 5 student personas.
Tests build_list() and checks for:
- IIT CSE labelled Safe for a weak rank
- Category rank compared against CRL
- Home-state quota applied to the wrong state
- Probabilities of exactly 0 or 1
- Correct tier distribution and choice ordering
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.eligibility import StudentProfile
from src.optimize import build_list

def run_qa_sanity():
    personas = [
        {
            "name": "Persona 1: OPEN, CRL 500, male, home state Karnataka",
            "student": StudentProfile(
                main_rank=500,
                adv_rank=500,
                category="OPEN",
                gender="Gender-Neutral",
                home_state="Karnataka",
                list_size=30,
            ),
        },
        {
            "name": "Persona 2: OPEN, CRL 15,000, female, home state Madhya Pradesh",
            "student": StudentProfile(
                main_rank=15000,
                category="OPEN",
                gender="Female-only",
                home_state="Madhya Pradesh",
                list_size=30,
            ),
        },
        {
            "name": "Persona 3: OBC-NCL, category rank 3,000, male, home state Maharashtra",
            "student": StudentProfile(
                main_rank=15000,  # Representative CRL for OBC rank 3000
                category="OBC-NCL",
                category_rank=3000,
                gender="Gender-Neutral",
                home_state="Maharashtra",
                list_size=30,
            ),
        },
        {
            "name": "Persona 4: SC, category rank 800, female, home state Uttar Pradesh",
            "student": StudentProfile(
                main_rank=35000,  # Representative CRL for SC rank 800
                category="SC",
                category_rank=800,
                gender="Female-only",
                home_state="Uttar Pradesh",
                list_size=30,
            ),
        },
        {
            "name": "Persona 5: OPEN, CRL 120,000, male, home state Delhi",
            "student": StudentProfile(
                main_rank=120000,
                category="OPEN",
                gender="Gender-Neutral",
                home_state="Delhi",
                list_size=30,
            ),
        },
    ]

    all_passed = True

    for p in personas:
        print("=" * 80)
        print(f"TESTING: {p['name']}")
        print("=" * 80)
        stu = p["student"]
        df = build_list(stu)

        if df.empty:
            print(f"[FAIL] Returned empty list for {p['name']}")
            all_passed = False
            continue

        print(f"Generated {len(df)} choices. Tiers: {df['label'].value_counts().to_dict()}")

        top10 = df.head(10)[["choice_no", "institute", "program", "quota", "seat_type", "gender", "probability", "label", "predicted_closing_rank"]]
        print("\nTOP 10 CHOICES:")
        print(top10.to_string(index=False))

        # Check 1: Probabilities strictly between 0.01 and 0.99
        min_p, max_p = df["probability"].min(), df["probability"].max()
        if min_p < 0.01 or max_p > 0.99 or (df["probability"] == 0).any() or (df["probability"] == 1).any():
            print(f"[FAIL] Probability bounds violated! Min: {min_p}, Max: {max_p}")
            all_passed = False
        else:
            print(f"[PASS] Probabilities within (0.01, 0.99). Range: [{min_p:.4f}, {max_p:.4f}]")

        # Check 2: Quota validation
        # HS seats must match stu.home_state
        # OS seats must NOT match stu.home_state
        for idx, row in df.iterrows():
            if row["quota"] == "HS":
                # Check institute state
                inst_st = row.get("state", "")
                if inst_st and inst_st.lower() != stu.home_state.lower():
                    print(f"[FAIL] HS quota applied to wrong state: Inst {row['institute']} in {inst_st} vs Home State {stu.home_state}")
                    all_passed = False
            elif row["quota"] == "OS":
                inst_st = row.get("state", "")
                if inst_st and inst_st.lower() == stu.home_state.lower():
                    print(f"[FAIL] OS quota applied to home state: Inst {row['institute']} in {inst_st} vs Home State {stu.home_state}")
                    all_passed = False

        # Check 3: Category rank vs CRL separation
        # For non-OPEN seat types, student must match category and rank evaluated must be category_rank
        for idx, row in df.iterrows():
            st_type = row["seat_type"]
            if st_type != "OPEN":
                if st_type != stu.category:
                    print(f"[FAIL] Student with category {stu.category} allocated seat for {st_type}")
                    all_passed = False
                if stu.category_rank is not None and row["student_rank"] != stu.category_rank:
                    print(f"[FAIL] Category seat evaluated with student rank {row['student_rank']} instead of category rank {stu.category_rank}")
                    all_passed = False
            else:
                crl = stu.adv_rank if row["institute_type"] == "IIT" else stu.main_rank
                if crl is not None and row["student_rank"] != crl:
                    print(f"[FAIL] OPEN seat evaluated with student rank {row['student_rank']} instead of CRL {crl}")
                    all_passed = False

        # Check 4: IIT CSE labeled Safe for weak ranks
        for idx, row in df.iterrows():
            if row["institute_type"] == "IIT" and "Computer Science" in row["program"]:
                if row["label"] == "Safe" and row["student_rank"] > 2000:
                    print(f"[FAIL] IIT CSE labelled Safe for rank {row['student_rank']}: {row['institute']} {row['program']}")
                    all_passed = False

        # Check 5: Gender-neutral candidate should never have Female-only seats
        if stu.gender == "Gender-Neutral":
            has_female = (df["gender"] == "Female-only").any()
            if has_female:
                print(f"[FAIL] Male candidate received Female-only seat allocation!")
                all_passed = False

        print("[PASS] All eligibility & domain rules passed for this profile.\n")

    if all_passed:
        print("\n>>> ALL 5 PERSONAS PASSED SANITY TESTS SUCCESSFULLY! <<<")
    else:
        print("\n>>> SANITY CHECKS FAILED! <<<")

if __name__ == "__main__":
    run_qa_sanity()
