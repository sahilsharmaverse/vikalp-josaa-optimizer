"""Sanity test script for 5 candidate profiles.
Evaluates top 10 options and portfolio seat chance for each.
"""

from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from src.eligibility import StudentProfile
from src.optimize import build_list
from src.simulation import simulate_at_least_one_seat_probability

students = [
    {
        "name": "Student 1: OPEN CRL 500 male Karnataka",
        "profile": StudentProfile(
            main_rank=500,
            adv_rank=500,  # Top rank qualified for IITs
            category="OPEN",
            category_rank=0,
            gender="Gender-Neutral",
            home_state="Karnataka",
            list_size=50,
            priority_mode="College-first",
        ),
    },
    {
        "name": "Student 2: OPEN CRL 15,000 female Madhya Pradesh",
        "profile": StudentProfile(
            main_rank=15000,
            adv_rank=0,
            category="OPEN",
            category_rank=0,
            gender="Female-only",
            home_state="Madhya Pradesh",
            list_size=50,
            priority_mode="College-first",
        ),
    },
    {
        "name": "Student 3: OBC-NCL category rank 3,000 male Maharashtra",
        "profile": StudentProfile(
            main_rank=18000,
            adv_rank=0,
            category="OBC-NCL",
            category_rank=3000,
            gender="Gender-Neutral",
            home_state="Maharashtra",
            list_size=50,
            priority_mode="College-first",
        ),
    },
    {
        "name": "Student 4: SC category rank 800 female Uttar Pradesh",
        "profile": StudentProfile(
            main_rank=35000,
            adv_rank=0,
            category="SC",
            category_rank=800,
            gender="Female-only",
            home_state="Uttar Pradesh",
            list_size=50,
            priority_mode="College-first",
        ),
    },
    {
        "name": "Student 5: OPEN CRL 120,000 male Delhi",
        "profile": StudentProfile(
            main_rank=120000,
            adv_rank=0,
            category="OPEN",
            category_rank=0,
            gender="Gender-Neutral",
            home_state="Delhi",
            list_size=50,
            priority_mode="College-first",
        ),
    },
]

for s in students:
    p = s["profile"]
    df = build_list(p)
    seat_chance = simulate_at_least_one_seat_probability(
        student_ranks=df["student_rank"].values,
        y_hats=df["y_hat"].values,
        seed=42,
    )
    print("=" * 80)
    print(f"{s['name']}")
    print(f"Total choices generated: {len(df)} | Seat Admission Chance: {seat_chance:.1%}")
    print("-" * 80)
    top10 = df.head(10)[["choice_no", "institute", "program", "quota", "seat_type", "gender", "probability", "label", "warning"]]
    for _, r in top10.iterrows():
        p_str = f"{r['probability']:.1%}"
        warn_str = f" [WARN: {r['warning']}]" if r["warning"] else ""
        print(f"#{int(r['choice_no'])}: {r['institute']} | {r['program']} | {r['quota']} {r['seat_type']} ({r['gender']}) | P={p_str} [{r['label']}]{warn_str}")
    
    # Anomaly checks:
    # 1. Any probability exactly 0 or 1?
    exact_0_1 = df[(df["probability"] <= 0.0) | (df["probability"] >= 1.0)]
    if not exact_0_1.empty:
        print(f"  FLAG: Found {len(exact_0_1)} options with probability exactly 0 or 1!")
    
    # 2. Category rank check: For category seats, is student_rank equal to category_rank?
    if p.category != "OPEN":
        cat_rows = df[df["seat_type"] == p.category]
        if not cat_rows.empty:
            wrong_rank = cat_rows[cat_rows["student_rank"] != p.category_rank]
            if not wrong_rank.empty:
                print(f"  FLAG: Category seats compared to wrong rank instead of category rank {p.category_rank}!")
            else:
                print(f"  PASS: Category seats correctly evaluated using category rank {p.category_rank}.")

    # 3. Home State check: Are HS seats only for home_state?
    hs_rows = df[df["quota"] == "HS"]
    if not hs_rows.empty:
        wrong_hs = hs_rows[hs_rows["state"] != p.home_state]
        if not wrong_hs.empty:
            print(f"  FLAG: HS quota applied to wrong state (expected {p.home_state})!")
        else:
            print(f"  PASS: HS quota strictly applied only to institutes in {p.home_state}.")

    # 4. Weak rank IIT CSE safe check
    if p.main_rank > 10000:
        bad_safe = df[(df["institute_type"] == "IIT") & (df["program"].str.contains("Computer Science")) & (df["label"] == "Safe")]
        if not bad_safe.empty:
            print("  FLAG: IIT CSE labelled Safe for a weak rank!")
        else:
            print("  PASS: No spurious Safe IIT CSE options for weak rank.")
    print("\n")
