"""Detailed sanity student evaluation script.

Runs the 6 required candidate profiles:
(a) OPEN CRL 500 male Karnataka (Adv rank blank)
(b) OPEN CRL 500 male Karnataka (Adv rank 500)
(c) OPEN CRL 15,000 female Madhya Pradesh
(d) OBC-NCL with CRL 18,000 and category rank 3,000 male Maharashtra
(e) SC category rank 800 with CRL 35,000 female Uttar Pradesh
(f) OPEN CRL 120,000 male Delhi

Computes:
- Top 10 choices
- Counts of Dream (0.05 <= P < 0.30), Target (0.30 <= P < 0.80), Safe (P >= 0.80)
- Seat admission chance (Monte Carlo, 2000 runs, rho=0.5, seed=42)
- How many options sit after the first option with P >= 95%
"""

from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.eligibility import StudentProfile
from src.optimize import build_list
from src.simulation import simulate_at_least_one_seat_probability

test_cases = [
    {
        "id": "(a)",
        "title": "Case (a): OPEN CRL 500 male Karnataka (Advanced rank BLANK)",
        "inputs": {
            "main_rank": 500,
            "adv_rank": None,
            "category": "OPEN",
            "category_rank": None,
            "gender": "Gender-Neutral",
            "home_state": "Karnataka",
        },
    },
    {
        "id": "(b)",
        "title": "Case (b): OPEN CRL 500 male Karnataka (Advanced rank 500)",
        "inputs": {
            "main_rank": 500,
            "adv_rank": 500,
            "category": "OPEN",
            "category_rank": None,
            "gender": "Gender-Neutral",
            "home_state": "Karnataka",
        },
    },
    {
        "id": "(c)",
        "title": "Case (c): OPEN CRL 15,000 female Madhya Pradesh",
        "inputs": {
            "main_rank": 15000,
            "adv_rank": None,
            "category": "OPEN",
            "category_rank": None,
            "gender": "Female-only",
            "home_state": "Madhya Pradesh",
        },
    },
    {
        "id": "(d)",
        "title": "Case (d): OBC-NCL with CRL and category rank 3,000 male Maharashtra",
        "inputs": {
            "main_rank": 18000,
            "adv_rank": None,
            "category": "OBC-NCL",
            "category_rank": 3000,
            "gender": "Gender-Neutral",
            "home_state": "Maharashtra",
        },
    },
    {
        "id": "(e)",
        "title": "Case (e): SC category rank 800 with CRL female Uttar Pradesh",
        "inputs": {
            "main_rank": 35000,
            "adv_rank": None,
            "category": "SC",
            "category_rank": 800,
            "gender": "Female-only",
            "home_state": "Uttar Pradesh",
        },
    },
    {
        "id": "(f)",
        "title": "Case (f): OPEN CRL 120,000 male Delhi",
        "inputs": {
            "main_rank": 120000,
            "adv_rank": None,
            "category": "OPEN",
            "category_rank": None,
            "gender": "Gender-Neutral",
            "home_state": "Delhi",
        },
    },
]

for tc in test_cases:
    inp = tc["inputs"]
    student = StudentProfile(
        main_rank=inp["main_rank"],
        adv_rank=inp["adv_rank"] if inp["adv_rank"] else 0,
        category=inp["category"],
        category_rank=inp["category_rank"] if inp["category_rank"] else 0,
        gender=inp["gender"],
        home_state=inp["home_state"],
        list_size=50,
        priority_mode="College-first",
    )

    df = build_list(student)

    # Portfolio seat chance
    if not df.empty:
        seat_chance = simulate_at_least_one_seat_probability(
            student_ranks=df["student_rank"].values,
            y_hats=df["y_hat"].values,
            seed=42,
        )
    else:
        seat_chance = 0.0

    # Dream, Target, Safe counts
    dream_cnt = int((df["label"] == "Dream").sum()) if not df.empty else 0
    target_cnt = int((df["label"] == "Target").sum()) if not df.empty else 0
    safe_cnt = int((df["label"] == "Safe").sum()) if not df.empty else 0
    total_cnt = len(df)

    # Count options sitting after first option with P >= 0.95
    after_95_cnt = 0
    first_95_cno = None
    if not df.empty:
        p_vals = df["probability"].values
        cno_vals = df["choice_no"].values
        found_95 = False
        for p, cno in zip(p_vals, cno_vals):
            if found_95:
                after_95_cnt += 1
            elif p >= 0.95:
                found_95 = True
                first_95_cno = cno

    print("=" * 85)
    print(f"{tc['title']}")
    print(f"Inputs: Main CRL={inp['main_rank']:,} | Adv Rank={inp['adv_rank'] or 'BLANK'} | "
          f"Category={inp['category']} | Cat Rank={inp['category_rank'] or 'N/A'} | "
          f"Gender={inp['gender']} | Home State={inp['home_state']}")
    print(f"Summary: Total Choices={total_cnt} | Dream={dream_cnt} | Target={target_cnt} | Safe={safe_cnt} | "
          f"Seat Admission Chance={seat_chance:.1%}")
    if first_95_cno is not None:
        print(f"Unreachable Alert: Choice #{first_95_cno} has P >= 95%. "
              f"Options sitting after it: {after_95_cnt} out of {total_cnt} choices.")
    else:
        print("Unreachable Alert: None (No choice in list has P >= 95%). Options after 95%: 0.")
    print("-" * 85)

    if df.empty:
        print("  [EMPTY LIST]: No options met the 5% minimum probability threshold!")
    else:
        top10 = df.head(10)
        for _, r in top10.iterrows():
            cno = int(r["choice_no"])
            inst = r["institute"]
            prog = r["program"]
            quota = r["quota"]
            stype = r["seat_type"]
            gnd = r["gender"]
            p_val = float(r["probability"])
            lbl = r["label"]
            warn = f" [WARN: {r['warning']}]" if r["warning"] else ""
            print(f"  #{cno:2d}: {inst[:38]} | {prog[:32]} | {quota} {stype} ({gnd}) | P={p_val:5.1%} [{lbl:6s}]{warn}")

    # Specific assertions / flags:
    if tc["id"] == "(a)":
        iit_count = int((df["institute_type"] == "IIT").sum()) if not df.empty else 0
        if iit_count > 0:
            print(f"  >>> FLAG: {iit_count} IIT options appeared when Advanced rank was blank!")
        else:
            print("  >>> PASS: Zero IIT options appeared with blank Advanced rank.")

    if tc["id"] == "(b)":
        iit_count = int((df["institute_type"] == "IIT").sum()) if not df.empty else 0
        print(f"  >>> PASS: {iit_count} IIT options correctly appeared when Advanced rank=500 entered.")

    # Check for any B.Arch / B.Plan
    if not df.empty:
        arch_cnt = int(df["program"].str.contains("Architecture|Planning", case=False).sum())
        naval_cnt = int(df["program"].str.contains("Naval Architecture", case=False).sum())
        if arch_cnt > naval_cnt:
            print(f"  >>> FLAG: Found {arch_cnt - naval_cnt} B.Arch / B.Plan options in choice list!")
        else:
            print("  >>> PASS: B.Arch and B.Planning correctly excluded.")

    # Check for P < 5%
    if not df.empty:
        sub_5 = int((df["probability"] < 0.05).sum())
        if sub_5 > 0:
            print(f"  >>> FLAG: Found {sub_5} options below MIN_LIST_PROB (0.05)!")
        else:
            print(f"  >>> PASS: All options strictly have P >= 5.0% (min in list: {df['probability'].min():.1%}).")

    print("\n")
