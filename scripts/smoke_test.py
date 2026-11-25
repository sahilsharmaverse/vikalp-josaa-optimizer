import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import pypdf
import pandas as pd
from src.eligibility import StudentProfile
from src.optimize import build_list
from src.pdf_report import build_pdf_report
from src.simulation import simulate_at_least_one_seat_probability

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

profiles = [
    (
        "Profile A (CRL 45,000, CSE, NIT/IIIT, General Male)",
        StudentProfile(
            main_rank=45000,
            adv_rank=None,
            category="OPEN",
            gender="Gender-Neutral",
            home_state="Delhi",
            preferred_branches=["CSE"],
            institute_types=["NIT", "IIIT"],
            list_size=30,
        ),
        "profile_a_crl45k_cse_male.pdf"
    ),
    (
        "Profile B (CRL 25,000, General Female)",
        StudentProfile(
            main_rank=25000,
            adv_rank=None,
            category="OPEN",
            gender="Female-only",
            home_state="Maharashtra",
            list_size=30,
        ),
        "profile_b_crl25k_female.pdf"
    ),
    (
        "Profile C (JEE Main Only, adv_rank=None, CRL 12,000, General Male)",
        StudentProfile(
            main_rank=12000,
            adv_rank=None,
            category="OPEN",
            gender="Gender-Neutral",
            home_state="Uttar Pradesh",
            list_size=30,
        ),
        "profile_c_main_only.pdf"
    ),
]

print("=" * 70)
print("VIKALP SMOKE TEST EXECUTION")
print("=" * 70)

for name, prof, pdf_filename in profiles:
    print(f"\n--- {name} ---")
    df = build_list(prof)
    total_choices = len(df)
    
    if not df.empty:
        counts = df["label"].value_counts().to_dict()
        dream_count = counts.get("Dream", 0)
        target_count = counts.get("Target", 0)
        safe_count = counts.get("Safe", 0)
        lowest_prob = df["probability"].min()
        warning_triggered = safe_count < 5
    else:
        dream_count = target_count = safe_count = 0
        lowest_prob = None
        warning_triggered = True

    print(f"Total Choices: {total_choices}")
    print(f"Dream: {dream_count} | Target: {target_count} | Safe: {safe_count}")
    print(f"Lowest Probability: {f'{lowest_prob:.4f}' if lowest_prob is not None else 'N/A'}")
    print(f"'< 5 Safe choices' Warning: {'TRIGGERED' if warning_triggered else 'NOT TRIGGERED'}")
    
    # Calculate portfolio chance
    seat_chance = (
        simulate_at_least_one_seat_probability(
            student_ranks=df["student_rank"].values,
            y_hats=df["y_hat"].values,
            sigma=0.2285,
            rho=0.5,
            n_sims=2000,
            seed=42,
        )
        if not df.empty
        else 0.0
    )
    
    # Generate PDF report
    pdf_bytes = build_pdf_report(df, prof, seat_chance=seat_chance)
    pdf_path = output_dir / pdf_filename
    pdf_path.write_bytes(pdf_bytes)
    
    # Confirm PDF opens and is non-empty
    file_size = pdf_path.stat().st_size
    assert file_size > 0, f"PDF file {pdf_filename} is empty!"
    reader = pypdf.PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    assert num_pages > 0, f"PDF file {pdf_filename} has 0 pages!"
    print(f"PDF Saved: {pdf_path} ({file_size:,} bytes, {num_pages} pages) - VALID & NON-EMPTY")

print("\n" + "=" * 70)
print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
