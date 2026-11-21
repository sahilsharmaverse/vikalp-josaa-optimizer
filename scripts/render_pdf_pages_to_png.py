"""Script to render Page 1 and the Last Page of the generated Vikalp PDF report to PNG.

Uses PyMuPDF (fitz) to render high-resolution PNG images for visual inspection.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pymupdf
import pandas as pd

from src.eligibility import StudentProfile
from src.optimize import build_list
from src.pdf_report import build_pdf_report
from src.simulation import simulate_at_least_one_seat_probability


def main():
    print("Generating choice list and PDF report...")
    student = StudentProfile(
        main_rank=15000,
        adv_rank=0,
        category="OPEN",
        category_rank=0,
        gender="Gender-Neutral",
        home_state="Maharashtra",
        list_size=50,
        priority_mode="College-first",
    )

    choice_df = build_list(student)
    cutoffs_df = pd.read_parquet("data/processed/cutoffs.parquet")

    # Compute portfolio seat chance
    seat_chance = simulate_at_least_one_seat_probability(
        student_ranks=choice_df["student_rank"].values,
        y_hats=choice_df["y_hat"].values,
        n_sims=2000,
        seed=42,
    )

    # Generate PDF in memory
    pdf_bytes = build_pdf_report(
        choice_df=choice_df,
        student_profile=student,
        seat_chance=seat_chance,
        rank_shift_pct=0,
        active_rank=15000,
        history_df=cutoffs_df,
    )

    print(f"Generated PDF size: {len(pdf_bytes):,} bytes")

    # Open PDF with PyMuPDF
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    print(f"Total PDF pages: {num_pages}")

    out_dir = Path("docs/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render Page 1 (index 0) at 2x resolution (144 dpi) for crisp viewing
    page1 = doc[0]
    pix1 = page1.get_pixmap(dpi=150)
    p1_path = out_dir / "feature5_pdf_page1.png"
    pix1.save(str(p1_path))
    print(f"Saved Page 1 to: {p1_path}")

    # Render Last Page (index num_pages - 1)
    last_page = doc[num_pages - 1]
    pix_last = last_page.get_pixmap(dpi=150)
    last_path = out_dir / "feature5_pdf_last_page.png"
    pix_last.save(str(last_path))
    print(f"Saved Last Page to: {last_path}")

    # Also render Page 2 (showing choice table continuation with repeated headers and dagger if any)
    if num_pages > 2:
        page2 = doc[1]
        pix2 = page2.get_pixmap(dpi=150)
        p2_path = out_dir / "feature5_pdf_page2_table.png"
        pix2.save(str(p2_path))
        print(f"Saved Page 2 to: {p2_path}")

    doc.close()
    print("PDF rendering completed successfully!")


if __name__ == "__main__":
    main()
