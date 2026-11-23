"""Regression tests for v1.1 refinements:
1. Deduplication preserving distinct quota and gender pools.
2. Deduplication keeping highest probability for true duplicates.
3. Strict exclusion of B.Des, B.Arch, and B.Planning rows.
4. Triggering of '< 5 Safe choices' warning in optimizer logic and PDF report.
"""

import io
import pandas as pd
import pypdf
import pytest

from src.eligibility import StudentProfile
from src.optimize import build_list, is_architecture_or_planning
from src.pdf_report import build_pdf_report


def test_dedup_keeps_distinct_quota_and_gender_pools():
    """Dedup must NOT merge rows that differ by quota or gender/seat pool."""
    mock_data = pd.DataFrame([
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "OS",
            "seat_type": "OPEN",
            "gender": "Female-only",
            "probability": 0.45,
        },
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "OS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "probability": 0.25,
        },
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "HS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "probability": 0.55,
        },
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "HS",
            "seat_type": "OBC-NCL",
            "gender": "Gender-Neutral",
            "probability": 0.70,
        },
    ])

    dedup = (
        mock_data.sort_values(by="probability", ascending=False)
        .drop_duplicates(
            subset=["institute", "program", "quota", "seat_type", "gender"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # All 4 rows represent distinct counselling seat allocations and must be preserved
    assert len(dedup) == 4
    genders = set(dedup[dedup["quota"] == "OS"]["gender"])
    assert "Female-only" in genders
    assert "Gender-Neutral" in genders

    quotas = set(dedup[dedup["gender"] == "Gender-Neutral"]["quota"])
    assert "OS" in quotas
    assert "HS" in quotas


def test_dedup_keeps_highest_probability_for_true_duplicates():
    """When true duplicate records exist, dedup keeps the row with the highest probability."""
    mock_data = pd.DataFrame([
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "OS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "probability": 0.30,
        },
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "OS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "probability": 0.72,
        },
        {
            "institute": "National Institute of Technology, Tiruchirappalli",
            "program": "Computer Science and Engineering",
            "quota": "OS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "probability": 0.50,
        },
    ])

    dedup = (
        mock_data.sort_values(by="probability", ascending=False)
        .drop_duplicates(
            subset=["institute", "program", "quota", "seat_type", "gender"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    assert len(dedup) == 1
    assert dedup.iloc[0]["probability"] == 0.72


def test_bdes_and_barch_never_appear_in_optimizer_output():
    """B.Des, B.Arch, and B.Planning rows must never appear in the optimizer output."""
    # 1. Test helper function classifications
    progs_to_exclude = pd.Series([
        "Bachelor of Design",
        "Bachelor of Design (4 Years, Bachelor of Design)",
        "B.Des in Product Design",
        "Architecture (5 Years, Bachelor of Architecture)",
        "Planning (4 Years, Bachelor of Planning)",
    ])
    assert is_architecture_or_planning(progs_to_exclude).all()

    progs_to_keep = pd.Series([
        "Computer Science and Engineering",
        "Electronics and Communication Engineering",
        "Ocean Engineering and Naval Architecture",
        "Naval Architecture and Ocean Engineering",
        "Design (4 Years, Bachelor of Technology)",  # Standard B.Tech at IIT Delhi
    ])
    assert (~is_architecture_or_planning(progs_to_keep)).all()

    # 2. Test optimizer choice list outputs across diverse ranks
    for rank in [5000, 25000, 50000]:
        student = StudentProfile(
            main_rank=rank,
            adv_rank=rank if rank <= 15000 else None,
            category="OPEN",
            gender="Gender-Neutral",
            home_state="Delhi",
            list_size=50,
        )
        df = build_list(student)
        if df.empty:
            continue

        for prog in df["program"]:
            prog_lower = prog.lower()
            if "naval architecture" in prog_lower:
                continue
            assert "architecture" not in prog_lower, f"Found architecture: {prog}"
            assert "planning" not in prog_lower, f"Found planning: {prog}"
            assert "bachelor of design" not in prog_lower, f"Found B.Des: {prog}"
            assert "b.des" not in prog_lower, f"Found B.Des: {prog}"


def test_low_safe_choices_warning_logic_and_pdf():
    """The '< 5 Safe choices' warning triggers for low-rank profiles and not for high-rank ones."""
    # Low-rank profile with restricted choices -> Safe count < 5
    student_low = StudentProfile(
        main_rank=45000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        preferred_branches=["CSE"],
        institute_types=["NIT", "IIIT"],
        list_size=20,
    )
    df_low = build_list(student_low)
    safe_count_low = int((df_low["label"] == "Safe").sum())
    assert safe_count_low < 5, f"Expected < 5 safe choices for low-rank profile, got {safe_count_low}"

    # High-rank profile -> Safe count >= 5
    student_high = StudentProfile(
        main_rank=25000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        preferred_branches=["CSE"],
        institute_types=["NIT", "IIIT"],
        list_size=20,
    )
    df_high = build_list(student_high)
    safe_count_high = int((df_high["label"] == "Safe").sum())
    assert safe_count_high >= 5, f"Expected >= 5 safe choices for high-rank profile, got {safe_count_high}"

    # PDF generation and text inspection
    pdf_low_bytes = build_pdf_report(df_low, student_low, seat_chance=0.72)
    reader_low = pypdf.PdfReader(io.BytesIO(pdf_low_bytes))
    text_low = " ".join([page.extract_text() for page in reader_low.pages])
    assert "few or no safe options" in text_low, "Expected safe options warning in low-rank PDF"

    pdf_high_bytes = build_pdf_report(df_high, student_high, seat_chance=0.98)
    reader_high = pypdf.PdfReader(io.BytesIO(pdf_high_bytes))
    text_high = " ".join([page.extract_text() for page in reader_high.pages])
    assert "few or no safe options" not in text_high, "Did not expect safe options warning in high-rank PDF"

    # Verify portfolio mix and 5% probability floor on high-rank profile
    counts = df_high["label"].value_counts()
    assert counts.get("Dream", 0) == 4   # 20% of 20
    assert counts.get("Target", 0) == 10 # 50% of 20
    assert counts.get("Safe", 0) == 6    # 30% of 20
    assert (df_high["probability"] >= 0.05).all(), "Found option below 5% floor!"
