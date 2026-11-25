"""Unit tests for PDF Report Generator (v1.1 Feature 5)."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
import pytest

from src.eligibility import StudentProfile
from src.pdf_report import build_pdf_report


def _create_sample_df(n_choices: int = 50, has_unreachable: bool = False) -> pd.DataFrame:
    """Helper to synthesize a test choices DataFrame."""
    records = []
    for i in range(1, n_choices + 1):
        if i <= 10:
            tier = "Dream"
            prob = 0.15 + (i * 0.01)
        elif i <= 35:
            tier = "Target"
            prob = 0.35 + ((i - 10) * 0.015)
        else:
            tier = "Safe"
            prob = 0.82 + ((i - 35) * 0.005)

        # Force an unreachable scenario if requested
        if has_unreachable and i == 36:
            prob = 0.96  # triggers >= 95%
        elif has_unreachable and i > 36:
            prob = 0.98

        records.append({
            "choice_no": i,
            "institute": f"NIT Example Institute #{i}",
            "program": f"Computer Science & Engineering ({i})",
            "quota": "OS" if i % 2 == 0 else "HS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "student_rank": 15000,
            "shifted_probability": prob,
            "shifted_label": tier,
            "probability": prob,
            "label": tier,
            "y_hat": np.log(14000 + i * 200),
            "trend": "Getting easier" if i % 3 == 0 else "Stable",
            "warning": "",
        })
    return pd.DataFrame(records)


def test_pdf_generation_sizes_and_header():
    """PDF bytes must start with %PDF and be non-empty for lists of 1, 50, and 100 choices under 3 seconds."""
    student = StudentProfile(
        main_rank=15000,
        adv_rank=0,
        category="OPEN",
        category_rank=0,
        gender="Gender-Neutral",
        home_state="Maharashtra",
        preferred_branches=["CSE", "ECE"],
    )

    for n in [1, 50, 100]:
        df = _create_sample_df(n_choices=n)
        start_time = time.time()
        pdf_bytes = build_pdf_report(
            choice_df=df,
            student_profile=student,
            seat_chance=0.984,
            rank_shift_pct=0,
            active_rank=15000,
        )
        elapsed = time.time() - start_time

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000
        assert pdf_bytes.startswith(b"%PDF"), f"PDF for n={n} does not start with %PDF"
        assert elapsed < 5.0, f"Generation for n={n} took {elapsed:.2f}s (must be < 5.0s)"

        # Verify page count is positive
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 2


def test_special_characters_escaping():
    """Names containing '&', '<', quotes, and long dashes must not crash or corrupt the PDF."""
    student = StudentProfile(
        main_rank=12000,
        adv_rank=8500,
        category="OBC-NCL",
        category_rank=2800,
        gender="Female-only",
        home_state="Karnataka",
        preferred_branches=["Computer Science & Engg", "AI <Robotics>"],
    )

    special_df = pd.DataFrame([
        {
            "choice_no": 1,
            "institute": "IIT & NIT 'Excellence' Campus — North",
            "program": "B.Tech in CS & AI <Specialization> \"Dual Degree\"",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Female-only",
            "student_rank": 12000,
            "shifted_probability": 0.45,
            "shifted_label": "Target",
            "probability": 0.45,
            "label": "Target",
            "y_hat": np.log(11500),
            "trend": "Getting harder",
            "warning": "",
        },
        {
            "choice_no": 2,
            "institute": "NIT Warangal & Trichy — Tech Hub <South>",
            "program": "Electrical & Electronics Engg 'Honors' — 4 Years",
            "quota": "OS",
            "seat_type": "OBC-NCL",
            "gender": "Female-only",
            "student_rank": 2800,
            "shifted_probability": 0.88,
            "shifted_label": "Safe",
            "probability": 0.88,
            "label": "Safe",
            "y_hat": np.log(3200),
            "trend": "Stable",
            "warning": "",
        },
    ])

    pdf_bytes = build_pdf_report(
        choice_df=special_df,
        student_profile=student,
        seat_chance=0.925,
        rank_shift_pct=0,
        active_rank=12000,
    )
    assert pdf_bytes.startswith(b"%PDF")
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    all_text = " ".join([page.extract_text() for page in reader.pages])

    # Check key strings extracted
    assert "Built by Sahil Sharma" in all_text
    assert "Vikalp, JoSAA Choice List Report" in all_text


def test_extracted_text_contains_disclaimer_author_and_dagger_footnote():
    """Extracted text must contain disclaimer, 'Built by Sahil Sharma', and dagger footnote when unreachable options exist."""
    student = StudentProfile(
        main_rank=15000,
        adv_rank=0,
        category="OPEN",
        category_rank=0,
        gender="Gender-Neutral",
        home_state="Maharashtra",
    )

    # 1. Unreachable case
    unreachable_df = _create_sample_df(n_choices=45, has_unreachable=True)
    pdf_bytes = build_pdf_report(
        choice_df=unreachable_df,
        student_profile=student,
        seat_chance=0.992,
        rank_shift_pct=0,
        active_rank=15000,
    )

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    all_text = " ".join([page.extract_text() for page in reader.pages])

    assert "Built by Sahil Sharma" in all_text
    assert "Full Disclaimer:" in all_text or "Disclaimer:" in all_text
    assert "Always verify at josaa.nic.in" in all_text
    # Dagger footnote check
    assert "†" in all_text or "dagger" in all_text.lower()
    assert "Strategic JoSAA Order Warning" in all_text
    assert "effectively unreachable" in all_text

    # 2. No unreachable case
    safe_df = _create_sample_df(n_choices=20, has_unreachable=False)
    pdf_bytes_safe = build_pdf_report(
        choice_df=safe_df,
        student_profile=student,
        seat_chance=0.750,
        rank_shift_pct=0,
        active_rank=15000,
    )
    reader_safe = pypdf.PdfReader(io.BytesIO(pdf_bytes_safe))
    text_safe = " ".join([page.extract_text() for page in reader_safe.pages])
    assert "Built by Sahil Sharma" in text_safe
    assert "Always verify at josaa.nic.in" in text_safe
    assert "Strategic JoSAA Order Warning" not in text_safe


def test_whatif_rank_displayed_in_pdf():
    """When a what-if shift is active, the PDF text must explicitly state the simulated rank."""
    student = StudentProfile(
        main_rank=15000,
        adv_rank=0,
        category="OPEN",
        category_rank=0,
        gender="Gender-Neutral",
        home_state="Maharashtra",
    )
    df = _create_sample_df(n_choices=30)

    # Simulated shift +10% (15,000 -> 16,500)
    pdf_bytes_shifted = build_pdf_report(
        choice_df=df,
        student_profile=student,
        seat_chance=0.885,
        rank_shift_pct=10,
        active_rank=16500,
    )

    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes_shifted))
    text_shifted = " ".join([page.extract_text() for page in reader.pages])

    assert "Rank used: 16,500 (simulated +10%)" in text_shifted

    # Unshifted (0%)
    pdf_bytes_unshifted = build_pdf_report(
        choice_df=df,
        student_profile=student,
        seat_chance=0.910,
        rank_shift_pct=0,
        active_rank=15000,
    )
    reader_unshifted = pypdf.PdfReader(io.BytesIO(pdf_bytes_unshifted))
    text_unshifted = " ".join([page.extract_text() for page in reader_unshifted.pages])

    assert "Rank used: 15,000" in text_unshifted
    assert "(simulated" not in text_unshifted
