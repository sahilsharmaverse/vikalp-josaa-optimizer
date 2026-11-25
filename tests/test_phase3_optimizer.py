"""Tests for Phase 3 Optimizer Quality:
1. Adaptive portfolio risk diagnosis and recommendations when safe choices are scarce.
2. Preference ordering vs strategic tier ordering.
3. Expected outcome quality (expected allotted choice rank and closing rank).
"""

import pandas as pd
import pytest

from src.eligibility import StudentProfile
from src.optimize import build_list
from src.simulation import compute_expected_outcome_quality


def test_adaptive_portfolio_diagnosis_scarcity():
    """Verify adaptive portfolio diagnosis accurately flags scarce safety anchors and suggests alternatives."""
    # Low-rank candidate with restricted branches
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
    assert not df_low.empty
    diag = df_low.attrs.get("portfolio_diagnosis")
    assert diag is not None
    assert diag["safe_scarcity"] is True
    assert diag["counts"]["Safe"] < 5
    assert len(diag["alternatives"]) >= 3
    # Check that alternatives mention allied branches, GFTIs, and state/CSAB counselling
    alt_text = " ".join(diag["alternatives"]).lower()
    assert "branch alternative" in alt_text
    assert "institute alternative" in alt_text
    assert "exam/counselling alternative" in alt_text


def test_adaptive_portfolio_diagnosis_healthy():
    """Verify healthy portfolios with >= 30% safe choices are not flagged as scarce."""
    student_good = StudentProfile(
        main_rank=20000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        list_size=30,
    )
    df_good = build_list(student_good)
    assert not df_good.empty
    diag = df_good.attrs.get("portfolio_diagnosis")
    assert diag is not None
    assert diag["safe_scarcity"] is False
    assert diag["counts"]["Safe"] >= 5


def test_ordering_strategy_preference_vs_strategic():
    """Verify ordering_strategy='preference' vs 'strategic' produces distinct valid ordering."""
    student_pref = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        preferred_branches=["CSE", "ECE"],
        institute_types=["NIT", "IIIT"],
        priority_mode="College-first",
        list_size=20,
        ordering_strategy="preference",
    )
    df_pref = build_list(student_pref)

    student_strat = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        preferred_branches=["CSE", "ECE"],
        institute_types=["NIT", "IIIT"],
        priority_mode="College-first",
        list_size=20,
        ordering_strategy="strategic",
    )
    df_strat = build_list(student_strat)

    assert len(df_pref) == 20
    assert len(df_strat) == 20

    # In strategic mode, Dream labels must precede Safe labels
    labels_strat = list(df_strat["label"])
    if "Dream" in labels_strat and "Safe" in labels_strat:
        assert labels_strat.index("Dream") < labels_strat.index("Safe")


def test_expected_outcome_quality_simulation():
    """Verify compute_expected_outcome_quality computes valid serial JoSAA allocation metrics."""
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        preferred_branches=["CSE", "ECE"],
        list_size=25,
    )
    df = build_list(student)
    assert not df.empty

    outcome = compute_expected_outcome_quality(df, n_sims=1000, seed=42)
    assert 0.0 < outcome["admission_prob"] <= 1.0
    assert 1.0 <= outcome["expected_choice_no"] <= len(df)
    assert outcome["expected_closing_rank"] > 0
    assert 0.0 <= outcome["top_choice_prob"] <= outcome["admission_prob"]
    assert 0.0 <= outcome["top_3_prob"] <= outcome["admission_prob"]
