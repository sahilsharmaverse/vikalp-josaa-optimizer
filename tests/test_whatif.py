"""Tests for What-If Rank Sensitivity Simulation."""

import numpy as np
import pytest

from src.eligibility import StudentProfile
from src.optimize import build_list
from src.whatif import shift_rank, shift_student_profile, rescore_choice_list


def test_rank_one_stays_one_with_negative_shift():
    """A rank of 1 with -20% stays at 1."""
    assert shift_rank(1, -20.0) == 1
    assert shift_rank(1, -10.0) == 1
    assert shift_rank(1, 0.0) == 1
    assert shift_rank(1, -50.0) == 1


def test_zero_shift_equals_unshifted_result():
    """At 0% the result equals the unshifted result exactly."""
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        preferred_branches=["CSE", "ECE"],
        list_size=20,
    )
    df = build_list(student)
    assert not df.empty

    rescored_df, p_at_least_one = rescore_choice_list(df, pct=0.0, seed=42)

    # Individual probabilities and labels must match exactly
    np.testing.assert_allclose(
        rescored_df["shifted_probability"].values,
        df["probability"].values,
        rtol=1e-5,
    )
    assert (rescored_df["shifted_label"] == df["label"]).all()
    assert (rescored_df["shifted_rank"] == df["student_rank"]).all()
    # No label changes at 0%
    assert (rescored_df["tier_change"] == "").all()


def test_monotonicity_of_rank_shifts():
    """A worse rank (higher number) never increases any option's probability or seat admission chance.

    A better rank never decreases them.
    """
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        preferred_branches=["CSE", "ECE"],
        list_size=20,
    )
    df = build_list(student)

    # Evaluate at -20% (better rank), 0% (base rank), and +20% (worse rank)
    df_better, p_better = rescore_choice_list(df, pct=-20.0, seed=42)
    df_base, p_base = rescore_choice_list(df, pct=0.0, seed=42)
    df_worse, p_worse = rescore_choice_list(df, pct=20.0, seed=42)

    # Better rank gives higher or equal option probabilities
    assert (df_better["shifted_probability"].values >= df_base["shifted_probability"].values - 1e-6).all()
    # Worse rank gives lower or equal option probabilities
    assert (df_worse["shifted_probability"].values <= df_base["shifted_probability"].values + 1e-6).all()

    # Cumulative seat admission chance monotonicity
    assert p_better >= p_base - 1e-4
    assert p_worse <= p_base + 1e-4


def test_shift_student_profile():
    """All student ranks (main, adv, category) shift consistently."""
    student = StudentProfile(
        main_rank=10000,
        adv_rank=5000,
        category="OBC-NCL",
        category_rank=2000,
    )
    shifted = shift_student_profile(student, pct=10.0)
    assert shifted.main_rank == 11000
    assert shifted.adv_rank == 5500
    assert shifted.category_rank == 2200

    better = shift_student_profile(student, pct=-20.0)
    assert better.main_rank == 8000
    assert better.adv_rank == 4000
    assert better.category_rank == 1600


def test_no_reordering_during_rescore():
    """Slider alone must NOT reorder the original list."""
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        preferred_branches=["CSE", "ECE"],
        list_size=25,
    )
    df = build_list(student)

    for pct in [-20.0, -10.0, 5.0, 15.0, 20.0]:
        rescored, _ = rescore_choice_list(df, pct=pct, seed=42)
        # Choice order and institute-program sequence must remain identical
        assert (rescored["choice_no"] == df["choice_no"]).all()
        assert (rescored["institute"] == df["institute"]).all()
        assert (rescored["program"] == df["program"]).all()
