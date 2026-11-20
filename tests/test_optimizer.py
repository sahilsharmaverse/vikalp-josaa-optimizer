"""Unit tests for eligibility filtering and choice list optimizer."""

import pandas as pd
import pytest

from src.eligibility import (
    StudentProfile,
    filter_eligible_options,
    is_gender_eligible,
    is_quota_eligible,
)
from src.optimize import build_list


def test_quota_eligibility():
    """Verify AI, HS, and OS quota logic."""
    # AI is always eligible
    assert is_quota_eligible("AI", "Maharashtra", "Delhi") is True
    assert is_quota_eligible("AI", "Delhi", "Delhi") is True

    # HS is eligible only when states match
    assert is_quota_eligible("HS", "Maharashtra", "Maharashtra") is True
    assert is_quota_eligible("HS", "Maharashtra", "Delhi") is False

    # OS is eligible only when states differ
    assert is_quota_eligible("OS", "Maharashtra", "Delhi") is True
    assert is_quota_eligible("OS", "Maharashtra", "Maharashtra") is False


def test_gender_eligibility():
    """Verify female students qualify for both gender pools, male only for gender-neutral."""
    # Female student
    assert is_gender_eligible("Gender-Neutral", "Female-only") is True
    assert is_gender_eligible("Female-only", "Female-only") is True
    assert is_gender_eligible("Female-only", "Female") is True

    # Male / Gender-Neutral student
    assert is_gender_eligible("Gender-Neutral", "Gender-Neutral") is True
    assert is_gender_eligible("Female-only", "Gender-Neutral") is False


def test_build_list_risk_mix_and_order():
    """Verify build_list produces target risk mix and proper Dream -> Target -> Safe order."""
    student = StudentProfile(
        main_rank=8000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        preferred_branches=["CSE", "ECE"],
        institute_types=["NIT", "IIIT"],
        priority_mode="College-first",
        list_size=20,
    )

    df_list = build_list(student)
    assert len(df_list) == 20

    labels = list(df_list["label"])
    # Dream options should appear before Target, and Target before Safe
    dream_indices = [i for i, l in enumerate(labels) if l == "Dream"]
    target_indices = [i for i, l in enumerate(labels) if l == "Target"]
    safe_indices = [i for i, l in enumerate(labels) if l == "Safe"]

    if dream_indices and target_indices:
        assert max(dream_indices) < min(target_indices)
    if target_indices and safe_indices:
        assert max(target_indices) < min(safe_indices)

    counts = df_list["label"].value_counts()
    assert counts.get("Dream", 0) == 4
    assert counts.get("Target", 0) == 10
    assert counts.get("Safe", 0) == 6


def test_unreachable_warning_flag():
    """Choices placed after an option with P >= 0.95 should carry an unreachable warning."""
    student = StudentProfile(
        main_rank=3000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        institute_types=["NIT", "IIIT", "GFTI"],
        list_size=30,
    )

    df_list = build_list(student)
    probs = df_list["probability"].values
    warnings = df_list["warning"].values

    has_certain = False
    for p, w in zip(probs, warnings):
        if has_certain:
            assert "Unreachable" in w
        else:
            assert w == ""
            if p >= 0.95:
                has_certain = True
