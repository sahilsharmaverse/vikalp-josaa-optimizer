"""Test suite for edge cases in Vikalp:
1. Empty rank / 0 rank
2. Negative rank
3. Rank > 1,000,000
4. No branch selected (empty preferred_branches list)
5. Filters that match nothing (empty result DataFrame)
"""

import pytest
import pandas as pd
from src.eligibility import StudentProfile, filter_eligible_options
from src.optimize import build_list
from src.predict import admit_probability

def test_empty_and_zero_rank():
    # Both main and adv rank are None or 0
    student = StudentProfile(main_rank=0, adv_rank=0)
    # filter_eligible_options should return empty DataFrame because no valid exam rank is provided
    cutoffs = pd.read_parquet("data/processed/cutoffs.parquet")
    from src.optimize import prepare_latest_options
    active = prepare_latest_options(cutoffs)
    eligible = filter_eligible_options(active, student)
    assert eligible.empty, "Zero rank should result in 0 eligible options"

def test_negative_rank_handling():
    # Negative rank should raise ValueError or be caught by validation
    student = StudentProfile(main_rank=-500, adv_rank=-100)
    cutoffs = pd.read_parquet("data/processed/cutoffs.parquet")
    from src.optimize import prepare_latest_options
    active = prepare_latest_options(cutoffs)
    eligible = filter_eligible_options(active, student)
    assert eligible.empty, "Negative rank should result in 0 eligible options"

    with pytest.raises(ValueError):
        admit_probability(-500, 10.0, 0.2285)

def test_rank_above_one_million():
    # Student with rank 1,500,000
    student = StudentProfile(
        main_rank=1500000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",
        list_size=20,
    )
    df = build_list(student)
    # With MIN_LIST_PROB = 0.05, rank 1.5M correctly has 0 options meeting the 5% threshold without crashing
    assert df.empty or (df["probability"] >= 0.05).all()

def test_no_branch_selected():
    # Empty list for preferred_branches should default to all branches
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Karnataka",
        preferred_branches=[],
        list_size=30,
    )
    df = build_list(student)
    assert not df.empty
    branches_present = set(df["branch_group"])
    assert len(branches_present) > 1, f"Expected multiple branch groups, got: {branches_present}"

def test_filters_match_nothing():
    # Institute types filter with an empty or impossible filter
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Karnataka",
        institute_types=["NON_EXISTENT_TYPE"],
        list_size=30,
    )
    df = build_list(student)
    assert df.empty, "Should return empty dataframe when filters match nothing"

if __name__ == "__main__":
    test_empty_and_zero_rank()
    test_negative_rank_handling()
    test_rank_above_one_million()
    test_no_branch_selected()
    test_filters_match_nothing()
    print("ALL EDGE CASES PASSED SUCCESSFULLY!")
