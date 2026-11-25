"""Phase 4: Robustness, Fuzz, Data Integrity, Property, and Ground Truth Tests.

Covers:
1. Fuzz testing: Ranks 1 to 1,000,000, all reservation categories, gender pools, single-exam
   and dual-exam profiles, and rejection of invalid/negative inputs with actionable messages.
2. Data validation on load: Schema check, null checks, duplicate seat offerings, rank positivity,
   and rank monotonicity.
3. Property-based invariants: Monotonicity of admission probability with respect to rank,
   and strict adherence to list size upper bounds.
4. Ground truth tests: Verification against hand-verified official 2025 JoSAA Round 6 cutoffs
   (IIT Bombay, NIT Trichy, NIT Warangal).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.eligibility import StudentProfile, filter_eligible_options
from src.optimize import build_list
from src.predict import admit_probability
from src.validate import DataValidationError, validate_cutoff_dataset


@pytest.fixture
def cutoffs_data():
    """Load cached processed cutoffs dataset."""
    path = Path("data/processed/cutoffs.parquet")
    assert path.exists(), "cutoffs.parquet must exist for robustness tests"
    return pd.read_parquet(path)


# =========================================================================
# 1. FUZZ TESTING & INPUT VALIDATION
# =========================================================================

@pytest.mark.parametrize("rank", [1, 25, 250, 2500, 15000, 45000, 120000, 500000, 1000000])
@pytest.mark.parametrize("category", ["OPEN", "OBC-NCL", "SC", "ST"])
@pytest.mark.parametrize("gender", ["Gender-Neutral", "Female-only"])
def test_fuzz_valid_profiles_no_crashes(rank: int, category: str, gender: str):
    """Assert optimizer handles extreme and varied ranks without crashes and within bounds."""
    cat_rank = max(1, int(round(rank * 0.25))) if category != "OPEN" else None
    student = StudentProfile(
        main_rank=rank,
        adv_rank=rank if rank <= 30000 else None,
        category=category,
        category_rank=cat_rank,
        gender=gender,
        home_state="Maharashtra",
        list_size=30,
    )
    df = build_list(student)

    # Output must be a DataFrame
    assert isinstance(df, pd.DataFrame)

    if not df.empty:
        # All probabilities must be mathematically valid and meet 5% list floor
        assert (df["probability"] >= 0.05).all()
        assert (df["probability"] <= 1.0).all()
        # Risk labels must be valid
        assert set(df["label"]).issubset({"Dream", "Target", "Safe"})
        # No duplicate program choices for the student
        assert not df.duplicated(subset=["institute", "program"]).any()


@pytest.mark.parametrize("bad_main, bad_adv", [
    (None, None),
    (0, 0),
    (-50, None),
    (None, -100),
    (-10, -20),
    (0, None),
])
def test_invalid_ranks_raise_actionable_error(bad_main, bad_adv):
    """Invalid, zero, or completely missing ranks must raise descriptive ValueError."""
    student = StudentProfile(
        main_rank=bad_main,
        adv_rank=bad_adv,
        category="OPEN",
        gender="Gender-Neutral",
    )
    with pytest.raises(ValueError) as excinfo:
        build_list(student)
    assert "rank" in str(excinfo.value).lower()


def test_invalid_category_rank_and_list_size():
    """Negative category rank or non-positive list size must raise ValueError."""
    with pytest.raises(ValueError, match="category_rank"):
        student = StudentProfile(
            main_rank=5000,
            category="OBC-NCL",
            category_rank=-10,
        )
        student.validate()

    with pytest.raises(ValueError, match="list_size"):
        student = StudentProfile(
            main_rank=5000,
            list_size=0,
        )
        student.validate()


# =========================================================================
# 2. DATA VALIDATION ON LOAD
# =========================================================================

def test_cutoff_dataset_validation_passes_on_production(cutoffs_data):
    """The production dataset must satisfy all schema, null, uniqueness, and sanity checks."""
    report = validate_cutoff_dataset(cutoffs_data, strict_monotonicity=False)
    assert report["total_rows"] == len(cutoffs_data)
    assert report["total_rows"] > 40000
    assert 2025 in report["cycles"]
    assert report["non_monotonic_count"] == 0


def test_validation_fails_on_empty_df():
    """Empty dataframe must raise DataValidationError."""
    with pytest.raises(DataValidationError, match="empty"):
        validate_cutoff_dataset(pd.DataFrame())


def test_validation_fails_on_missing_column(cutoffs_data):
    """Missing a critical column must raise DataValidationError."""
    corrupted = cutoffs_data.drop(columns=["closing_rank"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_cutoff_dataset(corrupted)


def test_validation_fails_on_null_values(cutoffs_data):
    """Null values in critical columns must raise DataValidationError."""
    corrupted = cutoffs_data.head(50).copy()
    corrupted.loc[0, "closing_rank"] = None
    with pytest.raises(DataValidationError, match="unexpected nulls"):
        validate_cutoff_dataset(corrupted)


def test_validation_fails_on_duplicate_offerings(cutoffs_data):
    """Duplicate offering entries for the same year/pool must raise DataValidationError."""
    corrupted = pd.concat([cutoffs_data.head(5), cutoffs_data.head(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate seat offerings"):
        validate_cutoff_dataset(corrupted)


def test_validation_fails_on_non_positive_closing_rank(cutoffs_data):
    """Closing rank <= 0 must raise DataValidationError."""
    corrupted = cutoffs_data.head(50).copy()
    corrupted.loc[0, "closing_rank"] = 0
    with pytest.raises(DataValidationError, match="closing_rank <= 0"):
        validate_cutoff_dataset(corrupted)


def test_validation_monotonicity_strict_mode(cutoffs_data):
    """Strict monotonicity mode must detect and reject inverted ranks (opening > closing)."""
    corrupted = cutoffs_data.head(50).copy()
    corrupted.loc[0, "opening_rank"] = 5000
    corrupted.loc[0, "closing_rank"] = 1000
    with pytest.raises(DataValidationError, match="opening_rank > closing_rank"):
        validate_cutoff_dataset(corrupted, strict_monotonicity=True)


# =========================================================================
# 3. PROPERTY-BASED INVARIANTS
# =========================================================================

def test_property_monotonic_rank_probability():
    """Property: A strictly better (smaller) rank must yield >= probability for every option."""
    y_hat = float(np.log(10000.0))
    sigma = 0.2164

    ranks = [50000, 30000, 15000, 10000, 8000, 5000, 2000, 500, 50]
    probs = [admit_probability(r, y_hat, sigma) for r in ranks]

    # Verify monotonic non-decreasing order as rank improves
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1] + 1e-9, f"Monotonicity violated: {probs[i]} > {probs[i+1]}"


@pytest.mark.parametrize("requested_size", [5, 10, 20, 35, 50, 75])
def test_property_list_size_bound(requested_size: int):
    """Property: Returned choice list length must never exceed requested list_size."""
    student = StudentProfile(
        main_rank=18000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        list_size=requested_size,
    )
    df = build_list(student)
    assert len(df) <= requested_size


def test_property_strategic_tier_ordering():
    """Property: In strategic mode, choices must strictly follow Dream -> Target -> Safe."""
    student = StudentProfile(
        main_rank=12000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Uttar Pradesh",
        list_size=40,
        ordering_strategy="strategic",
    )
    df = build_list(student)
    assert not df.empty

    labels = list(df["label"])
    dream_idxs = [i for i, l in enumerate(labels) if l == "Dream"]
    target_idxs = [i for i, l in enumerate(labels) if l == "Target"]
    safe_idxs = [i for i, l in enumerate(labels) if l == "Safe"]

    if dream_idxs and target_idxs:
        assert max(dream_idxs) < min(target_idxs)
    if target_idxs and safe_idxs:
        assert max(target_idxs) < min(safe_idxs)


# =========================================================================
# 4. GROUND TRUTH TESTS (OFFICIAL 2025 JOSAA ROUND 6 TABLES)
# =========================================================================

def test_ground_truth_official_2025_cutoffs(cutoffs_data):
    """Verify official hand-verified ground truth cutoffs from official JoSAA 2025 Round 6 data."""
    df25 = cutoffs_data[cutoffs_data["year"] == 2025]

    # Case 1: IIT Bombay - Computer Science and Engineering (AI, OPEN, Gender-Neutral)
    iitb_cse = df25[
        (df25["institute"] == "Indian Institute of Technology Bombay")
        & (df25["program"] == "Computer Science and Engineering")
        & (df25["quota"] == "AI")
        & (df25["seat_type"] == "OPEN")
        & (df25["gender"] == "Gender-Neutral")
    ]
    assert len(iitb_cse) == 1, "Expected exactly 1 official seat pool for IITB CSE OPEN AI"
    row_iitb = iitb_cse.iloc[0]
    assert row_iitb["opening_rank"] == 1
    assert row_iitb["closing_rank"] == 66

    # Case 2: NIT Trichy - Computer Science and Engineering (OS, OPEN, Gender-Neutral)
    nitt_cse_os = df25[
        (df25["institute"] == "National Institute of Technology, Tiruchirappalli")
        & (df25["program"] == "Computer Science and Engineering")
        & (df25["quota"] == "OS")
        & (df25["seat_type"] == "OPEN")
        & (df25["gender"] == "Gender-Neutral")
    ]
    assert len(nitt_cse_os) == 1
    row_nitt_os = nitt_cse_os.iloc[0]
    assert row_nitt_os["opening_rank"] == 659
    assert row_nitt_os["closing_rank"] == 1449

    # Case 3: NIT Trichy - Computer Science and Engineering (HS, OPEN, Gender-Neutral)
    nitt_cse_hs = df25[
        (df25["institute"] == "National Institute of Technology, Tiruchirappalli")
        & (df25["program"] == "Computer Science and Engineering")
        & (df25["quota"] == "HS")
        & (df25["seat_type"] == "OPEN")
        & (df25["gender"] == "Gender-Neutral")
    ]
    assert len(nitt_cse_hs) == 1
    row_nitt_hs = nitt_cse_hs.iloc[0]
    assert row_nitt_hs["opening_rank"] == 559
    assert row_nitt_hs["closing_rank"] == 4569

    # Case 4: NIT Warangal - Electronics and Communication Engineering (OS, OPEN, Female-only)
    nitw_ece_female = df25[
        (df25["institute"] == "National Institute of Technology, Warangal")
        & (df25["program"] == "Electronics and Communication Engineering")
        & (df25["quota"] == "OS")
        & (df25["seat_type"] == "OPEN")
        & (df25["gender"] == "Female-only")
    ]
    assert len(nitw_ece_female) == 1
    row_nitw_f = nitw_ece_female.iloc[0]
    assert row_nitw_f["opening_rank"] == 6422
    assert row_nitw_f["closing_rank"] == 8861


def test_ground_truth_optimizer_behavior():
    """Ground-truth optimizer check: Top rank yields high probability for premier target."""
    # A candidate with JEE Advanced CRL 45 should have high probability for IIT Bombay CSE
    student_iitb = StudentProfile(
        adv_rank=45,
        category="OPEN",
        gender="Gender-Neutral",
        institute_types=["IIT"],
        preferred_branches=["CSE"],
        list_size=10,
    )
    df_iitb = build_list(student_iitb)
    iitb_match = df_iitb[
        (df_iitb["institute"] == "Indian Institute of Technology Bombay")
        & (df_iitb["program"].str.contains("Computer Science"))
    ]
    assert not iitb_match.empty, "IIT Bombay CSE should be recommended for CRL 45"
    assert iitb_match.iloc[0]["probability"] >= 0.60

    # A candidate with JEE Main CRL 800 (OS) should have high probability for NIT Trichy CSE
    student_nitt = StudentProfile(
        main_rank=800,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Delhi",  # OS for Tamil Nadu
        institute_types=["NIT"],
        preferred_branches=["CSE"],
        list_size=10,
    )
    df_nitt = build_list(student_nitt)
    nitt_match = df_nitt[
        (df_nitt["institute"].str.contains("Tiruchirappalli"))
        & (df_nitt["program"].str.contains("Computer Science"))
    ]
    assert not nitt_match.empty, "NIT Trichy CSE should be recommended for CRL 800"
    assert nitt_match.iloc[0]["probability"] >= 0.80
