"""Comprehensive tests for pipeline, feature engineering, cleaning, and model evaluation modules.
Ensures total test coverage across src/ reaches >= 85%.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.baseline import compute_weighted_log_trend, evaluate_baseline_on_held_out
from src.clean import (
    clean_all_raw_files,
    clean_single_year_file,
    infer_institute_metadata,
    normalize_institute_name,
    normalize_program_name,
    parse_clean_rank,
)
from src.features import assign_branch_group, compute_lag_features
from src.scrape import (
    extract_table_dataframe,
    fetch_available_years,
    get_existing_final_round_file,
    select_year_and_get_final_round,
    trigger_cascade_and_submit,
)
from src.train import ALL_FEATURES, prepare_features, train_and_validate


# =========================================================================
# 1. CLEANING MODULE TESTS (src/clean.py)
# =========================================================================

def test_parse_clean_rank_all_cases():
    """Verify parse_clean_rank handles valid, invalid, P-suffix, negative, and null ranks."""
    assert parse_clean_rank("1500") == 1500
    assert parse_clean_rank(25000) == 25000
    assert parse_clean_rank("12,450") == 12450
    # Preparatory ranks with P-suffix
    assert parse_clean_rank("123P") is None
    assert parse_clean_rank("P45") is None
    assert parse_clean_rank("p10") is None
    # Nulls, empty strings, and non-numeric
    assert parse_clean_rank(None) is None
    assert parse_clean_rank(np.nan) is None
    assert parse_clean_rank("") is None
    assert parse_clean_rank("N/A") is None
    assert parse_clean_rank("-500") is None
    assert parse_clean_rank("0") is None


def test_normalize_institute_name():
    """Verify institute name canonicalization and mapping."""
    # Mapped variant
    raw_name = "Indian Institute of Information Technology(IIIT) Kottayam"
    assert normalize_institute_name(raw_name) == "Indian Institute of Information Technology Kottayam"
    # Whitespace cleanup
    assert normalize_institute_name("  Indian Institute of Technology Bombay   ") == "Indian Institute of Technology Bombay"
    # Unmapped name passes through stripped
    assert normalize_institute_name("Unknown Institute of Technology") == "Unknown Institute of Technology"


def test_infer_institute_metadata():
    """Verify rule-based inference of institute type and state."""
    # Exact known mappings
    iit_type, iit_state = infer_institute_metadata("Indian Institute of Technology Bombay")
    assert iit_type == "IIT"
    assert iit_state == "Maharashtra"

    nit_type, nit_state = infer_institute_metadata("National Institute of Technology, Tiruchirappalli")
    assert nit_type == "NIT"
    assert nit_state == "Tamil Nadu"

    # Inferred fallback types
    iiit_t, _ = infer_institute_metadata("Indian Institute of Information Technology Pune")
    assert iiit_t == "IIIT"

    gfti_t, gfti_s = infer_institute_metadata("Birla Institute of Technology, Mesra, Ranchi")
    assert gfti_t == "GFTI"

    # State matching in name
    _, delhi_s = infer_institute_metadata("Some College of Engineering, Delhi")
    assert delhi_s == "Delhi"


def test_normalize_program_name_degrees():
    """Verify degree duration stripping and canonical suffix assignment."""
    # B.Tech standard 4-year duration
    prog1 = "Computer Science and Engineering (4 Years, Bachelor of Technology)"
    assert normalize_program_name(prog1) == "Computer Science and Engineering"

    # Dual Degree
    prog2 = "Electrical Engineering and M.Tech in VLSI (Dual Degree)"
    assert normalize_program_name(prog2) == "Electrical Engineering and M.Tech in VLSI (Dual Degree)"

    # Integrated M.Tech
    prog3 = "Mathematics and Computing (Integrated Master of Technology)"
    assert normalize_program_name(prog3) == "Mathematics and Computing (Integrated M.Tech)"

    # Integrated M.Sc
    prog4 = "Chemistry (Integrated Master of Science)"
    assert normalize_program_name(prog4) == "Chemistry (Integrated M.Sc)"

    # BS-MS Dual
    prog5 = "Biological Sciences (BS-MS Dual Degree)"
    assert normalize_program_name(prog5) == "Biological Sciences (BS-MS Dual Degree)"

    # Bachelor of Science
    prog6 = "Bachelor of Science in Chemistry"
    assert "BS" in normalize_program_name(prog6)

    # B.Arch and B.Plan
    assert "B.Arch" in normalize_program_name("Bachelor of Architecture (5 Years)")
    assert "B.Plan" in normalize_program_name("Bachelor of Planning (4 Years)")

    # Non-string input
    assert normalize_program_name(None) == ""


def test_clean_single_year_file(tmp_path):
    """Test cleaning a mock raw JoSAA CSV file."""
    # Invalid filename pattern
    bad_file = tmp_path / "invalid_name.csv"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError, match="does not match expected pattern"):
        clean_single_year_file(bad_file)

    # Valid CSV file
    good_file = tmp_path / "2024_round5.csv"
    csv_content = (
        "Institute,Academic Program Name,Quota,Seat Type,Gender,Opening Rank,Closing Rank\n"
        "Indian Institute of Technology Bombay,Computer Science and Engineering (4 Years),AI,OPEN,Gender-Neutral,1,66\n"
        "Indian Institute of Technology Delhi,Mathematics and Computing,AI,OPEN,Gender-Neutral,100,350\n"
        "National Institute of Technology Tiruchirappalli,Civil Engineering,OS,OPEN,Gender-Neutral,5000,12000\n"
        "Some Institute,Bachelor of Architecture (5 Years),AI,OPEN,Gender-Neutral,10,50\n"  # Excluded B.Arch
        "Invalid Institute,Computer Science,AI,OPEN,Gender-Neutral,100P,200P\n"  # Dropped preparatory
    )
    good_file.write_text(csv_content, encoding="utf-8")

    cleaned_df = clean_single_year_file(good_file)
    assert not cleaned_df.empty
    assert len(cleaned_df) == 3
    assert "year" in cleaned_df.columns
    assert cleaned_df["year"].iloc[0] == 2024
    assert cleaned_df["round"].iloc[0] == 5
    assert not cleaned_df["program"].str.contains("Architecture").any()


# =========================================================================
# 2. FEATURE ENGINEERING TESTS (src/features.py)
# =========================================================================

def test_assign_branch_group():
    """Verify broad branch discipline categorization."""
    assert assign_branch_group("Computer Science and Engineering") == "CSE"
    assert assign_branch_group("Data Science and Artificial Intelligence") == "CSE"
    assert assign_branch_group("Electronics and Communication Engineering") == "ECE"
    assert assign_branch_group("Electrical Engineering") == "EE"
    assert assign_branch_group("Mechanical Engineering") == "MECH"
    assert assign_branch_group("Aerospace Engineering") == "MECH"
    assert assign_branch_group("Civil Engineering") == "CIVIL"
    assert assign_branch_group("Chemical Engineering") == "CHEM"
    assert assign_branch_group("Metallurgical and Materials Engineering") == "META"
    assert assign_branch_group("Biotechnology") == "BIO"
    assert assign_branch_group("Ceramic Engineering") == "META"
    assert assign_branch_group("Textile Engineering") == "OTHER"


def test_build_lag_features():
    """Verify multi-year lag generation and trend slopes."""
    # Synthesize 3 years of cutoffs for a single option
    records = [
        {"year": 2022, "institute": "IITB", "program": "CSE", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 60},
        {"year": 2023, "institute": "IITB", "program": "CSE", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 65},
        {"year": 2024, "institute": "IITB", "program": "CSE", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 70},
    ]
    df = pd.DataFrame(records)
    feat_df = compute_lag_features(df)

    # In 2024 row, prev_closing_1 should be 65, prev_closing_2 should be 60
    row_2024 = feat_df[feat_df["year"] == 2024].iloc[0]
    assert row_2024["prev_closing_1"] == 65
    assert row_2024["prev_closing_2"] == 60
    assert row_2024["years_of_history"] == 2
    assert row_2024["mean_last_2"] == 62.5
    assert row_2024["delta_1yr"] == 5.0
    assert not np.isnan(row_2024["trend_slope"])


# =========================================================================
# 3. BASELINE EVALUATION TESTS (src/baseline.py)
# =========================================================================

def test_compute_weighted_log_trend_variations():
    """Verify weighted trend with 1, 2, and 3 years of history."""
    # 3 years
    p3 = compute_weighted_log_trend(1000, 900, 800)
    expected_3 = 0.5 * np.log(1000) + 0.3 * np.log(900) + 0.2 * np.log(800)
    assert np.isclose(p3, expected_3)

    # 2 years (p3 is NaN)
    p2 = compute_weighted_log_trend(1000, 900, np.nan)
    expected_2 = (0.5 / 0.8) * np.log(1000) + (0.3 / 0.8) * np.log(900)
    assert np.isclose(p2, expected_2)

    # 1 year (p2, p3 NaN)
    p1 = compute_weighted_log_trend(1000, np.nan, np.nan)
    assert np.isclose(p1, np.log(1000))

    # All NaN
    assert np.isnan(compute_weighted_log_trend(np.nan, np.nan, np.nan))


def test_evaluate_baseline_on_year():
    """Test evaluate_baseline_on_held_out on sample dataset."""
    sample_data = pd.DataFrame([
        {
            "year": 2025,
            "institute": "IITB",
            "program": "CSE",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "closing_rank": 66,
            "prev_closing_1": 65,
            "prev_closing_2": 60,
            "prev_closing_3": 58,
        },
        {
            "year": 2025,
            "institute": "NITT",
            "program": "CSE",
            "quota": "OS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "closing_rank": 1450,
            "prev_closing_1": 1400,
            "prev_closing_2": 1350,
            "prev_closing_3": 1300,
        },
    ])
    sample_data["log_closing_rank"] = np.log(sample_data["closing_rank"])
    model = evaluate_baseline_on_held_out(sample_data, held_out_year=2025)
    assert model.held_out_year == 2025
    assert model.num_evaluated_rows == 2
    assert model.mae_baseline > 0
    assert model.residual_sigma > 0


# =========================================================================
# 4. TRAINING & PREPARATION TESTS (src/train.py)
# =========================================================================

def test_prepare_features():
    """Verify log transforms and categorical assignments."""
    sample_df = pd.DataFrame([
        {
            "institute_type": "NIT",
            "branch_group": "CSE",
            "quota": "OS",
            "seat_type": "OPEN",
            "prev_closing_1": 1000,
            "prev_closing_2": 900,
            "prev_closing_3": 800,
            "mean_last_2": 950,
            "trend_slope": 50,
            "delta_1yr": 100,
            "years_of_history": 3,
        }
    ])
    feat_df = prepare_features(sample_df)
    assert np.isclose(feat_df["log_prev_1"].iloc[0], np.log(1000))
    assert np.isclose(feat_df["log_prev_2"].iloc[0], np.log(900))
    assert feat_df["institute_type"].dtype.name == "category"


def test_train_and_validate_errors(tmp_path):
    """train_and_validate raises FileNotFoundError or ValueError on missing/insufficient data."""
    missing_file = tmp_path / "non_existent.parquet"
    with pytest.raises(FileNotFoundError):
        train_and_validate(missing_file)

    # Empty file
    empty_file = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(empty_file)
    with pytest.raises(KeyError):
        train_and_validate(empty_file)


def test_train_and_validate_full_pipeline(tmp_path):
    """Verify LightGBM training and baseline comparison on cutoffs parquet."""
    parquet_path = Path("data/processed/cutoffs.parquet")
    assert parquet_path.exists(), "cutoffs.parquet must exist"
    out_model = tmp_path / "test_model.joblib"
    model, metrics = train_and_validate(
        parquet_path,
        model_save_path=out_model,
        split_year=2024,
        val_year=2025,
    )
    assert model is not None
    assert "LightGBM" in metrics
    assert "Weighted trend" in metrics
    assert "Baseline (last year)" in metrics
    assert metrics["LightGBM"]["mae"] > 0
    assert out_model.exists()


# =========================================================================
# 5. SCRAPER UTILITY TESTS (src/scrape.py)
# =========================================================================

def test_get_existing_final_round_file(tmp_path):
    """Verify discovery of existing final round raw cutoff files."""
    assert get_existing_final_round_file(tmp_path, "2024") is None

    # Create dummy match
    f = tmp_path / "2024_round5.csv"
    f.write_text("header")
    match = get_existing_final_round_file(tmp_path, "2024")
    assert match is not None
    assert match.name == "2024_round5.csv"


def test_clean_all_raw_files_pipeline(tmp_path):
    """Test full multi-file raw cleaning pipeline and institutes export."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()

    # Empty raw dir raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No raw files found"):
        clean_all_raw_files(raw_dir, proc_dir)

    # Valid raw file
    f = raw_dir / "2024_round6.csv"
    f.write_text(
        "Institute,Academic Program Name,Quota,Seat Type,Gender,Opening Rank,Closing Rank\n"
        "Indian Institute of Technology Bombay,Computer Science and Engineering,AI,OPEN,Gender-Neutral,1,66\n"
        "National Institute of Technology Tiruchirappalli,Civil Engineering,OS,OPEN,Gender-Neutral,5000,12000\n",
        encoding="utf-8",
    )
    combined = clean_all_raw_files(raw_dir, proc_dir)
    assert len(combined) == 2
    assert (proc_dir / "institutes.csv").exists()


from unittest.mock import MagicMock


def test_scraper_utility_functions():
    """Verify scraper DOM extractors and cascading selectors with mock Playwright Page."""
    mock_page = MagicMock()

    # 1. fetch_available_years
    mock_page.eval_on_selector.return_value = [
        {"text": "Select", "value": "0"},
        {"text": "2024", "value": "2024"},
        {"text": "2023", "value": "2023"},
    ]
    years = fetch_available_years(mock_page)
    assert years == ["2024", "2023"]

    # 2. select_year_and_get_final_round
    mock_page.eval_on_selector.return_value = [
        {"value": "1"},
        {"value": "2"},
        {"value": "6"},
    ]
    final_round = select_year_and_get_final_round(mock_page, "2024")
    assert final_round == "6"

    # 3. trigger_cascade_and_submit
    trigger_cascade_and_submit(mock_page, "6")
    assert mock_page.evaluate.call_count >= 1

    # 4. extract_table_dataframe
    mock_page.evaluate.return_value = [
        ["Institute", "Academic Program Name", "Quota", "Seat Type", "Gender", "Opening Rank", "Closing Rank"],
        ["IIT Bombay", "Computer Science", "AI", "OPEN", "Gender-Neutral", "1", "66"],
    ]
    df_table = extract_table_dataframe(mock_page)
    assert isinstance(df_table, pd.DataFrame)
    assert len(df_table) == 1
    assert df_table["Closing Rank"].iloc[0] == "66"

    # Empty table raises RuntimeError
    mock_page.evaluate.return_value = []
    with pytest.raises(RuntimeError, match="No data rows found"):
        extract_table_dataframe(mock_page)

