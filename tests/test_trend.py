"""Unit tests for 5-year closing rank trend classification and deep-dive probability explanation."""

from __future__ import annotations

import numpy as np
import pytest

from src.eligibility import StudentProfile
from src.optimize import build_list
from src.predict import admit_probability
from src.simulation import get_model_metrics, get_model_sigma
from src.trend import (
    EASIER_MEAN_THRESHOLD,
    HARDER_MEAN_THRESHOLD,
    MIN_HISTORY_YEARS,
    TREND_COLORS,
    TREND_GETTING_EASIER,
    TREND_GETTING_HARDER,
    TREND_NOT_ENOUGH_DATA,
    TREND_STABLE,
    TREND_VOLATILE,
    VOLATILE_STD_THRESHOLD,
    compute_prediction_explanation,
    compute_trend,
    get_trend_badge,
)
from src.whatif import rescore_choice_list


def test_trend_threshold_constants_defined():
    """Verify that all required threshold constants are named and set properly."""
    assert MIN_HISTORY_YEARS == 3
    assert VOLATILE_STD_THRESHOLD == 0.20
    assert EASIER_MEAN_THRESHOLD == 0.05
    assert HARDER_MEAN_THRESHOLD == -0.05


def test_trend_fewer_than_3_points():
    """Fewer than 3 years of data must return 'Not enough history' with grey color."""
    # 0 points
    badge, color = get_trend_badge([])
    assert badge == TREND_NOT_ENOUGH_DATA
    assert color == TREND_COLORS[TREND_NOT_ENOUGH_DATA]

    # 1 point
    badge, color = get_trend_badge([5000])
    assert badge == TREND_NOT_ENOUGH_DATA
    assert color == "#9CA3AF"

    # 2 points
    res = compute_trend([10000, 11000])
    assert res["badge"] == TREND_NOT_ENOUGH_DATA
    assert res["color"] == "#9CA3AF"
    assert res["n_years"] == 2
    assert res["mean_d"] is None
    assert res["std_d"] is None


def test_trend_rising_getting_easier():
    """Rising closing rank numbers (relaxation of cutoffs) must classify as 'Getting easier' (green)."""
    # Closing ranks rising steadily: 1000 -> 1200 -> 1500
    ranks = [1000, 1200, 1500]
    res = compute_trend(ranks)

    assert res["badge"] == TREND_GETTING_EASIER
    assert res["color"] == "#22C55E"
    assert res["n_years"] == 3
    assert res["std_d"] <= VOLATILE_STD_THRESHOLD
    assert res["mean_d"] > EASIER_MEAN_THRESHOLD


def test_trend_falling_getting_harder():
    """Falling closing rank numbers (tightening of cutoffs) must classify as 'Getting harder' (red)."""
    # Closing ranks falling steadily: 1500 -> 1200 -> 1000
    ranks = [1500, 1200, 1000]
    res = compute_trend(ranks)

    assert res["badge"] == TREND_GETTING_HARDER
    assert res["color"] == "#EF4444"
    assert res["n_years"] == 3
    assert res["std_d"] <= VOLATILE_STD_THRESHOLD
    assert res["mean_d"] < HARDER_MEAN_THRESHOLD


def test_trend_volatile():
    """Series with high year-on-year volatility (std(d) > 0.20) must classify as 'Volatile' (grey)."""
    # Severe fluctuation: 1000 -> 2500 -> 700
    ranks = [1000, 2500, 700]
    res = compute_trend(ranks)

    assert res["badge"] == TREND_VOLATILE
    assert res["color"] == "#9CA3AF"
    assert res["n_years"] == 3
    assert res["std_d"] > VOLATILE_STD_THRESHOLD


def test_trend_stable():
    """Series with low std and small drift (|mean| <= 0.05) must classify as 'Stable' (grey)."""
    # Ranks nearly constant: 10000 -> 10100 -> 10050
    ranks = [10000, 10100, 10050]
    res = compute_trend(ranks)

    assert res["badge"] == TREND_STABLE
    assert res["color"] == "#9CA3AF"
    assert res["n_years"] == 3
    assert res["std_d"] <= VOLATILE_STD_THRESHOLD
    assert HARDER_MEAN_THRESHOLD <= res["mean_d"] <= EASIER_MEAN_THRESHOLD


def test_probability_consistency_with_choice_list():
    """Verify that probability shown in deep dive matches the choice list table exactly."""
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        list_size=30,
    )
    df = build_list(student)
    assert not df.empty

    sigma = get_model_sigma()
    metrics = get_model_metrics()

    for _, row in df.iterrows():
        student_rank = int(row["student_rank"])
        y_hat = float(row["y_hat"])
        expected_prob = float(row["probability"])

        # Compute via compute_prediction_explanation
        explanation = compute_prediction_explanation(
            student_rank=student_rank,
            y_hat=y_hat,
            sigma=sigma,
            metrics=metrics,
        )

        # Exact match within floating tolerance
        assert np.isclose(explanation["prob"], expected_prob, atol=1e-6)
        assert explanation["pred_rank"] == int(row["predicted_closing_rank"])

    # Test also with What-If shift applied
    rescored_df, _ = rescore_choice_list(df, pct=10.0, sigma=sigma)
    for _, row in rescored_df.iterrows():
        shifted_rank = int(row["shifted_rank"])
        y_hat = float(row["y_hat"])
        expected_shifted_prob = float(row["shifted_probability"])

        explanation_shifted = compute_prediction_explanation(
            student_rank=shifted_rank,
            y_hat=y_hat,
            sigma=sigma,
            metrics=metrics,
        )
        assert np.isclose(explanation_shifted["prob"], expected_shifted_prob, atol=1e-6)
