"""Unit tests for admission probability and prediction functions."""

import numpy as np
import pytest

from src.predict import (
    admit_probability,
    get_prediction_interval,
    label_risk,
)


def test_rank_equal_to_prediction_gives_half():
    """Rank exactly equal to predicted cutoff gives P ~ 0.50."""
    cutoff = 5000
    y_hat = float(np.log(cutoff))
    sigma = 0.25

    prob = admit_probability(student_rank=cutoff, y_hat=y_hat, sigma=sigma)
    assert pytest.approx(0.50, abs=1e-4) == prob


def test_better_rank_gives_higher_probability():
    """Numerically smaller (better) rank yields higher probability of admission."""
    cutoff = 5000
    y_hat = float(np.log(cutoff))
    sigma = 0.25

    p_equal = admit_probability(student_rank=5000, y_hat=y_hat, sigma=sigma)
    p_better = admit_probability(student_rank=3000, y_hat=y_hat, sigma=sigma)
    p_much_better = admit_probability(student_rank=1000, y_hat=y_hat, sigma=sigma)
    p_worse = admit_probability(student_rank=8000, y_hat=y_hat, sigma=sigma)

    assert p_equal == pytest.approx(0.50, abs=1e-4)
    assert p_better > p_equal
    assert p_much_better > p_better
    assert p_worse < p_equal


def test_probability_monotonicity():
    """Probability strictly increases as student rank improves (decreases)."""
    cutoff = 10000
    y_hat = float(np.log(cutoff))
    sigma = 0.20

    ranks = [25000, 15000, 12000, 10000, 8000, 5000, 1000]
    probs = [admit_probability(r, y_hat, sigma) for r in ranks]

    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1]


def test_clipping_bounds():
    """Probabilities are strictly clipped to [0.01, 0.99] and never 0 or 1."""
    sigma = 0.20

    # Near-impossible dream seat
    p_dream = admit_probability(student_rank=100000, y_hat=np.log(100), sigma=sigma)
    assert p_dream == 0.01

    # Guaranteed safe seat
    p_safe = admit_probability(student_rank=1, y_hat=np.log(50000), sigma=sigma)
    assert p_safe == 0.99


def test_invalid_inputs():
    """Invalid student ranks (<=0) or non-positive sigmas raise ValueError."""
    y_hat = np.log(5000)

    with pytest.raises(ValueError, match="student_rank must be a positive number"):
        admit_probability(student_rank=0, y_hat=y_hat, sigma=0.25)

    with pytest.raises(ValueError, match="student_rank must be a positive number"):
        admit_probability(student_rank=-50, y_hat=y_hat, sigma=0.25)

    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        admit_probability(student_rank=5000, y_hat=y_hat, sigma=0.0)

    with pytest.raises(ValueError, match="sigma must be strictly positive"):
        admit_probability(student_rank=5000, y_hat=y_hat, sigma=-0.1)


def test_vectorized_array_inputs():
    """Vectorized inputs return identical results to scalar iterations."""
    ranks = np.array([2000, 5000, 8000])
    y_hats = np.array([np.log(5000), np.log(5000), np.log(5000)])
    sigma = 0.25

    vec_probs = admit_probability(ranks, y_hats, sigma)
    scalar_probs = np.array([
        admit_probability(int(r), float(y), sigma) for r, y in zip(ranks, y_hats)
    ])

    assert np.allclose(vec_probs, scalar_probs, atol=1e-5)


def test_risk_labels():
    """Risk tiers match PRD and rules.md thresholds: Dream < 0.30, Target 0.30-0.80, Safe >= 0.80."""
    assert label_risk(0.01) == "Dream"
    assert label_risk(0.29) == "Dream"
    assert label_risk(0.30) == "Target"
    assert label_risk(0.50) == "Target"
    assert label_risk(0.79) == "Target"
    assert label_risk(0.80) == "Safe"
    assert label_risk(0.99) == "Safe"


def test_prediction_interval():
    """80% prediction interval contains predicted rank and bounds are monotonic."""
    y_hat = float(np.log(10000))
    sigma = 0.25

    pred, lower, upper = get_prediction_interval(y_hat, sigma)
    assert lower < pred < upper
    assert pred == 10000
    assert lower > 1
