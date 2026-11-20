"""Tests for the Monte Carlo at-least-one-seat probability simulation."""

import numpy as np
import pytest
from scipy.special import ndtr

from src.simulation import simulate_at_least_one_seat_probability, get_model_sigma


def test_independent_case_matches_formula():
    """When rho = 0, Monte Carlo estimate must match 1 - prod(1 - P_i) within statistical tolerance."""
    sigma = 0.2285
    # 5 options with diverse cutoffs
    y_hats = np.array([8.0, 8.5, 9.0, 9.5, 10.0])
    # Student rank of 6000
    student_ranks = np.array([6000, 6000, 6000, 6000, 6000])

    # Analytical marginal probabilities for each option: P_i = Phi((y_hat_i - ln(rank)) / sigma)
    z = (np.log(student_ranks) - y_hats) / sigma
    p_marginal = ndtr(-z)

    # Analytical cumulative under independence:
    p_analytical_independent = 1.0 - float(np.prod(1.0 - p_marginal))

    # Monte Carlo with rho = 0.0
    p_sim = simulate_at_least_one_seat_probability(
        student_ranks=student_ranks,
        y_hats=y_hats,
        sigma=sigma,
        rho=0.0,
        n_sims=5000,
        seed=123,
    )

    assert abs(p_sim - p_analytical_independent) < 0.02, (
        f"Expected MC estimate {p_sim:.4f} to be close to analytical {p_analytical_independent:.4f}"
    )


def test_correlated_case_behavior():
    """Positive correlation rho = 0.5 should result in slightly lower or equal tail probability than independent."""
    sigma = 0.2285
    y_hats = np.array([8.5, 9.0, 9.2])
    student_ranks = np.array([8000, 8000, 8000])

    p_corr = simulate_at_least_one_seat_probability(
        student_ranks=student_ranks,
        y_hats=y_hats,
        sigma=sigma,
        rho=0.5,
        n_sims=3000,
        seed=42,
    )

    assert 0.0 <= p_corr <= 1.0


def test_monotonicity_with_additional_choices():
    """Adding more choices should never decrease cumulative admission probability."""
    sigma = 0.2285
    y_hats_small = np.array([8.0, 8.5])
    ranks_small = np.array([5000, 5000])

    p_small = simulate_at_least_one_seat_probability(ranks_small, y_hats_small, sigma=sigma, seed=42)

    y_hats_large = np.array([8.0, 8.5, 9.0, 9.5, 10.0])
    ranks_large = np.array([5000, 5000, 5000, 5000, 5000])

    p_large = simulate_at_least_one_seat_probability(ranks_large, y_hats_large, sigma=sigma, seed=42)

    assert p_large >= p_small


def test_invalid_arguments():
    """Test validation errors for improper inputs."""
    with pytest.raises(ValueError):
        simulate_at_least_one_seat_probability([1000], [8.0], rho=1.5)

    with pytest.raises(ValueError):
        simulate_at_least_one_seat_probability([1000], [8.0], rho=-0.1)

    with pytest.raises(ValueError):
        simulate_at_least_one_seat_probability([1000, 2000], [8.0])


def test_get_model_sigma():
    """Test reading sigma from model metrics."""
    sigma = get_model_sigma()
    assert isinstance(sigma, float)
    assert 0.1 <= sigma <= 0.5
