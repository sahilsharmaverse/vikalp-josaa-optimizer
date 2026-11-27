"""Monte Carlo simulation engine for portfolio-level admission probability.

Calculates the probability of getting at least one seat across a choice list,
modeling correlated exam-level and counselling cutoff shocks:
- Shared shock s ~ N(0, 1) across all programs in a counselling cycle
- Idiosyncratic shock e_i ~ N(0, 1) per program option
- Correlated latent shock: z_i = sqrt(rho)*s + sqrt(1 - rho)*e_i (default rho = 0.5)
- Simulated cutoff: closing_{k, i} = exp(y_hat_i + sigma * z_i)
- Student admitted to option i if student_rank_i <= closing_{k, i}
- Overall probability: share of simulations where at least one option is admitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def get_model_metrics() -> dict:
    """Read model metrics and metadata from saved json."""
    metrics_path = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
    default_metrics = {
        "val_year": 2025,
        "mae": 0.1360,
        "sigma": 0.2278,
        "coverage_80": 0.9026,
        "model_name": "Weighted Trend Predictor (weights 0.5, 0.3, 0.2)",
    }
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_metrics.update(data)
                return default_metrics
        except Exception:
            return default_metrics
    return default_metrics


def get_model_sigma(default: float = 0.2278) -> float:
    """Read calibrated residual standard error from saved model metrics."""
    return float(get_model_metrics().get("sigma", default))


def simulate_at_least_one_seat_probability(
    student_ranks: Union[list, np.ndarray],
    y_hats: Union[list, np.ndarray],
    sigma: Optional[float] = None,
    rho: float = 0.5,
    n_sims: int = 2000,
    seed: int = 42,
) -> float:
    """Estimate the probability of getting at least one seat via Monte Carlo simulation.

    Parameters
    ----------
    student_ranks : list or np.ndarray
        Array of applicable student ranks for each choice option.
    y_hats : list or np.ndarray
        Array of predicted log closing ranks for each choice option.
    sigma : float, optional
        Residual standard deviation in log space. If None, loaded from saved metrics.
    rho : float, default 0.5
        Correlation coefficient among options due to macro exam/counselling shocks (in [0, 1]).
    n_sims : int, default 2000
        Number of Monte Carlo simulations to run.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    float
        Monte Carlo estimate of the probability of receiving at least one seat (in [0.0, 1.0]).
    """
    ranks = np.asarray(student_ranks, dtype=float)
    y_hat_arr = np.asarray(y_hats, dtype=float)

    if len(ranks) == 0 or len(y_hat_arr) == 0:
        return 0.0

    if len(ranks) != len(y_hat_arr):
        raise ValueError(
            f"student_ranks and y_hats must have the same length ({len(ranks)} vs {len(y_hat_arr)})"
        )

    if sigma is None:
        sigma = get_model_sigma()

    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}")

    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be between 0.0 and 1.0, got {rho}")

    n_options = len(ranks)
    rng = np.random.default_rng(seed)

    # s: shape (n_sims, 1) - common shock across all options in a cycle
    s = rng.standard_normal((n_sims, 1))

    # e: shape (n_sims, n_options) - option-specific idiosyncratic shock
    e = rng.standard_normal((n_sims, n_options))

    # Correlated shock: z_i = sqrt(rho)*s + sqrt(1 - rho)*e_i
    z = np.sqrt(rho) * s + np.sqrt(1.0 - rho) * e

    # Simulated closing ranks: exp(y_hat_i + sigma * z_{k, i})
    sim_log_closing = y_hat_arr[np.newaxis, :] + sigma * z
    sim_closing = np.exp(sim_log_closing)

    # Student admitted to option i if rank_i <= sim_closing
    admitted_matrix = ranks[np.newaxis, :] <= sim_closing

    # Simulation k has at least one admission if any option admitted
    any_admitted_per_sim = np.any(admitted_matrix, axis=1)

    prob = float(np.mean(any_admitted_per_sim))
    return prob


def compute_expected_outcome_quality(
    choice_df: pd.DataFrame,
    sigma: Optional[float] = None,
    rho: float = 0.5,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """Calculate expected outcome quality under JoSAA serial allocation semantics.

    JoSAA evaluates choices strictly from Choice #1 to #K. The applicant is allotted
    the FIRST choice for which student_rank <= simulated_closing_rank.

    Parameters
    ----------
    choice_df : pd.DataFrame
        Ordered choice list containing 'student_rank', 'y_hat', and 'predicted_closing_rank'.
    sigma : float, optional
        Residual standard deviation in log space.
    rho : float, default 0.5
        Shock correlation coefficient.
    n_sims : int, default 2000
        Number of simulations.
    seed : int, default 42
        Simulation random seed.

    Returns
    -------
    dict
        Dictionary with admission probability, expected allotted choice number,
        expected admitted closing rank, top-choice probability, and top-3 probability.
    """
    if choice_df.empty:
        return {
            "admission_prob": 0.0,
            "expected_choice_no": None,
            "expected_closing_rank": None,
            "top_choice_prob": 0.0,
            "top_3_prob": 0.0,
            "allotted_sims_count": 0,
        }

    ranks = np.asarray(choice_df["student_rank"].values, dtype=float)
    y_hat_arr = np.asarray(choice_df["y_hat"].values, dtype=float)
    pred_ranks = np.asarray(choice_df["predicted_closing_rank"].values, dtype=float)

    if sigma is None:
        sigma = get_model_sigma()

    n_options = len(ranks)
    rng = np.random.default_rng(seed)

    # Correlated shocks
    s = rng.standard_normal((n_sims, 1))
    e = rng.standard_normal((n_sims, n_options))
    z = np.sqrt(rho) * s + np.sqrt(1.0 - rho) * e

    sim_closing = np.exp(y_hat_arr[np.newaxis, :] + sigma * z)
    admitted_matrix = ranks[np.newaxis, :] <= sim_closing

    any_admitted = np.any(admitted_matrix, axis=1)
    allotted_count = int(np.sum(any_admitted))

    if allotted_count == 0:
        return {
            "admission_prob": 0.0,
            "expected_choice_no": None,
            "expected_closing_rank": None,
            "top_choice_prob": 0.0,
            "top_3_prob": 0.0,
            "allotted_sims_count": 0,
        }

    # First choice that qualified per simulation
    first_choice_indices = np.argmax(admitted_matrix[any_admitted], axis=1)
    allotted_choice_numbers = first_choice_indices + 1
    allotted_cutoffs = pred_ranks[first_choice_indices]

    expected_choice_no = float(np.mean(allotted_choice_numbers))
    expected_closing_rank = float(np.mean(allotted_cutoffs))

    # Top choice (Choice #1) allotted rate
    top_choice_count = np.sum(first_choice_indices == 0)
    top_choice_prob = float(top_choice_count / n_sims)

    # Top 3 choices allotted rate
    top_3_count = np.sum(first_choice_indices < min(3, n_options))
    top_3_prob = float(top_3_count / n_sims)

    admission_prob = float(allotted_count / n_sims)

    return {
        "admission_prob": admission_prob,
        "expected_choice_no": round(expected_choice_no, 1),
        "expected_closing_rank": int(round(expected_closing_rank)),
        "top_choice_prob": round(top_choice_prob, 3),
        "top_3_prob": round(top_3_prob, 3),
        "allotted_sims_count": allotted_count,
    }
