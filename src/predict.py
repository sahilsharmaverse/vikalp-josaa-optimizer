"""Admission probability and cutoff prediction engine.

Implements the probability calculation:
P(admit) = 1 - Phi((ln(student_rank) - y_hat) / sigma)
Clipped to [0.01, 0.99] as per rules.md.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy.special import ndtr

MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
MIN_LIST_PROB = 0.05
DREAM_MAX_PROB = 0.30
TARGET_MAX_PROB = 0.80
Z_80 = 1.28155  # Critical value for 80% central prediction interval


def admit_probability(
    student_rank: Union[int, float, np.ndarray],
    y_hat: Union[float, np.ndarray],
    sigma: float,
) -> Union[float, np.ndarray]:
    """Calculate admission probability for a student rank given predicted log cutoff.

    Formula:
        z = (ln(student_rank) - y_hat) / sigma
        P(admit) = 1 - Phi(z) = Phi(-z)
    Clipped to [0.01, 0.99].

    Parameters
    ----------
    student_rank : int, float, or np.ndarray
        The student's rank (CRL or category rank). Must be strictly positive.
    y_hat : float or np.ndarray
        Predicted log(closing_rank).
    sigma : float
        Standard deviation of residuals in log space. Must be strictly positive.

    Returns
    -------
    float or np.ndarray
        Admission probability clipped to [0.01, 0.99].
    """
    sigma_arr = np.asarray(sigma, dtype=float)
    if np.any(sigma_arr <= 0):
        raise ValueError(f"sigma must be strictly positive, got {sigma}")

    rank_arr = np.asarray(student_rank, dtype=float)
    if np.any(rank_arr <= 0):
        raise ValueError("student_rank must be a positive number")

    y_hat_arr = np.asarray(y_hat, dtype=float)

    z = (np.log(rank_arr) - y_hat_arr) / sigma
    # 1 - Phi(z) = Phi(-z) using numerically stable ndtr
    prob = ndtr(-z)
    prob_clipped = np.clip(prob, MIN_PROBABILITY, MAX_PROBABILITY)

    if isinstance(student_rank, (int, float)) and isinstance(y_hat, (int, float)):
        return float(prob_clipped.item())

    return prob_clipped


def get_prediction_interval(
    y_hat: Union[float, np.ndarray], sigma: float
) -> Tuple[int, int, int]:
    """Return (predicted_rank, lower_80_rank, upper_80_rank) for a predicted log cutoff."""
    pred_rank = int(np.round(np.exp(y_hat)))
    lower_rank = int(np.round(np.exp(y_hat - Z_80 * sigma)))
    upper_rank = int(np.round(np.exp(y_hat + Z_80 * sigma)))
    return pred_rank, max(1, lower_rank), max(1, upper_rank)


def label_risk(prob: float) -> str:
    """Classify probability into JoSAA decision tiers per rules.md.

    - Dream: MIN_LIST_PROB <= P < DREAM_MAX_PROB (0.05 <= P < 0.30)
    - Target: DREAM_MAX_PROB <= P < TARGET_MAX_PROB (0.30 <= P < 0.80)
    - Safe: P >= TARGET_MAX_PROB (P >= 0.80)
    """
    if prob < DREAM_MAX_PROB:
        return "Dream"
    elif prob < TARGET_MAX_PROB:
        return "Target"
    else:
        return "Safe"
