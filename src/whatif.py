"""What-If rank sensitivity simulation module for Vikalp.

Enables real-time exploration of rank variations (+/- 20%):
- shift_rank: scales rank by percentage, ensuring rank never drops below 1.
- rescore_choice_list: recomputes individual admission probabilities, tiers,
  and portfolio seat admission chance for an existing choice list WITHOUT reordering.
- shift_student_profile: creates a new StudentProfile with shifted ranks for full list regeneration.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.eligibility import StudentProfile
from src.predict import admit_probability, label_risk
from src.simulation import get_model_sigma, simulate_at_least_one_seat_probability


def shift_rank(rank: Optional[int], pct: float) -> Optional[int]:
    """Shift rank by percentage.

    Parameters
    ----------
    rank : int or None
        The base rank.
    pct : float
        Percentage shift (-20.0 to +20.0). +10% means 10% higher rank number (worse rank).

    Returns
    -------
    int or None
        Shifted rank, bounded below by 1. Returns None if rank is None or <= 0.
    """
    if rank is None or rank <= 0:
        return None

    shift_factor = 1.0 + (pct / 100.0)
    shifted = int(np.round(rank * shift_factor))
    return max(1, shifted)


def shift_student_profile(student: StudentProfile, pct: float) -> StudentProfile:
    """Create a new StudentProfile with all applicable ranks shifted by pct."""
    new_student = copy.deepcopy(student)
    new_student.main_rank = shift_rank(student.main_rank, pct)
    new_student.adv_rank = shift_rank(student.adv_rank, pct)
    new_student.category_rank = shift_rank(student.category_rank, pct)
    return new_student


def rescore_choice_list(
    df: pd.DataFrame,
    pct: float,
    sigma: Optional[float] = None,
    rho: float = 0.5,
    n_sims: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, float]:
    """Re-score an existing choice list with shifted ranks in-place without reordering.

    Parameters
    ----------
    df : pd.DataFrame
        Existing choice list containing columns 'student_rank', 'y_hat', 'probability', 'label'.
    pct : float
        Percentage rank shift.
    sigma : float, optional
        Log-scale residual standard error.
    rho : float, default 0.5
        Cross-option shock correlation for Monte Carlo simulation.
    n_sims : int, default 2000
        Number of Monte Carlo simulations.
    seed : int, default 42
        Simulation random seed.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        Rescored DataFrame with new probability, new label, and tier_change marker,
        plus the updated seat admission chance (in [0.0, 1.0]).
    """
    if df.empty:
        return pd.DataFrame(), 0.0

    if sigma is None:
        sigma = get_model_sigma()

    rescored = df.copy()

    # Shift each option's effective rank
    shifted_ranks = [shift_rank(int(r), pct) for r in rescored["student_rank"].values]
    rescored["shifted_rank"] = shifted_ranks

    # Recalculate probabilities using option-specific sigma if present
    if "sigma_used" in rescored.columns:
        effective_sigma = rescored["sigma_used"].values
    else:
        effective_sigma = sigma if sigma is not None else get_model_sigma()

    new_probs = admit_probability(
        student_rank=np.array(shifted_ranks, dtype=float),
        y_hat=rescored["y_hat"].values,
        sigma=effective_sigma,
    )
    rescored["shifted_probability"] = new_probs

    # Recalculate tier labels
    new_labels = [label_risk(p) for p in new_probs]
    rescored["shifted_label"] = new_labels

    # Track tier changes
    tier_changes = []
    for old_lbl, new_lbl in zip(rescored["label"].values, new_labels):
        if old_lbl != new_lbl:
            tier_changes.append(f"{old_lbl} → {new_lbl}")
        else:
            tier_changes.append("")
    rescored["tier_change"] = tier_changes

    # Recompute Monte Carlo portfolio seat admission chance
    new_p_at_least_one = simulate_at_least_one_seat_probability(
        student_ranks=shifted_ranks,
        y_hats=rescored["y_hat"].values,
        sigma=sigma,
        rho=rho,
        n_sims=n_sims,
        seed=seed,
    )

    return rescored, new_p_at_least_one
