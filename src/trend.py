"""5-year closing rank trend badge and trajectory calculation module.

Thresholds are defined as named constants at the top:
- MIN_HISTORY_YEARS: Minimum number of historical years required to compute trend.
- VOLATILE_STD_THRESHOLD: Standard deviation threshold of year-on-year log changes for volatility.
- EASIER_MEAN_THRESHOLD: Mean log cutoff delta threshold for "Getting easier".
- HARDER_MEAN_THRESHOLD: Mean log cutoff delta threshold for "Getting harder".
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Named thresholds per specification
MIN_HISTORY_YEARS: int = 3
VOLATILE_STD_THRESHOLD: float = 0.20
EASIER_MEAN_THRESHOLD: float = 0.05
HARDER_MEAN_THRESHOLD: float = -0.05

# Trend badge names
TREND_NOT_ENOUGH_DATA: str = "Not enough history"
TREND_VOLATILE: str = "Volatile"
TREND_GETTING_EASIER: str = "Getting easier"
TREND_GETTING_HARDER: str = "Getting harder"
TREND_STABLE: str = "Stable"

# Hex color definitions
COLOR_GREY: str = "#9CA3AF"
COLOR_GREEN: str = "#22C55E"
COLOR_RED: str = "#EF4444"

TREND_COLORS: Dict[str, str] = {
    TREND_NOT_ENOUGH_DATA: COLOR_GREY,
    TREND_VOLATILE: COLOR_GREY,
    TREND_GETTING_EASIER: COLOR_GREEN,
    TREND_GETTING_HARDER: COLOR_RED,
    TREND_STABLE: COLOR_GREY,
}


def compute_trend(
    closing_ranks: Sequence[Union[int, float]] | np.ndarray,
    ddof: int = 1,
) -> Dict[str, Any]:
    """Calculate trend badge and summary statistics from chronological closing ranks.

    Year-on-year changes are defined as:
        d_t = ln(closing_t) - ln(closing_{t-1})

    Classification rules:
    - Fewer than 3 years of data -> "Not enough history" (grey)
    - Else if std(d) > 0.20 -> "Volatile" (grey)
    - Else if mean(d) > 0.05 -> "Getting easier" (green, closing rank number is rising)
    - Else if mean(d) < -0.05 -> "Getting harder" (red, closing rank number is falling)
    - Else -> "Stable" (grey)

    Parameters
    ----------
    closing_ranks : Sequence[float] or np.ndarray
        Historical closing ranks in chronological order (oldest to newest).
    ddof : int, default 1
        Delta degrees of freedom for sample standard deviation.

    Returns
    -------
    dict
        Dictionary containing:
        - badge: str
        - color: str
        - mean_d: Optional[float]
        - std_d: Optional[float]
        - n_years: int
    """
    ranks = np.asarray(closing_ranks, dtype=float)
    ranks = ranks[~np.isnan(ranks)]
    ranks = ranks[ranks > 0]

    n_years = len(ranks)
    if n_years < MIN_HISTORY_YEARS:
        return {
            "badge": TREND_NOT_ENOUGH_DATA,
            "color": TREND_COLORS[TREND_NOT_ENOUGH_DATA],
            "mean_d": None,
            "std_d": None,
            "n_years": n_years,
        }

    log_ranks = np.log(ranks)
    d = np.diff(log_ranks)

    n_diffs = len(d)
    actual_ddof = ddof if n_diffs > ddof else 0
    std_d = float(np.std(d, ddof=actual_ddof))
    mean_d = float(np.mean(d))

    if std_d > VOLATILE_STD_THRESHOLD:
        badge = TREND_VOLATILE
    elif mean_d > EASIER_MEAN_THRESHOLD:
        badge = TREND_GETTING_EASIER
    elif mean_d < HARDER_MEAN_THRESHOLD:
        badge = TREND_GETTING_HARDER
    else:
        badge = TREND_STABLE

    return {
        "badge": badge,
        "color": TREND_COLORS[badge],
        "mean_d": mean_d,
        "std_d": std_d,
        "n_years": n_years,
    }


def get_trend_badge(
    closing_ranks: Sequence[Union[int, float]] | np.ndarray,
) -> Tuple[str, str]:
    """Return (badge_text, hex_color) for an option's closing rank history."""
    res = compute_trend(closing_ranks)
    return res["badge"], res["color"]


def render_trend_badge_html(badge: str, color: Optional[str] = None) -> str:
    """Render an accessible HTML trend badge with colored dot and label text."""
    if color is None:
        color = TREND_COLORS.get(badge, COLOR_GREY)

    bg_color = (
        "rgba(34, 197, 94, 0.12)"
        if color == COLOR_GREEN
        else "rgba(239, 68, 68, 0.12)"
        if color == COLOR_RED
        else "rgba(156, 163, 175, 0.12)"
    )
    border_color = (
        "rgba(34, 197, 94, 0.3)"
        if color == COLOR_GREEN
        else "rgba(239, 68, 68, 0.3)"
        if color == COLOR_RED
        else "rgba(156, 163, 175, 0.3)"
    )

    return (
        f'<span style="display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; '
        f'border-radius: 9999px; background: {bg_color}; border: 1px solid {border_color}; '
        f'color: {color}; font-size: 13px; font-weight: 600; letter-spacing: 0.01em;">'
        f'<span style="width: 8px; height: 8px; border-radius: 50%; background: {color};"></span>'
        f'{badge}'
        f'</span>'
    )


def compute_prediction_explanation(
    student_rank: int,
    y_hat: float,
    sigma: float,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute mathematical and plain-words explanation components for deep dive.

    Parameters
    ----------
    student_rank : int
        Current effective rank (including What-If shift if active).
    y_hat : float
        Predicted log closing rank.
    sigma : float
        Residual standard deviation in log space.
    metrics : dict, optional
        Loaded metrics dictionary from models/metrics.json.

    Returns
    -------
    dict
        Detailed explanation fields including z-score, 80% interval, and formatted text.
    """
    if metrics is None:
        metrics = {
            "val_year": 2025,
            "mae": 0.1393,
            "sigma": sigma,
            "coverage_80": 0.9006,
            "model_name": "Weighted Trend Predictor (weights 0.5, 0.3, 0.2)",
        }

    from src.predict import admit_probability

    pred_rank = int(np.round(np.exp(y_hat)))
    low_80 = int(np.round(np.exp(y_hat - 1.28 * sigma)))
    high_80 = int(np.round(np.exp(y_hat + 1.28 * sigma)))

    z_score = float((np.log(float(student_rank)) - y_hat) / sigma)
    prob = float(admit_probability(student_rank, y_hat, sigma))

    rank_diff = student_rank - pred_rank
    rank_diff_abs = abs(rank_diff)

    if student_rank < pred_rank:
        diff_text = f"Your rank ({student_rank:,}) is {rank_diff_abs:,} ranks better (lower number) than the predicted median cutoff ({pred_rank:,})."
    elif student_rank > pred_rank:
        diff_text = f"Your rank ({student_rank:,}) is {rank_diff_abs:,} ranks behind (higher number) than the predicted median cutoff ({pred_rank:,})."
    else:
        diff_text = f"Your rank ({student_rank:,}) matches the predicted median cutoff ({pred_rank:,}) exactly."

    model_name = metrics.get(
        "model_name", "Weighted Trend Predictor (weights 0.5, 0.3, 0.2)"
    )
    mae = float(metrics.get("mae", 0.1393))
    coverage = float(metrics.get("coverage_80", 0.9006)) * 100.0

    return {
        "student_rank": student_rank,
        "pred_rank": pred_rank,
        "low_80": low_80,
        "high_80": high_80,
        "z_score": z_score,
        "prob": prob,
        "prob_pct": prob * 100.0,
        "rank_diff": rank_diff,
        "rank_diff_abs": rank_diff_abs,
        "diff_text": diff_text,
        "model_name": model_name,
        "mae": mae,
        "sigma": sigma,
        "coverage_80_pct": coverage,
    }
