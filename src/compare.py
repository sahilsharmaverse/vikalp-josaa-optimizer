"""Side-by-side comparison engine for Vikalp (v1.1 Feature 4).

Enables comparing 2 or 3 options from the current choice list:
- Validation: exactly 2 or 3 options allowed.
- Order check: detects if an option is placed after an option with P >= 95%.
- Rule-based quick facts: strictly deterministic, non-AI statements.
- SVG sparklines and circular probability gauges for glass cards.
- Multi-series Plotly comparison chart with line styles and marker shapes for accessibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.trend import (
    COLOR_GREEN,
    COLOR_GREY,
    COLOR_RED,
    TREND_GETTING_EASIER,
    TREND_GETTING_HARDER,
    TREND_NOT_ENOUGH_DATA,
    TREND_STABLE,
    TREND_VOLATILE,
    compute_trend,
    render_trend_badge_html,
)

# Color-blind friendly styles for up to 3 options
OPTION_STYLES = [
    {
        "color": "#0EA5E9",  # Electric Cyan
        "dash": "solid",
        "symbol": "circle",
        "name_prefix": "Option A",
    },
    {
        "color": "#F59E0B",  # Vibrant Amber
        "dash": "dash",
        "symbol": "square",
        "name_prefix": "Option B",
    },
    {
        "color": "#10B981",  # Emerald Green
        "dash": "dot",
        "symbol": "triangle-up",
        "name_prefix": "Option C",
    },
]


def validate_selection_count(count: int) -> Tuple[bool, str]:
    """Validate that exactly 2 or 3 options are selected.

    Parameters
    ----------
    count : int
        Number of selected options.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, helper_or_warning_message)
    """
    if count == 0:
        return False, "Pick 2 or 3 options from your list above to compare them side by side."
    elif count == 1:
        return False, "Select at least one more option (2 or 3 total) to enable comparison."
    elif count > 3:
        return False, "Please select at most 3 options to compare."
    return True, ""


def check_order_unreachable(
    selected_rows: Union[List[Dict[str, Any]], pd.DataFrame],
    prob_col: str = "shifted_probability",
) -> List[Dict[str, Any]]:
    """Check JoSAA preference order: flag options sitting after another selected option with P >= 0.95.

    Under JoSAA rules, allocation stops at the first qualified option. If option J has P >= 0.95,
    any option K placed after J (choice_no_k > choice_no_j) is effectively unreachable.

    Parameters
    ----------
    selected_rows : list of dicts or DataFrame
        Selected option rows containing 'choice_no', 'institute', 'program', and prob_col.
    prob_col : str, default 'shifted_probability'
        Column name holding active admission probabilities.

    Returns
    -------
    list of dict
        List of warning dictionaries for any unreachable options.
    """
    if isinstance(selected_rows, pd.DataFrame):
        rows = selected_rows.to_dict(orient="records")
    else:
        rows = list(selected_rows)

    if len(rows) < 2:
        return []

    # Sort strictly by preference order in the student's list (choice_no)
    sorted_rows = sorted(rows, key=lambda r: int(r.get("choice_no", 0)))
    warnings: List[Dict[str, Any]] = []

    for i in range(len(sorted_rows)):
        current_opt = sorted_rows[i]
        curr_choice_no = int(current_opt.get("choice_no", 0))

        # Check if any earlier option had P >= 0.95
        for j in range(i):
            earlier_opt = sorted_rows[j]
            earlier_choice_no = int(earlier_opt.get("choice_no", 0))
            earlier_prob = float(earlier_opt.get(prob_col, earlier_opt.get("probability", 0.0)))

            if earlier_prob >= 0.95 and curr_choice_no > earlier_choice_no:
                msg = (
                    f"Choice #{curr_choice_no} ({current_opt.get('institute', '')} — {current_opt.get('program', '')}) "
                    f"sits after Choice #{earlier_choice_no} ({earlier_opt.get('institute', '')} — {earlier_opt.get('program', '')}), "
                    f"which has a {earlier_prob:.1%} admission chance. Because JoSAA allocates strictly top-to-bottom, "
                    f"you will almost certainly be allotted Choice #{earlier_choice_no} first, making Choice #{curr_choice_no} "
                    f"effectively unreachable."
                )
                warnings.append({
                    "blocked_choice": curr_choice_no,
                    "blocked_name": f"{current_opt.get('institute', '')} — {current_opt.get('program', '')}",
                    "blocking_choice": earlier_choice_no,
                    "blocking_name": f"{earlier_opt.get('institute', '')} — {earlier_opt.get('program', '')}",
                    "blocking_prob": earlier_prob,
                    "message": msg,
                })
                break  # Record first blocking option

    return warnings


def generate_quick_facts(
    selected_rows: Union[List[Dict[str, Any]], pd.DataFrame],
    trends: List[Dict[str, Any]],
    prob_col: str = "shifted_probability",
    label_col: str = "shifted_label",
) -> List[str]:
    """Generate rule-based comparison sentences without AI-generated text.

    Parameters
    ----------
    selected_rows : list of dicts or DataFrame
        Selected options.
    trends : list of dict
        Calculated trend dictionaries from compute_trend for each option.
    prob_col : str
        Column name holding active admission probabilities.
    label_col : str
        Column name holding active tier labels.

    Returns
    -------
    list of str
        Deterministic factual comparison sentences plus mandatory disclaimer.
    """
    if isinstance(selected_rows, pd.DataFrame):
        rows = selected_rows.to_dict(orient="records")
    else:
        rows = list(selected_rows)

    if len(rows) < 2:
        return []

    facts: List[str] = []

    # 1. Probability Comparison
    probs = [float(r.get(prob_col, r.get("probability", 0.0))) for r in rows]
    choice_nos = [int(r.get("choice_no", 0)) for r in rows]
    insts = [str(r.get("institute", "")) for r in rows]

    if len(rows) == 2:
        p1, p2 = probs[0], probs[1]
        c1, c2 = choice_nos[0], choice_nos[1]
        if abs(p1 - p2) >= 0.001:
            if p1 > p2:
                facts.append(
                    f"Choice #{c1} ({insts[0]}) has the higher admission chance ({p1:.1%} vs {p2:.1%})."
                )
            else:
                facts.append(
                    f"Choice #{c2} ({insts[1]}) has the higher admission chance ({p2:.1%} vs {p1:.1%})."
                )
        else:
            facts.append(
                f"Choice #{c1} and Choice #{c2} have identical estimated admission chances ({p1:.1%})."
            )
    elif len(rows) == 3:
        max_idx = int(np.argmax(probs))
        min_idx = int(np.argmin(probs))
        all_equal = np.allclose(probs, probs[0], atol=0.001)

        if not all_equal:
            facts.append(
                f"Choice #{choice_nos[max_idx]} offers the highest admission chance ({probs[max_idx]:.1%}), "
                f"while Choice #{choice_nos[min_idx]} has the lowest ({probs[min_idx]:.1%})."
            )
        else:
            facts.append(
                f"All 3 selected options have identical estimated admission chances ({probs[0]:.1%})."
            )

    # 2. Tier Diversity Fact
    tiers = [str(r.get(label_col, r.get("label", ""))) for r in rows]
    unique_tiers = sorted(list(set(tiers)))
    if len(unique_tiers) > 1:
        tier_summary = ", ".join(f"Choice #{c} is {t}" for c, t in zip(choice_nos, tiers))
        facts.append(f"Strategic risk tier mix: {tier_summary}.")
    else:
        facts.append(f"All selected options belong to the {unique_tiers[0]} tier.")

    # 3. 5-Year Cutoff Trend Facts
    for r, t in zip(rows, trends):
        c_no = int(r.get("choice_no", 0))
        badge = t.get("badge", "")
        if badge == TREND_GETTING_HARDER:
            facts.append(
                f"Choice #{c_no}'s cutoff has been getting harder (closing rank numbers decreasing)."
            )
        elif badge == TREND_GETTING_EASIER:
            facts.append(
                f"Choice #{c_no}'s cutoff has been getting easier (closing rank numbers increasing)."
            )
        elif badge == TREND_VOLATILE:
            facts.append(
                f"Choice #{c_no}'s cutoff has been volatile across recent counselling cycles."
            )

    # 4. Mandatory Decision Support Line
    facts.append(
        "This compares admission chances only. Branch, placement, location and fees also matter."
    )

    return facts


def render_svg_circular_gauge(
    probability: float,
    tier_label: str,
    size: int = 90,
) -> str:
    """Render a clean SVG circular gauge with percentage inside and tier color arc."""
    prob = max(0.01, min(0.99, probability))
    pct_text = f"{prob * 100.0:.0f}%"

    stroke_color = (
        "#EF4444"
        if tier_label == "Dream"
        else "#F59E0B"
        if tier_label == "Target"
        else "#22C55E"
    )

    radius = 36
    circ = 2 * math.pi * radius
    stroke_dash = prob * circ

    return (
        f'<div style="display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:{size}px;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 100 100">'
        f'<circle cx="50" cy="50" r="{radius}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="8" />'
        f'<circle cx="50" cy="50" r="{radius}" fill="none" stroke="{stroke_color}" stroke-width="8" '
        f'stroke-dasharray="{stroke_dash:.1f} {circ:.1f}" stroke-linecap="round" transform="rotate(-90 50 50)" />'
        f'<text x="50" y="55" font-family="Inter, sans-serif" font-size="19" font-weight="800" '
        f'fill="#FFFFFF" text-anchor="middle" dominant-baseline="middle">{pct_text}</text>'
        f'</svg>'
        f'</div>'
    )


def render_svg_sparkline(
    closing_ranks: Sequence[Union[int, float]],
    width: int = 140,
    height: int = 38,
    color: str = "#0EA5E9",
) -> str:
    """Render a lightweight SVG sparkline showing cutoff trajectory without crashing on sparse data."""
    ranks = [float(r) for r in closing_ranks if not np.isnan(r) and r > 0]
    n = len(ranks)

    if n == 0:
        return f'<div style="width:{width}px;height:{height}px;display:flex;align-items:center;justify-content:center;color:#64748B;font-size:11px;">No history</div>'
    elif n == 1:
        return f'<div style="width:{width}px;height:{height}px;display:flex;align-items:center;justify-content:center;color:#94A3B8;font-size:11px;">1 year: {int(ranks[0]):,}</div>'

    min_val = min(ranks)
    max_val = max(ranks)
    range_val = max_val - min_val if max_val != min_val else 1.0

    pad_x = 6
    pad_y = 6
    inner_w = width - 2 * pad_x
    inner_h = height - 2 * pad_y

    pts = []
    circles = []
    for i, r in enumerate(ranks):
        x = pad_x + (i / (n - 1)) * inner_w
        norm = (r - min_val) / range_val
        y = pad_y + (1.0 - norm) * inner_h
        pts.append(f"{x:.1f},{y:.1f}")
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" />'
        )

    poly_pts = " ".join(pts)
    circles_html = "".join(circles)

    return (
        f'<svg width="{width}" height="{height}" style="overflow:visible;display:block;">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="{poly_pts}" />'
        f'{circles_html}'
        f'</svg>'
    )


def create_comparison_chart(
    selected_df: pd.DataFrame,
    history_df: pd.DataFrame,
    current_student_rank: int,
    rank_shift_pct: int,
) -> go.Figure:
    """Construct multi-series Plotly dark theme chart with color-blind line styles and markers.

    Parameters
    ----------
    selected_df : pd.DataFrame
        DataFrame of 2 or 3 selected choices.
    history_df : pd.DataFrame
        Historical cutoffs DataFrame.
    current_student_rank : int
        Active rank (including What-If shift).
    rank_shift_pct : int
        Current What-If rank shift percentage.

    Returns
    -------
    go.Figure
        Configured Plotly figure with log y-axis, thousands separators, and accessible series.
    """
    fig = go.Figure()

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        style = OPTION_STYLES[idx % len(OPTION_STYLES)]

        # Query historical data
        hist_match = history_df[
            (history_df["institute"] == row["institute"])
            & (history_df["program"] == row["program"])
            & (history_df["quota"] == row["quota"])
            & (history_df["seat_type"] == row["seat_type"])
            & (history_df["gender"] == row["gender"])
        ].sort_values("year")

        choice_no = int(row["choice_no"])
        label_text = f"#{choice_no}: {row['institute'][:25]}.. ({row['program'][:20]}..)"

        if not hist_match.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist_match["year"],
                    y=hist_match["closing_rank"],
                    mode="lines+markers",
                    name=label_text,
                    line=dict(
                        color=style["color"],
                        width=2.8,
                        dash=style["dash"],
                    ),
                    marker=dict(
                        size=8,
                        symbol=style["symbol"],
                        color=style["color"],
                    ),
                    hovertemplate=f"Choice #{choice_no}<br>Year: %{{x}}<br>Closing Rank: %{{y:,}}<extra></extra>",
                )
            )

            # Add projected point at 2026
            pred_val = float(np.exp(row["y_hat"]))
            fig.add_trace(
                go.Scatter(
                    x=[2026],
                    y=[pred_val],
                    mode="markers",
                    name=f"#{choice_no} Projected ({int(round(pred_val)):,})",
                    marker=dict(
                        size=10,
                        symbol="star",
                        color=style["color"],
                    ),
                    hovertemplate=f"Choice #{choice_no} 2026 Projected: {int(round(pred_val)):,}<extra></extra>",
                    showlegend=False,
                )
            )

    # Student active rank horizontal reference line
    annot_text = (
        f"Your Simulated Rank: {current_student_rank:,}"
        if rank_shift_pct != 0
        else f"Your Rank: {current_student_rank:,}"
    )
    fig.add_hline(
        y=current_student_rank,
        line_dash="dashdot",
        line_color="#E2E8F0",
        line_width=2.2,
        annotation_text=annot_text,
        annotation_position="bottom right",
        annotation_font=dict(color="#E2E8F0", size=11.5, family="Inter"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        height=350,
        margin=dict(l=40, r=30, t=30, b=40),
        xaxis=dict(
            title="Counselling Year",
            tickmode="linear",
            dtick=1,
            gridcolor="rgba(255, 255, 255, 0.08)",
        ),
        yaxis=dict(
            type="log",
            title="Closing Rank (Log Scale)",
            tickformat=",d",
            gridcolor="rgba(255, 255, 255, 0.08)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )

    return fig
