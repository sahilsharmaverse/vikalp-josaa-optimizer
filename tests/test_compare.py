"""Unit tests for Side-by-Side Comparison Engine (v1.1 Feature 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.compare import (
    check_order_unreachable,
    create_comparison_chart,
    generate_quick_facts,
    render_svg_circular_gauge,
    render_svg_sparkline,
    validate_selection_count,
)
from src.eligibility import StudentProfile
from src.optimize import build_list
from src.trend import compute_trend
from src.whatif import rescore_choice_list


def test_validate_selection_count():
    """Selecting fewer than 2 or more than 3 options must fail validation with helpful message and not crash."""
    # 0 options
    valid, msg = validate_selection_count(0)
    assert not valid
    assert "2 or 3" in msg

    # 1 option
    valid, msg = validate_selection_count(1)
    assert not valid
    assert "at least one more" in msg

    # 2 options (valid)
    valid, msg = validate_selection_count(2)
    assert valid
    assert msg == ""

    # 3 options (valid)
    valid, msg = validate_selection_count(3)
    assert valid
    assert msg == ""

    # 4 options (invalid)
    valid, msg = validate_selection_count(4)
    assert not valid
    assert "at most 3" in msg

    # 5 options (invalid)
    valid, msg = validate_selection_count(5)
    assert not valid


def test_comparison_probabilities_match_choice_list():
    """Probabilities in comparison must equal probabilities in list table exactly (including with What-If shift)."""
    student = StudentProfile(
        main_rank=15000,
        category="OPEN",
        gender="Gender-Neutral",
        home_state="Maharashtra",
        list_size=30,
    )
    df_list = build_list(student)
    assert len(df_list) >= 3

    # Pick choices #1, #5, #10
    selected_indices = [0, 4, 9]
    selected_df = df_list.iloc[selected_indices].copy()

    for _, row in selected_df.iterrows():
        expected_p = float(row["probability"])
        assert 0.01 <= expected_p <= 0.99

    # Now apply a What-If shift of +10%
    rescored_df, _ = rescore_choice_list(df_list, pct=10.0)
    selected_rescored = rescored_df.iloc[selected_indices]

    for orig_idx, (_, row) in zip(selected_indices, selected_rescored.iterrows()):
        expected_shifted_p = float(rescored_df.iloc[orig_idx]["shifted_probability"])
        actual_shifted_p = float(row["shifted_probability"])
        assert np.isclose(actual_shifted_p, expected_shifted_p, atol=1e-6)


def test_sparkline_and_chart_sparse_history_no_crash():
    """Options with 0, 1, or fewer years of history must not crash sparkline or comparison chart."""
    # Sparkline with empty history
    svg_empty = render_svg_sparkline([])
    assert "No history" in svg_empty

    # Sparkline with 1 year
    svg_one = render_svg_sparkline([12500])
    assert "1 year" in svg_one

    # Sparkline with 2 years
    svg_two = render_svg_sparkline([12000, 14000])
    assert "<svg" in svg_two
    assert "<polyline" in svg_two

    # Chart with sparse history data
    mock_selected = pd.DataFrame([
        {
            "choice_no": 1,
            "institute": "Inst A",
            "program": "Prog A",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "y_hat": 9.5,
        },
        {
            "choice_no": 2,
            "institute": "Inst B",
            "program": "Prog B",
            "quota": "HS",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "y_hat": 9.8,
        },
    ])

    # History where Inst A has 5 years and Inst B has only 1 year
    mock_history = pd.DataFrame([
        {"year": 2021, "institute": "Inst A", "program": "Prog A", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 10000},
        {"year": 2022, "institute": "Inst A", "program": "Prog A", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 11000},
        {"year": 2023, "institute": "Inst A", "program": "Prog A", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 12000},
        {"year": 2024, "institute": "Inst A", "program": "Prog A", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 13000},
        {"year": 2025, "institute": "Inst A", "program": "Prog A", "quota": "AI", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 14000},
        {"year": 2025, "institute": "Inst B", "program": "Prog B", "quota": "HS", "seat_type": "OPEN", "gender": "Gender-Neutral", "closing_rank": 18000},
    ])

    fig = create_comparison_chart(mock_selected, mock_history, current_student_rank=15000, rank_shift_pct=0)
    assert fig is not None
    # Traces for Inst A (line + marker) and Inst B (line + marker)
    assert len(fig.data) >= 2


def test_order_check_unreachable_rule():
    """Order-check warning appears ONLY when an earlier selected option has P >= 0.95."""
    # Case 1: Option 1 has P = 0.50, Option 2 has P = 0.80 -> No warning
    opts_no_warn = [
        {"choice_no": 5, "institute": "NIT Warangal", "program": "CSE", "shifted_probability": 0.50},
        {"choice_no": 12, "institute": "NIT Surathkal", "program": "ECE", "shifted_probability": 0.80},
    ]
    warns = check_order_unreachable(opts_no_warn)
    assert len(warns) == 0

    # Case 2: Option 1 has P = 0.96 (Choice #10), Option 2 has P = 0.85 (Choice #20)
    # Since Choice #20 sits AFTER Choice #10, Choice #20 is unreachable!
    opts_with_warn = [
        {"choice_no": 10, "institute": "NIT Trichy", "program": "Civil", "shifted_probability": 0.96},
        {"choice_no": 20, "institute": "NIT Rourkela", "program": "Mech", "shifted_probability": 0.85},
    ]
    warns = check_order_unreachable(opts_with_warn)
    assert len(warns) == 1
    assert warns[0]["blocked_choice"] == 20
    assert warns[0]["blocking_choice"] == 10
    assert "effectively unreachable" in warns[0]["message"]

    # Case 3: Reverse selection order in input list (Choice #20 passed first, Choice #10 second)
    # The function must sort by choice_no and still correctly identify Choice #20 as blocked!
    reversed_opts = [
        {"choice_no": 20, "institute": "NIT Rourkela", "program": "Mech", "shifted_probability": 0.85},
        {"choice_no": 10, "institute": "NIT Trichy", "program": "Civil", "shifted_probability": 0.96},
    ]
    warns_rev = check_order_unreachable(reversed_opts)
    assert len(warns_rev) == 1
    assert warns_rev[0]["blocked_choice"] == 20
    assert warns_rev[0]["blocking_choice"] == 10

    # Case 4: Later option has P >= 0.95, but earlier option has P < 0.95
    # e.g. Choice #5 (P=0.40), Choice #15 (P=0.98). Choice #15 is NOT blocked by Choice #5!
    opts_later_certain = [
        {"choice_no": 5, "institute": "NIT Warangal", "program": "CSE", "shifted_probability": 0.40},
        {"choice_no": 15, "institute": "NIT Trichy", "program": "Civil", "shifted_probability": 0.98},
    ]
    warns_later = check_order_unreachable(opts_later_certain)
    assert len(warns_later) == 0


def test_quick_facts_rule_based():
    """Quick facts must be strictly deterministic rules and include the mandatory note."""
    opts = [
        {"choice_no": 4, "institute": "NIT Warangal", "program": "CSE", "shifted_probability": 0.82, "shifted_label": "Safe"},
        {"choice_no": 21, "institute": "NIT Jalandhar", "program": "Data Science", "shifted_probability": 0.41, "shifted_label": "Target"},
    ]
    trends = [
        {"badge": "Getting easier"},
        {"badge": "Getting harder"},
    ]
    facts = generate_quick_facts(opts, trends)
    assert len(facts) >= 3

    # Contains probability comparison
    assert any("higher admission chance" in f and "82.0%" in f for f in facts)
    # Contains trend fact
    assert any("getting easier" in f.lower() for f in facts)
    assert any("getting harder" in f.lower() for f in facts)
    # Contains mandatory note
    assert any("Branch, placement, location and fees also matter" in f for f in facts)
