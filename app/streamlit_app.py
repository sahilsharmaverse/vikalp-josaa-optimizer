"""Vikalp: Counselling Choice-List Optimizer for JoSAA.

Author: Sahil Sharma
UI designed per design.md with Inter typography, clean metric cards,
quota-aware eligibility filtering, risk-balanced list generation, and history plots.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

# Add project root to sys.path to ensure modules import cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.eligibility import StudentProfile
from src.optimize import build_list
from src.simulation import (
    get_model_metrics,
    get_model_sigma,
    simulate_at_least_one_seat_probability,
)
from src.trend import (
    TREND_GETTING_EASIER,
    TREND_GETTING_HARDER,
    TREND_NOT_ENOUGH_DATA,
    TREND_STABLE,
    TREND_VOLATILE,
    compute_prediction_explanation,
    compute_trend,
    render_trend_badge_html,
)
from src.whatif import rescore_choice_list, shift_rank, shift_student_profile

# Page Configuration
st.set_page_config(
    page_title="Vikalp - JoSAA Choice-List Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto",
)

# Load External Stylesheet
css_path = ROOT_DIR / "app" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def get_optimized_choice_list(
    main_rank: Optional[int],
    adv_rank: Optional[int],
    category: str,
    category_rank: Optional[int],
    gender: str,
    home_state: str,
    preferred_branches: Optional[Tuple[str, ...]],
    institute_types: Optional[Tuple[str, ...]],
    list_size: int,
    priority_mode: str,
) -> pd.DataFrame:
    """Cached generator for baseline choice list to ensure sub-second What-If updates."""
    student_obj = StudentProfile(
        main_rank=main_rank,
        adv_rank=adv_rank,
        category=category,
        category_rank=category_rank,
        gender=gender,
        home_state=home_state,
        preferred_branches=list(preferred_branches) if preferred_branches else None,
        institute_types=list(institute_types) if institute_types else None,
        list_size=list_size,
        priority_mode=priority_mode,
    )
    return build_list(student_obj)


@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Load cleaned historical parquet dataset for plotting."""
    parquet_path = ROOT_DIR / "data" / "processed" / "cutoffs.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def build_plotly_trend_chart(
    selected_row: pd.Series,
    hist_match: pd.DataFrame,
    current_student_rank: int,
    rank_shift_pct: int,
    sigma: float,
) -> go.Figure:
    """Construct Plotly dark theme closing rank history chart with shaded 80% interval."""
    fig = go.Figure()

    # Historical closing ranks line
    if not hist_match.empty:
        hist_years = hist_match["year"].tolist()
        hist_ranks = hist_match["closing_rank"].tolist()

        fig.add_trace(
            go.Scatter(
                x=hist_years,
                y=hist_ranks,
                mode="lines+markers",
                name="Actual Closing Rank",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=8, color="#38BDF8", symbol="circle"),
                hovertemplate="Year: %{x}<br>Closing Rank: %{y:,}<extra></extra>",
            )
        )
        last_year = hist_years[-1]
        last_rank = hist_ranks[-1]
    else:
        last_year = 2025
        last_rank = int(round(np.exp(selected_row["y_hat"])))

    # Prediction for next cycle (2026)
    y_hat = float(selected_row["y_hat"])
    pred_val = float(np.exp(y_hat))
    low_80 = float(np.exp(y_hat - 1.28 * sigma))
    high_80 = float(np.exp(y_hat + 1.28 * sigma))

    # 80% Prediction interval as shaded band
    fig.add_trace(
        go.Scatter(
            x=[last_year, 2026, 2026, last_year],
            y=[last_rank, high_80, low_80, last_rank],
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.18)",
            line=dict(color="rgba(245, 158, 11, 0.4)", width=1, dash="dot"),
            name="80% Prediction Interval Band",
            hoverinfo="skip",
        )
    )

    # Error bar at 2026
    fig.add_trace(
        go.Scatter(
            x=[2026, 2026],
            y=[low_80, high_80],
            mode="lines",
            line=dict(color="#F59E0B", width=4),
            name="80% Prediction Range",
            hovertemplate=f"80% Interval: {int(round(low_80)):,} – {int(round(high_80)):,}<extra></extra>",
            showlegend=False,
        )
    )

    # Predicted closing rank marker
    fig.add_trace(
        go.Scatter(
            x=[2026],
            y=[pred_val],
            mode="markers",
            name=f"Predicted Closing Rank ({int(round(pred_val)):,})",
            marker=dict(size=12, color="#EF4444", symbol="star"),
            hovertemplate=f"2026 Projected: {int(round(pred_val)):,}<extra></extra>",
        )
    )

    # Student's rank horizontal reference line
    annot_text = (
        f"Your Simulated Rank: {current_student_rank:,}"
        if rank_shift_pct != 0
        else f"Your Rank: {current_student_rank:,}"
    )
    fig.add_hline(
        y=current_student_rank,
        line_dash="dash",
        line_color="#0EA5E9",
        line_width=2.5,
        annotation_text=annot_text,
        annotation_position="bottom right",
        annotation_font=dict(color="#0EA5E9", size=12, family="Inter"),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        height=300,
        margin=dict(l=35, r=30, t=25, b=35),
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
            font=dict(size=10.5),
        ),
    )
    return fig


@st.dialog("Why this probability? · Deep Dive", width="large")
def show_why_dialog(
    selected_row: pd.Series,
    history_df: pd.DataFrame,
    rank_shift_pct: int,
    model_metrics: dict,
    all_display_df: pd.DataFrame,
):
    """Interactive modal dialog displaying mathematical breakdown, 5-year trend, and historical plot."""
    choice_names = [
        f"Choice #{r['choice_no']}: {r['institute']} — {r['program']} ({r['quota']}, {r['seat_type']})"
        for _, r in all_display_df.iterrows()
    ]
    current_choice_no = int(selected_row["choice_no"])
    current_idx = current_choice_no - 1

    selected_choice_idx = st.selectbox(
        "Option being inspected:",
        options=range(len(all_display_df)),
        index=current_idx if 0 <= current_idx < len(all_display_df) else 0,
        format_func=lambda i: choice_names[i],
        key=f"why_dialog_selector_{current_choice_no}",
    )
    row = all_display_df.iloc[selected_choice_idx]

    current_student_rank = int(row["shifted_rank"])
    sigma = float(model_metrics.get("sigma", 0.2285))

    # Option Header & Meta Chips
    tier_color = (
        "#EF4444"
        if row["shifted_label"] == "Dream"
        else "#F59E0B"
        if row["shifted_label"] == "Target"
        else "#22C55E"
    )
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div style="font-size: 19px; font-weight: 800; color: #FFFFFF;">
                Choice #{row['choice_no']}: {row['institute']}
            </div>
            <div style="font-size: 15px; color: #38BDF8; font-weight: 600; margin-top: 2px;">
                {row['program']}
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; align-items: center;">
                <span class="model-chip">🏛️ Quota: <strong>{row['quota']}</strong></span>
                <span class="model-chip">💺 Seat: <strong>{row['seat_type']}</strong></span>
                <span class="model-chip">👥 Pool: <strong>{row['gender']}</strong></span>
                <span style="background: {tier_color}22; border: 1px solid {tier_color}66; color: {tier_color}; padding: 3px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700;">
                    {row['shifted_label']} ({row['shifted_probability']:.1%})
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CURRENT rank callout
    rank_callout = (
        f"Using your simulated rank: {current_student_rank:,}"
        if rank_shift_pct != 0
        else f"Using your rank: {current_student_rank:,}"
    )
    st.markdown(
        f"""
        <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 8px; padding: 8px 14px; margin-bottom: 14px; font-size: 13.5px; color: #38BDF8; font-weight: 600;">
            📊 <strong>{rank_callout}</strong>
            {' &nbsp;·&nbsp; <span style="color: #94A3B8; font-weight: 400;">(Shifted by ' + f"{rank_shift_pct:+d}%" + f" from {row['student_rank']:,})</span>" if rank_shift_pct != 0 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Trend Badge & Historical Query
    hist_match = history_df[
        (history_df["institute"] == row["institute"])
        & (history_df["program"] == row["program"])
        & (history_df["quota"] == row["quota"])
        & (history_df["seat_type"] == row["seat_type"])
        & (history_df["gender"] == row["gender"])
    ].sort_values("year")

    hist_ranks = hist_match["closing_rank"].tolist()
    trend_res = compute_trend(hist_ranks)
    trend_badge_html = render_trend_badge_html(trend_res["badge"], trend_res["color"])

    trend_desc = {
        TREND_GETTING_HARDER: "Closing ranks are falling year-on-year (competition rising / cutoffs tightening).",
        TREND_GETTING_EASIER: "Closing ranks are rising year-on-year (competition easing / cutoffs relaxing).",
        TREND_VOLATILE: "Closing ranks show wide year-on-year fluctuations (std(d) > 0.20 in log-space).",
        TREND_NOT_ENOUGH_DATA: "Fewer than 3 years of historical cutoffs are available in JoSAA records.",
        TREND_STABLE: "Closing ranks have remained relatively stable over the observed cycles.",
    }.get(trend_res["badge"], "")

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
            <span style="font-size: 14px; font-weight: 700; color: #E2E8F0;">5-Year Cutoff Trend:</span>
            {trend_badge_html}
            <span style="font-size: 12.5px; color: #94A3B8;">{trend_desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chart (Plotly, dark theme, log-scale, thousands separators, shaded band, no unlabeled colors)
    fig = build_plotly_trend_chart(
        row, hist_match, current_student_rank, rank_shift_pct, sigma
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plain Words & Mathematical Breakdown
    explanation = compute_prediction_explanation(
        student_rank=current_student_rank,
        y_hat=float(row["y_hat"]),
        sigma=sigma,
        metrics=model_metrics,
    )

    # Summary Metrics Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="glass-metric" style="padding: 12px 14px;">
                <div class="metric-title" style="color: #38BDF8;">Your Active Rank</div>
                <div style="font-size: 22px; font-weight: 800; color: #FFFFFF;">{current_student_rank:,}</div>
                <div class="metric-desc">{'Simulated rank' if rank_shift_pct != 0 else 'Entered rank'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="glass-metric" style="padding: 12px 14px;">
                <div class="metric-title" style="color: #F59E0B;">Predicted Closing Rank</div>
                <div style="font-size: 22px; font-weight: 800; color: #FFFFFF;">{explanation['pred_rank']:,}</div>
                <div class="metric-desc">80% Interval: {explanation['low_80']:,} – {explanation['high_80']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="glass-metric" style="padding: 12px 14px;">
                <div class="metric-title" style="color: {tier_color};">Admit Probability</div>
                <div style="font-size: 22px; font-weight: 800; color: #FFFFFF;">{explanation['prob_pct']:.1f}%</div>
                <div class="metric-desc">Tier: <strong>{row['shifted_label']}</strong> &middot; z = {explanation['z_score']:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed plain-words walkthrough
    st.markdown(
        f"""
        <div class="formula-box">
            <div style="font-weight: 700; color: #38BDF8; margin-bottom: 6px;">📐 Plain-Words Prediction Breakdown</div>
            <div>• <strong>Cutoff & Range:</strong> Predicted closing rank is <strong>{explanation['pred_rank']:,}</strong> with an 80% prediction interval of <strong>{explanation['low_80']:,}</strong> to <strong>{explanation['high_80']:,}</strong>.</div>
            <div>• <strong>Student Rank Position:</strong> {explanation['diff_text']}</div>
            <div>• <strong>Standardized Distance (z-score):</strong> <code>z = (ln(rank) - y&#770;) / &sigma; = ({np.log(current_student_rank):.4f} - {row['y_hat']:.4f}) / {sigma:.4f} = {explanation['z_score']:+.3f}</code>.</div>
            <div>• <strong>Resulting Probability:</strong> <code>P(admit) = 1 - &Phi;((ln(rank) - y&#770;) / &sigma;) = &Phi;(-z) = &Phi;({-explanation['z_score']:+.3f}) = {explanation['prob_pct']:.1f}%</code>.</div>
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1); color: #CBD5E1;">
                • <strong>Model & Error:</strong> Typical error is <strong>MAE = {explanation['mae']:.4f}</strong> in log-rank space, predicted using the <strong>{explanation['model_name']}</strong>.
            </div>
            <div style="margin-top: 4px; color: #94A3B8; font-size: 12.5px;">
                • <strong>Category & Quota:</strong> This cutoff belongs to <strong>{row['quota']}</strong> quota and <strong>{row['seat_type']}</strong> seat type ({row['gender']}).
            </div>
            <div style="margin-top: 6px; font-size: 12px; color: #F59E0B;">
                ⚠️ <strong>Note:</strong> Cutoffs change with seats added, JEE difficulty and policy. Verify at <a href="https://josaa.nic.in" target="_blank" style="color: #0EA5E9; text-decoration: underline;">josaa.nic.in</a>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load historical data
history_df = load_historical_data()
model_metrics = get_model_metrics()
model_sigma = float(model_metrics.get("sigma", 0.2285))

# Available state & branch choices
ALL_STATES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu and Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]
ALL_BRANCHES = [
    "CSE",
    "ECE",
    "EE",
    "MECH",
    "CIVIL",
    "CHEM",
    "META",
    "ARCH",
    "BIO",
    "OTHER",
]
ALL_INST_TYPES = ["IIT", "NIT", "IIIT", "GFTI"]

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.markdown("### 📋 Student Information")

    main_rank_input = st.number_input(
        "JEE Main CRL Rank (NIT/IIIT/GFTI)",
        value=15000,
        step=100,
        help="Common Rank List (CRL) rank in JEE Main.",
    )

    adv_rank_input = st.number_input(
        "JEE Advanced Rank (IITs - Optional)",
        value=0,
        step=50,
        help="Leave 0 if not qualified or not applying to IITs.",
    )

    category = st.selectbox(
        "Category",
        options=["OPEN", "EWS", "OBC-NCL", "SC", "ST"],
        index=0,
        help="Seat allocation category.",
    )

    category_rank_input = 0
    if category != "OPEN":
        category_rank_input = st.number_input(
            f"{category} Category Rank",
            value=3500,
            step=50,
            help=f"Your category-specific rank for {category} seats.",
        )

    gender = st.selectbox(
        "Gender",
        options=["Gender-Neutral", "Female-only"],
        index=0,
        help="Female candidates are eligible for both Gender-Neutral and Female-only supernumerary pools.",
    )

    home_state = st.selectbox(
        "Home State",
        options=ALL_STATES,
        index=ALL_STATES.index("Maharashtra") if "Maharashtra" in ALL_STATES else 0,
        help="State of 12th schooling (determines Home State quota in NITs).",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Preferences & Filters")

    priority_mode = st.selectbox(
        "Priority Mode",
        options=["College-first", "Branch-first"],
        index=0,
        help="College-first prioritizes institutional prestige; Branch-first prioritizes your preferred branch hierarchy.",
    )

    preferred_branches = st.multiselect(
        "Preferred Branches",
        options=ALL_BRANCHES,
        default=["CSE", "ECE"],
        help="Leave empty to consider all branches.",
    )

    institute_types = st.multiselect(
        "Institute Types",
        options=ALL_INST_TYPES,
        default=["IIT", "NIT", "IIIT"]
        if adv_rank_input > 0
        else ["NIT", "IIIT", "GFTI"],
        help="Select which institutional systems to include.",
    )

    list_size = st.slider(
        "Choice List Size",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        help="Number of choices to generate (balanced across Dream / Target / Safe).",
    )

    generate_btn = st.button(
        "🚀 Generate Choice List", type="primary", use_container_width=True
    )

# --- HERO SECTION ---
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-headline">Your rank. Your seats. Your smartest list.</div>
        <div class="hero-subhead">Data-driven JoSAA choice optimization powered by 5 years of cutoffs, quota logic, and uncertainty modeling.</div>
        <div class="hero-chip-row">
            <span class="hero-chip">🎯 AI / HS / OS Quota Aware</span>
            <span class="hero-chip">⚖️ 20 / 50 / 30 Strategic Mix</span>
            <span class="hero-chip">📊 80% Prediction Intervals</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Validate ranks
if (
    (main_rank_input is None or main_rank_input <= 0)
    and (adv_rank_input is None or adv_rank_input <= 0)
) or (main_rank_input is not None and main_rank_input < 0) or (adv_rank_input is not None and adv_rank_input < 0) or (category_rank_input is not None and category_rank_input < 0):
    st.error("Please enter a valid rank (a positive whole number).")
    st.stop()

# Build student profile
student = StudentProfile(
    main_rank=int(main_rank_input) if main_rank_input > 0 else None,
    adv_rank=int(adv_rank_input) if adv_rank_input > 0 else None,
    category=category,
    category_rank=int(category_rank_input) if category_rank_input > 0 else None,
    gender=gender,
    home_state=home_state,
    preferred_branches=preferred_branches if preferred_branches else None,
    institute_types=institute_types if institute_types else None,
    list_size=int(list_size),
    priority_mode=priority_mode,
)

# Check if a rebuild rank shift was requested in session state
if "rebuild_shift_pct" not in st.session_state:
    st.session_state["rebuild_shift_pct"] = 0

active_student = student
if st.session_state["rebuild_shift_pct"] != 0:
    active_student = shift_student_profile(
        student, float(st.session_state["rebuild_shift_pct"])
    )

# Generate choice list (cached for instant What-If slider responsiveness)
try:
    with st.spinner("Analyzing cutoffs and optimizing choice list..."):
        pref_branches_tuple = (
            tuple(active_student.preferred_branches)
            if active_student.preferred_branches
            else None
        )
        inst_types_tuple = (
            tuple(active_student.institute_types)
            if active_student.institute_types
            else None
        )
        df_list = get_optimized_choice_list(
            main_rank=active_student.main_rank,
            adv_rank=active_student.adv_rank,
            category=active_student.category,
            category_rank=active_student.category_rank,
            gender=active_student.gender,
            home_state=active_student.home_state,
            preferred_branches=pref_branches_tuple,
            institute_types=inst_types_tuple,
            list_size=active_student.list_size,
            priority_mode=active_student.priority_mode,
        )
except Exception as e:
    st.error(f"Error generating choice list: {e}")
    st.stop()

if df_list.empty:
    st.warning(
        "No options match your filters. Try adding more branches or institute types."
    )
    st.stop()

# Feature 1 Baseline Simulation (Quick fix a: 99.9% cap, 1 decimal, fixed seed)
base_p_at_least_one = simulate_at_least_one_seat_probability(
    student_ranks=df_list["student_rank"].values,
    y_hats=df_list["y_hat"].values,
    sigma=model_sigma,
    rho=0.5,
    n_sims=2000,
    seed=42,
)
base_p_pct = min(99.9, round(base_p_at_least_one * 100.0, 1))

base_dream_count = int((df_list["label"] == "Dream").sum())
base_target_count = int((df_list["label"] == "Target").sum())
base_safe_count = int((df_list["label"] == "Safe").sum())

# --- FEATURE 2: WHAT-IF RANK SENSITIVITY TOOLBAR ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; margin-bottom: 4px;">
        <span style="font-size: 15px; font-weight: 700; color: #38BDF8;">🎚️ What-If Rank Sensitivity Analysis</span>
        <span style="font-size: 13px; color: #94A3B8;">Simulate cutoff & tier shifts (±20%)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

wi_col1, wi_col2 = st.columns([3, 1])

default_slider_val = 0
if "shift" in st.query_params:
    try:
        parsed_shift = int(st.query_params["shift"])
        if parsed_shift in [-20, -15, -10, -5, 0, 5, 10, 15, 20]:
            default_slider_val = parsed_shift
    except Exception:
        default_slider_val = 0

with wi_col1:
    rank_shift_pct = st.slider(
        "Rank Shift Percentage",
        min_value=-20,
        max_value=20,
        value=default_slider_val,
        step=5,
        format="%+d%%",
        help="Simulate how cutoffs and admission chances shift if your rank were higher (worse) or lower (better). +10% means 10% higher rank number.",
        label_visibility="collapsed",
    )

with wi_col2:
    rebuild_btn = st.button("🔄 Rebuild list at this rank", use_container_width=True)

if rebuild_btn:
    st.session_state["rebuild_shift_pct"] = int(rank_shift_pct)
    st.rerun()

if st.session_state["rebuild_shift_pct"] != 0:
    st.info(
        f"⚡ Choice list is currently rebuilt at **{st.session_state['rebuild_shift_pct']:+d}%** rank shift. "
        "Move slider to explore or adjust sidebar to reset."
    )

# Compute shifted rank numbers for display
shifted_main = (
    shift_rank(active_student.main_rank, float(rank_shift_pct))
    if active_student.main_rank
    else None
)
shifted_adv = (
    shift_rank(active_student.adv_rank, float(rank_shift_pct))
    if active_student.adv_rank
    else None
)
shifted_cat = (
    shift_rank(active_student.category_rank, float(rank_shift_pct))
    if active_student.category_rank
    else None
)

rank_badge_parts = []
if shifted_main:
    diff_text = f" ({rank_shift_pct:+d}%)" if rank_shift_pct != 0 else ""
    orig_text = f" from {active_student.main_rank:,}" if rank_shift_pct != 0 else ""
    rank_badge_parts.append(
        f"JEE Main CRL: <span class='rank-badge'>{shifted_main:,}</span>{orig_text}{diff_text}"
    )
if shifted_adv:
    diff_text = f" ({rank_shift_pct:+d}%)" if rank_shift_pct != 0 else ""
    orig_text = f" from {active_student.adv_rank:,}" if rank_shift_pct != 0 else ""
    rank_badge_parts.append(
        f"JEE Adv CRL: <span class='rank-badge'>{shifted_adv:,}</span>{orig_text}{diff_text}"
    )
if shifted_cat:
    diff_text = f" ({rank_shift_pct:+d}%)" if rank_shift_pct != 0 else ""
    orig_text = (
        f" from {active_student.category_rank:,}" if rank_shift_pct != 0 else ""
    )
    rank_badge_parts.append(
        f"{active_student.category} Rank: <span class='rank-badge'>{shifted_cat:,}</span>{orig_text}{diff_text}"
    )

badges_html = " &nbsp;|&nbsp; ".join(rank_badge_parts)
st.markdown(
    f"""
    <div style="background: rgba(14, 165, 233, 0.05); border: 1px solid rgba(14, 165, 233, 0.2); border-radius: 10px; padding: 10px 14px; margin-top: 4px; margin-bottom: 20px; font-size: 13px; color: #94A3B8;">
        Simulated Rank: {badges_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# Re-score CURRENT list with shifted rank (in-place, strictly preserving choice ordering)
rescored_df, shifted_p_at_least_one = rescore_choice_list(
    df_list,
    pct=float(rank_shift_pct),
    sigma=model_sigma,
    rho=0.5,
    n_sims=2000,
    seed=42,
)
shifted_p_pct = min(99.9, round(shifted_p_at_least_one * 100.0, 1))

# New tier counts
new_dream_count = int((rescored_df["shifted_label"] == "Dream").sum())
new_target_count = int((rescored_df["shifted_label"] == "Target").sum())
new_safe_count = int((rescored_df["shifted_label"] == "Safe").sum())

# Metric summary cards (Quick fix a: one decimal, Quick fix b: count diff e.g. "10 to 25, +15")
col0, col1, col2, col3 = st.columns(4)
with col0:
    if rank_shift_pct == 0:
        val_display = f"{base_p_pct:.1f}%"
        sub_desc = (
            "Monte Carlo estimate (2,000 runs, &rho;=0.5) of getting &ge;1 seat"
        )
    else:
        diff_p = shifted_p_pct - base_p_pct
        val_display = f"{base_p_pct:.1f}% → {shifted_p_pct:.1f}%"
        sub_desc = f"Simulated shift ({diff_p:+.1f}%) with rank shifted by {rank_shift_pct:+d}%"

    st.markdown(
        f"""
        <div class="glass-metric metric-highlight">
            <div class="metric-title" style="color: #0EA5E9;">Seat Admission Chance</div>
            <div class="metric-value" style="color: #38BDF8; font-size: {'24px' if rank_shift_pct != 0 else '32px'};">{val_display}</div>
            <div class="metric-desc">{sub_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col1:
    diff_dream = new_dream_count - base_dream_count
    dream_display = (
        f"{base_dream_count}"
        if rank_shift_pct == 0
        else f"{base_dream_count} to {new_dream_count}, {diff_dream:+d}"
    )
    st.markdown(
        f"""
        <div class="glass-metric metric-dream">
            <div class="metric-title" style="color: #EF4444;">Dream Options</div>
            <div class="metric-value" style="font-size: {'22px' if rank_shift_pct != 0 else '32px'};">{dream_display}</div>
            <div class="metric-desc">P &lt; 30% &middot; Aspirational reach</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    diff_target = new_target_count - base_target_count
    target_display = (
        f"{base_target_count}"
        if rank_shift_pct == 0
        else f"{base_target_count} to {new_target_count}, {diff_target:+d}"
    )
    st.markdown(
        f"""
        <div class="glass-metric metric-target">
            <div class="metric-title" style="color: #F59E0B;">Target Options</div>
            <div class="metric-value" style="font-size: {'22px' if rank_shift_pct != 0 else '32px'};">{target_display}</div>
            <div class="metric-desc">30% &le; P &lt; 80% &middot; Competitive zone</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    diff_safe = new_safe_count - base_safe_count
    safe_display = (
        f"{base_safe_count}"
        if rank_shift_pct == 0
        else f"{base_safe_count} to {new_safe_count}, {diff_safe:+d}"
    )
    st.markdown(
        f"""
        <div class="glass-metric metric-safe">
            <div class="metric-title" style="color: #22C55E;">Safe Options</div>
            <div class="metric-value" style="font-size: {'22px' if rank_shift_pct != 0 else '32px'};">{safe_display}</div>
            <div class="metric-desc">P &ge; 80% &middot; Reliable safety anchors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Table display preparation
display_df = rescored_df.copy()

# Tier change impact callout if any options changed tier
changed_count = int((rescored_df["tier_change"] != "").sum())
if rank_shift_pct != 0 and changed_count > 0:
    st.info(
        f"⚡ **What-If Impact ({rank_shift_pct:+d}% shift):** **{changed_count} options** changed tier! "
        f"(Dream: {base_dream_count} → {new_dream_count} | Target: {base_target_count} → {new_target_count} | Safe: {base_safe_count} → {new_safe_count}). "
        "Check the 'Shift Marker' column in the table below to see which options moved."
    )

# Table header and action bar
tbl_col1, tbl_col2 = st.columns([3, 1])
with tbl_col1:
    st.markdown("### 📑 Optimized JoSAA Choice List")
with tbl_col2:
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Choice List (CSV)",
        data=csv_data,
        file_name=f"vikalp_choice_list_{active_student.category}_{active_student.main_rank or active_student.adv_rank}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Render interactive DataFrame with shifted probability, tier, and tier change markers
st.dataframe(
    display_df[[
        "choice_no",
        "institute",
        "program",
        "quota",
        "seat_type",
        "shifted_probability",
        "shifted_label",
        "tier_change",
        "predicted_range",
        "warning",
    ]],
    column_config={
        "choice_no": st.column_config.NumberColumn("#", format="%d", width="small"),
        "institute": st.column_config.TextColumn("Institute", width="medium"),
        "program": st.column_config.TextColumn("Academic Program", width="medium"),
        "quota": st.column_config.TextColumn("Quota", width="small"),
        "seat_type": st.column_config.TextColumn("Seat", width="small"),
        "shifted_probability": st.column_config.ProgressColumn(
            "Admit Chance",
            format="%.0f%%",
            min_value=0.0,
            max_value=1.0,
            width="small",
        ),
        "shifted_label": st.column_config.TextColumn("Tier", width="small"),
        "tier_change": st.column_config.TextColumn("Shift Marker", width="medium"),
        "predicted_range": st.column_config.TextColumn(
            "Est. Range (80%)", width="small"
        ),
        "warning": st.column_config.TextColumn("Notes / Warnings", width="small"),
    },
    hide_index=True,
    use_container_width=True,
)

# Unreachable Options Banner if any
unreachable_rows = display_df[display_df["warning"] != ""]
if not unreachable_rows.empty:
    first_warn_choice = unreachable_rows.iloc[0]["choice_no"]
    st.warning(
        f"⚠️ **Strategic Note:** Choices starting from #{first_warn_choice} are placed after a choice with ≥95% admission chance. "
        "Because JoSAA allots the first choice you qualify for, choices placed below a near-certain option are effectively unreachable. "
        "Review your ordering if you prefer any of those options!"
    )

# --- FEATURE 3: "WHY THIS PROBABILITY?" DEEP-DIVE LAUNCHER & PANEL ---
st.markdown("---")
st.markdown("### 🔬 Inspect Option: 'Why this probability?' Deep Dive")

choice_options_list = [
    f"Choice #{r['choice_no']}: {r['institute']} — {r['program']} ({r['quota']}, {r['seat_type']}) [{r['shifted_label']}]"
    for _, r in display_df.iterrows()
]

# Check default selection from query param
default_dd_idx = 0
if "why" in st.query_params:
    try:
        param_choice_no = int(st.query_params["why"])
        if 1 <= param_choice_no <= len(display_df):
            default_dd_idx = param_choice_no - 1
    except Exception:
        default_dd_idx = 0

dd_col1, dd_col2 = st.columns([3, 1])
with dd_col1:
    selected_deepdive_idx = st.selectbox(
        "Select an option to inspect its 5-year trend, statistical model breakdown, and prediction intervals:",
        options=range(len(display_df)),
        index=default_dd_idx,
        format_func=lambda i: choice_options_list[i],
        key="main_why_selector",
    )

with dd_col2:
    st.write("")
    st.write("")
    open_why_btn = st.button(
        "🔍 Open Deep Dive Modal", type="primary", use_container_width=True
    )

# Open st.dialog when button clicked or when query param ?open_dialog=1
if open_why_btn or st.query_params.get("open_dialog") == "1":
    show_why_dialog(
        selected_row=display_df.iloc[selected_deepdive_idx],
        history_df=history_df,
        rank_shift_pct=int(rank_shift_pct),
        model_metrics=model_metrics,
        all_display_df=display_df,
    )

# Also render the full Deep Dive panel inline for seamless accessibility
selected_row = display_df.iloc[selected_deepdive_idx]
current_student_rank = int(selected_row["shifted_rank"])

# Query historical closing ranks
hist_match = history_df[
    (history_df["institute"] == selected_row["institute"])
    & (history_df["program"] == selected_row["program"])
    & (history_df["quota"] == selected_row["quota"])
    & (history_df["seat_type"] == selected_row["seat_type"])
    & (history_df["gender"] == selected_row["gender"])
].sort_values("year")

hist_ranks = hist_match["closing_rank"].tolist()
trend_res = compute_trend(hist_ranks)
trend_badge_html = render_trend_badge_html(
    trend_res["badge"], trend_res["color"]
)

trend_desc = {
    TREND_GETTING_HARDER: "Closing ranks are falling year-on-year (competition rising / cutoffs tightening).",
    TREND_GETTING_EASIER: "Closing ranks are rising year-on-year (competition easing / cutoffs relaxing).",
    TREND_VOLATILE: "Closing ranks show wide year-on-year fluctuations (std(d) > 0.20 in log-space).",
    TREND_NOT_ENOUGH_DATA: "Fewer than 3 years of historical cutoffs are available in JoSAA records.",
    TREND_STABLE: "Closing ranks have remained relatively stable over the observed cycles.",
}.get(trend_res["badge"], "")

rank_callout = (
    f"Using your simulated rank: {current_student_rank:,}"
    if rank_shift_pct != 0
    else f"Using your rank: {current_student_rank:,}"
)

st.markdown(
    f"""
    <div class="deep-dive-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
            <div>
                <span style="font-size: 16px; font-weight: 800; color: #FFFFFF;">Choice #{selected_row['choice_no']}: {selected_row['institute']}</span>
                <div style="font-size: 14px; color: #38BDF8; font-weight: 600;">{selected_row['program']}</div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="model-chip">🏛️ {selected_row['quota']}</span>
                <span class="model-chip">💺 {selected_row['seat_type']}</span>
                <span class="model-chip">👥 {selected_row['gender']}</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px; margin-top: 6px; margin-bottom: 10px; flex-wrap: wrap;">
            <span style="font-size: 13.5px; font-weight: 700; color: #38BDF8;">📊 {rank_callout}</span>
            <span style="color: #64748B;">|</span>
            <span style="font-size: 13px; font-weight: 600; color: #E2E8F0;">5-Year Trajectory:</span>
            {trend_badge_html}
            <span style="font-size: 12.5px; color: #94A3B8;">({trend_desc})</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Plotly historical & projected trajectory chart
fig = build_plotly_trend_chart(
    selected_row, hist_match, current_student_rank, rank_shift_pct, model_sigma
)
st.plotly_chart(fig, use_container_width=True)

# Explanation in plain words
explanation = compute_prediction_explanation(
    student_rank=current_student_rank,
    y_hat=float(selected_row["y_hat"]),
    sigma=model_sigma,
    metrics=model_metrics,
)

st.markdown(
    f"""
    <div class="formula-box">
        <div style="font-weight: 700; color: #38BDF8; margin-bottom: 6px;">📐 Plain-Words Prediction Breakdown</div>
        <div>• <strong>Cutoff & Range:</strong> Predicted closing rank is <strong>{explanation['pred_rank']:,}</strong> with an 80% prediction interval of <strong>{explanation['low_80']:,}</strong> to <strong>{explanation['high_80']:,}</strong>.</div>
        <div>• <strong>Student Rank Position:</strong> {explanation['diff_text']}</div>
        <div>• <strong>Standardized Distance (z-score):</strong> <code>z = (ln(rank) - y&#770;) / &sigma; = ({np.log(current_student_rank):.4f} - {selected_row['y_hat']:.4f}) / {model_sigma:.4f} = {explanation['z_score']:+.3f}</code>.</div>
        <div>• <strong>Resulting Probability:</strong> <code>P(admit) = 1 - &Phi;((ln(rank) - y&#770;) / &sigma;) = &Phi;(-z) = &Phi;({-explanation['z_score']:+.3f}) = {explanation['prob_pct']:.1f}%</code>.</div>
        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.1); color: #CBD5E1;">
            • <strong>Model & Error:</strong> Typical error is <strong>MAE = {explanation['mae']:.4f}</strong> in log-rank space, predicted using the <strong>{explanation['model_name']}</strong>.
        </div>
        <div style="margin-top: 4px; color: #94A3B8; font-size: 12.5px;">
            • <strong>Category & Quota:</strong> This cutoff belongs to <strong>{selected_row['quota']}</strong> quota and <strong>{selected_row['seat_type']}</strong> seat type ({selected_row['gender']}).
        </div>
        <div style="margin-top: 6px; font-size: 12px; color: #F59E0B;">
            ⚠️ <strong>Note:</strong> Cutoffs change with seats added, JEE difficulty and policy. Verify at <a href="https://josaa.nic.in" target="_blank" style="color: #0EA5E9; text-decoration: underline;">josaa.nic.in</a>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Disclaimer and Methodology Expander
st.markdown(
    """
    <div class="disclaimer-glass">
        <strong>⚠️ Disclaimer:</strong> These are statistical estimates based on past JoSAA data, not official predictions or guarantees.
        Seat matrix variations, candidate preference shifts, and exam difficulty fluctuations affect actual allotments. Always verify at 
        <a href="https://josaa.nic.in" target="_blank" style="color: #0EA5E9; text-decoration: underline;">josaa.nic.in</a>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How Vikalp Works"):
    st.markdown(
        """
        - **Quota & Category Precision**: Evaluates All India (AI), Home State (HS), and Other State (OS) quotas accurately based on your 12th schooling state. OPEN seats are evaluated with CRL, while category seats use category ranks.
        - **Uncertainty Quantification**: Calculates $P(\\text{admit}) = 1 - \\Phi\\left(\\frac{\\ln(\\text{rank}) - \\hat{y}}{\\sigma}\\right)$ where $\\hat{y}$ is the weighted trend of past cutoffs and $\\sigma$ is the residual standard error read directly from calibrated model metrics.
        - **Strategic Risk Balance**: Constructs a balanced list with **20% Dream** ($P < 30\\%$), **50% Target** ($30\\% \\le P < 80\\%$), and **30% Safe** ($P \\ge 80\\%$) options.
        - **Portfolio Admission Chance**: Estimates the probability of securing at least one seat across your entire list via a 2,000-run Monte Carlo simulation modeling correlated exam/counselling shocks ($\rho = 0.5$).
        - **Unreachable Choice Detection**: Highlights choices ordered after a near-certain option ($P \\ge 95\\%$) to protect you from wasted choices.
        """
    )

st.markdown(
    '<div class="footer-glass">Built by Sahil Sharma</div>',
    unsafe_allow_html=True,
)
