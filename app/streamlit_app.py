"""Vikalp: Counselling Choice-List Optimizer for JoSAA.

Author: Sahil Sharma
UI designed per design.md with Inter typography, clean metric cards,
quota-aware eligibility filtering, risk-balanced list generation, and history plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from src.simulation import simulate_at_least_one_seat_probability, get_model_sigma

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
def load_historical_data() -> pd.DataFrame:
    """Load cleaned historical parquet dataset for plotting."""
    parquet_path = ROOT_DIR / "data" / "processed" / "cutoffs.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


# Load historical data
history_df = load_historical_data()

# Available state & branch choices
ALL_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal"
]
ALL_BRANCHES = ["CSE", "ECE", "EE", "MECH", "CIVIL", "CHEM", "META", "ARCH", "BIO", "OTHER"]
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
        default=["IIT", "NIT", "IIIT"] if adv_rank_input > 0 else ["NIT", "IIIT", "GFTI"],
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

    generate_btn = st.button("🚀 Generate Choice List", type="primary", use_container_width=True)

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

# Generate choice list
try:
    with st.spinner("Analyzing cutoffs and optimizing choice list..."):
        df_list = build_list(student)
except Exception as e:
    st.error(f"Error generating choice list: {e}")
    st.stop()

if df_list.empty:
    st.warning("No options match your filters. Try adding more branches or institute types.")
    st.stop()

# Feature 1: Portfolio-level Monte Carlo Simulation
model_sigma = get_model_sigma()
p_at_least_one = simulate_at_least_one_seat_probability(
    student_ranks=df_list["student_rank"].values,
    y_hats=df_list["y_hat"].values,
    sigma=model_sigma,
    rho=0.5,
    n_sims=2000,
    seed=42,
)
p_at_least_one_pct = min(99, int(round(p_at_least_one * 100)))

# Metric summary cards
dream_count = int((df_list["label"] == "Dream").sum())
target_count = int((df_list["label"] == "Target").sum())
safe_count = int((df_list["label"] == "Safe").sum())
total_count = len(df_list)

col0, col1, col2, col3 = st.columns(4)
with col0:
    st.markdown(
        f"""
        <div class="glass-metric metric-highlight">
            <div class="metric-title" style="color: #0EA5E9;">Seat Admission Chance</div>
            <div class="metric-value" style="color: #38BDF8;">{p_at_least_one_pct}%</div>
            <div class="metric-desc">Monte Carlo estimate (2,000 runs, &rho;=0.5) of getting &ge;1 seat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col1:
    st.markdown(
        f"""
        <div class="glass-metric metric-dream">
            <div class="metric-title" style="color: #EF4444;">Dream Options</div>
            <div class="metric-value">{dream_count}</div>
            <div class="metric-desc">P &lt; 30% &middot; Aspirational reach</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="glass-metric metric-target">
            <div class="metric-title" style="color: #F59E0B;">Target Options</div>
            <div class="metric-value">{target_count}</div>
            <div class="metric-desc">30% &le; P &lt; 80% &middot; Competitive zone</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="glass-metric metric-safe">
            <div class="metric-title" style="color: #22C55E;">Safe Options</div>
            <div class="metric-value">{safe_count}</div>
            <div class="metric-desc">P &ge; 80% &middot; Reliable safety anchors</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Table display preparation
display_df = df_list.copy()

# Table header and action bar
tbl_col1, tbl_col2 = st.columns([3, 1])
with tbl_col1:
    st.markdown("### 📑 Optimized JoSAA Choice List")
with tbl_col2:
    csv_data = df_list.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Choice List (CSV)",
        data=csv_data,
        file_name=f"vikalp_choice_list_{student.category}_{student.main_rank or student.adv_rank}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Render interactive DataFrame
st.dataframe(
    display_df[[
        "choice_no",
        "institute",
        "program",
        "quota",
        "seat_type",
        "probability",
        "label",
        "predicted_range",
        "warning",
    ]],
    column_config={
        "choice_no": st.column_config.NumberColumn("#", format="%d", width="small"),
        "institute": st.column_config.TextColumn("Institute", width="large"),
        "program": st.column_config.TextColumn("Academic Program", width="large"),
        "quota": st.column_config.TextColumn("Quota", width="small"),
        "seat_type": st.column_config.TextColumn("Seat Type", width="small"),
        "probability": st.column_config.ProgressColumn(
            "Admit Chance",
            format="%.0f%%",
            min_value=0.0,
            max_value=1.0,
            width="medium",
        ),
        "label": st.column_config.TextColumn("Tier", width="small"),
        "predicted_range": st.column_config.TextColumn("Est. Closing Range (80%)", width="medium"),
        "warning": st.column_config.TextColumn("Notes / Warnings", width="large"),
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

# --- HISTORICAL TREND PLOTTER ---
st.markdown("---")
st.markdown("### 📈 Closing Rank History & Projected Range")

choice_options = [
    f"Choice #{r['choice_no']}: {r['institute']} - {r['program']} ({r['quota']}, {r['seat_type']})"
    for _, r in display_df.iterrows()
]

selected_choice_str = st.selectbox(
    "Select an option to inspect its 5-year closing rank trajectory:",
    options=choice_options,
    index=0,
)

if selected_choice_str:
    choice_idx = int(selected_choice_str.split(":")[0].replace("Choice #", "").strip()) - 1
    selected_row = display_df.iloc[choice_idx]

    # Query historical data
    inst_match = selected_row["institute"]
    prog_match = selected_row["program"]
    quota_match = selected_row["quota"]
    seat_match = selected_row["seat_type"]
    gender_match = selected_row["gender"]

    hist_match = history_df[
        (history_df["institute"] == inst_match)
        & (history_df["program"] == prog_match)
        & (history_df["quota"] == quota_match)
        & (history_df["seat_type"] == seat_match)
        & (history_df["gender"] == gender_match)
    ].sort_values("year")

    fig = go.Figure()

    if not hist_match.empty:
        # Historical line
        fig.add_trace(
            go.Scatter(
                x=hist_match["year"],
                y=hist_match["closing_rank"],
                mode="lines+markers",
                name="Historical Closing Rank",
                line=dict(color="#1F2A44", width=3),
                marker=dict(size=8, color="#F59E0B"),
            )
        )

    # Projected marker and range
    pred_val = selected_row["predicted_closing_rank"]
    range_parts = selected_row["predicted_range"].replace(",", "").split("-")
    if len(range_parts) == 2:
        low_val = float(range_parts[0].strip())
        high_val = float(range_parts[1].strip())

        # Predicted point
        fig.add_trace(
            go.Scatter(
                x=[2026],
                y=[pred_val],
                mode="markers",
                name="Projected Closing Rank",
                marker=dict(size=12, color="#DC2626", symbol="star"),
            )
        )

        # 80% error bar
        fig.add_trace(
            go.Scatter(
                x=[2026, 2026],
                y=[low_val, high_val],
                mode="lines",
                name="80% Prediction Range",
                line=dict(color="#DC2626", width=4, dash="dot"),
            )
        )

    # Student rank horizontal reference line
    student_rank_val = student.adv_rank if selected_row["institute_type"] == "IIT" else student.main_rank
    if selected_row["seat_type"] != "OPEN" and student.category_rank:
        student_rank_val = student.category_rank

    if student_rank_val:
        fig.add_hline(
            y=student_rank_val,
            line_dash="dash",
            line_color="#16A34A",
            annotation_text=f"Your Rank ({student_rank_val:,})",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"Cutoff Trend: {inst_match} · {prog_match}",
        xaxis_title="Counselling Year",
        yaxis_title="Closing Rank (Lower is Better)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Reverse y-axis because in competitive exams lower rank number is superior
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

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

st.markdown('<div class="footer-glass">Built by Sahil Sharma</div>', unsafe_allow_html=True)
