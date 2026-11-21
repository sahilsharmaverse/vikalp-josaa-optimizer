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
from src.whatif import shift_rank, shift_student_profile, rescore_choice_list

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

# Check if a rebuild rank shift was requested in session state
if "rebuild_shift_pct" not in st.session_state:
    st.session_state["rebuild_shift_pct"] = 0

active_student = student
if st.session_state["rebuild_shift_pct"] != 0:
    active_student = shift_student_profile(student, float(st.session_state["rebuild_shift_pct"]))

# Generate choice list (cached for instant What-If slider responsiveness)
try:
    with st.spinner("Analyzing cutoffs and optimizing choice list..."):
        pref_branches_tuple = tuple(active_student.preferred_branches) if active_student.preferred_branches else None
        inst_types_tuple = tuple(active_student.institute_types) if active_student.institute_types else None
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
    st.warning("No options match your filters. Try adding more branches or institute types.")
    st.stop()

# Feature 1 Baseline Simulation
model_sigma = get_model_sigma()
base_p_at_least_one = simulate_at_least_one_seat_probability(
    student_ranks=df_list["student_rank"].values,
    y_hats=df_list["y_hat"].values,
    sigma=model_sigma,
    rho=0.5,
    n_sims=2000,
    seed=42,
)
base_p_pct = min(99, int(round(base_p_at_least_one * 100)))

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
shifted_main = shift_rank(active_student.main_rank, float(rank_shift_pct)) if active_student.main_rank else None
shifted_adv = shift_rank(active_student.adv_rank, float(rank_shift_pct)) if active_student.adv_rank else None
shifted_cat = shift_rank(active_student.category_rank, float(rank_shift_pct)) if active_student.category_rank else None

rank_badge_parts = []
if shifted_main:
    diff_text = f" ({rank_shift_pct:+d}%)" if rank_shift_pct != 0 else ""
    orig_text = f" from {active_student.main_rank:,}" if rank_shift_pct != 0 else ""
    rank_badge_parts.append(f"JEE Main CRL: <span class='rank-badge'>{shifted_main:,}</span>{orig_text}{diff_text}")
if shifted_adv:
    diff_text = f" ({rank_shift_pct:+d}%)" if rank_shift_pct != 0 else ""
    orig_text = f" from {active_student.adv_rank:,}" if rank_shift_pct != 0 else ""
    rank_badge_parts.append(f"JEE Adv CRL: <span class='rank-badge'>{shifted_adv:,}</span>{orig_text}{diff_text}")
if shifted_cat:
    diff_text = f" ({rank_shift_pct:+d}%)" if rank_shift_pct != 0 else ""
    orig_text = f" from {active_student.category_rank:,}" if rank_shift_pct != 0 else ""
    rank_badge_parts.append(f"{active_student.category} Rank: <span class='rank-badge'>{shifted_cat:,}</span>{orig_text}{diff_text}")

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
shifted_p_pct = min(99, int(round(shifted_p_at_least_one * 100)))

# New tier counts
new_dream_count = int((rescored_df["shifted_label"] == "Dream").sum())
new_target_count = int((rescored_df["shifted_label"] == "Target").sum())
new_safe_count = int((rescored_df["shifted_label"] == "Safe").sum())

# Metric summary cards (showing old vs new when shifted)
col0, col1, col2, col3 = st.columns(4)
with col0:
    if rank_shift_pct == 0:
        val_display = f"{base_p_pct}%"
        sub_desc = "Monte Carlo estimate (2,000 runs, &rho;=0.5) of getting &ge;1 seat"
    else:
        diff_p = shifted_p_pct - base_p_pct
        val_display = f"{base_p_pct}% → {shifted_p_pct}%"
        sub_desc = f"Simulated shift ({diff_p:+d}%) with rank shifted by {rank_shift_pct:+d}%"

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
    dream_display = f"{base_dream_count}" if rank_shift_pct == 0 else f"{base_dream_count} → {new_dream_count}"
    st.markdown(
        f"""
        <div class="glass-metric metric-dream">
            <div class="metric-title" style="color: #EF4444;">Dream Options</div>
            <div class="metric-value" style="font-size: {'24px' if rank_shift_pct != 0 else '32px'};">{dream_display}</div>
            <div class="metric-desc">P &lt; 30% &middot; Aspirational reach</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    target_display = f"{base_target_count}" if rank_shift_pct == 0 else f"{base_target_count} → {new_target_count}"
    st.markdown(
        f"""
        <div class="glass-metric metric-target">
            <div class="metric-title" style="color: #F59E0B;">Target Options</div>
            <div class="metric-value" style="font-size: {'24px' if rank_shift_pct != 0 else '32px'};">{target_display}</div>
            <div class="metric-desc">30% &le; P &lt; 80% &middot; Competitive zone</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    safe_display = f"{base_safe_count}" if rank_shift_pct == 0 else f"{base_safe_count} → {new_safe_count}"
    st.markdown(
        f"""
        <div class="glass-metric metric-safe">
            <div class="metric-title" style="color: #22C55E;">Safe Options</div>
            <div class="metric-value" style="font-size: {'24px' if rank_shift_pct != 0 else '32px'};">{safe_display}</div>
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
        "predicted_range": st.column_config.TextColumn("Est. Range (80%)", width="small"),
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

    # Student rank horizontal reference line (reflects What-If shift if active)
    student_rank_val = selected_row["shifted_rank"]
    if student_rank_val:
        annot_text = f"Your Rank ({student_rank_val:,})" if rank_shift_pct == 0 else f"Shifted Rank ({student_rank_val:,}, {rank_shift_pct:+d}%)"
        fig.add_hline(
            y=student_rank_val,
            line_dash="dash",
            line_color="#0EA5E9" if rank_shift_pct != 0 else "#16A34A",
            annotation_text=annot_text,
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
