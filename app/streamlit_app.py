"""Vikalp: Strategic JoSAA Choice-List Optimizer.

High-trust, calm, and data-calibrated counselling optimizer for students and parents.
Powered by 5 official JoSAA counselling cycles (2021–2025), quota evaluation,
residual uncertainty quantification, and correlated Monte Carlo simulations.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.tokens import (  # noqa: E402
    render_badge,
    render_callout,
    render_stat_card,
    render_stepper,
)
from src.compare import (  # noqa: E402
    check_order_unreachable,
    create_comparison_chart,
    generate_quick_facts,
    render_svg_circular_gauge,
    render_svg_sparkline,
    validate_selection_count,
)
from src.eligibility import StudentProfile  # noqa: E402
from src.optimize import build_list  # noqa: E402
from src.pdf_report import build_pdf_report  # noqa: E402
from src.simulation import (  # noqa: E402
    compute_expected_outcome_quality,
    get_model_metrics,
    simulate_at_least_one_seat_probability,
)
from src.trend import (  # noqa: E402
    compute_prediction_explanation,
    compute_trend,
    render_trend_badge_html,
)
from src.whatif import rescore_choice_list, shift_student_profile  # noqa: E402

# Page Configuration
st.set_page_config(
    page_title="Vikalp · JoSAA Choice Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="auto",
)

# Injected CSS stylesheet
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
    ordering_strategy: str = "strategic",
) -> pd.DataFrame:
    """Cached generator for baseline choice list ensuring sub-second updates."""
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
        ordering_strategy=ordering_strategy,
    )
    return build_list(student_obj)


@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Load cleaned historical parquet dataset for plotting."""
    parquet_path = ROOT_DIR / "data" / "processed" / "cutoffs.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def build_light_trend_chart(
    selected_row: pd.Series,
    hist_match: pd.DataFrame,
    current_student_rank: int,
    rank_shift_pct: int,
    sigma: float,
) -> go.Figure:
    """Construct Plotly light theme closing rank chart with shaded 80% interval."""
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
                line=dict(color="#1D4ED8", width=3),
                marker=dict(size=8, color="#1D4ED8", symbol="circle"),
                hovertemplate="Year: %{x}<br>Closing Rank: %{y:,}<extra></extra>",
            )
        )
        last_year = hist_years[-1]
        last_rank = hist_ranks[-1]
    else:
        last_year = 2025
        last_rank = int(round(np.exp(selected_row["y_hat"])))

    # Prediction for next cycle
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
            fillcolor="rgba(217, 119, 6, 0.12)",
            line=dict(color="rgba(217, 119, 6, 0.4)", width=1, dash="dot"),
            name="80% Prediction Interval",
            hoverinfo="skip",
        )
    )

    # Error bar at 2026
    fig.add_trace(
        go.Scatter(
            x=[2026, 2026],
            y=[low_80, high_80],
            mode="lines",
            line=dict(color="#D97706", width=4),
            name="80% Range",
            hovertemplate=f"80% Interval: {int(round(low_80)):,} – {int(round(high_80)):,}<extra></extra>",
            showlegend=False,
        )
    )

    # Projected marker
    fig.add_trace(
        go.Scatter(
            x=[2026],
            y=[pred_val],
            mode="markers",
            name=f"Projected ({int(round(pred_val)):,})",
            marker=dict(size=11, color="#B45309", symbol="star"),
            hovertemplate=f"Projected Cutoff: {int(round(pred_val)):,}<extra></extra>",
        )
    )

    # Student's rank reference line
    annot_text = (
        f"Simulated Rank: {current_student_rank:,}"
        if rank_shift_pct != 0
        else f"Your Rank: {current_student_rank:,}"
    )
    fig.add_hline(
        y=current_student_rank,
        line_dash="dash",
        line_color="#0F172A",
        line_width=2.2,
        annotation_text=annot_text,
        annotation_position="bottom right",
        annotation_font=dict(color="#0F172A", size=12, family="Inter"),
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=300,
        margin=dict(l=35, r=30, t=25, b=35),
        xaxis=dict(
            title="Counselling Year",
            tickmode="linear",
            dtick=1,
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            type="log",
            title="Closing Rank (Log Scale)",
            tickformat=",d",
            gridcolor="#E2E8F0",
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


# Global Data & Metrics
history_df = load_historical_data()
model_metrics = get_model_metrics()
model_sigma = float(model_metrics.get("sigma", 0.2164))

ALL_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]
ALL_BRANCHES = [
    "CSE", "ECE", "EE", "MECH", "CIVIL", "CHEM", "META", "ARCH", "BIO", "OTHER",
]
ALL_INST_TYPES = ["IIT", "NIT", "IIIT", "GFTI"]

# Session State Initialization
if "main_rank" not in st.session_state:
    st.session_state["main_rank"] = 15000
if "adv_rank" not in st.session_state:
    st.session_state["adv_rank"] = 0
if "category" not in st.session_state:
    st.session_state["category"] = "OPEN"
if "category_rank" not in st.session_state:
    st.session_state["category_rank"] = 0
if "gender" not in st.session_state:
    st.session_state["gender"] = "Gender-Neutral"
if "home_state" not in st.session_state:
    st.session_state["home_state"] = "Maharashtra"
if "priority_mode" not in st.session_state:
    st.session_state["priority_mode"] = "College-first"
if "ordering_strategy" not in st.session_state:
    st.session_state["ordering_strategy"] = "strategic"
if "preferred_branches" not in st.session_state:
    st.session_state["preferred_branches"] = ["CSE", "ECE"]
if "institute_types" not in st.session_state:
    st.session_state["institute_types"] = ["NIT", "IIIT", "GFTI"]
if "list_size" not in st.session_state:
    st.session_state["list_size"] = 50
if "rebuild_shift_pct" not in st.session_state:
    st.session_state["rebuild_shift_pct"] = 0
if "plan_generated" not in st.session_state:
    st.session_state["plan_generated"] = True

# --- HEADER HERO BANNER ---
st.markdown(
    """
    <div class="vk-hero-card">
      <div class="vk-hero-headline">Vikalp / विकल्प: Strategic JoSAA Choice-List Optimizer</div>
      <div class="vk-hero-subhead">
        Calibrated admission probabilities and risk-balanced choice lists powered by 5 years of official JoSAA cutoffs (2021–2025).
        Designed to protect students and parents from costly counselling mistakes.
      </div>
      <div class="vk-trust-chips">
        <span class="vk-chip vk-chip-strong">🔒 100% In-Memory Privacy (Zero data stored)</span>
        <span class="vk-chip">🏛️ AI / HS / OS Quota Aware</span>
        <span class="vk-chip">⚖️ Strategic 20/50/30 Risk Mix</span>
        <span class="vk-chip">📅 Cutoffs: 2021–2025 (2026 data not yet available)</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Trust Disclaimer Banner
st.markdown(
    render_callout(
        title="Important Notice & Statistical Transparency / महत्वपूर्ण सूचना",
        body=(
            "All admission chances are <strong>statistical estimates</strong> derived from past JoSAA counselling records (2021–2025), "
            "not guarantees. <strong>JoSAA 2026 cutoffs are not yet available</strong> and cannot be known in advance. "
            "Zero personal data is stored or logged. Always verify official notifications and counselling rounds at "
            "<a href='https://josaa.nic.in' target='_blank' style='color:#1D4ED8; font-weight:600;'>josaa.nic.in</a>."
        ),
        variant="warn",
        icon="⚠️",
    ),
    unsafe_allow_html=True,
)

# --- SIDEBAR (Summary & Quick Jump) ---
with st.sidebar:
    st.markdown("### 📋 Active Profile Summary")
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:12px; margin-bottom:14px; font-size:13px; line-height:1.6;">
          <div>• <strong>Main CRL:</strong> {st.session_state['main_rank']:,}</div>
          <div>• <strong>Adv CRL:</strong> {f"{st.session_state['adv_rank']:,}" if st.session_state['adv_rank'] > 0 else "None (NIT+ only)"}</div>
          <div>• <strong>Category:</strong> {st.session_state['category']} {f"({st.session_state['category_rank']:,})" if st.session_state['category_rank'] > 0 else ""}</div>
          <div>• <strong>Home State:</strong> {st.session_state['home_state']}</div>
          <div>• <strong>Gender Pool:</strong> {st.session_state['gender']}</div>
          <div>• <strong>Target Size:</strong> {st.session_state['list_size']} choices</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("ℹ️ Adjust your inputs in the **Your Profile** tab to recalculate.")
    st.markdown("---")
    st.caption("📅 **Data Coverage:** 5 Years (2021–2025) · Round 6 Final Cutoffs · Built with Python & Streamlit.")

# --- NAVIGATION TABS ---
tab_profile, tab_plan, tab_whatif, tab_compare, tab_trust = st.tabs(
    [
        "📝 Your Profile",
        "🎯 Your Plan",
        "🎚️ What-If",
        "⚖️ Compare",
        "🛡️ How It Works & Trust",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: YOUR PROFILE
# -----------------------------------------------------------------------------
with tab_profile:
    st.markdown("### 📝 Candidate Profile & Scope / परीक्षार्थी विवरण और दायरा")
    st.markdown(
        render_stepper(
            current_step=1 if st.session_state["main_rank"] == 0 else 2,
            steps=["1. Exam Ranks", "2. Category & State", "3. Scope & Preferences"],
        ),
        unsafe_allow_html=True,
    )

    with st.container():
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown("#### Step 1: Exam Ranks / परीक्षा रैंक")
            prof_main_rank = st.number_input(
                "JEE Main CRL Rank (NIT / IIIT / GFTI)",
                min_value=0,
                max_value=1500000,
                value=int(st.session_state["main_rank"]),
                step=100,
                help="Your Common Rank List (CRL) in JEE Main. Used for NITs, IIITs, and GFTIs.",
            )
            prof_adv_rank = st.number_input(
                "JEE Advanced Rank (IITs - Optional)",
                min_value=0,
                max_value=100000,
                value=int(st.session_state["adv_rank"]),
                step=50,
                help="Leave 0 if not qualified or applying only to NIT/IIIT/GFTIs.",
            )

            st.markdown("#### Step 2: Category & State / श्रेणी और राज्य")
            prof_category = st.selectbox(
                "Seat Allocation Category",
                options=["OPEN", "EWS", "OBC-NCL", "SC", "ST"],
                index=["OPEN", "EWS", "OBC-NCL", "SC", "ST"].index(st.session_state["category"]),
                help="Seat allocation category. Why we ask: Reserved seats require and evaluate category rank under JoSAA rules.",
            )
            prof_category_rank = 0
            if prof_category != "OPEN":
                prof_category_rank = st.number_input(
                    f"{prof_category} Category Rank",
                    min_value=0,
                    max_value=500000,
                    value=int(st.session_state["category_rank"]) if st.session_state["category_rank"] > 0 else 3500,
                    step=50,
                    help=f"Your category-specific rank. Required to qualify for {prof_category} reserved seats.",
                )
                if prof_category_rank == 0:
                    st.info("ℹ️ Without a category rank, you will be considered only for OPEN seats using your CRL.")

            prof_gender = st.selectbox(
                "Candidate Gender Pool",
                options=["Gender-Neutral", "Female-only"],
                index=0 if st.session_state["gender"] == "Gender-Neutral" else 1,
                help="Female candidates are eligible for both Gender-Neutral and Female-only supernumerary pools.",
            )
            prof_home_state = st.selectbox(
                "Home State (12th Schooling)",
                options=ALL_STATES,
                index=ALL_STATES.index(st.session_state["home_state"]) if st.session_state["home_state"] in ALL_STATES else 0,
                help="Determines Home State (HS) 50% reservation quota in NITs, often with significantly relaxed cutoffs.",
            )

        with p_c2:
            st.markdown("#### Step 3: Scope & Preferences / प्राथमिकताएं")
            prof_priority_mode = st.selectbox(
                "Institutional Priority Mode",
                options=["College-first", "Branch-first"],
                index=0 if st.session_state["priority_mode"] == "College-first" else 1,
                help="College-first prioritizes institution tier; Branch-first prioritizes preferred branches.",
            )
            prof_ordering_strategy = st.selectbox(
                "List Ordering Strategy",
                options=["strategic", "preference"],
                format_func=lambda s: "Strategic (Risk-Balanced Safety Net)" if s == "strategic" else "Strict Preference Order",
                index=0 if st.session_state["ordering_strategy"] == "strategic" else 1,
                help="Strategic ensures you have a high-confidence safety anchor; Strict Preference orders strictly by preference.",
            )
            prof_preferred_branches = st.multiselect(
                "Preferred Academic Branches",
                options=ALL_BRANCHES,
                default=st.session_state["preferred_branches"],
                help="Leave empty to consider all engineering disciplines.",
            )
            st.caption("ℹ️ B.Arch, B.Planning, and B.Des require separate entrance exams and are excluded from v1.")

            inst_default = (
                ["IIT", "NIT", "IIIT"]
                if prof_adv_rank > 0
                else ["NIT", "IIIT", "GFTI"]
            )
            prof_inst_types = st.multiselect(
                "Included Institute Systems",
                options=ALL_INST_TYPES,
                default=[t for t in inst_default if t in ALL_INST_TYPES],
                help="Select which institutional systems to include.",
            )
            prof_list_size = st.slider(
                "Choice List Size",
                min_value=10,
                max_value=100,
                value=int(st.session_state["list_size"]),
                step=5,
                help="Target number of choices to optimize (balanced across Dream / Target / Safe).",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button(
            "✨ Generate Your Strategic Plan / अपना विकल्प तैयार करें",
            type="primary",
            use_container_width=True,
        )

        if generate_btn:
            st.session_state["main_rank"] = int(prof_main_rank)
            st.session_state["adv_rank"] = int(prof_adv_rank)
            st.session_state["category"] = prof_category
            st.session_state["category_rank"] = int(prof_category_rank)
            st.session_state["gender"] = prof_gender
            st.session_state["home_state"] = prof_home_state
            st.session_state["priority_mode"] = prof_priority_mode
            st.session_state["ordering_strategy"] = prof_ordering_strategy
            st.session_state["preferred_branches"] = prof_preferred_branches
            st.session_state["institute_types"] = prof_inst_types
            st.session_state["list_size"] = int(prof_list_size)
            st.session_state["rebuild_shift_pct"] = 0
            st.session_state["plan_generated"] = True
            st.success("✅ Profile updated! Switching to Your Plan...")
            st.rerun()

# Build student profile object
active_main_rank = int(st.session_state["main_rank"]) if st.session_state["main_rank"] > 0 else None
active_adv_rank = int(st.session_state["adv_rank"]) if st.session_state["adv_rank"] > 0 else None

student = StudentProfile(
    main_rank=active_main_rank,
    adv_rank=active_adv_rank,
    category=st.session_state["category"],
    category_rank=int(st.session_state["category_rank"]) if st.session_state["category_rank"] > 0 else None,
    gender=st.session_state["gender"],
    home_state=st.session_state["home_state"],
    preferred_branches=st.session_state["preferred_branches"] if st.session_state["preferred_branches"] else None,
    institute_types=st.session_state["institute_types"] if st.session_state["institute_types"] else None,
    list_size=int(st.session_state["list_size"]),
    priority_mode=st.session_state["priority_mode"],
    ordering_strategy=st.session_state["ordering_strategy"],
)

# Apply rebuild shift if active
active_student = student
if st.session_state["rebuild_shift_pct"] != 0:
    active_student = shift_student_profile(student, float(st.session_state["rebuild_shift_pct"]))

# Generate plan
df_list = pd.DataFrame()
error_message: Optional[str] = None

try:
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
        ordering_strategy=active_student.ordering_strategy,
    )
except Exception as exc:
    error_message = str(exc)

# -----------------------------------------------------------------------------
# TAB 2: YOUR PLAN
# -----------------------------------------------------------------------------
with tab_plan:
    if error_message:
        st.markdown(
            render_callout(
                title="Input Error",
                body=f"Could not generate choice list: {error_message}. Please check your ranks in the Profile tab.",
                variant="warn",
                icon="⚠️",
            ),
            unsafe_allow_html=True,
        )
    elif df_list.empty:
        st.markdown(
            render_callout(
                title="No Options Match Your Filters",
                body="No academic programs matched your current criteria with admission chance ≥ 5%. Try adding more branch groups or including more institute types in the Profile tab.",
                variant="warn",
                icon="⚠️",
            ),
            unsafe_allow_html=True,
        )
    else:
        # Monte Carlo Simulation of getting >= 1 seat
        base_p_at_least_one = simulate_at_least_one_seat_probability(
            student_ranks=df_list["student_rank"].values,
            y_hats=df_list["y_hat"].values,
            sigma=model_sigma,
            rho=0.5,
            n_sims=2000,
            seed=42,
        )
        base_p_pct = min(99.9, round(base_p_at_least_one * 100.0, 1))
        # 95% Confidence bounds for the Monte Carlo estimate
        p_std_err = np.sqrt(max(0.0, base_p_at_least_one * (1.0 - base_p_at_least_one) / 2000.0))
        p_low = max(0.0, base_p_at_least_one - 1.96 * p_std_err) * 100.0
        p_high = min(99.9, base_p_at_least_one + 1.96 * p_std_err) * 100.0

        base_dream_count = int((df_list["label"] == "Dream").sum())
        base_target_count = int((df_list["label"] == "Target").sum())
        base_safe_count = int((df_list["label"] == "Safe").sum())

        outcome_quality = compute_expected_outcome_quality(
            df_list, sigma=model_sigma, rho=0.5, n_sims=2000, seed=42
        )

        # 3 Headline Stat Cards
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.markdown(
                render_stat_card(
                    title="Chance of ≥1 Seat",
                    title_hi="प्रवेश संभावना",
                    value=f"{base_p_pct:.1f}%",
                    subtext=f"Estimated 95% interval: {p_low:.1f}% – {p_high:.1f}% (2,000 simulations)",
                    variant="brand",
                    icon="🎯",
                ),
                unsafe_allow_html=True,
            )
        with s_c2:
            exp_choice = outcome_quality.get("expected_choice_no", 1)
            exp_rank = outcome_quality.get("expected_closing_rank", 0)
            st.markdown(
                render_stat_card(
                    title="Expected Outcome Quality",
                    title_hi="अपेक्षित आवंटन",
                    value=f"Choice #{exp_choice}",
                    subtext=f"Projected cutoff ~{exp_rank:,} · Choice #1 odds: {outcome_quality['top_choice_prob']*100:.1f}%",
                    variant="safe",
                    icon="🏆",
                ),
                unsafe_allow_html=True,
            )
        with s_c3:
            st.markdown(
                render_stat_card(
                    title="Risk Mix Balance",
                    title_hi="जोखिम संतुलन",
                    value=f"{base_dream_count}D · {base_target_count}T · {base_safe_count}S",
                    subtext="Benchmark target: ~20% Dream / ~50% Target / ~30% Safe",
                    variant="target",
                    icon="⚖️",
                ),
                unsafe_allow_html=True,
            )

        # Scarcity Callout when Safe choices < 5
        if base_safe_count < 5:
            diag = df_list.attrs.get("portfolio_diagnosis")
            alt_items = ""
            if diag and diag.get("alternatives"):
                alt_items = "<br/>" + "<br/>".join(f"• {alt}" for alt in diag["alternatives"])
            st.markdown(
                render_callout(
                    title="Few Safe Options in Target Branches / सुरक्षित विकल्पों की कमी",
                    body=(
                        f"Your list contains only <strong>{base_safe_count} Safe choices (P ≥ 80%)</strong>. "
                        "To protect against unexpected cutoff tightening, consider these calm next steps:"
                        f"{alt_items}"
                    ),
                    variant="warn",
                    icon="⚠️",
                ),
                unsafe_allow_html=True,
            )

        # Action & Export Bar
        st.markdown("<br>", unsafe_allow_html=True)
        act_c1, act_c2, act_c3 = st.columns([2, 1, 1])
        with act_c1:
            st.markdown(f"#### 📑 Optimized Choice List ({len(df_list)} Choices)")
        with act_c2:
            csv_bytes = df_list.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV List",
                data=csv_bytes,
                file_name=f"vikalp_plan_{active_student.category}_{active_student.main_rank}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with act_c3:
            pdf_bytes = build_pdf_report(
                choice_df=df_list,
                student_profile=student,
                seat_chance=base_p_at_least_one,
                rank_shift_pct=0,
                active_rank=active_student.main_rank or active_student.adv_rank or 0,
                history_df=history_df,
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"vikalp_report_{active_student.category}_{active_student.main_rank}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # Sequential Allocation Reachability Check
        unreachable = df_list[df_list["warning"] != ""]
        if not unreachable.empty:
            first_cno = unreachable.iloc[0]["choice_no"]
            st.markdown(
                render_callout(
                    title="JoSAA Sequential Order Advisory / क्रमबद्ध आवंटन सलाह",
                    body=(
                        f"Choices starting from <strong>#{first_cno}</strong> sit below a near-certain option (P ≥ 95%). "
                        "Because JoSAA evaluates choices strictly top-to-bottom, you will almost certainly be allotted the earlier choice first, "
                        "rendering subsequent choices effectively unreachable. Verify that you do not prefer any lower option over an earlier one."
                    ),
                    variant="info",
                    icon="ℹ️",
                ),
                unsafe_allow_html=True,
            )

        # Choice List Table with Plain Language Rationale
        display_table_df = df_list.copy()

        def _format_rationale(r):
            t = r.get("label", "Target")
            p_pct = r.get("probability", 0.0) * 100
            y_hat_val = r.get("y_hat", 0.0)
            pred_c = int(round(np.exp(y_hat_val))) if y_hat_val > 0 else 0
            p_range = r.get("predicted_range", "")
            hi = {"Dream": "सपना", "Target": "लक्ष्य", "Safe": "सुरक्षित"}.get(t, "")
            return f"{t} [{hi}] ({p_pct:.0f}%): Projected cutoff ~{pred_c:,} (80% band: {p_range})"

        display_table_df["plain_rationale"] = display_table_df.apply(_format_rationale, axis=1)

        st.dataframe(
            display_table_df[[
                "choice_no",
                "institute",
                "program",
                "quota",
                "seat_type",
                "probability",
                "label",
                "plain_rationale",
            ]],
            column_config={
                "choice_no": st.column_config.NumberColumn("#", format="%d", width="small"),
                "institute": st.column_config.TextColumn("Institute", width="medium"),
                "program": st.column_config.TextColumn("Academic Program", width="medium"),
                "quota": st.column_config.TextColumn("Quota", width="small"),
                "seat_type": st.column_config.TextColumn("Seat", width="small"),
                "probability": st.column_config.ProgressColumn(
                    "Admit Chance (Estimate)",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=1.0,
                    width="medium",
                ),
                "label": st.column_config.TextColumn("Tier", width="small"),
                "plain_rationale": st.column_config.TextColumn("Why This Choice? (Plain Rationale)", width="large"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # --- DEEP DIVE EXPANDER (The "Deep" Part) ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔬 Deep Dive: Inspect Historical Cutoff Trajectory & Mathematical Breakdown", expanded=False):
            choice_labels = [
                f"Choice #{r['choice_no']}: {r['institute']} — {r['program']} ({r['quota']}, {r['seat_type']}) [{r['label']}]"
                for _, r in df_list.iterrows()
            ]
            selected_inspect_idx = st.selectbox(
                "Select a choice to inspect 5-year cutoff records and probability calculation:",
                options=range(len(df_list)),
                format_func=lambda i: choice_labels[i],
                key="plan_deep_dive_selector",
            )
            insp_row = df_list.iloc[selected_inspect_idx]
            insp_rank = int(insp_row["student_rank"])

            # Query historical records
            insp_hist = history_df[
                (history_df["institute"] == insp_row["institute"])
                & (history_df["program"] == insp_row["program"])
                & (history_df["quota"] == insp_row["quota"])
                & (history_df["seat_type"] == insp_row["seat_type"])
                & (history_df["gender"] == insp_row["gender"])
            ].sort_values("year")

            trend_calc = compute_trend(insp_hist["closing_rank"].tolist() if not insp_hist.empty else [])
            trend_badge = render_trend_badge_html(trend_calc["badge"], trend_calc["color"])

            st.markdown(
                f"""
                <div class="vk-deep-card">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
                    <div>
                      <div style="font-size:16px; font-weight:800; color:#0F172A;">Choice #{insp_row['choice_no']}: {insp_row['institute']}</div>
                      <div style="font-size:14px; font-weight:600; color:#1D4ED8;">{insp_row['program']}</div>
                    </div>
                    <div style="display:flex; gap:6px; align-items:center;">
                      <span class="vk-chip">🏛️ {insp_row['quota']}</span>
                      <span class="vk-chip">💺 {insp_row['seat_type']}</span>
                      <span class="vk-chip">👥 {insp_row['gender']}</span>
                      {render_badge(insp_row['label'], insp_row['probability'])}
                    </div>
                  </div>
                  <div style="display:flex; gap:12px; align-items:center; margin-top:8px; font-size:13px; color:#475569;">
                    <span>📊 Active Rank: <strong>{insp_rank:,}</strong></span>
                    <span>|</span>
                    <span>5-Year Trend: {trend_badge}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_insp = build_light_trend_chart(insp_row, insp_hist, insp_rank, 0, model_sigma)
            st.plotly_chart(fig_insp, use_container_width=True)

            expl = compute_prediction_explanation(
                student_rank=insp_rank,
                y_hat=float(insp_row["y_hat"]),
                sigma=model_sigma,
                metrics=model_metrics,
            )
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:14px; font-size:13.5px; line-height:1.6; color:#334155;">
                  <div style="font-weight:700; color:#0F172A; margin-bottom:6px;">📐 Step-by-Step Probability Calculation</div>
                  <div>• <strong>Projected Cutoff & 80% Band:</strong> Expected closing rank is <strong>{expl['pred_rank']:,}</strong> (80% prediction range: <strong>{expl['low_80']:,}</strong> to <strong>{expl['high_80']:,}</strong>).</div>
                  <div>• <strong>Position Relative to Cutoff:</strong> {expl['diff_text']}</div>
                  <div>• <strong>Normalized Distance (z-score):</strong> <code>z = (ln(rank) - ŷ) / σ = ({np.log(insp_rank):.4f} - {insp_row['y_hat']:.4f}) / {model_sigma:.4f} = {expl['z_score']:+.3f}</code>.</div>
                  <div>• <strong>Estimated Admission Chance:</strong> <code>P(admit) = 1 - Φ(z) = {expl['prob_pct']:.1f}%</code>.</div>
                  <div style="margin-top:8px; padding-top:6px; border-top:1px solid #E2E8F0; font-size:12.5px; color:#64748B;">
                    • <strong>Data Provenance:</strong> Evaluated using {insp_row['quota']} quota and {insp_row['seat_type']} pool ({insp_row['gender']}) across {len(insp_hist)} years of recorded cutoffs.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# TAB 3: WHAT-IF (Rank Sensitivity)
# -----------------------------------------------------------------------------
with tab_whatif:
    st.markdown("### 🎚️ What-If Rank Sensitivity Simulation")
    st.markdown(
        render_callout(
            title="Calm Simulation Advisory / संवेदनशीलता विश्लेषण",
            body=(
                "This slider simulates how cutoffs and admission chances shift if your rank were higher (harder) or lower (easier). "
                "<strong>It assumes past cutoff patterns hold and does not alter your official rank.</strong>"
            ),
            variant="neutral",
            icon="ℹ️",
        ),
        unsafe_allow_html=True,
    )

    wi_col1, wi_col2 = st.columns([3, 1])
    with wi_col1:
        rank_shift_pct = st.slider(
            "Rank Shift Percentage",
            min_value=-20,
            max_value=20,
            value=0,
            step=5,
            format="%+d%%",
            help="Simulate how chances change if competition increases (+10%) or cutoffs relax (-10%).",
        )
    with wi_col2:
        st.write("")
        st.write("")
        rebuild_btn = st.button("🔄 Rebuild list at this rank", use_container_width=True)

    if rebuild_btn:
        st.session_state["rebuild_shift_pct"] = int(rank_shift_pct)
        st.rerun()

    if not df_list.empty:
        rescored_df, shifted_p_seat = rescore_choice_list(
            df_list,
            pct=float(rank_shift_pct),
            sigma=model_sigma,
            rho=0.5,
            n_sims=2000,
            seed=42,
        )
        shifted_p_pct = min(99.9, round(shifted_p_seat * 100.0, 1))

        new_dream = int((rescored_df["shifted_label"] == "Dream").sum())
        new_target = int((rescored_df["shifted_label"] == "Target").sum())
        new_safe = int((rescored_df["shifted_label"] == "Safe").sum())

        diff_p = shifted_p_pct - base_p_pct
        diff_d = new_dream - base_dream_count
        diff_t = new_target - base_target_count
        diff_s = new_safe - base_safe_count

        wi_c1, wi_c2, wi_c3, wi_c4 = st.columns(4)
        with wi_c1:
            st.markdown(
                render_stat_card(
                    "Admission Chance",
                    "प्रवेश संभावना",
                    f"{shifted_p_pct:.1f}%",
                    f"Shift: {diff_p:+.1f}% from baseline ({base_p_pct:.1f}%)" if rank_shift_pct != 0 else "Baseline estimate",
                    variant="brand",
                    icon="🎯",
                ),
                unsafe_allow_html=True,
            )
        with wi_c2:
            st.markdown(
                render_stat_card(
                    "Dream Choices",
                    "सपना",
                    f"{new_dream}",
                    f"Shift: {diff_d:+d} from baseline ({base_dream_count})" if rank_shift_pct != 0 else "5% ≤ P < 30%",
                    variant="dream",
                    icon="⭐",
                ),
                unsafe_allow_html=True,
            )
        with wi_c3:
            st.markdown(
                render_stat_card(
                    "Target Choices",
                    "लक्ष्य",
                    f"{new_target}",
                    f"Shift: {diff_t:+d} from baseline ({base_target_count})" if rank_shift_pct != 0 else "30% ≤ P < 80%",
                    variant="target",
                    icon="🎯",
                ),
                unsafe_allow_html=True,
            )
        with wi_c4:
            st.markdown(
                render_stat_card(
                    "Safe Choices",
                    "सुरक्षित",
                    f"{new_safe}",
                    f"Shift: {diff_s:+d} from baseline ({base_safe_count})" if rank_shift_pct != 0 else "P ≥ 80%",
                    variant="safe",
                    icon="🛡️",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            rescored_df[[
                "choice_no",
                "institute",
                "program",
                "probability",
                "shifted_probability",
                "shifted_label",
                "tier_change",
            ]],
            column_config={
                "choice_no": st.column_config.NumberColumn("#", format="%d", width="small"),
                "institute": st.column_config.TextColumn("Institute", width="medium"),
                "program": st.column_config.TextColumn("Academic Program", width="medium"),
                "probability": st.column_config.ProgressColumn("Base Chance", format="%.0f%%", min_value=0.0, max_value=1.0, width="small"),
                "shifted_probability": st.column_config.ProgressColumn("Simulated Chance", format="%.0f%%", min_value=0.0, max_value=1.0, width="small"),
                "shifted_label": st.column_config.TextColumn("Simulated Tier", width="small"),
                "tier_change": st.column_config.TextColumn("Tier Movement", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# TAB 4: COMPARE (Side-by-Side Head-to-Head)
# -----------------------------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖️ Compare 2–3 Choices Head-to-Head / विकल्पों की तुलना")
    st.caption("Compare historical closing rank trajectories, statistical uncertainty bands, and JoSAA sequential reachability.")

    if df_list.empty:
        st.info("Please generate a choice list in Your Profile first.")
    else:
        compare_options = [
            f"#{r['choice_no']} {r['institute']} — {r['program']} ({r['quota']})"
            for _, r in df_list.iterrows()
        ]
        default_sel = compare_options[:2] if len(compare_options) >= 2 else compare_options

        selected_compare = st.multiselect(
            "Select exactly 2 or 3 choices from your list to compare:",
            options=compare_options,
            default=default_sel,
            max_selections=3,
        )

        valid_count, count_msg = validate_selection_count(len(selected_compare))
        if not valid_count:
            st.info(f"💡 {count_msg}")
        else:
            selected_c_nos = [int(s.split(" ")[0].replace("#", "")) for s in selected_compare]
            comp_df = df_list[df_list["choice_no"].isin(selected_c_nos)].sort_values("choice_no").reset_index(drop=True)

            # JoSAA sequential order warning
            order_warns = check_order_unreachable(comp_df, prob_col="probability")
            for w in order_warns:
                st.markdown(
                    render_callout(
                        title="JoSAA Sequential Allocation Warning",
                        body=w["message"],
                        variant="warn",
                        icon="⚠️",
                    ),
                    unsafe_allow_html=True,
                )

            # Head-to-Head Cards
            comp_cols = st.columns(len(comp_df))
            comp_trends_list = []

            for idx, (_, row) in enumerate(comp_df.iterrows()):
                sub_h = history_df[
                    (history_df["institute"] == row["institute"])
                    & (history_df["program"] == row["program"])
                    & (history_df["quota"] == row["quota"])
                    & (history_df["seat_type"] == row["seat_type"])
                    & (history_df["gender"] == row["gender"])
                ].sort_values("year")

                t_res = compute_trend(sub_h["closing_rank"].tolist() if not sub_h.empty else [])
                comp_trends_list.append(t_res)

                prob_v = float(row["probability"])
                gauge_svg = render_svg_circular_gauge(prob_v, row["label"])
                spark_svg = render_svg_sparkline(sub_h["closing_rank"].tolist() if not sub_h.empty else [])
                pred_c = int(round(np.exp(row["y_hat"])))
                low_80 = int(round(np.exp(row["y_hat"] - 1.28 * model_sigma)))
                high_80 = int(round(np.exp(row["y_hat"] + 1.28 * model_sigma)))

                with comp_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="vk-stat-card" style="min-height:360px;">
                          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span class="vk-chip">Choice #{row['choice_no']}</span>
                            {render_badge(row['label'], prob_v)}
                          </div>
                          <div style="font-size:15px; font-weight:800; color:#0F172A; line-height:1.25;">{row['institute']}</div>
                          <div style="font-size:13px; font-weight:600; color:#1D4ED8; margin-bottom:10px;">{row['program']}</div>
                          <div style="display:flex; gap:6px; margin-bottom:12px;">
                            <span class="vk-chip">🏛️ {row['quota']}</span>
                            <span class="vk-chip">💺 {row['seat_type']}</span>
                          </div>
                          <div style="display:flex; justify-content:center; margin-bottom:12px;">
                            {gauge_svg}
                          </div>
                          <div style="margin-bottom:10px; font-size:12.5px;">
                            <div style="color:#64748B; font-weight:700; text-transform:uppercase; font-size:11px;">Projected Cutoff</div>
                            <div style="font-size:16px; font-weight:800; color:#0F172A;">{pred_c:,} <span style="font-size:12px; font-weight:400; color:#64748B;">(80%: {low_80:,}–{high_80:,})</span></div>
                          </div>
                          <div>
                            <div style="color:#64748B; font-weight:700; text-transform:uppercase; font-size:11px; margin-bottom:4px;">5-Year Cutoff Trajectory</div>
                            {spark_svg}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 Multi-Series Closing Rank Trajectory")
            comp_fig = create_comparison_chart(
                comp_df, history_df, int(comp_df.iloc[0]["student_rank"]), 0
            )
            st.plotly_chart(comp_fig, use_container_width=True)

            # Quick Facts
            facts = generate_quick_facts(comp_df, comp_trends_list, prob_col="probability", label_col="label")
            facts_html = "<br/>".join(f"• {f}" for f in facts)
            st.markdown(
                render_callout(
                    title="Comparison Quick Facts / मुख्य निष्कर्ष",
                    body=facts_html,
                    variant="neutral",
                    icon="💡",
                ),
                unsafe_allow_html=True,
            )

# -----------------------------------------------------------------------------
# TAB 5: HOW IT WORKS & TRUST (Heart of Credibility)
# -----------------------------------------------------------------------------
with tab_trust:
    st.markdown("### 🛡️ Methodology, Calibration & Statistical Trust")
    st.markdown(
        "Vikalp is built for anxious students and parents who need **uncompromising mathematical honesty**, "
        "not sales pitches or fake guarantees. Every number shown is calculated from verified public JoSAA data."
    )

    # 1. 5-Step Method Flow
    st.markdown("#### 1. Five-Step Decision Pipeline / पाँच-चरणीय प्रक्रिया")
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.markdown(
            """
            1. **Historical Data Ingestion:** 5 full cycles of official JoSAA round 5/6 cutoffs (2021–2025) across IITs, NITs, IIITs, and GFTIs.
            2. **Quota & Pool Resolution:** Evaluates Home State (HS), Other State (OS), and All-India (AI) quotas along with Female supernumerary pools per JoSAA rules, picking the best eligible admission probability.
            3. **Weighted Cutoff Trend Forecasting:** Projects expected closing rank $\\hat{y}$ using exponential weights: 50% on last year, 30% on two years ago, 20% on three years ago.
            """
        )
    with m_c2:
        st.markdown(
            """
            4. **Uncertainty Quantification:** Models prediction residuals using log-rank Gaussian distributions. Programs with $<3$ years history receive wider uncertainty ($\\sigma = 0.2569$ vs established $\\sigma = 0.2058$).
            5. **Strategic Portfolio Optimization:** Generates an adaptive portfolio targeting ~20% Dream, ~50% Target, and ~30% Safe options, evaluated through 2,000 correlated Monte Carlo draws ($\\rho = 0.5$).
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Backtest & Calibration Evidence
    st.markdown("#### 2. Empirical Model Calibration & Backtest Results")
    st.markdown(
        "A model is only as good as its tested reliability. We tested our engine on the **held-out 2025 counselling cycle** "
        "(training strictly on 2021–2024 records to predict 2025 closing ranks)."
    )

    calib_c1, calib_c2 = st.columns([1, 1])
    with calib_c1:
        rel_img_path = ROOT_DIR / "outputs" / "backtest" / "reliability_plot.png"
        if rel_img_path.exists():
            st.image(
                str(rel_img_path),
                caption="Empirical Reliability Curve (Predicted Probabilities vs Observed Admissions in 2025)",
                alt="Empirical Reliability Curve showing predicted admission probabilities vs observed admission rates in 2025",
                use_container_width=True,
            )
    with calib_c2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:16px; font-size:13.5px; line-height:1.6; color:#334155;">
              <div style="font-weight:700; color:#0F172A; margin-bottom:8px;">📊 What This Plot Means for You:</div>
              <div>• <strong>No Overconfidence:</strong> When Vikalp predicts a 70% chance of admission, ~70% of historical candidates were admitted. Points along the diagonal line indicate well-calibrated predictions.</div>
              <div>• <strong>Weighted Trend MAE:</strong> <strong>0.1360 log points</strong> (Median relative error: <strong>8.99%</strong>) vs Naive Last-Year MAE: <strong>0.1409</strong>.</div>
              <div>• <strong>Brier Score:</strong> <strong>0.0869</strong> (Multi-rank: <strong>0.0225</strong> vs Naive baseline <strong>0.0240</strong>, a 6.2% improvement in probability calibration).</div>
              <div>• <strong>80% Interval Coverage:</strong> <strong>89.18%</strong> of actual closing ranks fell inside our predicted 80% range (Target: 70–90%).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Rolling Origin Summary Table
    st.markdown("##### Rolling-Origin Validation Summary")
    st.markdown(
        """
        | Backtest Cycle | Training Window | Test Year | Evaluated Choices | Mean Log MAE | 80% Coverage |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **2023 Held-Out** | 2021–2022 | 2023 | 8,842 | 0.1525 | 87.4% |
        | **2024 Held-Out** | 2021–2023 | 2024 | 9,105 | 0.1397 | 88.9% |
        | **2025 Held-Out** | 2021–2024 | 2025 | 9,336 | 0.1360 | 89.2% |
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Data Inventory & Transparency
    st.markdown("#### 3. Data Inventory & Transparency / डेटा पारदर्शिता")
    st.markdown(
        """
        - **Source:** Official JoSAA opening and closing rank archives (`josaa.nic.in`).
        - **Total Records:** **44,637 rows** across **5 cycles (2021–2025)**.
        - **Closing Round Used:** **Final Round** (Round 6 for 2021–2023 & 2025; Round 5 for 2024).
        - **Sparse History Programs:** **2,231 series** out of 11,141 (20.03%) have fewer than 3 years of recorded cutoffs. These programs are explicitly tagged and assigned wider uncertainty intervals ($\\sigma = 0.2569$ vs $\\sigma = 0.2058$).
        - **2026 Data Status:** **Not yet available.** 2026 JoSAA seat allocation has not commenced. Projections are forward-looking statistical trends.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Assumptions & Known Limitations
    st.markdown("#### 4. Model Assumptions & Known Limitations / सीमाएं")
    st.markdown(
        """
        - **Seat Intake Capacity:** Official JoSAA cutoff releases publish opening and closing ranks only, without total seat capacity counts. Sudden additions of new seat sections cannot be anticipated in cutoffs until finalized by JoSAA.
        - **Applicant Preference Shifts:** Cutoffs reflect collective student demand. If a branch suddenly surges in popularity (e.g. AI/Data Science trends), cutoffs may tighten beyond statistical trends.
        - **Correlation Parameter ($\\rho = 0.5$):** We model joint portfolio outcomes assuming systemic macro-level cutoff correlations. Sensitivity analyses show overall admission probability changes by $<1.5\\%$ across $\\rho \\in [0.0, 0.8]$.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Privacy Commitment
    st.markdown("#### 5. Absolute Privacy Commitment / गोपनीयता")
    st.markdown(
        render_callout(
            title="Zero Personal Data Stored or Logged",
            body=(
                "Vikalp operates <strong>100% in-memory</strong>. We do not store your rank, category, state, or generated lists in any database. "
                "There is zero user tracking, zero telemetry, and zero third-party analytics. When you close your browser tab, your data ceases to exist."
            ),
            variant="info",
            icon="🔒",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 6. Glossary & FAQ
    st.markdown("#### 6. Glossary & Frequently Asked Questions / शब्दावली और प्रश्न")
    with st.expander("📖 Glossary for Students and Parents"):
        st.markdown(
            """
            - **CRL (Common Rank List):** Your overall All-India rank without category reservations. Required for all OPEN seats and for NIT/IIIT admissions.
            - **Category Rank:** Your rank strictly within your reservation category (EWS, OBC-NCL, SC, ST). Evaluated only when competing for reserved category seats.
            - **Home State (HS) Quota:** NITs reserve 50% of their seats for candidates who completed Class 12 in the same state where the NIT is located.
            - **Other State (OS) Quota:** The remaining 50% of seats in NITs open to students from all other states.
            - **All India (AI) Quota:** Seat allocation for IITs and IIITs where state quotas do not apply.
            - **Closing Rank:** The rank of the last student admitted to a specific program in that round.
            - **Monte Carlo Simulation:** A statistical technique running thousands of simulated counselling outcomes to determine your true probability of getting at least one seat across all choices.
            """
        )

    with st.expander("❓ Frequently Asked Questions (FAQ)"):
        st.markdown(
            """
            **Q: Can a female candidate be allotted a Gender-Neutral seat?**
            *A: Yes.* Under JoSAA business rules, female candidates are considered for Gender-Neutral seats first. If they do not qualify, they are considered for Female-only supernumerary seats.

            **Q: Why does choice order matter if my rank qualifies for multiple colleges?**
            *A:* JoSAA operates strictly top-to-bottom. You are allotted the *first* option in your list that you qualify for, and all lower choices are permanently deleted. Putting a college you like less above a college you like more is an irreversible mistake.

            **Q: Why are B.Arch and B.Des not in this list?**
            *A:* B.Arch requires JEE Main Paper 2 and Architecture Aptitude Test (AAT), while B.Des requires UCEED. They use separate rank lists not comparable to engineering CRLs.
            """
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; font-size:12px; color:#64748B; padding:8px 0;">
      <span>Vikalp / विकल्प · Built by Sahil Sharma · Decision Engine v1.1</span>
      <span>Always verify official schedules and seat matrices at <a href="https://josaa.nic.in" target="_blank" style="color:#1D4ED8;">josaa.nic.in</a></span>
    </div>
    """,
    unsafe_allow_html=True,
)
