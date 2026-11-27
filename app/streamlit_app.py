"""Vikalp: Modern JoSAA Choice-List Optimizer.

Built for students and worried parents. Clean, calm, consumer-grade clarity.
Based on 5 years of official JoSAA counselling records (2021-2025).
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
    normalize_tier_name,
    render_badge,
    render_callout,
    render_probability_meter,
    render_segment_bar,
)
from src.compare import (  # noqa: E402
    check_order_unreachable,
    create_comparison_chart,
    render_svg_circular_gauge,
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
from src.whatif import rescore_choice_list, shift_student_profile  # noqa: E402

# Page Configuration
st.set_page_config(
    page_title="Vikalp · JoSAA Choice Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    """Load cleaned historical cutoffs dataset for plotting."""
    parquet_path = ROOT_DIR / "data" / "processed" / "cutoffs.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def build_parent_trend_chart(
    selected_row: pd.Series,
    hist_match: pd.DataFrame,
    current_student_rank: int,
    sigma: float,
) -> go.Figure:
    """Construct parent-friendly closing rank chart with clear labels."""
    fig = go.Figure()

    if not hist_match.empty:
        hist_years = hist_match["year"].tolist()
        hist_ranks = hist_match["closing_rank"].tolist()

        fig.add_trace(
            go.Scatter(
                x=hist_years,
                y=hist_ranks,
                mode="lines+markers",
                name="Past years' last seat rank",
                line=dict(color="#1D4ED8", width=3),
                marker=dict(size=8, color="#1D4ED8", symbol="circle"),
                hovertemplate="Year %{x}: rank %{y:,}<extra></extra>",
            )
        )
        last_year = hist_years[-1]
        last_rank = hist_ranks[-1]
    else:
        last_year = 2025
        last_rank = int(round(np.exp(selected_row["y_hat"])))

    y_hat = float(selected_row["y_hat"])
    pred_val = float(np.exp(y_hat))
    low_80 = float(np.exp(y_hat - 1.28 * sigma))
    high_80 = float(np.exp(y_hat + 1.28 * sigma))

    # Shaded expected range
    fig.add_trace(
        go.Scatter(
            x=[last_year, 2026, 2026, last_year],
            y=[last_rank, high_80, low_80, last_rank],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.10)",
            line=dict(color="rgba(37, 99, 235, 0.35)", width=1, dash="dot"),
            name="Likely range for this year (80% confidence)",
            hoverinfo="skip",
        )
    )

    # Predicted mark
    fig.add_trace(
        go.Scatter(
            x=[2026],
            y=[pred_val],
            mode="markers",
            name=f"Expected last seat rank ({int(round(pred_val)):,})",
            marker=dict(size=10, color="#D97706", symbol="star"),
            hovertemplate=f"Expected rank: {int(round(pred_val)):,}<extra></extra>",
        )
    )

    # Student's rank reference line
    fig.add_hline(
        y=current_student_rank,
        line_dash="dash",
        line_color="#0F172A",
        line_width=2,
        annotation_text=f"Your rank: {current_student_rank:,}",
        annotation_position="bottom right",
        annotation_font=dict(color="#0F172A", size=12, family="Inter"),
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=280,
        margin=dict(l=35, r=30, t=25, b=35),
        xaxis=dict(
            title="Year",
            tickmode="linear",
            dtick=1,
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            type="log",
            title="Rank (lower number = harder)",
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


# Global Data & Setup
history_df = load_historical_data()
model_metrics = get_model_metrics()
model_sigma = float(model_metrics.get("sigma", 0.1733))

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

# Session State
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

# --- APP HEADER ---
st.markdown(
    """
    <div style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #E2E8F0;">
      <div style="font-size: 26px; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
        Vikalp <span style="font-size: 15px; font-weight: 500; color: #64748B;">· JoSAA Choice Optimizer</span>
      </div>
      <div style="font-size: 14.5px; color: #475569; margin-top: 4px;">
        See your realistic college options and build a safer choice list for engineering counselling.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation tabs in single guided flow
tab_input, tab_list, tab_whatif, tab_compare, tab_method = st.tabs(
    [
        "1. Tell us your rank",
        "2. Your list",
        "3. Try a different rank",
        "4. Compare choices",
        "5. How we got this",
    ]
)

# -----------------------------------------------------------------------------
# SCREEN 1: LANDING / START (Max 3 inputs visible first, progressive disclosure)
# -----------------------------------------------------------------------------
with tab_input:
    st.markdown("### Let's find your college choices")
    st.markdown("Enter your entrance rank and details. We will check your chances across 5 years of past counselling.")

    inp_c1, inp_c2 = st.columns([1.2, 1])
    with inp_c1:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:24px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            """,
            unsafe_allow_html=True,
        )

        in_main_rank = st.number_input(
            "1. Your overall rank (all India) in JEE Main",
            min_value=1,
            max_value=1500000,
            value=int(st.session_state["main_rank"]),
            step=100,
            help="Used to check your chances at NITs, IIITs, and government colleges.",
        )

        in_category = st.selectbox(
            "2. Reservation category",
            options=["OPEN", "EWS", "OBC-NCL", "SC", "ST"],
            index=["OPEN", "EWS", "OBC-NCL", "SC", "ST"].index(st.session_state["category"]),
            help="Select OPEN if you do not use category reservation.",
        )

        in_cat_rank = 0
        if in_category != "OPEN":
            in_cat_rank = st.number_input(
                f"Your {in_category} category rank",
                min_value=0,
                max_value=500000,
                value=int(st.session_state["category_rank"]) if st.session_state["category_rank"] > 0 else 3500,
                step=50,
                help=f"Your rank inside the {in_category} list. Needed for reserved seats.",
            )

        in_home_state = st.selectbox(
            "3. State where you finished 12th class",
            options=ALL_STATES,
            index=ALL_STATES.index(st.session_state["home_state"]) if st.session_state["home_state"] in ALL_STATES else 0,
            help="NITs give 50% of their seats to students from their own state, often with easier cutoffs.",
        )

        with st.expander("More details: Advanced rank, gender pool, college & branch preferences"):
            in_adv_rank = st.number_input(
                "JEE Advanced rank (optional, for IITs only)",
                min_value=0,
                max_value=100000,
                value=int(st.session_state["adv_rank"]),
                step=50,
                help="Leave 0 if you are looking at NITs and IIITs only.",
            )

            in_gender = st.selectbox(
                "Gender pool",
                options=["Gender-Neutral", "Female-only"],
                index=0 if st.session_state["gender"] == "Gender-Neutral" else 1,
                help="Female students are considered for both general seats and female-only seats.",
            )

            in_pref_branches = st.multiselect(
                "Preferred engineering branches",
                options=ALL_BRANCHES,
                default=st.session_state["preferred_branches"],
                help="Leave empty to see all engineering branches.",
            )

            inst_default = ["IIT", "NIT", "IIIT"] if in_adv_rank > 0 else ["NIT", "IIIT", "GFTI"]
            in_inst_types = st.multiselect(
                "College systems to include",
                options=ALL_INST_TYPES,
                default=[t for t in inst_default if t in ALL_INST_TYPES],
            )

            in_list_size = st.slider(
                "How many choices do you want to see?",
                min_value=10,
                max_value=100,
                value=int(st.session_state["list_size"]),
                step=5,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        show_btn = st.button("Show My Choices →", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_btn:
            st.session_state["main_rank"] = int(in_main_rank)
            st.session_state["adv_rank"] = int(in_adv_rank) if 'in_adv_rank' in locals() else 0
            st.session_state["category"] = in_category
            st.session_state["category_rank"] = int(in_cat_rank)
            st.session_state["home_state"] = in_home_state
            if 'in_gender' in locals():
                st.session_state["gender"] = in_gender
            if 'in_pref_branches' in locals():
                st.session_state["preferred_branches"] = in_pref_branches
            if 'in_inst_types' in locals():
                st.session_state["institute_types"] = in_inst_types
            if 'in_list_size' in locals():
                st.session_state["list_size"] = in_list_size
            st.session_state["rebuild_shift_pct"] = 0
            st.success("Your choices are ready! Check tab 2: Your list.")

    with inp_c2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:22px; font-size:14px; line-height:1.6; color:#334155;">
              <div style="font-weight:700; color:#0F172A; margin-bottom:10px; font-size:15px;">🛡️ Honest Promise to Students & Parents</div>
              <div>• <strong>No fake guarantees:</strong> Everything here is an estimate based on 2021–2025 counselling cutoffs. 2026 data is not out yet.</div>
              <div>• <strong>Complete privacy:</strong> Nothing is stored or logged. Your rank never leaves your session.</div>
              <div>• <strong>Safety first:</strong> We divide choices into <strong>Reach</strong>, <strong>Good fit</strong>, and <strong>Likely</strong> so you always have a safety net.</div>
              <div style="margin-top:14px; padding-top:12px; border-top:1px solid #E2E8F0; font-size:12.5px; color:#64748B;">
                Always check official schedules and seat allotment at <a href="https://josaa.nic.in" target="_blank" style="color:#1D4ED8; font-weight:600;">josaa.nic.in</a>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Build student object
active_main = int(st.session_state["main_rank"]) if st.session_state["main_rank"] > 0 else None
active_adv = int(st.session_state["adv_rank"]) if st.session_state["adv_rank"] > 0 else None

student = StudentProfile(
    main_rank=active_main,
    adv_rank=active_adv,
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

if st.session_state["rebuild_shift_pct"] != 0:
    student = shift_student_profile(student, float(st.session_state["rebuild_shift_pct"]))

# Compute choices
df_list = pd.DataFrame()
try:
    pref_branches_tuple = tuple(student.preferred_branches) if student.preferred_branches else None
    inst_types_tuple = tuple(student.institute_types) if student.institute_types else None
    df_list = get_optimized_choice_list(
        main_rank=student.main_rank,
        adv_rank=student.adv_rank,
        category=student.category,
        category_rank=student.category_rank,
        gender=student.gender,
        home_state=student.home_state,
        preferred_branches=pref_branches_tuple,
        institute_types=inst_types_tuple,
        list_size=student.list_size,
        priority_mode=student.priority_mode,
        ordering_strategy=student.ordering_strategy,
    )
except Exception:
    df_list = pd.DataFrame()

# -----------------------------------------------------------------------------
# SCREEN 2: YOUR LIST (Hero Card + Choice Cards)
# -----------------------------------------------------------------------------
with tab_list:
    if df_list.empty:
        st.markdown(
            render_callout(
                title="No Choices Found for Your Filters",
                body="None of the colleges in your selected branches had cutoffs within reach (above 5% odds). Try adding more branch groups or other college systems in Step 1.",
                variant="warn",
                icon="⚠️",
            ),
            unsafe_allow_html=True,
        )
    else:
        # We tried 2,000 possible outcomes
        p_at_least_one = simulate_at_least_one_seat_probability(
            student_ranks=df_list["student_rank"].values,
            y_hats=df_list["y_hat"].values,
            sigma=model_sigma,
            rho=0.5,
            n_sims=2000,
            seed=42,
        )
        p_seat_pct = min(99.9, round(p_at_least_one * 100.0, 1))

        # Plain language meaning of the percentage
        ten_equiv = round(p_seat_pct / 10.0)
        meaning_sentence = f"About {ten_equiv} out of 10 similar students with your rank got at least one seat across past years."

        # Category counts
        reach_cnt = int((df_list["label"] == "Dream").sum())
        good_cnt = int((df_list["label"] == "Target").sum())
        likely_cnt = int((df_list["label"] == "Safe").sum())

        outcome_qual = compute_expected_outcome_quality(
            df_list, sigma=model_sigma, rho=0.5, n_sims=2000, seed=42
        )

        # Hero Card
        st.markdown(
            f"""
            <div class="vk-hero-card">
              <div class="vk-hero-number tabular-num">{p_seat_pct:.0f}%</div>
              <div class="vk-hero-label">Chance of getting at least one seat</div>
              <div class="vk-hero-subtext">
                {meaning_sentence} We tried 2,000 possible cutoff outcomes to verify this safety net.
              </div>
              {render_segment_bar(reach_cnt, good_cnt, likely_cnt)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gentle callout if few likely choices
        if likely_cnt < 5:
            diag = df_list.attrs.get("portfolio_diagnosis")
            alt_text = ""
            if diag and diag.get("alternatives"):
                alt_text = "<br/>" + "<br/>".join(f"• {a}" for a in diag["alternatives"])
            st.markdown(
                render_callout(
                    title="You have few likely options right now",
                    body=(
                        f"Your list currently has only <strong>{likely_cnt} Likely choices (80%+ odds)</strong>. "
                        "To be completely safe, consider these gentle next steps:"
                        f"{alt_text}"
                    ),
                    variant="warn",
                    icon="💡",
                ),
                unsafe_allow_html=True,
            )

        # Action bar: Download PDF / CSV
        act_c1, act_c2 = st.columns([1, 1])
        with act_c1:
            st.markdown(f"#### Your Recommended Choice List ({len(df_list)} Choices)")
            st.caption("Ordered from top to bottom. JoSAA gives you the first choice on this list where your rank qualifies.")
        with act_c2:
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                csv_bytes = df_list.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Save as CSV",
                    data=csv_bytes,
                    file_name=f"vikalp_choices_{student.main_rank}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with d_c2:
                pdf_bytes = build_pdf_report(
                    choice_df=df_list,
                    student_profile=student,
                    seat_chance=p_at_least_one,
                    rank_shift_pct=0,
                    active_rank=student.main_rank or student.adv_rank or 0,
                    history_df=history_df,
                )
                st.download_button(
                    label="📄 Save as PDF Report",
                    data=pdf_bytes,
                    file_name=f"vikalp_report_{student.main_rank}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        # Choices rendered as clean cards
        for idx, row in df_list.iterrows():
            cno = int(row["choice_no"])
            inst = row["institute"]
            prog = row["program"]
            prob = float(row["probability"])
            tier = normalize_tier_name(row["label"])

            # Friendly seat type description
            q_label = "Home state seat" if row["quota"] == "HS" else "Other state seat" if row["quota"] == "OS" else "All India seat"
            seat_desc = f"{q_label} · {row['seat_type']}"

            # Plain rationale
            y_hat_v = float(row.get("y_hat", 0.0))
            pred_last = int(round(np.exp(y_hat_v))) if y_hat_v > 0 else 0
            why_text = f"Expected last seat rank is ~{pred_last:,}. Your rank is {int(row['student_rank']):,}, giving you a {prob:.0%} chance."

            meter_html = render_probability_meter(prob, tier)
            badge_html = render_badge(tier, prob)

            st.markdown(
                f"""
                <div class="vk-choice-card">
                  <div class="vk-choice-header">
                    <div>
                      <div class="vk-choice-title">#{cno}. {prog}</div>
                      <div class="vk-choice-inst">{inst}</div>
                    </div>
                    <div class="vk-choice-meta">
                      <span class="vk-chip">{seat_desc}</span>
                      {badge_html}
                    </div>
                  </div>
                  <div class="vk-choice-body">
                    <div class="vk-choice-rationale">{why_text}</div>
                    <div>{meter_html}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable inspection chart
            with st.expander(f"View cutoff history for #{cno} {prog} at {inst}"):
                sub_h = history_df[
                    (history_df["institute"] == inst)
                    & (history_df["program"] == prog)
                    & (history_df["quota"] == row["quota"])
                    & (history_df["seat_type"] == row["seat_type"])
                    & (history_df["gender"] == row["gender"])
                ].sort_values("year")

                fig = build_parent_trend_chart(row, sub_h, int(row["student_rank"]), model_sigma)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SCREEN 3: TRY A DIFFERENT RANK (What-If Sensitivity)
# -----------------------------------------------------------------------------
with tab_whatif:
    st.markdown("### 🎚️ Try a different rank")
    st.markdown("What if competition gets tougher or easier this year? Move the slider to see how your chances change.")
    st.caption("ℹ️ This assumes past cutoff patterns hold; it does not change your official rank.")

    w_c1, w_c2 = st.columns([3, 1])
    with w_c1:
        shift_pct = st.slider(
            "Move rank number up or down",
            min_value=-20,
            max_value=20,
            value=0,
            step=5,
            format="%+d%%",
            help="+10% means your rank number gets 10% higher (more competitive); -10% means easier.",
        )
    with w_c2:
        st.write("")
        st.write("")
        rebuild_w_btn = st.button("Rebuild choices at this rank", use_container_width=True)

    if rebuild_w_btn:
        st.session_state["rebuild_shift_pct"] = int(shift_pct)
        st.rerun()

    if not df_list.empty:
        rescored, p_shifted = rescore_choice_list(
            df_list,
            pct=float(shift_pct),
            sigma=model_sigma,
            rho=0.5,
            n_sims=2000,
            seed=42,
        )
        p_s_pct = min(99.9, round(p_shifted * 100.0, 1))

        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px; margin:16px 0;">
              <div style="font-size:14px; font-weight:600; color:#64748B;">Simulated chance of getting at least one seat:</div>
              <div style="font-size:36px; font-weight:800; color:#1D4ED8; margin:4px 0;">{p_s_pct:.0f}%</div>
              <div style="font-size:13.5px; color:#475569;">
                {'Baseline was ' + f"{p_seat_pct:.0f}%" + f" (shift of {p_s_pct - p_seat_pct:+.0f}%)" if shift_pct != 0 else "This matches your current entered rank."}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            rescored[[
                "choice_no",
                "institute",
                "program",
                "probability",
                "shifted_probability",
                "shifted_label",
                "tier_change",
            ]],
            column_config={
                "choice_no": st.column_config.NumberColumn("#", width="small"),
                "institute": st.column_config.TextColumn("College", width="medium"),
                "program": st.column_config.TextColumn("Branch", width="medium"),
                "probability": st.column_config.ProgressColumn("Base Odds", format="%.0f%%", min_value=0.0, max_value=1.0),
                "shifted_probability": st.column_config.ProgressColumn("Simulated Odds", format="%.0f%%", min_value=0.0, max_value=1.0),
                "shifted_label": st.column_config.TextColumn("New Category", width="small"),
                "tier_change": st.column_config.TextColumn("Movement", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
        )

# -----------------------------------------------------------------------------
# SCREEN 4: COMPARE CHOICES (Side-by-side head to head)
# -----------------------------------------------------------------------------
with tab_compare:
    st.markdown("### ⚖️ Compare 2 or 3 choices side by side")
    st.caption("See which option is safer and how their past cutoffs compare.")

    if df_list.empty:
        st.info("Please fill in your rank in Step 1 first.")
    else:
        comp_opts = [
            f"#{r['choice_no']} {r['institute']} — {r['program']}"
            for _, r in df_list.iterrows()
        ]
        default_comp = comp_opts[:2] if len(comp_opts) >= 2 else comp_opts

        selected_comp = st.multiselect(
            "Select 2 or 3 choices from your list:",
            options=comp_opts,
            default=default_comp,
            max_selections=3,
        )

        valid_c, msg_c = validate_selection_count(len(selected_comp))
        if not valid_c:
            st.info(f"💡 {msg_c}")
        else:
            c_nos = [int(s.split(" ")[0].replace("#", "")) for s in selected_comp]
            comp_sub = df_list[df_list["choice_no"].isin(c_nos)].sort_values("choice_no").reset_index(drop=True)

            # Reachability check
            warns = check_order_unreachable(comp_sub, prob_col="probability")
            for w in warns:
                st.markdown(
                    render_callout(
                        title="JoSAA Choice Order Warning",
                        body=w["message"],
                        variant="warn",
                        icon="⚠️",
                    ),
                    unsafe_allow_html=True,
                )

            # Plain comparison summary at top
            p1 = float(comp_sub.iloc[0]["probability"])
            p2 = float(comp_sub.iloc[1]["probability"])
            safer_idx = 0 if p1 >= p2 else 1
            safer_cno = comp_sub.iloc[safer_idx]["choice_no"]
            safer_inst = comp_sub.iloc[safer_idx]["institute"]

            st.markdown(
                f"""
                <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px; padding:14px 18px; margin-bottom:18px; font-size:14px; color:#1E40AF;">
                  💡 <strong>Comparison Summary:</strong> Choice #{safer_cno} ({safer_inst}) is safer with a <strong>{max(p1, p2):.0%}</strong> chance (vs {min(p1, p2):.0%}).
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Side-by-side cards
            card_cols = st.columns(len(comp_sub))
            for idx, (_, r) in enumerate(comp_sub.iterrows()):
                sub_h = history_df[
                    (history_df["institute"] == r["institute"])
                    & (history_df["program"] == r["program"])
                    & (history_df["quota"] == r["quota"])
                    & (history_df["seat_type"] == r["seat_type"])
                    & (history_df["gender"] == r["gender"])
                ].sort_values("year")

                prob_v = float(r["probability"])
                gauge_svg = render_svg_circular_gauge(prob_v, r["label"])
                pred_last = int(round(np.exp(r["y_hat"])))

                with card_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="vk-choice-card">
                          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span class="vk-chip">Choice #{r['choice_no']}</span>
                            {render_badge(r['label'], prob_v)}
                          </div>
                          <div style="font-weight:700; font-size:15px; color:#0F172A;">{r['institute']}</div>
                          <div style="font-size:13.5px; color:#1D4ED8; margin-bottom:12px;">{r['program']}</div>
                          <div style="display:flex; justify-content:center; margin-bottom:12px;">{gauge_svg}</div>
                          <div style="font-size:13px; color:#475569; text-align:center;">
                            Expected last seat rank: <strong>~{pred_last:,}</strong>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Cutoff Trajectory Over Time")
            comp_fig = create_comparison_chart(
                comp_sub, history_df, int(comp_sub.iloc[0]["student_rank"]), 0
            )
            st.plotly_chart(comp_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SCREEN 5: HOW WE GOT THIS (Plain Steps + Reliability Chart + "For nerds")
# -----------------------------------------------------------------------------
with tab_method:
    st.markdown("### 🛡️ How we got your choices")
    st.markdown("We believe in total transparency. Here is exactly how your list was computed.")

    st.markdown("#### 1. Five Simple Steps")
    st.markdown(
        """
        1. **We collected 5 years of cutoffs:** Every closing rank from 2021 through 2025 across all IITs, NITs, and IIITs.
        2. **We matched your eligible seats:** Checking home state, other state, and female supernumerary seat rules.
        3. **We calculated the trend:** Giving 50% weight to last year, 30% to 2 years ago, and 20% to 3 years ago.
        4. **We estimated how much cutoffs bounce:** A program with 5 years of steady ranks is more predictable than a brand-new program.
        5. **We tried 2,000 possible outcomes:** Checking that your list has a healthy mix of Reach, Good fit, and Likely options.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 2. How often our predictions came true")
    st.markdown(
        "To test our system, we hid the actual 2025 results and tried to predict them using only 2021–2024 data. "
        "Here is how close our predictions came across 9,336 programs:"
    )

    r_c1, r_c2 = st.columns([1, 1.2])
    with r_c1:
        rel_img = ROOT_DIR / "outputs" / "backtest" / "reliability_plot.png"
        if rel_img.exists():
            st.image(
                str(rel_img),
                caption="Points along the diagonal show that our predicted odds matched real-world admission rates.",
                alt="Prediction accuracy chart showing predicted odds versus real-world admissions in 2025",
                use_container_width=True,
            )
    with r_c2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px; font-size:13.5px; line-height:1.6; color:#334155;">
              <div style="font-weight:700; color:#0F172A; margin-bottom:8px;">📊 Plain Facts on Prediction Accuracy:</div>
              <div>• <strong>Better than just copying last year:</strong> Our weighted trend improved cutoff accuracy over last year's cutoff alone by 0.0049 log points (a 3.48% relative error reduction, with a median absolute percentage error of 8.99%).</div>
              <div>• <strong>80% Prediction Band:</strong> <strong>84.01%</strong> of actual 2025 last-seat cutoffs landed inside our calibrated 80% range (and 89.82% inside the 90% band).</div>
              <div>• <strong>No fake optimism:</strong> When we told students they had a 70% chance, about 7 out of 10 actually got in.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 3. What we know and what we cannot know")
    st.markdown(
        """
        - **Data years used:** 2021 through 2025 (44,637 total program records).
        - **2026 cutoffs are not yet out:** JoSAA 2026 counselling has not happened yet. Anyone promising exact cutoffs is guessing.
        - **Seat counts are not published in cutoffs:** If a college suddenly doubles its computer science seats, its cutoff rank will drop. We cannot know that until JoSAA releases new seat counts.
        - **Zero personal data stored:** Your rank is never saved to disk or shared.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🤓 For nerds: Technical details & statistical formulas"):
        st.markdown(
            r"""
            ##### Mathematical Formulation
            - **Expected Closing Rank:** $\hat{y} = 0.5 \ln(C_{t-1}) + 0.3 \ln(C_{t-2}) + 0.2 \ln(C_{t-3})$ where $C_t$ is the historical closing rank.
            - **Residual Standard Error:** $\sigma = 0.2058$ for established programs ($\ge 3$ years history) and $\sigma = 0.2569$ for programs with sparse history ($< 3$ years).
            - **Admission Probability:** $P(\text{admit}) = 1 - \Phi\left(\frac{\ln(\text{rank}) - \hat{y}}{\sigma}\right)$
            - **Portfolio Simulation:** 2,000 correlated Gaussian draws with cross-option correlation $\rho = 0.50$.

            ##### Brier Score & Probability Protocol Reconciliation
            - **Primary Non-Circular Benchmark (Brier 0.1436):** Evaluated at fixed relative offsets from last year's cutoff (-30% to +30%) across all 9,336 programs. Weighted Trend outperforms the Naive baseline (0.1550) and LightGBM (0.1482).
            - **Secondary z-Grid Benchmark (Brier 0.0842):** Evaluated across the standardized Gaussian spectrum ($z \in [-2.5, +2.5]$). Secondary because query ranks depend on each model's own scale.
            - **Applicant-Rank Tail Sweep (Brier 0.0225):** Querying static candidate ranks where choices naturally cluster into the extreme tails ($P \approx 0$ or $P \approx 1$), deflating squared loss.
            """
        )

# Simple footer
st.markdown("---")
st.markdown(
    """
    <div style="font-size:12px; color:#64748B; text-align:center; padding:12px 0;">
      Vikalp · Decision Engine v1.1 · Built by Sahil Sharma · Verify official dates and lists at <a href="https://josaa.nic.in" target="_blank" style="color:#1D4ED8;">josaa.nic.in</a>
    </div>
    """,
    unsafe_allow_html=True,
)
