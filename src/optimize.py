"""Choice list optimizer and preference scoring engine.

Constructs an optimized, risk-balanced JoSAA choice list:
- Categorizes options: Dream (< 0.30), Target (0.30 to 0.80), Safe (>= 0.80)
- Enforces 20% Dream / 50% Target / 30% Safe target mix
- Scores by priority mode: College-first (prestige proxy) vs Branch-first
- Orders choices from Dream to Safe so near-certain choices don't eliminate higher aspirations
- Flags choices placed after an option with P >= 0.95 as unreachable
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.baseline import compute_weighted_log_trend
from src.eligibility import StudentProfile, filter_eligible_options
from src.predict import admit_probability, get_prediction_interval, label_risk

DEFAULT_SIGMA = 0.2285  # Residual sigma calibrated from Phase 3 & 4
UNREACHABLE_THRESHOLD = 0.95
MIN_LIST_PROB = 0.05
DREAM_MAX_PROB = 0.30
TARGET_MAX_PROB = 0.80
DEFAULT_CUTOFFS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "cutoffs.parquet"
)


def is_architecture_or_planning(program_series: pd.Series) -> pd.Series:
    """Detect B.Arch, B.Planning, and B.Des programs that do not use JEE Main Paper 1 CRL."""
    prog = program_series.astype(str)
    is_arch_or_plan = prog.str.contains("Architecture|Planning", case=False, na=False)
    # Naval Architecture is standard B.Tech engineering using Paper 1, not Paper 2
    is_naval = prog.str.contains("Naval Architecture", case=False, na=False)
    # Bachelor of Design / B.Des uses separate aptitude / UCEED evaluation
    is_bdes = (
        prog.str.strip().str.lower() == "bachelor of design"
    ) | prog.str.contains(r"\b(?:b\.?des|bachelor of design)\b", case=False, na=False)
    return (is_arch_or_plan & (~is_naval)) | is_bdes


def compute_institute_prestige_scores(cutoffs_df: pd.DataFrame) -> Dict[str, float]:
    """Compute prestige proxy based on historical median closing rank of CSE / premier branch."""
    cse_df = cutoffs_df[
        (cutoffs_df["branch_group"] == "CSE")
        & (cutoffs_df["seat_type"] == "OPEN")
        & (cutoffs_df["gender"] == "Gender-Neutral")
    ]
    if not cse_df.empty:
        prestige = cse_df.groupby("institute")["closing_rank"].median().to_dict()
    else:
        prestige = {}

    # For institutes without CSE, use overall minimum median closing rank
    overall = cutoffs_df[
        (cutoffs_df["seat_type"] == "OPEN")
        & (cutoffs_df["gender"] == "Gender-Neutral")
    ].groupby("institute")["closing_rank"].median().to_dict()

    all_institutes = cutoffs_df["institute"].unique()
    scores: Dict[str, float] = {}
    for inst in all_institutes:
        scores[inst] = prestige.get(inst, overall.get(inst, 100000.0))

    return scores


def prepare_latest_options(cutoffs_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare latest cycle seat choices and their weighted-trend predicted cutoffs."""
    # Use the most recent year's offerings as active choice catalog
    max_year = cutoffs_df["year"].max()
    latest = cutoffs_df[cutoffs_df["year"] == max_year].copy()

    # Exclude B.Arch and B.Planning (they use JEE Main Paper 2 ranks, not Paper 1 CRL)
    latest = latest[~is_architecture_or_planning(latest["program"])].copy()

    # Predict upcoming cycle cutoff using latest available years:
    # last year = closing_rank, 2 yrs ago = prev_closing_1, 3 yrs ago = prev_closing_2
    latest["y_hat"] = compute_weighted_log_trend(
        latest["closing_rank"].values,
        latest["prev_closing_1"].values,
        latest["prev_closing_2"].values,
    )

    # Clean fallback for any NaN y_hat: use current closing_rank
    latest["y_hat"] = latest["y_hat"].fillna(np.log(latest["closing_rank"].astype(float)))

    return latest


def score_preference(
    df: pd.DataFrame,
    student: StudentProfile,
    prestige_map: Dict[str, float],
) -> pd.DataFrame:
    """Calculate preference rank score based on College-first or Branch-first mode."""
    scored = df.copy()

    # College prestige rank
    scored["prestige_score"] = scored["institute"].map(
        lambda inst: prestige_map.get(inst, 100000.0)
    )

    # Branch preference score
    branch_order: Dict[str, int] = {}
    if student.preferred_branches:
        for idx, b in enumerate(student.preferred_branches):
            branch_order[b.strip().lower()] = idx

    def get_branch_rank(branch_grp: str, prog_name: str) -> int:
        b_low = branch_grp.strip().lower()
        if b_low in branch_order:
            return branch_order[b_low]
        # Check if any preferred branch keyword is in program name
        for b_name, idx in branch_order.items():
            if b_name in prog_name.lower():
                return idx
        return 999

    scored["branch_score"] = scored.apply(
        lambda r: get_branch_rank(r["branch_group"], r["program"]), axis=1
    )

    if student.priority_mode.lower() == "branch-first":
        # Branch-first: prioritize branch preference, then college prestige
        scored = scored.sort_values(
            by=["branch_score", "prestige_score", "probability"],
            ascending=[True, True, False],
        )
    else:
        # College-first (default): prioritize college prestige, then branch preference
        scored = scored.sort_values(
            by=["prestige_score", "branch_score", "probability"],
            ascending=[True, True, False],
        )

    return scored


def apply_risk_mix_balancing(
    scored_df: pd.DataFrame, list_size: int
) -> pd.DataFrame:
    """Enforce 20% Dream / 50% Target / 30% Safe target mix.

    Safe and Target slots are reserved first (30% and 50%) and Dream fills the rest,
    so Dream can never crowd out Safe. If fewer Safe options exist than the quota,
    fill from the next tier (Target, then Dream).
    """
    actual_list_size = min(len(scored_df), list_size)
    target_safe = int(round(0.30 * actual_list_size))
    target_target = int(round(0.50 * actual_list_size))
    target_dream = actual_list_size - target_safe - target_target

    safe_pool = scored_df[scored_df["label"] == "Safe"]
    target_pool = scored_df[scored_df["label"] == "Target"]
    dream_pool = scored_df[scored_df["label"] == "Dream"]

    # 1. Reserve Safe first (30%)
    selected_safe = safe_pool.head(target_safe)
    safe_deficit = target_safe - len(selected_safe)

    # 2. Reserve Target second (50% + Safe deficit filled from next tier)
    needed_target = target_target + max(0, safe_deficit)
    selected_target = target_pool.head(needed_target)
    target_deficit = needed_target - len(selected_target)

    # 3. Dream fills the rest (20% + any deficit from Safe/Target)
    needed_dream = target_dream + max(0, target_deficit)
    selected_dream = dream_pool.head(needed_dream)

    # 4. If list is still not full (e.g. Dream had fewer options), backfill from remaining Target, then Safe
    selected_indices = (
        set(selected_safe.index)
        | set(selected_target.index)
        | set(selected_dream.index)
    )
    remaining_needed = actual_list_size - len(selected_indices)

    if remaining_needed > 0:
        remaining_candidates = scored_df[~scored_df.index.isin(selected_indices)]
        backfill_target = remaining_candidates[
            remaining_candidates["label"] == "Target"
        ].head(remaining_needed)
        selected_target = pd.concat([selected_target, backfill_target])
        selected_indices.update(backfill_target.index)
        remaining_needed = actual_list_size - len(selected_indices)

    if remaining_needed > 0:
        remaining_candidates = scored_df[~scored_df.index.isin(selected_indices)]
        backfill_safe = remaining_candidates[
            remaining_candidates["label"] == "Safe"
        ].head(remaining_needed)
        selected_safe = pd.concat([selected_safe, backfill_safe])
        selected_indices.update(backfill_safe.index)
        remaining_needed = actual_list_size - len(selected_indices)

    # Assemble choice list in proper JoSAA strategic order: Dream -> Target -> Safe
    ordered_list = pd.concat([selected_dream, selected_target, selected_safe]).reset_index(drop=True)
    return ordered_list


def flag_unreachable_options(ordered_df: pd.DataFrame) -> pd.DataFrame:
    """Flag any choice placed after a choice with P >= 0.95 as unreachable."""
    df = ordered_df.copy()
    warnings: List[str] = []
    first_certain_idx: Optional[int] = None

    for idx, prob in enumerate(df["probability"], start=1):
        if first_certain_idx is not None:
            warnings.append(
                f"Unreachable: choice placed after choice #{first_certain_idx} with >=95% probability"
            )
        else:
            warnings.append("")
            if prob >= UNREACHABLE_THRESHOLD:
                first_certain_idx = idx

    df["warning"] = warnings
    return df


def build_list(
    student: Union[StudentProfile, dict],
    cutoffs_path: Optional[Union[Path, str]] = None,
    sigma: Optional[float] = None,
) -> pd.DataFrame:
    """Build an optimized JoSAA choice list for a student profile."""
    if isinstance(student, dict):
        student_obj = StudentProfile(**student)
    else:
        student_obj = student

    if student_obj.round_type.lower() != "final":
        raise ValueError(
            f"Unsupported round_type '{student_obj.round_type}'. Vikalp uses official JoSAA final-round cutoffs "
            "(Round 6 for 2021-2023, 2025; Round 5 for 2024), which represent the ultimate admission threshold."
        )

    resolved_path = (
        Path(cutoffs_path) if cutoffs_path is not None else DEFAULT_CUTOFFS_PATH
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"{resolved_path} not found. Run Phase 2 first.")

    cutoffs_df = pd.read_parquet(resolved_path)
    prestige_map = compute_institute_prestige_scores(cutoffs_df)
    active_options = prepare_latest_options(cutoffs_df)

    # 1. Filter to eligible seats
    eligible = filter_eligible_options(active_options, student_obj)
    if eligible.empty:
        return pd.DataFrame()

    # 2. Compute probabilities and intervals
    sigma_val = sigma if sigma is not None else DEFAULT_SIGMA
    eligible["probability"] = admit_probability(
        student_rank=eligible["effective_student_rank"].values,
        y_hat=eligible["y_hat"].values,
        sigma=sigma_val,
    )

    # JoSAA Rule: In JoSAA, a candidate's choice is strictly (institute, program).
    # All eligible pools (Quota: HS/OS/AI, Gender: Gender-Neutral/Female-only, Seat Type: OPEN/Category)
    # are evaluated under the SAME choice. If a student is eligible for multiple pools,
    # they qualify if they meet the threshold of ANY eligible pool.
    # Therefore, collapse to ONE choice per (institute, program), retaining the best eligible pool
    # (highest admission probability) as the effective admission chance.
    eligible = (
        eligible.sort_values(by="probability", ascending=False)
        .drop_duplicates(subset=["institute", "program"], keep="first")
        .reset_index(drop=True)
    )

    # Enforce minimum probability floor (MIN_LIST_PROB = 0.05)
    # Options below 5% must never appear in the choice list!
    eligible = eligible[eligible["probability"] >= MIN_LIST_PROB].copy()
    if eligible.empty:
        return pd.DataFrame()

    eligible["label"] = [label_risk(p) for p in eligible["probability"]]

    intervals = [get_prediction_interval(y, sigma_val) for y in eligible["y_hat"].values]
    eligible["predicted_closing_rank"] = [i[0] for i in intervals]
    eligible["predicted_range"] = [f"{i[1]:,} - {i[2]:,}" for i in intervals]

    # 3. Score by student priority preference (College-first vs Branch-first)
    scored = score_preference(eligible, student_obj, prestige_map)

    # 4. Enforce 20/50/30 Dream/Target/Safe mix
    balanced = apply_risk_mix_balancing(scored, student_obj.list_size)

    # 5. Check and flag unreachable choices
    final_list = flag_unreachable_options(balanced)

    # 6. Format final presentation columns
    final_list["choice_no"] = range(1, len(final_list) + 1)
    final_list["student_rank"] = final_list["effective_student_rank"]
    output_cols = [
        "choice_no",
        "institute",
        "institute_type",
        "state",
        "program",
        "branch_group",
        "quota",
        "seat_type",
        "gender",
        "student_rank",
        "y_hat",
        "probability",
        "label",
        "predicted_closing_rank",
        "predicted_range",
        "warning",
    ]
    return final_list[output_cols]
