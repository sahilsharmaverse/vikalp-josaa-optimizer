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
    """Enforce 20% Dream / 50% Target / 30% Safe target mix with backfilling."""
    target_dream = int(round(0.20 * list_size))
    target_target = int(round(0.50 * list_size))
    target_safe = list_size - target_dream - target_target

    dream_pool = scored_df[scored_df["label"] == "Dream"]
    target_pool = scored_df[scored_df["label"] == "Target"]
    safe_pool = scored_df[scored_df["label"] == "Safe"]

    # Initial selection
    selected_dream = dream_pool.head(target_dream)
    selected_target = target_pool.head(target_target)
    selected_safe = safe_pool.head(target_safe)

    selected_indices = set(selected_dream.index) | set(selected_target.index) | set(selected_safe.index)
    remaining_needed = list_size - len(selected_indices)

    # Backfill if any pool had fewer options than its allocation
    if remaining_needed > 0:
        remaining_candidates = scored_df[~scored_df.index.isin(selected_indices)]
        # Prefer backfilling Target, then Safe, then Dream
        backfill_target = remaining_candidates[remaining_candidates["label"] == "Target"].head(remaining_needed)
        selected_target = pd.concat([selected_target, backfill_target])
        selected_indices.update(backfill_target.index)
        remaining_needed = list_size - len(selected_indices)

        if remaining_needed > 0:
            remaining_candidates = scored_df[~scored_df.index.isin(selected_indices)]
            backfill_safe = remaining_candidates[remaining_candidates["label"] == "Safe"].head(remaining_needed)
            selected_safe = pd.concat([selected_safe, backfill_safe])
            selected_indices.update(backfill_safe.index)
            remaining_needed = list_size - len(selected_indices)

        if remaining_needed > 0:
            remaining_candidates = scored_df[~scored_df.index.isin(selected_indices)]
            backfill_dream = remaining_candidates[remaining_candidates["label"] == "Dream"].head(remaining_needed)
            selected_dream = pd.concat([selected_dream, backfill_dream])

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
    cutoffs_path: Path = Path("data/processed/cutoffs.parquet"),
    sigma: float = DEFAULT_SIGMA,
) -> pd.DataFrame:
    """Build an optimized JoSAA choice list for a student profile."""
    if isinstance(student, dict):
        student_obj = StudentProfile(**student)
    else:
        student_obj = student

    if not cutoffs_path.exists():
        raise FileNotFoundError(f"{cutoffs_path} not found. Run Phase 2 first.")

    cutoffs_df = pd.read_parquet(cutoffs_path)
    prestige_map = compute_institute_prestige_scores(cutoffs_df)
    active_options = prepare_latest_options(cutoffs_df)

    # 1. Filter to eligible seats
    eligible = filter_eligible_options(active_options, student_obj)
    if eligible.empty:
        return pd.DataFrame()

    # 2. Compute probabilities and intervals
    eligible["probability"] = admit_probability(
        student_rank=eligible["effective_student_rank"].values,
        y_hat=eligible["y_hat"].values,
        sigma=sigma,
    )
    eligible["label"] = [label_risk(p) for p in eligible["probability"]]

    intervals = [get_prediction_interval(y, sigma) for y in eligible["y_hat"].values]
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
