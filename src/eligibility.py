"""Eligibility filtering logic for JoSAA applicants.

Enforces official JoSAA counselling allotment rules:
- Quota: AI is always eligible; HS if institute state == student home state; OS otherwise.
- Exam Rank: IIT requires JEE Advanced rank; NIT/IIIT/GFTI requires JEE Main rank.
- Category: OPEN seats evaluated with CRL; category seats evaluated with category rank.
- Gender: Female students eligible for Gender-Neutral and Female-only; male for Gender-Neutral only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class StudentProfile:
    """Represents a student's ranks, reservation category, gender, and preferences."""
    main_rank: Optional[int] = None           # JEE Main CRL (mandatory for NIT/IIIT/GFTI)
    adv_rank: Optional[int] = None            # JEE Advanced CRL (mandatory for IIT)
    category: str = "OPEN"                   # OPEN, EWS, OBC-NCL, SC, ST
    category_rank: Optional[int] = None       # Category rank (if not OPEN)
    gender: str = "Gender-Neutral"            # "Gender-Neutral" or "Female-only"
    home_state: str = ""                      # e.g. "Delhi", "Maharashtra"
    preferred_branches: Optional[List[str]] = None  # e.g. ["CSE", "ECE"] or specific names
    preferred_states: Optional[List[str]] = None    # e.g. ["Delhi", "Karnataka"]
    institute_types: Optional[List[str]] = None    # e.g. ["IIT", "NIT", "IIIT", "GFTI"]
    list_size: int = 50
    priority_mode: str = "College-first"      # "College-first" or "Branch-first"
    round_type: str = "final"                 # Cutoff round: "final" (default, terminal round)
    ordering_strategy: str = "strategic"      # "strategic" (Dream->Target->Safe) or "preference" (strict student priority)

    def validate(self) -> None:
        """Validate student profile parameters, raising descriptive ValueError if invalid."""
        has_main = self.main_rank is not None and self.main_rank > 0
        has_adv = self.adv_rank is not None and self.adv_rank > 0
        if not (has_main or has_adv):
            raise ValueError(
                "At least one valid positive rank must be provided (JEE Main CRL or JEE Advanced rank > 0)."
            )
        if self.main_rank is not None and self.main_rank <= 0:
            raise ValueError(f"Invalid main_rank: {self.main_rank}. Rank must be a positive integer.")
        if self.adv_rank is not None and self.adv_rank <= 0:
            raise ValueError(f"Invalid adv_rank: {self.adv_rank}. Rank must be a positive integer.")
        if self.category_rank is not None and self.category_rank <= 0:
            raise ValueError(f"Invalid category_rank: {self.category_rank}. Category rank must be a positive integer.")
        if self.list_size <= 0:
            raise ValueError(f"Invalid list_size: {self.list_size}. List size must be a positive integer.")
        valid_modes = {"college-first", "branch-first"}
        if self.priority_mode.lower() not in valid_modes:
            raise ValueError(f"Invalid priority_mode '{self.priority_mode}'. Must be one of {valid_modes}.")
        valid_strategies = {"strategic", "preference"}
        if self.ordering_strategy.lower() not in valid_strategies:
            raise ValueError(f"Invalid ordering_strategy '{self.ordering_strategy}'. Must be one of {valid_strategies}.")


def is_quota_eligible(quota: str, inst_state: str, home_state: str) -> bool:
    """Check quota eligibility for an institute and student home state."""
    q = quota.strip().upper()
    if q == "AI":
        return True

    inst_st = inst_state.strip().lower()
    home_st = home_state.strip().lower()

    if q == "HS":
        return bool(inst_st and home_st and inst_st == home_st)
    elif q == "OS":
        return bool(not home_st or inst_st != home_st)

    return False


def is_gender_eligible(seat_gender: str, student_gender: str) -> bool:
    """Check gender eligibility. Female students qualify for both gender pools."""
    s_gender = student_gender.strip().lower()
    seat_gen = seat_gender.strip().lower()

    if "female" in s_gender:
        return True  # Female students eligible for both Gender-Neutral and Female-only
    return "female" not in seat_gen  # Non-female students eligible only for Gender-Neutral


def get_effective_student_rank(
    student: StudentProfile, institute_type: str, seat_type: str
) -> Optional[int]:
    """Return the applicable rank for a given seat, strictly separating CRL vs Category rank."""
    is_iit = institute_type.strip().upper() == "IIT"

    # IIT seats require JEE Advanced rank
    if is_iit:
        if not student.adv_rank or student.adv_rank <= 0:
            return None
        crl_rank = student.adv_rank
    else:
        if not student.main_rank or student.main_rank <= 0:
            return None
        crl_rank = student.main_rank

    # OPEN seats always use CRL rank
    if seat_type.strip().upper() == "OPEN":
        return crl_rank

    # Non-OPEN seat: student must match the category AND provide a valid category rank
    if seat_type.strip().upper() == student.category.strip().upper():
        if student.category_rank and student.category_rank > 0:
            return student.category_rank
        # Without a category rank, candidate competes strictly for OPEN seats via CRL
        return None

    return None


def filter_eligible_options(
    options_df: pd.DataFrame, student: StudentProfile
) -> pd.DataFrame:
    """Filter seat options to only those the student is eligible to compete for."""
    df = options_df.copy()

    # 1. Exam rank filter (IIT vs non-IIT)
    if not student.adv_rank or student.adv_rank <= 0:
        df = df[df["institute_type"] != "IIT"]
    if not student.main_rank or student.main_rank <= 0:
        df = df[df["institute_type"] == "IIT"]

    if df.empty:
        return pd.DataFrame()

    # 2. Quota filter
    quota_mask = df.apply(
        lambda r: is_quota_eligible(r["quota"], r["state"], student.home_state),
        axis=1,
    )
    df = df[quota_mask]

    # 3. Gender filter
    gender_mask = df["gender"].apply(
        lambda g: is_gender_eligible(g, student.gender)
    )
    df = df[gender_mask]

    # 4. Seat Type / Category filter
    valid_seat_types = {"OPEN"}
    if student.category and student.category.upper() != "OPEN":
        valid_seat_types.add(student.category.upper())
    df = df[df["seat_type"].isin(valid_seat_types)]

    # 5. User Preference Filters: Institute Types
    if student.institute_types:
        chosen_types = {t.strip().upper() for t in student.institute_types}
        df = df[df["institute_type"].str.upper().isin(chosen_types)]

    # 6. User Preference Filters: States
    if student.preferred_states:
        chosen_states = {s.strip().lower() for s in student.preferred_states}
        df = df[df["state"].str.lower().isin(chosen_states)]

    # 7. User Preference Filters: Branches
    if student.preferred_branches:
        chosen_branches = [b.strip().lower() for b in student.preferred_branches]
        branch_mask = df["branch_group"].str.lower().isin(chosen_branches) | df["program"].apply(
            lambda p: any(b in p.lower() for b in chosen_branches)
        )
        df = df[branch_mask]

    if df.empty:
        return pd.DataFrame()

    # Assign effective student rank for each eligible row
    df["effective_student_rank"] = df.apply(
        lambda r: get_effective_student_rank(student, r["institute_type"], r["seat_type"]),
        axis=1,
    )
    df = df[df["effective_student_rank"].notna()].copy()
    df["effective_student_rank"] = df["effective_student_rank"].astype(int)

    return df
