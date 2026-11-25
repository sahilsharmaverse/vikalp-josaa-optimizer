"""PDF Report Generator for Vikalp (v1.1 Feature 5).

Builds an in-memory (BytesIO) executive PDF report of the student's optimized
JoSAA choice list using ReportLab:
- Print-friendly layout: clean white background, dark text, light tier fills.
- Page 1: Title, IST timestamp, Student Profile, Strategic Summary, Rank used, Choice table start.
- Choice table: repeated header row (repeatRows=1), 'Page x of y' footer via NumberedCanvas.
- Dagger (†) footnote for unreachable choices placed after an option with P >= 0.95.
- Last page: How to read tiers, probability calculation in 3 plain lines, model metrics,
  JoSAA allocation rule, full disclaimer, and official verification callout.
- Strictly in-memory generation with zero student data persisted to disk.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.sax.saxutils as saxutils

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.eligibility import StudentProfile
from src.simulation import get_model_metrics
from src.trend import (
    compute_trend,
)

# Color constants for print-friendly light tier fills
TIER_FILL_DREAM = colors.HexColor("#FEE2E2")  # Light red fill
TIER_TEXT_DREAM = colors.HexColor("#991B1B")  # Deep red text
TIER_FILL_TARGET = colors.HexColor("#FEF3C7")  # Light amber fill
TIER_TEXT_TARGET = colors.HexColor("#92400E")  # Deep amber text
TIER_FILL_SAFE = colors.HexColor("#DCFCE7")  # Light green fill
TIER_TEXT_SAFE = colors.HexColor("#166534")  # Deep green text

COLOR_NAVY = colors.HexColor("#0F172A")
COLOR_SLATE = colors.HexColor("#334155")
COLOR_MUTED = colors.HexColor("#64748B")
COLOR_BORDER = colors.HexColor("#CBD5E1")
COLOR_LIGHT_BG = colors.HexColor("#F8FAFC")
COLOR_WHITE = colors.HexColor("#FFFFFF")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that accumulates pages and stamps 'Page X of Y' and headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_MUTED)

        # Top running header on pages > 1
        if self._pageNumber > 1:
            self.drawString(36, 11 * 72 - 28, "Vikalp — JoSAA Choice List Report")
            self.drawRightString(8.5 * 72 - 36, 11 * 72 - 28, "Candidate Choice Plan")
            self.setStrokeColor(COLOR_BORDER)
            self.setLineWidth(0.5)
            self.line(36, 11 * 72 - 32, 8.5 * 72 - 36, 11 * 72 - 32)

        # Bottom running footer on all pages
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(36, 32, 8.5 * 72 - 36, 32)

        self.drawString(36, 20, "Vikalp · Built by Sahil Sharma · Verify at josaa.nic.in")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 36, 20, page_str)

        self.restoreState()


def safe_escape(val: Any) -> str:
    """Safely escape text for ReportLab Paragraph cells to prevent XML parser errors."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    text = str(val).strip()
    return saxutils.escape(text)


def build_pdf_report(
    choice_df: pd.DataFrame,
    student_profile: StudentProfile,
    seat_chance: float,
    rank_shift_pct: int = 0,
    active_rank: Optional[int] = None,
    history_df: Optional[pd.DataFrame] = None,
    metrics_path: Path = Path("models/metrics.json"),
) -> bytes:
    """Generate the complete executive JoSAA choice list PDF in memory.

    Parameters
    ----------
    choice_df : pd.DataFrame
        Active choice list DataFrame with columns: choice_no, institute, program,
        quota, seat_type, shifted_probability / probability, shifted_label / label,
        y_hat.
    student_profile : StudentProfile
        Student candidate details.
    seat_chance : float
        Monte Carlo portfolio seat admission chance (0.0 to 1.0).
    rank_shift_pct : int, default 0
        What-if percentage rank shift.
    active_rank : int, optional
        Effective rank used for rescoring.
    history_df : pd.DataFrame, optional
        Historical cutoffs for trend badge computation.
    metrics_path : Path, default models/metrics.json
        Path to model metrics.

    Returns
    -------
    bytes
        In-memory PDF file bytes.
    """
    buffer = io.BytesIO()

    # Document setup: 8.5 x 11 inches, 0.5 in margins (36 pt)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=COLOR_NAVY,
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=COLOR_MUTED,
    )
    style_section_h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=COLOR_NAVY,
        spaceBefore=6,
        spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=COLOR_SLATE,
    )
    style_body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=COLOR_NAVY,
    )
    style_meta_label = ParagraphStyle(
        "MetaLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=COLOR_MUTED,
    )
    style_meta_val = ParagraphStyle(
        "MetaVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=COLOR_NAVY,
    )
    style_tbl_header = ParagraphStyle(
        "TblHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=COLOR_WHITE,
        alignment=1,  # Center
    )
    style_tbl_cell = ParagraphStyle(
        "TblCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=COLOR_NAVY,
    )
    style_tbl_cell_center = ParagraphStyle(
        "TblCellCenter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=COLOR_NAVY,
        alignment=1,
    )
    style_tbl_cell_bold = ParagraphStyle(
        "TblCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=COLOR_NAVY,
        alignment=1,
    )
    style_footnote = ParagraphStyle(
        "FootnoteText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#B91C1C"),  # Red tone
    )

    story: List[Any] = []

    # 1. Header & Title Banner
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_tz).strftime("%d %b %Y, %I:%M %p IST")

    story.append(Paragraph("Vikalp, JoSAA Choice List Report", style_title))
    story.append(
        Paragraph(
            f"Built by Sahil Sharma &nbsp;·&nbsp; Generated on: {now_ist} &nbsp;·&nbsp; Historical Data: 2021–2025 (2026 data unavailable) &nbsp;·&nbsp; Model: Nov 2026",
            style_subtitle,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<b>Advisory Notice:</b> Probabilities are statistical estimates based on JoSAA 2021–2025 final-round cutoffs, not official guarantees. "
            "2026 cutoffs have not yet been released. Zero personal data is stored or logged. Always verify official allotment rules at <u>josaa.nic.in</u>.",
            style_footnote,
        )
    )
    story.append(Spacer(1, 6))

    # Determine effective rank used
    if active_rank is None:
        active_rank = (
            student_profile.main_rank
            if student_profile.main_rank and student_profile.main_rank > 0
            else student_profile.adv_rank
        )
        if (
            student_profile.category != "OPEN"
            and student_profile.category_rank
            and student_profile.category_rank > 0
        ):
            active_rank = student_profile.category_rank

    if rank_shift_pct != 0:
        rank_used_str = (
            f"Rank used: <strong>{active_rank:,}</strong> (simulated {rank_shift_pct:+d}%)"
        )
    else:
        rank_used_str = f"Rank used: <strong>{active_rank:,}</strong>"

    # 2. Student Profile Table (White background, light border)
    crl_str = (
        f"{student_profile.main_rank:,}"
        if student_profile.main_rank and student_profile.main_rank > 0
        else "Not entered"
    )
    adv_str = (
        f"{student_profile.adv_rank:,}"
        if student_profile.adv_rank and student_profile.adv_rank > 0
        else "Not entered"
    )
    cat_rank_str = (
        f"{student_profile.category_rank:,}"
        if (
            student_profile.category != "OPEN"
            and student_profile.category_rank
            and student_profile.category_rank > 0
        )
        else "N/A (OPEN / Not specified)"
    )
    branches_str = (
        ", ".join(student_profile.preferred_branches)
        if student_profile.preferred_branches
        else "All Branches (No branch restriction)"
    )

    profile_data = [
        [
            Paragraph("JEE Main CRL", style_meta_label),
            Paragraph(safe_escape(crl_str), style_meta_val),
            Paragraph("Category", style_meta_label),
            Paragraph(safe_escape(student_profile.category), style_meta_val),
        ],
        [
            Paragraph("JEE Advanced Rank", style_meta_label),
            Paragraph(safe_escape(adv_str), style_meta_val),
            Paragraph(f"{student_profile.category} Rank", style_meta_label),
            Paragraph(safe_escape(cat_rank_str), style_meta_val),
        ],
        [
            Paragraph("Gender Pool", style_meta_label),
            Paragraph(safe_escape(student_profile.gender), style_meta_val),
            Paragraph("Home State", style_meta_label),
            Paragraph(safe_escape(student_profile.home_state), style_meta_val),
        ],
        [
            Paragraph("Preferred Branches", style_meta_label),
            Paragraph(safe_escape(branches_str), style_meta_val),
            Paragraph("Active Rank Callout", style_meta_label),
            Paragraph(rank_used_str, style_meta_val),
        ],
        [
            Paragraph("Historical Data", style_meta_label),
            Paragraph("2021–2025 (2026 data not yet available)", style_meta_val),
            Paragraph("Privacy Guarantee", style_meta_label),
            Paragraph("Zero candidate data stored or logged", style_meta_val),
        ],
    ]

    profile_table = Table(
        profile_data,
        colWidths=[100, 170, 100, 170],
    )
    profile_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.75, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(profile_table)
    story.append(
        Paragraph(
            "<font size='7' color='#64748B'>* Note: B.Arch, B.Planning, and B.Des use separate entrance ranks and are not included in v1.</font>",
            style_body,
        )
    )
    story.append(Spacer(1, 6))

    # 3. Strategic Summary Box
    # Probability and Tier counts
    prob_col = (
        "shifted_probability"
        if "shifted_probability" in choice_df.columns
        else "probability"
    )
    label_col = (
        "shifted_label" if "shifted_label" in choice_df.columns else "label"
    )

    total_choices = len(choice_df)
    dream_count = (
        int((choice_df[label_col] == "Dream").sum())
        if not choice_df.empty
        else 0
    )
    target_count = (
        int((choice_df[label_col] == "Target").sum())
        if not choice_df.empty
        else 0
    )
    safe_count = (
        int((choice_df[label_col] == "Safe").sum()) if not choice_df.empty else 0
    )

    summary_data = [
        [
            Paragraph(
                f"Seat Admission Chance: <strong>{seat_chance * 100:.1f}%</strong> (estimate)",
                style_body_bold,
            ),
            Paragraph(
                f"Choice Mix: <strong>Dream {dream_count}</strong> &middot; "
                f"<strong>Target {target_count}</strong> &middot; "
                f"<strong>Safe {safe_count}</strong> "
                f"(Total: {total_choices} choices)",
                style_body_bold,
            ),
        ],
        [
            Paragraph(
                "Monte Carlo estimate (2,000 simulations) assuming a 50% shared exam/counselling shock across all choices.",
                style_body,
            ),
            Paragraph(
                "Strategic allocation target: 20% Dream (reach), 50% Target (core), 30% Safe (anchors).",
                style_body,
            ),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[270, 270],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_WHITE),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0284C7")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)

    # Low safe options warning (< 5 Safe options)
    if safe_count < 5:
        story.append(Spacer(1, 4))
        style_safe_warning = ParagraphStyle(
            "SafeWarningText",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#991B1B"),
        )
        warning_table = Table(
            [[Paragraph("⚠️ WARNING: Your list has few or no safe options. Add more branches, institute types or states so you have a fallback.", style_safe_warning)]],
            colWidths=[540],
        )
        warning_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEE2E2")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#EF4444")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(warning_table)

    story.append(Spacer(1, 8))

    # 4. Choice List Table
    story.append(
        Paragraph("Ordered JoSAA Preference Choice List", style_section_h2)
    )

    # Read calibrated model metrics for 80% range and last-page summary
    if metrics_path is not None and Path(metrics_path).exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = get_model_metrics()
    else:
        metrics = get_model_metrics()
    model_sigma = metrics.get("sigma", 0.2285)

    # Detect unreachable choices (sitting after an option with P >= 0.95)
    unreachable_flags: Dict[int, bool] = {}
    first_certain_cno: Optional[int] = None
    has_unreachable = False

    for idx, row in choice_df.iterrows():
        cno = int(row.get("choice_no", idx + 1))
        prob_val = float(row.get(prob_col, 0.0))
        if first_certain_cno is not None:
            unreachable_flags[cno] = True
            has_unreachable = True
        else:
            unreachable_flags[cno] = False
            if prob_val >= 0.95:
                first_certain_cno = cno

    # Table Column Widths: exactly 540 pt printable width
    # Col 0: # (24 pt)
    # Col 1: Institute (126 pt)
    # Col 2: Program (136 pt)
    # Col 3: Quota / Seat (55 pt)
    # Col 4: Prob % (44 pt)
    # Col 5: Tier (45 pt)
    # Col 6: Predicted (80% Range) (65 pt)
    # Col 7: Trend (45 pt)
    col_widths = [24, 126, 136, 55, 44, 45, 65, 45]

    table_data = [
        [
            Paragraph("<b>#</b>", style_tbl_header),
            Paragraph("<b>Institute</b>", style_tbl_header),
            Paragraph("<b>Academic Program</b>", style_tbl_header),
            Paragraph("<b>Quota / Seat</b>", style_tbl_header),
            Paragraph("<b>Admit %</b>", style_tbl_header),
            Paragraph("<b>Tier</b>", style_tbl_header),
            Paragraph("<b>Pred. Cutoff (80% Range)</b>", style_tbl_header),
            Paragraph("<b>Trend</b>", style_tbl_header),
        ]
    ]

    table_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]

    for row_idx, (_, r) in enumerate(choice_df.iterrows(), start=1):
        cno = int(r.get("choice_no", row_idx))
        is_blocked = unreachable_flags.get(cno, False)
        cno_display = f"{cno}†" if is_blocked else str(cno)

        inst_text = safe_escape(r.get("institute", ""))
        prog_text = safe_escape(r.get("program", ""))
        quota_text = safe_escape(f"{r.get('quota', '')} · {r.get('seat_type', '')}")

        p_val = float(r.get(prob_col, 0.0))
        p_str = f"{p_val * 100:.1f}%"

        tier_lbl = str(r.get(label_col, "Target"))

        # Predicted closing rank and 80% interval
        y_hat_val = float(r.get("y_hat", 0.0))
        if y_hat_val > 0:
            pred_cutoff = int(round(math.exp(y_hat_val)))
            low_80 = int(round(math.exp(y_hat_val - 1.28 * model_sigma)))
            high_80 = int(round(math.exp(y_hat_val + 1.28 * model_sigma)))
            pred_str = f"{pred_cutoff:,}<br/><font size='6' color='#64748B'>({low_80:,}–{high_80:,})</font>"
        else:
            pred_str = "—"

        # Trend badge text
        trend_badge = "Stable"
        if history_df is not None and not history_df.empty:
            inst_name = str(r.get("institute", ""))
            prog_name = str(r.get("program", ""))
            quota_name = str(r.get("quota", ""))
            seat_name = str(r.get("seat_type", ""))
            gender_name = str(r.get("gender", ""))

            sub_hist = history_df[
                (history_df["institute"] == inst_name)
                & (history_df["program"] == prog_name)
                & (history_df["quota"] == quota_name)
                & (history_df["seat_type"] == seat_name)
                & (history_df["gender"] == gender_name)
            ]
            if not sub_hist.empty:
                trend_res = compute_trend(
                    sub_hist.sort_values("year")["closing_rank"].tolist()
                )
                trend_badge = trend_res.get("badge", "Stable")
        elif "trend" in r:
            trend_badge = str(r["trend"])

        # Light tier fill formatting
        if tier_lbl == "Dream":
            fill_col = TIER_FILL_DREAM
            text_col = "#991B1B"
        elif tier_lbl == "Safe":
            fill_col = TIER_FILL_SAFE
            text_col = "#166534"
        else:
            fill_col = TIER_FILL_TARGET
            text_col = "#92400E"

        tier_cell_html = f"<font color='{text_col}'><b>{safe_escape(tier_lbl)}</b></font>"

        # Alternating row background
        bg_col = COLOR_WHITE if row_idx % 2 != 0 else COLOR_LIGHT_BG
        table_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg_col))
        table_styles.append(("BACKGROUND", (5, row_idx), (5, row_idx), fill_col))

        table_data.append(
            [
                Paragraph(cno_display, style_tbl_cell_bold),
                Paragraph(inst_text, style_tbl_cell),
                Paragraph(prog_text, style_tbl_cell),
                Paragraph(quota_text, style_tbl_cell_center),
                Paragraph(p_str, style_tbl_cell_center),
                Paragraph(tier_cell_html, style_tbl_cell_center),
                Paragraph(pred_str, style_tbl_cell_center),
                Paragraph(safe_escape(trend_badge), style_tbl_cell_center),
            ]
        )

    choice_table = Table(
        table_data,
        colWidths=col_widths,
        repeatRows=1,  # Repeated table header row on every page
    )
    choice_table.setStyle(TableStyle(table_styles))
    story.append(choice_table)
    story.append(Spacer(1, 8))

    # Dagger footnote if unreachable options exist
    if has_unreachable and first_certain_cno is not None:
        story.append(
            Paragraph(
                f"† <b>Strategic JoSAA Order Warning:</b> Options marked with a dagger (†) are placed below choice #{first_certain_cno}, "
                f"which has ≥95% admission probability. Under JoSAA's strict top-to-bottom allocation rules, candidates are allotted "
                f"the very first option they qualify for. Consequently, any choice placed below a near-certain option is effectively unreachable. "
                f"Review your list order if you prefer any of those options.",
                style_footnote,
            )
        )
        story.append(Spacer(1, 8))

    # 5. Last Page: Methodology, Model Metrics, Strategic Rules & Disclaimer
    story.append(PageBreak())

    story.append(
        Paragraph("Methodology, Reference Guide & Strategic Rules", style_title)
    )
    story.append(
        Paragraph(
            "Essential context on statistical models, risk tiers, JoSAA allocation mechanics, and official verification.",
            style_subtitle,
        )
    )
    story.append(Spacer(1, 10))

    # Subsection A: How to read Dream / Target / Safe
    story.append(Paragraph("1. How to Read Risk Tiers", style_section_h2))
    guide_data = [
        [
            Paragraph("<b>Dream</b><br/>(5% &le; P &lt; 30%)", style_tbl_cell_bold),
            Paragraph(
                "Aspirational, highly selective branches or institutes (5% &le; P &lt; 30%). Low allotment probability, but zero penalty for placing them at the top. Options below 5% are excluded to ensure list quality. Never omit them, but do not rely on them for your seat.",
                style_body,
            ),
        ],
        [
            Paragraph("<b>Target</b><br/>(30% &le; P &lt; 80%)", style_tbl_cell_bold),
            Paragraph(
                "Competitive zone tightly calibrated around your rank. These represent realistic matches where the majority of allotments occur across past counselling rounds.",
                style_body,
            ),
        ],
        [
            Paragraph("<b>Safe</b><br/>(P &ge; 80%)", style_tbl_cell_bold),
            Paragraph(
                "High-certainty safety anchors where previous closing cutoffs were comfortably relaxed. Essential towards the bottom of your list to guarantee you never exit counselling without an allotment.",
                style_body,
            ),
        ],
    ]
    guide_table = Table(guide_data, colWidths=[120, 420])
    guide_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), TIER_FILL_DREAM),
                ("BACKGROUND", (0, 1), (0, 1), TIER_FILL_TARGET),
                ("BACKGROUND", (0, 2), (0, 2), TIER_FILL_SAFE),
                ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(guide_table)
    story.append(Spacer(1, 8))

    # Subsection B: How probability is computed in 3 plain lines
    story.append(
        Paragraph("2. How Admission Probability is Computed", style_section_h2)
    )
    prob_lines = (
        "• <b>Line 1 (Expected Cutoff):</b> The expected closing rank (ŷ) for the upcoming cycle is predicted from past 5-year JoSAA closing ranks using weighted exponential trend regression.<br/>"
        "• <b>Line 2 (Uncertainty Quantification):</b> Annual fluctuations in applicant choices and seat matrix capacity are modeled with residual standard error σ read from calibrated validation metrics.<br/>"
        "• <b>Line 3 (Statistical Probability):</b> Individual admission chances are calculated as P(admit) = 1 − Φ((ln(rank) − ŷ) / σ), representing the cumulative normal probability of your rank meeting or beating the closing rank."
    )
    story.append(Paragraph(prob_lines, style_body))
    story.append(Spacer(1, 8))

    # Subsection C: Model Provenance & Metrics from models/metrics.json
    story.append(
        Paragraph("3. Predictive Model & Typical Error Metrics", style_section_h2)
    )
    model_name_val = metrics.get(
        "model_name", "Weighted Trend Predictor (weights 0.5, 0.3, 0.2)"
    )
    val_year_val = metrics.get("val_year", 2025)
    mae_val = metrics.get("mae", 0.1393)
    cov_val = metrics.get("coverage_80", 0.9006)

    metrics_data = [
        [
            Paragraph("Predictive Model", style_meta_label),
            Paragraph(safe_escape(model_name_val), style_meta_val),
            Paragraph("Validation Cycle", style_meta_label),
            Paragraph(f"Held-out Year {val_year_val}", style_meta_val),
        ],
        [
            Paragraph("Log-Rank MAE", style_meta_label),
            Paragraph(f"{mae_val:.4f} (typical log error)", style_meta_val),
            Paragraph("80% Interval Coverage", style_meta_label),
            Paragraph(f"{cov_val * 100:.2f}% (empirically calibrated)", style_meta_val),
        ],
        [
            Paragraph("Residual Sigma (σ)", style_meta_label),
            Paragraph(f"{model_sigma:.4f}", style_meta_val),
            Paragraph("Cross-Option Shock (ρ)", style_meta_label),
            Paragraph("0.50 (correlated portfolio shock)", style_meta_val),
        ],
    ]
    metrics_table = Table(metrics_data, colWidths=[110, 160, 110, 160])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metrics_table)
    story.append(Spacer(1, 8))

    # Subsection D: JoSAA Sequential Allotment Rule
    story.append(
        Paragraph("4. Crucial JoSAA Allocation Mechanism", style_section_h2)
    )
    josaa_note_text = (
        "<b>Important JoSAA Allocation Rule:</b> In every seat allotment round, the JoSAA allocation algorithm evaluates your submitted "
        "choices strictly sequentially from preference #1 downwards. <b>You are allotted the first option in your list that you qualify for.</b> "
        "Once an option is allotted, all lower-preference options in your list are permanently discarded for that round. "
        "You can never be allotted a lower choice if your rank qualifies for a higher one. "
        "Always order choices strictly in your true order of preference, irrespective of cutoffs or perceived odds."
    )
    story.append(Paragraph(josaa_note_text, style_body))
    story.append(Spacer(1, 8))

    # Subsection E: Full Disclaimer and Official Verification Note
    story.append(
        Paragraph("5. Official Verification & Legal Disclaimer", style_section_h2)
    )
    disclaimer_text = (
        "<b>Full Disclaimer:</b> Vikalp provides probabilistic decision-support estimates based strictly on historical JoSAA counselling data "
        "(2021–2025). JoSAA 2026 counselling data is not yet available and cannot be known in advance. These statistical estimates are modeled assuming log-normal "
        "closing rank distributions and do not constitute official allotment guarantees or formal admissions advice. "
        "Actual opening and closing ranks fluctuate every year due to seat matrix modifications, candidate preference trends, "
        "and examination score distributions. Zero candidate personal data is stored or logged. Neither Vikalp nor its creator assumes liability for admissions outcomes. "
        "<b>Always verify official cutoffs, seat matrices, and counselling rules at josaa.nic.in.</b>"
    )
    story.append(Paragraph(disclaimer_text, style_body))
    story.append(Spacer(1, 10))

    # Final Verification Banner
    final_banner_data = [
        [
            Paragraph(
                "<font color='#0284C7' size='10'><b>Always verify at josaa.nic.in</b></font><br/>"
                "<font size='8' color='#64748B'>Built by Sahil Sharma &middot; Vikalp JoSAA Decision Engine</font>",
                style_tbl_cell_center,
            )
        ]
    ]
    final_banner_table = Table(final_banner_data, colWidths=[540])
    final_banner_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#0284C7")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(final_banner_table)

    # Build PDF with two-pass NumberedCanvas for 'Page X of Y'
    doc.build(story, canvasmaker=NumberedCanvas)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
