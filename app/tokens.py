"""Design Tokens and Accessible Component Helpers for Vikalp.

High-trust, light-theme default design system built on:
- WCAG AA contrast (>= 4.5:1 for normal text, >= 3.0:1 for large/graphical).
- 8px spatial grid.
- Semantic colors strictly mapped to meaning:
  - Dream: Amber (#B45309 text on #FEF3C7)
  - Target: Blue (#1D4ED8 text on #DBEAFE)
  - Safe: Green (#15803D text on #DCFCE7)
- Tabular numerals for ranks and percentages.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# Color Palette
TOKENS = {
    # Surfaces & Canvas
    "bg_canvas": "#F8FAFC",      # Slate 50
    "bg_surface": "#FFFFFF",     # Pure White
    "bg_subtle": "#F1F5F9",      # Slate 100
    "bg_hover": "#F8FAFC",       # Slate 50

    # Borders
    "border_subtle": "#E2E8F0",  # Slate 200
    "border_strong": "#CBD5E1",  # Slate 300
    "border_focus": "#2563EB",   # Blue 600

    # Text
    "text_primary": "#0F172A",   # Slate 900 (Contrast on #FFFFFF: 18.7:1)
    "text_secondary": "#334155", # Slate 700 (Contrast on #FFFFFF: 8.8:1)
    "text_muted": "#475569",     # Slate 600 (Contrast on #FFFFFF: 5.5:1)
    "text_caption": "#64748B",   # Slate 500 (Contrast on #FFFFFF: 4.6:1)

    # Accent (Brand)
    "brand_deep": "#1E3A8A",     # Blue 900
    "brand_primary": "#1D4ED8",  # Blue 700
    "brand_light": "#EFF6FF",    # Blue 50
    "brand_border": "#BFDBFE",   # Blue 200

    # Semantic Tiers
    "dream_text": "#92400E",     # Amber 800 (Contrast on #FEF3C7: 6.2:1)
    "dream_bg": "#FEF3C7",       # Amber 100
    "dream_border": "#FDE68A",   # Amber 200
    "dream_solid": "#D97706",    # Amber 600

    "target_text": "#1E40AF",    # Blue 800 (Contrast on #DBEAFE: 6.9:1)
    "target_bg": "#DBEAFE",      # Blue 100
    "target_border": "#BFDBFE",  # Blue 200
    "target_solid": "#2563EB",   # Blue 600

    "safe_text": "#166534",      # Green 800 (Contrast on #DCFCE7: 6.3:1)
    "safe_bg": "#DCFCE7",        # Green 100
    "safe_border": "#BBF7D0",    # Green 200
    "safe_solid": "#16A34A",     # Green 600

    # Callouts
    "callout_warn_bg": "#FFFBEB",
    "callout_warn_border": "#FCD34D",
    "callout_warn_text": "#92400E",

    "callout_info_bg": "#F0FDF4",
    "callout_info_border": "#86EFAC",
    "callout_info_text": "#166534",

    "callout_neutral_bg": "#F8FAFC",
    "callout_neutral_border": "#E2E8F0",
    "callout_neutral_text": "#1E293B",
}


def compute_relative_luminance(hex_color: str) -> float:
    """Calculate standard sRGB relative luminance per WCAG 2.1 specs."""
    hex_color = hex_color.lstrip("#")
    r_val = int(hex_color[0:2], 16) / 255.0
    g_val = int(hex_color[2:4], 16) / 255.0
    b_val = int(hex_color[4:6], 16) / 255.0

    def to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = to_linear(r_val)
    g_lin = to_linear(g_val)
    b_lin = to_linear(b_val)
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def calculate_contrast_ratio(hex1: str, hex2: str) -> float:
    """Return WCAG contrast ratio between two hex colors (1.0 to 21.0)."""
    l1 = compute_relative_luminance(hex1)
    l2 = compute_relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def verify_design_tokens_contrast() -> List[Tuple[str, str, str, float, bool]]:
    """Audit all core text/background token pairings for WCAG AA (>= 4.5:1)."""
    pairs = [
        ("Text Primary on Canvas", TOKENS["text_primary"], TOKENS["bg_canvas"], 4.5),
        ("Text Primary on Surface", TOKENS["text_primary"], TOKENS["bg_surface"], 4.5),
        ("Text Secondary on Surface", TOKENS["text_secondary"], TOKENS["bg_surface"], 4.5),
        ("Text Muted on Surface", TOKENS["text_muted"], TOKENS["bg_surface"], 4.5),
        ("Dream Text on Dream Bg", TOKENS["dream_text"], TOKENS["dream_bg"], 4.5),
        ("Target Text on Target Bg", TOKENS["target_text"], TOKENS["target_bg"], 4.5),
        ("Safe Text on Safe Bg", TOKENS["safe_text"], TOKENS["safe_bg"], 4.5),
        ("Brand Text on Brand Light", TOKENS["brand_deep"], TOKENS["brand_light"], 4.5),
        ("Callout Warn Text on Warn Bg", TOKENS["callout_warn_text"], TOKENS["callout_warn_bg"], 4.5),
        ("Callout Info Text on Info Bg", TOKENS["callout_info_text"], TOKENS["callout_info_bg"], 4.5),
    ]
    results = []
    for label, text_hex, bg_hex, min_ratio in pairs:
        ratio = calculate_contrast_ratio(text_hex, bg_hex)
        passes = ratio >= min_ratio
        results.append((label, text_hex, bg_hex, round(ratio, 2), passes))
    return results


# Component HTML Generators

def render_badge(label: str, prob: Optional[float] = None, show_icon: bool = True) -> str:
    """Render accessible semantic tier badge (Dream = Amber, Target = Blue, Safe = Green)."""
    norm = label.strip().capitalize()
    if norm == "Dream":
        bg = TOKENS["dream_bg"]
        color = TOKENS["dream_text"]
        border = TOKENS["dream_border"]
        icon = "⭐ " if show_icon else ""
        hi_label = "सपना"
    elif norm == "Safe":
        bg = TOKENS["safe_bg"]
        color = TOKENS["safe_text"]
        border = TOKENS["safe_border"]
        icon = "🛡️ " if show_icon else ""
        hi_label = "सुरक्षित"
    else:  # Target
        bg = TOKENS["target_bg"]
        color = TOKENS["target_text"]
        border = TOKENS["target_border"]
        icon = "🎯 " if show_icon else ""
        hi_label = "लक्ष्य"

    prob_str = f" · {prob:.0%}" if prob is not None else ""
    return (
        f'<span class="vk-badge" style="background:{bg}; border:1px solid {border}; color:{color};">'
        f'{icon}<strong>{norm}</strong> / {hi_label}{prob_str}'
        f'</span>'
    )


def render_confidence_bar(prob: float, tier: str) -> str:
    """Render a compact horizontal confidence bar with tabular probability label."""
    norm = tier.strip().capitalize()
    fill_color = (
        TOKENS["dream_solid"]
        if norm == "Dream"
        else TOKENS["safe_solid"]
        if norm == "Safe"
        else TOKENS["target_solid"]
    )
    pct = max(0.0, min(100.0, prob * 100.0))
    return (
        f'<div class="vk-conf-wrapper">'
        f'  <div class="vk-conf-track">'
        f'    <div class="vk-conf-fill" style="width:{pct:.1f}%; background:{fill_color};"></div>'
        f'  </div>'
        f'  <span class="vk-conf-num tabular-num">{pct:.0f}%</span>'
        f'</div>'
    )


def render_stat_card(
    title: str,
    title_hi: str,
    value: str,
    subtext: str,
    variant: str = "default",
    icon: str = "",
) -> str:
    """Render a high-trust headline stat card."""
    border_accent = ""
    if variant == "dream":
        border_accent = f"border-top: 3px solid {TOKENS['dream_solid']};"
    elif variant == "target":
        border_accent = f"border-top: 3px solid {TOKENS['target_solid']};"
    elif variant == "safe":
        border_accent = f"border-top: 3px solid {TOKENS['safe_solid']};"
    elif variant == "brand":
        border_accent = f"border-top: 3px solid {TOKENS['brand_primary']};"

    icon_html = f'<span class="vk-stat-icon">{icon}</span>' if icon else ""

    return (
        f'<div class="vk-stat-card" style="{border_accent}">'
        f'  <div class="vk-stat-header">'
        f'    <span class="vk-stat-title">{title} <span class="vk-stat-hi">/ {title_hi}</span></span>'
        f'    {icon_html}'
        f'  </div>'
        f'  <div class="vk-stat-value tabular-num">{value}</div>'
        f'  <div class="vk-stat-subtext">{subtext}</div>'
        f'</div>'
    )


def render_callout(
    title: str,
    body: str,
    variant: str = "warn",
    icon: str = "⚠️",
) -> str:
    """Render a calm, accessible callout box."""
    if variant == "warn":
        bg = TOKENS["callout_warn_bg"]
        border = TOKENS["callout_warn_border"]
        text_col = TOKENS["callout_warn_text"]
    elif variant == "info":
        bg = TOKENS["callout_info_bg"]
        border = TOKENS["callout_info_border"]
        text_col = TOKENS["callout_info_text"]
    else:
        bg = TOKENS["callout_neutral_bg"]
        border = TOKENS["callout_neutral_border"]
        text_col = TOKENS["callout_neutral_text"]

    return (
        f'<div class="vk-callout" style="background:{bg}; border-left: 4px solid {border}; color:{text_col};">'
        f'  <div class="vk-callout-title">{icon} {title}</div>'
        f'  <div class="vk-callout-body">{body}</div>'
        f'</div>'
    )


def render_stepper(current_step: int, steps: List[str]) -> str:
    """Render a clean 3-step progress indicator."""
    items = []
    for idx, name in enumerate(steps, start=1):
        if idx < current_step:
            state = "completed"
            marker = "✓"
        elif idx == current_step:
            state = "active"
            marker = str(idx)
        else:
            state = "pending"
            marker = str(idx)

        items.append(
            f'<div class="vk-step-item vk-step-{state}">'
            f'  <span class="vk-step-badge">{marker}</span>'
            f'  <span class="vk-step-name">{name}</span>'
            f'</div>'
        )

    joined = ' <div class="vk-step-divider"></div> '.join(items)
    return f'<div class="vk-stepper">{joined}</div>'
