"""Design Tokens and Plain-Language Component Helpers for Vikalp.

High-trust, Stripe/Linear-grade design system:
- English only, zero Devanagari text.
- Plain language for a 10th-grade student and worried parent.
- Semantic categories:
  - Reach (formerly Dream): Amber
  - Good fit (formerly Target): Blue
  - Likely (formerly Safe): Green
- WCAG AA contrast (>= 4.5:1 for normal text).
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
    "text_primary": "#0F172A",   # Slate 900 (Contrast on #FFFFFF: 17.8:1)
    "text_secondary": "#334155", # Slate 700 (Contrast on #FFFFFF: 10.3:1)
    "text_muted": "#475569",     # Slate 600 (Contrast on #FFFFFF: 7.5:1)
    "text_caption": "#64748B",   # Slate 500 (Contrast on #FFFFFF: 4.6:1)

    # Accent (Brand)
    "brand_deep": "#1E3A8A",     # Blue 900
    "brand_primary": "#1D4ED8",  # Blue 700
    "brand_light": "#EFF6FF",    # Blue 50
    "brand_border": "#BFDBFE",   # Blue 200

    # Semantic Categories: Reach, Good fit, Likely
    "reach_text": "#92400E",     # Amber 800 (Contrast on #FEF3C7: 6.3:1)
    "reach_bg": "#FEF3C7",       # Amber 100
    "reach_border": "#FDE68A",   # Amber 200
    "reach_solid": "#D97706",    # Amber 600

    "goodfit_text": "#1E40AF",   # Blue 800 (Contrast on #DBEAFE: 7.1:1)
    "goodfit_bg": "#DBEAFE",     # Blue 100
    "goodfit_border": "#BFDBFE", # Blue 200
    "goodfit_solid": "#2563EB",  # Blue 600

    "likely_text": "#166534",    # Green 800 (Contrast on #DCFCE7: 6.4:1)
    "likely_bg": "#DCFCE7",      # Green 100
    "likely_border": "#BBF7D0",  # Green 200
    "likely_solid": "#16A34A",   # Green 600

    # Callouts
    "callout_warn_bg": "#FFFBEB",
    "callout_warn_border": "#FCD34D",
    "callout_warn_text": "#92400E",

    "callout_info_bg": "#EFF6FF",
    "callout_info_border": "#93C5FD",
    "callout_info_text": "#1E40AF",

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
        ("Reach Text on Reach Bg", TOKENS["reach_text"], TOKENS["reach_bg"], 4.5),
        ("Good Fit Text on Good Fit Bg", TOKENS["goodfit_text"], TOKENS["goodfit_bg"], 4.5),
        ("Likely Text on Likely Bg", TOKENS["likely_text"], TOKENS["likely_bg"], 4.5),
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


def normalize_tier_name(tier_raw: str) -> str:
    """Map legacy model tier names to friendly plain English names."""
    norm = str(tier_raw).strip().capitalize()
    if norm in ["Dream", "Reach"]:
        return "Reach"
    elif norm in ["Safe", "Likely"]:
        return "Likely"
    return "Good fit"


def get_tier_colors(tier_name: str) -> Tuple[str, str, str, str]:
    """Return (bg, text, border, solid) for a tier name."""
    t = normalize_tier_name(tier_name)
    if t == "Reach":
        return TOKENS["reach_bg"], TOKENS["reach_text"], TOKENS["reach_border"], TOKENS["reach_solid"]
    elif t == "Likely":
        return TOKENS["likely_bg"], TOKENS["likely_text"], TOKENS["likely_border"], TOKENS["likely_solid"]
    else:
        return TOKENS["goodfit_bg"], TOKENS["goodfit_text"], TOKENS["goodfit_border"], TOKENS["goodfit_solid"]


def render_badge(tier_raw: str, prob: Optional[float] = None) -> str:
    """Render friendly semantic category badge (Reach, Good fit, Likely)."""
    t_name = normalize_tier_name(tier_raw)
    bg, color, border, _ = get_tier_colors(t_name)
    prob_str = f" · {prob:.0%}" if prob is not None else ""
    return (
        f'<span class="vk-badge" style="background:{bg}; border:1px solid {border}; color:{color};">'
        f'<strong>{t_name}</strong>{prob_str}'
        f'</span>'
    )


def render_probability_meter(prob: float, tier_name: str = "Good fit") -> str:
    """Render clean horizontal probability meter with tabular percentage."""
    _, _, _, fill_col = get_tier_colors(tier_name)
    pct = max(0.0, min(100.0, prob * 100.0))
    return (
        f'<div class="vk-meter-wrap">'
        f'  <div class="vk-meter-track">'
        f'    <div class="vk-meter-fill" style="width:{pct:.1f}%; background:{fill_col};"></div>'
        f'  </div>'
        f'  <span class="vk-meter-num tabular-num">{pct:.0f}%</span>'
        f'</div>'
    )


def render_callout(
    title: str,
    body: str,
    variant: str = "warn",
    icon: str = "ℹ️",
) -> str:
    """Render calm, accessible notification card."""
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
        f'<div class="vk-callout" style="background:{bg}; border: 1px solid {border}; border-left: 4px solid {border}; color:{text_col};">'
        f'  <div class="vk-callout-title">{icon} {title}</div>'
        f'  <div class="vk-callout-body">{body}</div>'
        f'</div>'
    )


def render_segment_bar(reach_cnt: int, good_cnt: int, likely_cnt: int) -> str:
    """Render 3-segment horizontal bar showing list balance."""
    total = reach_cnt + good_cnt + likely_cnt
    if total == 0:
        total = 1
    pct_r = (reach_cnt / total) * 100
    pct_g = (good_cnt / total) * 100
    pct_l = (likely_cnt / total) * 100

    return (
        f'<div class="vk-segment-container">'
        f'  <div class="vk-segment-bar">'
        f'    <div class="vk-seg vk-seg-reach" style="width:{pct_r:.1f}%;" title="Reach: {reach_cnt}"></div>'
        f'    <div class="vk-seg vk-seg-goodfit" style="width:{pct_g:.1f}%;" title="Good fit: {good_cnt}"></div>'
        f'    <div class="vk-seg vk-seg-likely" style="width:{pct_l:.1f}%;" title="Likely: {likely_cnt}"></div>'
        f'  </div>'
        f'  <div class="vk-segment-legend">'
        f'    <span class="vk-legend-item"><span class="vk-dot" style="background:{TOKENS["reach_solid"]};"></span>Reach: <strong>{reach_cnt}</strong></span>'
        f'    <span class="vk-legend-item"><span class="vk-dot" style="background:{TOKENS["goodfit_solid"]};"></span>Good fit: <strong>{good_cnt}</strong></span>'
        f'    <span class="vk-legend-item"><span class="vk-dot" style="background:{TOKENS["likely_solid"]};"></span>Likely: <strong>{likely_cnt}</strong></span>'
        f'  </div>'
        f'</div>'
    )
