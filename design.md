# design.md: UI design for the Streamlit app (v1.1 Bold Edition)

Project: Vikalp | Author: Sahil Sharma

## Style: Bold, Modern & High-Confidence
High-energy, modern, dark-theme aesthetic designed to inspire confidence and clarity during high-stakes JoSAA counselling decisions. Students and parents experience a premium analytics dashboard vibe with rich glassmorphism, animated metrics, and intuitive visual hierarchies.

## Colour & Dark Glass Theme
| Role | Hex / Value | Description |
|---|---|---|
| Deep Dark Background | `#0B0F19` | Pitch canvas foundation |
| Secondary Background (Sidebar / Wells) | `#111827` | Deep slate container backdrop |
| Glass Card Background | `rgba(255, 255, 255, 0.04)` | Translucent frosted glass |
| Glass Card Border | `rgba(255, 255, 255, 0.08)` | Crisp subtle separation border |
| Primary Text | `#F9FAFB` | High-contrast crisp white |
| Secondary / Muted Text | `#9CA3AF` | Clean muted cool grey |
| Navy-to-Saffron Gradient | `linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #B45309 85%, #F59E0B 100%)` | Bold hero header canvas |
| Primary Accent / Interactive | `#0EA5E9` (Electric Cyan) | Crisp interactive accent (buttons, active sliders, links) — distinct from Target amber |
| Safe Tier | `#22C55E` (Emerald Green) | High-probability anchor options ($P \ge 80\%$) |
| Target Tier | `#F59E0B` (Vibrant Amber) | Core competitive zone ($30\% \le P < 80\%$) |
| Dream Tier | `#EF4444` (Vivid Crimson) | Aspirational reach options ($5\% \le P < 30\%$, floor $P \ge 5\%$) |
| Trend: Getting harder | `#EF4444` (Red) | Cutoffs tightening / competition rising |
| Trend: Getting easier | `#22C55E` (Green) | Cutoffs relaxing / admission chance expanding |
| Trend: Volatile | `#9CA3AF` (Grey) | Historical closing ranks fluctuate across cycles |

`.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#0EA5E9"
backgroundColor = "#0B0F19"
secondaryBackgroundColor = "#111827"
textColor = "#F9FAFB"
font = "sans serif"
```

## Fonts & Typography
- Font: **Inter** (400 / 500 / 600 / 700 / 800) loaded via Google Fonts with system fallbacks.
- Hero Headline: 34-38 px, extra-bold.
- Section Headings: 20-22 px, bold.
- Card Titles: 16-18 px, semi-bold.
- Metrics / Counters: 26-32 px, tabular bold numbers.
- Badges & Footers: 12-13 px, crisp uppercase tracking with label text displayed next to the color.

## Components & Visual Elements
1. **Hero Section:**
   - Wide container with navy-to-saffron gradient (`linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #B45309 85%, #F59E0B 100%)`).
   - Headline: **"Your rank. Your seats. Your smartest list."**
   - Subtitle: *"Data-driven JoSAA choice optimization powered by 5 years of cutoffs and uncertainty modeling."*
   - Highlights the cumulative chance of admission metric prominently.

2. **Cumulative Chance of Getting at Least One Seat:**
   - Prominent hero metric card reporting the share of Monte Carlo simulations (2,000 runs, $\rho = 0.5$) resulting in $\ge 1$ admission.
   - Display capped at 99%.
   - Clearly labeled as an **estimate** with a plain-words explanation:
     *"Monte Carlo estimate (2,000 simulations) assuming correlated exam/counselling cutoff shocks ($\rho = 0.5$)."*

3. **Animated Count-Up Numbers:**
   - Key metrics (cumulative probability, Dream/Target/Safe counts, total choices) rendered with smooth CSS or lightweight HTML transitions.

4. **Option Cards & Circular Probability Gauge:**
   - Glassmorphism option cards (`backdrop-filter: blur(12px)`).
   - SVG circular progress gauge displaying exact admission probability percentage with colored arc matching the tier.
   - Tier tags: `Dream` (Crimson `#EF4444`), `Target` (Amber `#F59E0B`), `Safe` (Emerald `#22C55E`).
   - Predicted 80% prediction interval pill with thousands separators (read dynamically using model metrics $\sigma$).

5. **Three-Lane Staircase View:**
   - Visual strategic lane showing:
     - **Lane 1: Dream** (Aspirational reach, $5\% \le P < 30\%$, top 20%, minimum floor $P \ge 5\%$)
     - **Lane 2: Target** (Sweet spot, $30\% \le P < 80\%$, core 50%)
     - **Lane 3: Safe** (Safety nets, $P \ge 80\%$, anchor 30%)
   - Shows candidate how choices cascade down the JoSAA preference engine.

6. **What-If Interactive Rank Sensitivity Slider:**
   - Interactive slider in an Electric Cyan styled toolbar to test rank shifts ($\pm 20\%$) with instant tier re-scoring.

7. **"Why This Probability?" Diagnostic Panel:**
   - Interactive deep-dive modal or drawer with 5-year cutoff Plotly chart.
   - **Trend Badges (with label text next to the colour):**
     - 🔴 **Getting harder** (Cutoffs tightening / rank dropping)
     - 🟢 **Getting easier** (Cutoffs easing / rank increasing)
     - ⚪ **Volatile** (Fluctuating across years)
   - Step-by-step probability explanation highlighting student rank relative to median cutoff $\hat{y}$ and 80% prediction interval $[\hat{y} - 1.28155\sigma, \hat{y} + 1.28155\sigma]$ with $\sigma$ read directly from saved metrics.

8. **Side-by-Side Comparison Deck:**
   - Clean 2-column or 3-column glass cards comparing selected programs side by side with radial gauges, 5-year trend history, quota badges, and institutional rating.

9. **Export Suite:**
   - **Download CSV** button.
   - **Download PDF Report** button: generates formatted printable PDF choice list.

10. **Mobile-First Responsive Layout:**
   - Single-column reflow for 390 px mobile devices (iPhone/Pixel width) with stacked glass cards, fluid padding (`12px - 16px`), touch-accessible buttons ($\ge 44\text{ px}$ height), and scalable SVG gauges.
   - Expands seamlessly to desktop width.

## UX Copy & Mandatory Branding
- Headline: "Your rank. Your seats. Your smartest list."
- Title: "Vikalp · JoSAA Choice-List Optimizer"
- Disclaimer (persistent below results):
  *"These are statistical estimates based on past JoSAA data, not official predictions or guarantees. Always verify at josaa.nic.in."*
- Footer Credit (persistent at bottom):
  *"Built by Sahil Sharma"*
- Empty State: "No options match your filters. Try adding more branches or institute types."
- Error State: "Please enter a valid rank (a positive whole number)."
