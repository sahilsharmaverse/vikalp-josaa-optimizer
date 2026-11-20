# design.md: UI design for the Streamlit app

Project: Vikalp | Author: Sahil Sharma

## Style
Clean, trustworthy, calm. Students are stressed, so the screen should feel clear and not flashy. Light theme, one accent colour, three status colours.

## Colour and theme
| Role | Hex |
|---|---|
| Primary text / headings | `#1F2A44` (deep navy) |
| Accent (buttons, highlights) | `#F59E0B` (saffron) |
| Background | `#FFFFFF` |
| Secondary background (cards, sidebar) | `#F4F6FA` |
| Safe | `#16A34A` (green) |
| Target | `#D97706` (amber) |
| Dream | `#DC2626` (red) |
| Muted text | `#6B7280` |

`.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#F59E0B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F6FA"
textColor = "#1F2A44"
font = "sans serif"
```

## Fonts and typography
- Font: **Inter** (400 / 600 / 700), loaded with a CSS `@import` from Google Fonts, fallback sans-serif
- Page title: 32 px, bold
- Section headings: 20 px, semi-bold
- Body and table text: 15-16 px
- Captions and disclaimer: 13 px, muted colour
- Always use numbers with thousands separators (e.g. 12,345), ranks right-aligned in tables

## Layout
1. **Sidebar (inputs):** rank(s), category, category rank, gender, home state, preferred branches, institute types, list size, priority mode, **Generate list** button
2. **Main area:**
   - Short summary row: count of Dream / Target / Safe options (three coloured metric cards)
   - Results table: `#`, Institute, Program, Quota, Probability (progress bar), Label (coloured pill), Predicted closing rank range
   - Expander per row or a selected-row panel with the closing-rank history chart
   - **Download CSV** button
3. **Footer:** disclaimer and a short "How this works" expander

## UX copy
- Title: "Vikalp"
- Subtitle: "Counselling Choice-List Optimizer. Estimate your admission chances and build a smarter JoSAA choice list."
- Footer credit: "Built by Sahil Sharma"
- Disclaimer (always visible under the results): "These are statistical estimates based on past JoSAA data, not official predictions or guarantees. Always verify at josaa.nic.in."
- Empty state: "No options match your filters. Try adding more branches or institute types."
- Error state: "Please enter a valid rank (a positive whole number)."

## Components
- `st.sidebar`, `st.number_input`, `st.selectbox`, `st.multiselect`, `st.slider`, `st.button`
- `st.dataframe` with `column_config.ProgressColumn` for probability
- `st.download_button` for CSV
- Plotly line chart for closing-rank history with a shaded predicted range
- Keep it to one page; no multi-page navigation today

## Accessibility
- Never rely on colour alone: the label text (Dream / Target / Safe) always appears next to the colour
- Contrast: dark navy text on white or light grey backgrounds
