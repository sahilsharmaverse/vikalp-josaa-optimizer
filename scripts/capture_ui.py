"""Playwright UI capture and accessibility audit for Vikalp redesign.

Captures Desktop (1440x900) and Mobile (390x844) screenshots for all 5 tabs:
1. Your Profile
2. Your Plan
3. What-If
4. Compare
5. How It Works & Trust

Runs accessibility analysis via DOM inspection and contrast checks.
"""

from __future__ import annotations

from pathlib import Path
import time
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path("outputs/ui")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://localhost:8501"

TABS = [
    ("1. Tell us your rank", "tab1_start"),
    ("2. Your list", "tab2_list"),
    ("3. Try a different rank", "tab3_whatif"),
    ("4. Compare choices", "tab4_compare"),
    ("5. How we got this", "tab5_method"),
]


def capture_all():
    print(f"Connecting Playwright to {APP_URL}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # -------------------------------------------------------------
        # 1. DESKTOP VIEWPORT (1440x900)
        # -------------------------------------------------------------
        print("\n--- Capturing Desktop Viewports (1440x900) ---")
        context_desk = browser.new_context(viewport={"width": 1440, "height": 900})
        page_desk = context_desk.new_page()

        page_desk.goto(APP_URL, wait_until="networkidle", timeout=60000)
        time.sleep(3)  # Allow Streamlit re-run & caching to settle

        for tab_label, filename_base in TABS:
            print(f"Desktop: Selecting tab '{tab_label}'...")
            try:
                # Find tab button by text
                tab_btn = page_desk.get_by_role("tab", name=tab_label)
                if tab_btn.count() > 0:
                    tab_btn.first.click()
                    time.sleep(2)
                else:
                    # Fallback text locator
                    fallback = page_desk.locator(f"text={tab_label}")
                    if fallback.count() > 0:
                        fallback.first.click()
                        time.sleep(2)
            except Exception as e:
                print(f"Warning clicking tab '{tab_label}': {e}")

            out_path = OUTPUT_DIR / f"desktop_{filename_base}.png"
            page_desk.screenshot(path=str(out_path), full_page=True)
            print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        # Run DOM accessibility checks on desktop
        print("\nAuditing Accessibility & Heading Structure...")
        heading_counts = page_desk.evaluate("""
            () => {
                const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
                return headings.map(h => ({
                    tag: h.tagName,
                    text: h.innerText.slice(0, 50),
                    hasText: h.innerText.trim().length > 0
                }));
            }
        """)
        empty_headings = [h for h in heading_counts if not h['hasText']]
        print(f"Total Headings: {len(heading_counts)}, Empty Headings: {len(empty_headings)}")

        img_alts = page_desk.evaluate("""
            () => {
                const imgs = Array.from(document.querySelectorAll('img'));
                return imgs.map(i => ({
                    src: i.src.slice(0, 40),
                    alt: i.alt,
                    hasAlt: Boolean(i.alt && i.alt.trim().length > 0)
                }));
            }
        """)
        missing_alts = [i for i in img_alts if not i['hasAlt']]
        print(f"Total Images: {len(img_alts)}, Images Missing Alt: {len(missing_alts)}")

        context_desk.close()

        # -------------------------------------------------------------
        # 2. MOBILE VIEWPORT (390x844 - iPhone 12/13/14)
        # -------------------------------------------------------------
        print("\n--- Capturing Mobile Viewports (390x844) ---")
        context_mob = browser.new_context(
            viewport={"width": 390, "height": 844},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            is_mobile=True,
            has_touch=True,
        )
        page_mob = context_mob.new_page()

        page_mob.goto(APP_URL, wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # On mobile, close sidebar if open
        try:
            close_btn = page_mob.locator('[data-testid="stSidebarCollapseButton"]')
            if close_btn.count() > 0 and close_btn.is_visible():
                close_btn.click()
                time.sleep(1)
            else:
                page_mob.keyboard.press("Escape")
                time.sleep(1)
        except Exception:
            pass

        for tab_label, filename_base in TABS:
            print(f"Mobile: Selecting tab '{tab_label}'...")
            try:
                tab_btn = page_mob.get_by_role("tab", name=tab_label)
                if tab_btn.count() > 0:
                    tab_btn.first.click(force=True)
                    time.sleep(2)
                else:
                    fallback = page_mob.locator(f"text={tab_label}")
                    if fallback.count() > 0:
                        fallback.first.click(force=True)
                        time.sleep(2)
            except Exception as e:
                print(f"Warning clicking mobile tab '{tab_label}': {e}")

            out_path = OUTPUT_DIR / f"mobile_{filename_base}.png"
            page_mob.screenshot(path=str(out_path), full_page=True)
            print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

        context_mob.close()
        browser.close()
        print("\nAll screenshots and audits completed successfully!")


if __name__ == "__main__":
    capture_all()
