"""Capture desktop (1280px) and mobile (390px) screenshots for Feature 2:
With What-If slider at -20%, 0%, and +20% for the same student.
"""

import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(tempfile.gettempdir()) / "vikalp_screenshots"
DOCS_DIR = Path("docs/screenshots")

DOCS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SHIFTS = [-20, 0, 20]

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch()

        for shift in SHIFTS:
            sign_str = f"minus_{abs(shift)}" if shift < 0 else (f"plus_{shift}" if shift > 0 else "0pct")
            url = f"http://localhost:8501/?shift={shift}"
            print(f"\n--- Capturing What-If Shift {shift}% ({url}) ---")

            # 1. Desktop capture (1280x900)
            ctx_desktop = browser.new_context(viewport={"width": 1280, "height": 900})
            page_desktop = ctx_desktop.new_page()
            page_desktop.goto(url, wait_until="networkidle")
            page_desktop.wait_for_timeout(3500)

            desktop_filename = f"feature2_whatif_{sign_str}_desktop.png"
            desktop_path = DOCS_DIR / desktop_filename
            page_desktop.screenshot(path=str(desktop_path), full_page=False)
            shutil.copy(desktop_path, ARTIFACT_DIR / desktop_filename)
            print(f"Captured: {desktop_filename}")
            ctx_desktop.close()

            # 2. Mobile capture (390x844 - iPhone 13/14 viewport)
            ctx_mobile = browser.new_context(
                viewport={"width": 390, "height": 844},
                is_mobile=True,
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            )
            page_mobile = ctx_mobile.new_page()
            page_mobile.goto(url, wait_until="networkidle")
            page_mobile.wait_for_timeout(3500)

            mobile_filename = f"feature2_whatif_{sign_str}_mobile_390px.png"
            mobile_path = DOCS_DIR / mobile_filename
            page_mobile.screenshot(path=str(mobile_path), full_page=False)
            shutil.copy(mobile_path, ARTIFACT_DIR / mobile_filename)
            print(f"Captured: {mobile_filename}")
            ctx_mobile.close()

        browser.close()
        print("\nAll 6 screenshots captured successfully!")

if __name__ == "__main__":
    capture_all()
