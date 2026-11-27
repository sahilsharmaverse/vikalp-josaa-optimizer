"""Capture desktop (1280px) and mobile (390px) screenshots of Vikalp."""

import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(tempfile.gettempdir()) / "vikalp_screenshots"
DOCS_DIR = Path("docs/screenshots")

DOCS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def capture_screenshots(feature_name: str = "feature1_at_least_one"):
    with sync_playwright() as p:
        browser = p.chromium.launch()

        # 1. Desktop capture (1280x850)
        context_desktop = browser.new_context(viewport={"width": 1280, "height": 850})
        page_desktop = context_desktop.new_page()
        page_desktop.goto("http://localhost:8501", wait_until="networkidle")
        page_desktop.wait_for_timeout(4000)  # Wait for calculations and render

        desktop_file = DOCS_DIR / f"{feature_name}_desktop.png"
        page_desktop.screenshot(path=str(desktop_file), full_page=False)
        shutil.copy(desktop_file, ARTIFACT_DIR / f"{feature_name}_desktop.png")
        print(f"Captured desktop screenshot: {desktop_file}")
        context_desktop.close()

        # 2. Mobile capture (390x844 - iPhone 13/14 viewport)
        context_mobile = browser.new_context(
            viewport={"width": 390, "height": 844},
            is_mobile=True,
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
        )
        page_mobile = context_mobile.new_page()
        page_mobile.goto("http://localhost:8501", wait_until="networkidle")
        page_mobile.wait_for_timeout(4000)

        mobile_file = DOCS_DIR / f"{feature_name}_mobile_390px.png"
        page_mobile.screenshot(path=str(mobile_file), full_page=False)
        shutil.copy(mobile_file, ARTIFACT_DIR / f"{feature_name}_mobile_390px.png")
        print(f"Captured mobile screenshot (390px): {mobile_file}")
        context_mobile.close()

        browser.close()

if __name__ == "__main__":
    capture_screenshots()
