import os
import shutil
import time
from playwright.sync_api import sync_playwright

DOCS_DIR = r"C:\Users\HP\Documents\SNAPDRAGON HACKTHONE\vikalp\docs\screenshots"
ARTIFACTS_DIR = r"C:\Users\HP\.gemini\antigravity\brain\63377d71-64cb-447f-b83c-c87e2b572b16\screenshots"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

test_scenarios = [
    ("compare_2options", "4,21"),
    ("compare_3options", "1,4,21"),
    ("compare_unreachable_warning", "37,48"),
]

viewports = [
    ("desktop", {"width": 1280, "height": 1150}),
    ("mobile_390px", {"width": 390, "height": 844, "is_mobile": True}),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for vp_name, vp_cfg in viewports:
        context = browser.new_context(
            viewport={"width": vp_cfg["width"], "height": vp_cfg["height"]},
            device_scale_factor=2.0,
        )
        page = context.new_page()

        for name, compare_param in test_scenarios:
            url = f"http://localhost:8501/?compare={compare_param}"
            print(f"Navigating to {url} ({vp_name})...")
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(3500)

            # Scroll to comparison section
            try:
                elem = page.locator("text=Compare 2–3 Options Side by Side")
                if elem.count() > 0:
                    elem.first.scroll_into_view_if_needed()
                    page.wait_for_timeout(1000)
            except Exception as e:
                print(f"Scroll note: {e}")

            fname = f"feature4_{name}_{vp_name}.png"
            docs_path = os.path.join(DOCS_DIR, fname)
            art_path = os.path.join(ARTIFACTS_DIR, fname)

            page.screenshot(path=docs_path)
            shutil.copy2(docs_path, art_path)
            print(f"Saved {fname} (size: {os.path.getsize(docs_path):,} bytes)")

        context.close()
    browser.close()

print("All Feature 4 screenshots captured successfully.")
