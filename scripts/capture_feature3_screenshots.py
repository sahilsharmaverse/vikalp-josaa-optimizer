import os
import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

DOCS_DIR = Path("docs/screenshots")
ARTIFACTS_DIR = Path(tempfile.gettempdir()) / "vikalp_screenshots"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 1. Getting harder: Choice #6 (NIT Warangal Math & Computing - ranks 4106, 3436, 3277)
# 2. Getting easier: Choice #21 (NIT Jalandhar Data Science & Eng - ranks 13373, 14016, 14814)
# 3. Volatile: Choice #1 (NIT Trichy CSE - ranks 714, 996, 1509, 1224, 1449)
targets = [
    ("getting_harder", 6),
    ("getting_easier", 21),
    ("volatile", 1),
]

viewports = [
    ("desktop", {"width": 1280, "height": 1180}),
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

        for name, choice_no in targets:
            url = f"http://localhost:8501/?why={choice_no}&open_dialog=1"
            print(f"Navigating to {url} ({vp_name})...")
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(3500)

            # Wait for dialog
            try:
                page.wait_for_selector('div[role="dialog"]', timeout=6000)
            except Exception:
                pass

            page.wait_for_timeout(1000)

            fname = f"feature3_{name}_{vp_name}.png"
            docs_path = os.path.join(DOCS_DIR, fname)
            art_path = os.path.join(ARTIFACTS_DIR, fname)

            page.screenshot(path=docs_path)
            shutil.copy2(docs_path, art_path)
            print(f"Saved {fname} (size: {os.path.getsize(docs_path):,} bytes)")

        context.close()
    browser.close()

print("All Feature 3 screenshots captured successfully.")
