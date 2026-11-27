import shutil
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

DOCS_DIR = Path("docs/screenshots")
ARTIFACT_DIR = Path(tempfile.gettempdir()) / "vikalp_screenshots"

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.goto("http://localhost:8501/?shift=20", wait_until="networkidle")
    page.wait_for_timeout(3500)
    page.locator("div[data-testid='stDataFrame']").scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    tbl_path = DOCS_DIR / "feature2_whatif_table_plus_20_desktop.png"
    page.screenshot(path=str(tbl_path), full_page=False)
    shutil.copy(tbl_path, ARTIFACT_DIR / "feature2_whatif_table_plus_20_desktop.png")
    browser.close()
    print("Table screenshot captured")
