"""Capture the 3 canonical pre-release UI screenshots for desktop (1440px) and mobile (390px).

Screenshots saved:
1. docs/screenshots/landing_desktop.png / landing_mobile.png
2. docs/screenshots/your_list_desktop.png / your_list_mobile.png
3. docs/screenshots/how_we_got_this_desktop.png / how_we_got_this_mobile.png
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = ROOT_DIR / "docs" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8503
URL = f"http://localhost:{PORT}"


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def capture():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)
    print(f"Started Streamlit on port {PORT} (PID {proc.pid}). Waiting for health check...")

    try:
        if not wait_for_server(URL):
            raise RuntimeError(f"Streamlit server did not become healthy at {URL} within timeout")
        print("Streamlit server is healthy. Launching Playwright...")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            configs = [
                ("desktop", {"width": 1440, "height": 900}),
                ("mobile", {"width": 390, "height": 844, "is_mobile": True}),
            ]

            for device_name, vp in configs:
                print(f"\n--- Capturing for {device_name} ({vp['width']}x{vp['height']}) ---")
                context = browser.new_context(viewport=vp)
                page = context.new_page()

                # A. Landing page
                page.goto(URL, wait_until="networkidle")
                page.wait_for_timeout(3000)
                landing_path = SCREENSHOT_DIR / f"landing_{device_name}.png"
                page.screenshot(path=str(landing_path), full_page=False)
                print(f"Saved {landing_path.name}")

                # B. Click 'Show My Choices →' then navigate to '2. Your list'
                show_btn = page.locator("button:has-text('Show My Choices')").first
                if show_btn.count() > 0:
                    show_btn.click()
                    page.wait_for_timeout(3000)

                tab_list = page.locator("[data-baseweb='tab']:has-text('2. Your list'), button:has-text('2. Your list')").first
                if tab_list.count() > 0:
                    tab_list.click()
                    page.wait_for_timeout(3000)

                your_list_path = SCREENSHOT_DIR / f"your_list_{device_name}.png"
                page.screenshot(path=str(your_list_path), full_page=False)
                print(f"Saved {your_list_path.name}")

                # C. Navigate to '5. How we got this'
                tab_trust = page.locator("[data-baseweb='tab']:has-text('5. How we got this'), button:has-text('5. How we got this')").first
                if tab_trust.count() > 0:
                    tab_trust.click()
                    page.wait_for_timeout(3000)

                how_path = SCREENSHOT_DIR / f"how_we_got_this_{device_name}.png"
                page.screenshot(path=str(how_path), full_page=False)
                print(f"Saved {how_path.name}")

                context.close()

            browser.close()
            print("\nAll 6 screenshots captured successfully.")

    finally:
        print(f"Terminating Streamlit process (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Streamlit process stopped.")


if __name__ == "__main__":
    capture()
