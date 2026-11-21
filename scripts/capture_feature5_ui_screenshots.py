"""Capture UI screenshot showing the new Download PDF Report button next to the CSV button."""

from __future__ import annotations

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def capture():
    out_dir = Path("docs/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # 1. Desktop Viewport (1280 x 900)
        context_dt = await browser.new_context(viewport={"width": 1280, "height": 900})
        page_dt = await context_dt.new_page()
        await page_dt.goto("http://localhost:8501", wait_until="networkidle")
        await page_dt.wait_for_timeout(3000)

        # Scroll to table action bar
        table_header = page_dt.locator("text=Optimized JoSAA Choice List")
        if await table_header.count() > 0:
            await table_header.scroll_into_view_if_needed()
            await page_dt.wait_for_timeout(1000)

        dt_path = out_dir / "feature5_app_action_bar_desktop.png"
        await page_dt.screenshot(path=str(dt_path), full_page=False)
        print(f"Captured: {dt_path}")
        await context_dt.close()

        # 2. Mobile Viewport (390 x 844)
        context_mb = await browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
        page_mb = await context_mb.new_page()
        await page_mb.goto("http://localhost:8501", wait_until="networkidle")
        await page_mb.wait_for_timeout(3000)

        table_header_mb = page_mb.locator("text=Optimized JoSAA Choice List")
        if await table_header_mb.count() > 0:
            await table_header_mb.scroll_into_view_if_needed()
            await page_mb.wait_for_timeout(1000)

        mb_path = out_dir / "feature5_app_action_bar_mobile_390px.png"
        await page_mb.screenshot(path=str(mb_path), full_page=False)
        print(f"Captured: {mb_path}")
        await context_mb.close()

        await browser.close()


if __name__ == "__main__":
    asyncio.run(capture())
