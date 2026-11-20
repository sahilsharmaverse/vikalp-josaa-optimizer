"""Scraper for JoSAA historical Opening and Closing Ranks.

Automates the archive ASP.NET form to download final-round cutoff tables
for available years and saves each as data/raw/<year>_round<N>.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from playwright.sync_api import Browser, Page, sync_playwright

ARCHIVE_URL = (
    "https://josaa.admissions.nic.in/applicant/seatmatrix/openingclosingrankarchieve.aspx"
)
REQUEST_DELAY_SECONDS = 1.5


def get_existing_final_round_file(raw_dir: Path, year: str) -> Optional[Path]:
    """Check if a final round file for the given year already exists in raw_dir."""
    matches = list(raw_dir.glob(f"{year}_round*.csv"))
    if matches:
        return matches[0]
    return None


def fetch_available_years(page: Page) -> List[str]:
    """Retrieve list of valid years from the Year dropdown."""
    year_options: List[Dict[str, str]] = page.eval_on_selector(
        "#ctl00_ContentPlaceHolder1_ddlYear",
        """select => Array.from(select.options)
            .map(opt => ({text: opt.text.trim(), value: opt.value.trim()}))
        """,
    )
    years = [opt["value"] for opt in year_options if opt["value"] not in ("0", "")]
    return sorted(years, reverse=True)


def select_year_and_get_final_round(page: Page, year: str) -> str:
    """Select a year via postback and determine the final (maximum) round number."""
    with page.expect_navigation(wait_until="networkidle"):
        page.evaluate(
            f"""() => {{
                document.querySelector('#ctl00_ContentPlaceHolder1_ddlYear').value = '{year}';
                __doPostBack('ctl00$ContentPlaceHolder1$ddlYear', '');
            }}"""
        )
    time.sleep(REQUEST_DELAY_SECONDS)

    rounds: List[Dict[str, str]] = page.eval_on_selector(
        "#ctl00_ContentPlaceHolder1_ddlroundno",
        """select => Array.from(select.options)
            .map(opt => ({text: opt.text.trim(), value: opt.value.trim()}))
        """,
    )
    valid_rounds = [
        int(r["value"]) for r in rounds if r["value"].isdigit() and int(r["value"]) > 0
    ]
    if not valid_rounds:
        raise RuntimeError(f"No valid rounds found for year {year}")

    final_round = str(max(valid_rounds))
    return final_round


def trigger_cascade_and_submit(page: Page, final_round: str) -> None:
    """Cascade through dropdowns to select Round, ALL institutes, branches, and seat types."""
    # 1. Select Round No
    with page.expect_navigation(wait_until="networkidle"):
        page.evaluate(
            f"""() => {{
                document.querySelector('#ctl00_ContentPlaceHolder1_ddlroundno').value = '{final_round}';
                __doPostBack('ctl00$ContentPlaceHolder1$ddlroundno', '');
            }}"""
        )
    time.sleep(REQUEST_DELAY_SECONDS)

    # 2. Select Institute Type = ALL
    with page.expect_navigation(wait_until="networkidle"):
        page.evaluate(
            """() => {
                document.querySelector('#ctl00_ContentPlaceHolder1_ddlInstype').value = 'ALL';
                __doPostBack('ctl00$ContentPlaceHolder1$ddlInstype', '');
            }"""
        )
    time.sleep(REQUEST_DELAY_SECONDS)

    # 3. Select Institute Name = ALL
    with page.expect_navigation(wait_until="networkidle"):
        page.evaluate(
            """() => {
                document.querySelector('#ctl00_ContentPlaceHolder1_ddlInstitute').value = 'ALL';
                __doPostBack('ctl00$ContentPlaceHolder1$ddlInstitute', '');
            }"""
        )
    time.sleep(REQUEST_DELAY_SECONDS)

    # 4. Select Academic Program = ALL
    with page.expect_navigation(wait_until="networkidle"):
        page.evaluate(
            """() => {
                document.querySelector('#ctl00_ContentPlaceHolder1_ddlBranch').value = 'ALL';
                __doPostBack('ctl00$ContentPlaceHolder1$ddlBranch', '');
            }"""
        )
    time.sleep(REQUEST_DELAY_SECONDS)

    # 5. Select Seat Type = ALL (no postback)
    page.evaluate(
        """() => {
            document.querySelector('#ctl00_ContentPlaceHolder1_ddlSeatType').value = 'ALL';
        }"""
    )

    # 6. Click Submit
    with page.expect_navigation(wait_until="networkidle", timeout=90000):
        page.evaluate(
            """() => {
                document.querySelector('#ctl00_ContentPlaceHolder1_btnSubmit').click();
            }"""
        )
    time.sleep(REQUEST_DELAY_SECONDS)


def extract_table_dataframe(page: Page) -> pd.DataFrame:
    """Extract results table rows directly from the DOM and return as a pandas DataFrame."""
    rows: List[List[str]] = page.evaluate(
        """() => {
            const table = document.querySelector('#ctl00_ContentPlaceHolder1_GridView1');
            if (!table) return [];
            return Array.from(table.rows).map(row => 
                Array.from(row.cells).map(cell => cell.innerText.trim())
            );
        }"""
    )
    if not rows or len(rows) < 2:
        raise RuntimeError("No data rows found in GridView1 table")

    headers = rows[0]
    data = rows[1:]
    return pd.DataFrame(data, columns=headers)


def scrape_year(page: Page, year: str, output_dir: Path) -> Path:
    """Scrape final round table for a single year and save to CSV."""
    existing_file = get_existing_final_round_file(output_dir, year)
    if existing_file:
        print(f"[{year}] Existing file found at {existing_file}. Skipping.")
        return existing_file

    print(f"[{year}] Navigating to archive form...")
    page.goto(ARCHIVE_URL, wait_until="networkidle")
    time.sleep(REQUEST_DELAY_SECONDS)

    print(f"[{year}] Selecting year and determining final round...")
    final_round = select_year_and_get_final_round(page, year)
    print(f"[{year}] Final round identified: Round {final_round}")

    target_file = output_dir / f"{year}_round{final_round}.csv"
    if target_file.exists():
        print(f"[{year}] Target file {target_file} already exists. Skipping.")
        return target_file

    print(f"[{year}] Setting cascaded filters (Round {final_round}, ALL, ALL, ALL, ALL)...")
    trigger_cascade_and_submit(page, final_round)

    print(f"[{year}] Extracting data table...")
    df = extract_table_dataframe(page)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_file, index=False)
    print(f"[{year}] Saved {len(df)} rows to {target_file}")
    return target_file


def run_scraper(
    target_year: Optional[str] = None,
    all_years: bool = False,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Execute scraping workflow for one or all available years."""
    if output_dir is None:
        output_dir = Path("data/raw")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(headless=True)
        page: Page = browser.new_page()

        page.goto(ARCHIVE_URL, wait_until="networkidle")
        available_years = fetch_available_years(page)
        print(f"Available archive years: {available_years}")

        years_to_process: List[str]
        if target_year:
            if target_year not in available_years:
                raise ValueError(
                    f"Requested year {target_year} not in available years: {available_years}"
                )
            years_to_process = [target_year]
        elif all_years:
            years_to_process = available_years
        else:
            # Default to the most recent completed year
            years_to_process = [available_years[0]]

        for y in years_to_process:
            file_path = scrape_year(page, y, output_dir)
            saved_files.append(file_path)

        browser.close()

    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape JoSAA final round cutoffs.")
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Specific year to scrape (e.g. 2024)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all available historical years",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw CSV files",
    )
    args = parser.parse_args()

    try:
        saved = run_scraper(
            target_year=args.year, all_years=args.all, output_dir=args.output_dir
        )
        print(f"\nScraping complete. Processed {len(saved)} files:")
        for f in saved:
            print(f"  - {f}")
    except Exception as exc:
        print(f"Scraping failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
