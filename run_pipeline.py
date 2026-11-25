"""Vikalp Pipeline Automation Runner.

Executes all end-to-end stages of data ingestion, feature engineering,
model training, statistical backtesting, CI verification, and app launching.

Usage:
    python run_pipeline.py --all           # Execute full pipeline end-to-end
    python run_pipeline.py --stage clean   # Run cleaning and feature engineering
    python run_pipeline.py --stage train   # Run model training & baseline evaluations
    python run_pipeline.py --stage test    # Run full test suite with coverage
    python run_pipeline.py --stage lint    # Run ruff and mypy static analysis
    python run_pipeline.py --stage app     # Launch Streamlit web app
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_command(cmd: list[str], description: str) -> None:
    """Execute command and fail loudly on non-zero return code."""
    print(f"\n========================================================")
    print(f"[*] {description}")
    print(f"[*] Command: {' '.join(cmd)}")
    print(f"========================================================")
    res = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if res.returncode != 0:
        print(f"\n[!] ERROR: {description} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✓] {description} completed successfully.")


def stage_clean() -> None:
    run_command([PYTHON, "src/clean.py"], "Stage 1: Clean Raw JoSAA Cutoffs")
    run_command([PYTHON, "src/features.py"], "Stage 2: Feature Engineering & Time-Lag Extraction")


def stage_train() -> None:
    run_command([PYTHON, "src/train.py"], "Stage 3: Train LightGBM & Validate Baselines")


def stage_backtest() -> None:
    run_command([PYTHON, "generate_reliability_plot.py"], "Stage 4: Generate Calibration & Backtest Outputs")


def stage_lint() -> None:
    run_command([PYTHON, "-m", "ruff", "check", "src/"], "Stage 5A: Ruff Lint Check")
    run_command([PYTHON, "-m", "mypy", "src/"], "Stage 5B: Mypy Static Type Check")


def stage_test() -> None:
    run_command(
        [PYTHON, "-m", "pytest", "--cov=src", "--cov-report=term"],
        "Stage 6: Pytest Suite & Coverage Verification (target >= 85%)",
    )


def stage_app() -> None:
    run_command([PYTHON, "-m", "streamlit", "run", "app/streamlit_app.py"], "Stage 7: Launch Vikalp Streamlit App")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vikalp Pipeline Automation Runner")
    parser.add_argument(
        "--stage",
        choices=["clean", "train", "backtest", "lint", "test", "app"],
        help="Run a specific pipeline stage.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all verification and training pipeline stages in order.",
    )
    args = parser.parse_args()

    if args.all or args.stage is None:
        stage_clean()
        stage_train()
        stage_backtest()
        stage_lint()
        stage_test()
        print("\n[✓] All pipeline stages verified successfully! Ready to run: python run_pipeline.py --stage app")
    elif args.stage == "clean":
        stage_clean()
    elif args.stage == "train":
        stage_train()
    elif args.stage == "backtest":
        stage_backtest()
    elif args.stage == "lint":
        stage_lint()
    elif args.stage == "test":
        stage_test()
    elif args.stage == "app":
        stage_app()


if __name__ == "__main__":
    main()
