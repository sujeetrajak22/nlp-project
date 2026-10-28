#!/usr/bin/env python3
"""
run_pipeline.py — Master pipeline script for Automatic News Topic Classification.

Runs all steps in order:
  1. Download dataset (download_dataset.py)
  2. Preprocess data (preprocess.py)
  3. Train baseline model (train_baseline.py)
  4. Evaluate models (evaluate.py)
  [Optionally] Train transformer (train_transformer.py) with --with-transformer flag

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --with-transformer
  python scripts/run_pipeline.py --skip-download

Author: NLP CS459 Project — SVNIT Surat
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"


def run_step(script_name: str, description: str) -> bool:
    """Run a pipeline step and return True if successful."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Script: {script_path}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR)
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"✓ {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"✗ {description} FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete news classification pipeline"
    )
    parser.add_argument(
        "--with-transformer",
        action="store_true",
        help="Also train DistilBERT transformer model (slow on CPU)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (if already downloaded)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AUTOMATIC NEWS TOPIC CLASSIFICATION — TRAINING PIPELINE")
    print("  CS459 NLP Project — SVNIT Surat")
    print("=" * 60)

    steps = []
    if not args.skip_download:
        steps.append(("download_dataset.py", "Download & explore AG News dataset"))
    steps.append(("preprocess.py", "Preprocess text data"))
    steps.append(("train_baseline.py", "Train TF-IDF + Logistic Regression baseline"))
    if args.with_transformer:
        steps.append(("train_transformer.py", "Fine-tune DistilBERT (this may take hours on CPU)"))
    steps.append(("evaluate.py", "Evaluate models and generate visualizations"))

    total_start = time.time()
    results = []

    for script, description in steps:
        success = run_step(script, description)
        results.append((description, success))
        if not success:
            print(f"\n⚠ Pipeline stopped at: {description}")
            print("Fix the error above and re-run, or use --skip-download if data exists.")
            break

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for desc, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {desc}")
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if all(s for _, s in results):
        print("\n✓ Pipeline completed successfully!")
        print("\nNext steps:")
        print("  1. Start backend:  cd backend && uvicorn app.main:app --reload")
        print("  2. Start frontend: cd frontend && npm install && npm run dev")
        print("  3. Open browser:   http://localhost:5173")
    else:
        print("\n✗ Pipeline completed with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
