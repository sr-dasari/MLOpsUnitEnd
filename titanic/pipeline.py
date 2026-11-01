"""
Titanic Survival — End-to-End MLOps Pipeline

Chains every stage of the workflow in order:
  1. Data preprocessing
  2. Model training & evaluation
  3. MLflow experiment tracking & model registry
  4. Monitoring reports (drift + performance)
  5. API smoke test (optional, if server is running)

Usage:
  python pipeline.py              # run full pipeline
  python pipeline.py --skip-api   # skip the API smoke test
  python pipeline.py --check      # run monitoring in headless/CI mode (exit 1 on alerts)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
APP = Path(__file__).resolve().parent / "app"

STEPS = [
    ("Data Preprocessing",          [sys.executable, str(SRC / "data_preprocessing.py")]),
    ("Model Training & Tracking",   [sys.executable, str(SRC / "model_tracking_mlflow.py")]),
]


def run_step(name, cmd, extra_args=None):
    full_cmd = cmd + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")
    result = subprocess.run(full_cmd, cwd=str(Path(__file__).resolve().parent.parent))
    if result.returncode != 0:
        print(f"\n  FAILED: {name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  PASSED: {name}")
    return result


def run_pipeline(check_only=False, skip_api=False):
    start = time.time()
    print("Starting Titanic MLOps Pipeline...\n")

    # Steps 1-3: preprocessing, training, tracking
    for name, cmd in STEPS:
        run_step(name, cmd)

    # Step 4: monitoring reports
    monitor_args = ["--check"] if check_only else []
    monitor_result = run_step(
        "Monitoring Reports",
        [sys.executable, str(SRC / "generate_report.py")],
        monitor_args,
    )

    # Step 5: API tests
    if not skip_api:
        run_step("API Tests", [sys.executable, "-m", "pytest", str(APP / "test_api.py"), "-v"])

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full MLOps pipeline")
    parser.add_argument("--check", action="store_true", help="Monitoring in CI mode (exit 1 on alerts)")
    parser.add_argument("--skip-api", action="store_true", help="Skip API smoke test")
    args = parser.parse_args()

    run_pipeline(check_only=args.check, skip_api=args.skip_api)
