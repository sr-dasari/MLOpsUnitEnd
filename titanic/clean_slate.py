"""
Titanic Survival — Clean Slate

Removes all generated artifacts so the project can be re-run from scratch:
  - Processed dataset (data/processed_titanic.csv)
  - MLflow database (mlflow.db)
  - MLflow run artifacts (mlruns/)
  - Monitoring reports (reports/*.html)
  - Python cache (__pycache__/)

The raw dataset (data/titanic.csv) and all source code are preserved.

Usage:
  python clean_slate.py          # interactive confirmation before deleting
  python clean_slate.py --yes    # skip confirmation (for automation)
"""

import argparse
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

TARGETS = [
    ("Processed dataset",   PROJECT_DIR / "data" / "processed_titanic.csv"),
    ("MLflow database",     PROJECT_DIR / "mlflow.db"),
    ("MLflow artifacts",    PROJECT_DIR / "mlruns"),
    ("Monitoring reports",  PROJECT_DIR / "reports"),
    ("pytest cache",        PROJECT_DIR.parent / ".pytest_cache"),
]

PYCACHE_DIRS = list(PROJECT_DIR.rglob("__pycache__"))


def find_targets():
    found = []
    for label, path in TARGETS:
        if path.exists():
            found.append((label, path))
    for p in PYCACHE_DIRS:
        if p.exists():
            found.append(("Python cache", p))
    return found


def remove(path):
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clean(skip_confirm=False):
    found = find_targets()

    if not found:
        print("Nothing to clean — project is already in a clean state.")
        return

    print("The following will be removed:\n")
    for label, path in found:
        size = ""
        if path.is_file():
            size = f" ({path.stat().st_size / 1024:.0f} KB)"
        elif path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            size = f" ({total / 1024:.0f} KB)"
        print(f"  - {label}: {path.relative_to(PROJECT_DIR.parent)}{size}")

    if not skip_confirm:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print()
    for label, path in found:
        remove(path)
        print(f"  Removed: {label}")

    # Recreate reports/ with .gitkeep
    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / ".gitkeep").touch()

    print("\nClean slate complete. Run the pipeline to regenerate everything:")
    print("  python titanic/pipeline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset project to a clean state")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    clean(skip_confirm=args.yes)
