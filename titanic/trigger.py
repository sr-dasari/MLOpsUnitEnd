"""
Titanic Survival — File-Watcher Trigger (Watchdog)

Monitors the data/ and src/ directories for file changes and automatically
re-runs the full MLOps pipeline when a change is detected.

Usage:
  python trigger.py                # watch and auto-run pipeline on changes
  python trigger.py --cooldown 60  # wait 60s between pipeline runs (default: 30s)
  python trigger.py --once         # run pipeline once on next change, then exit

Press Ctrl+C to stop watching.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_DIR / "pipeline.py"

WATCH_DIRS = [
    PROJECT_DIR / "data",
    PROJECT_DIR / "src",
]

WATCH_EXTENSIONS = {".csv", ".py"}


class PipelineTrigger(FileSystemEventHandler):
    def __init__(self, cooldown=30, run_once=False):
        super().__init__()
        self.cooldown = cooldown
        self.run_once = run_once
        self.last_run = 0
        self.triggered = False

    def on_modified(self, event):
        self._handle(event)

    def on_created(self, event):
        self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix not in WATCH_EXTENSIONS:
            return

        # Skip __pycache__ and temporary files
        if "__pycache__" in str(path) or path.name.startswith("."):
            return

        now = time.time()
        if now - self.last_run < self.cooldown:
            remaining = int(self.cooldown - (now - self.last_run))
            print(f"  [trigger] Change detected in {path.name}, cooldown active ({remaining}s remaining)")
            return

        print(f"\n  [trigger] Change detected: {path.relative_to(PROJECT_DIR)}")
        print(f"  [trigger] Running pipeline...\n")

        self.last_run = now
        result = subprocess.run(
            [sys.executable, str(PIPELINE_SCRIPT), "--skip-api"],
            cwd=str(PROJECT_DIR.parent),
        )

        if result.returncode == 0:
            print(f"\n  [trigger] Pipeline completed successfully.")
        else:
            print(f"\n  [trigger] Pipeline failed (exit code {result.returncode}).")

        if self.run_once:
            self.triggered = True

        print(f"  [trigger] Watching for changes... (Ctrl+C to stop)\n")


def main():
    parser = argparse.ArgumentParser(description="Watch for file changes and trigger pipeline")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Minimum seconds between pipeline runs (default: 30)")
    parser.add_argument("--once", action="store_true",
                        help="Run pipeline on next change, then exit")
    args = parser.parse_args()

    handler = PipelineTrigger(cooldown=args.cooldown, run_once=args.once)
    observer = Observer()

    for watch_dir in WATCH_DIRS:
        if watch_dir.exists():
            observer.schedule(handler, str(watch_dir), recursive=False)
            print(f"  [trigger] Watching: {watch_dir.relative_to(PROJECT_DIR)}/")
        else:
            print(f"  [trigger] Warning: {watch_dir} does not exist, skipping")

    print(f"  [trigger] Cooldown: {args.cooldown}s between runs")
    print(f"  [trigger] Waiting for changes... (Ctrl+C to stop)\n")

    observer.start()
    try:
        while True:
            if args.once and handler.triggered:
                print("  [trigger] --once mode: exiting after first pipeline run.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  [trigger] Stopped.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
