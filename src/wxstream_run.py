"""
wxstream_run.py
---------------
Full WxStream pipeline orchestrator. Runs each step in sequence,
stopping immediately if any step fails.

Pipeline steps:
  1. awos_test_run.py     - Place calls, wait for completion, download recordings
  2. transcribe.py        - Transcribe recordings with Whisper (word timestamps)
  3. parse_transcripts.py - Parse transcripts into structured weather data
  4. audio_trim.py        - Trim raw recordings to single clean broadcast loop
  5. generate_report.py   - Generate self-contained HTML weather report

Usage:
    python wxstream_run.py                        # run once
    python wxstream_run.py --from transcribe      # resume from a specific step
    python wxstream_run.py --dry-run              # print commands without running
    python wxstream_run.py --schedule 6           # run every 30 min, 6 times total
    python wxstream_run.py --schedule 6 --interval 60  # every 60 min, 6 times

Steps can be skipped individually:
    python wxstream_run.py --skip trim

Available step names:
    calls, transcribe, parse, trim, report

Scheduling notes:
  - Each run launches as a separate subprocess so runs can overlap if a
    previous run is still in progress when the next slot fires.
  - Use Ctrl+C to stop the scheduler between runs (a running pipeline
    will not be interrupted).

Requirements: Python 3.6+
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from threading import Thread

# ---------------------------------------------------------------------------
# Configuration — all paths relative to this script's directory
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def p(*parts):
    """Build a path relative to SCRIPT_DIR."""
    return os.path.join(SCRIPT_DIR, *parts)

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {
        "name":  "calls",
        "label": "Place calls, wait for completion, and download recordings",
        "cmd":   [sys.executable, p("awos_test_run.py"), "--fetch", "--download"],
    },
    {
        "name":  "silence_strip",
        "label": "Strip trailing silence from recordings before transcription",
        "cmd":   [sys.executable, p("silence_strip.py")],
    },
    {
        "name":  "transcribe",
        "label": "Transcribe recordings with Whisper",
        "cmd":   [sys.executable, p("transcribe_whisper.py"), "--local"],
    },
    {
        "name":  "parse",
        "label": "Parse transcripts into structured weather data",
        "cmd":   [
            sys.executable, p("parse_transcripts.py"),
            p("transcripts.json"),
            "--output", p("parsed_results.json"),
        ],
    },
    {
        "name":  "trim",
        "label": "Trim recordings to single broadcast loop",
        "cmd":   [sys.executable, p("audio_trim.py")],
    },
    {
        "name":  "report",
        "label": "Generate HTML weather report",
        "cmd":   [
            sys.executable, p("generate_report.py"),
            "--input",  p("parsed_results.json"),
            "--title",  "Missouri AWOS / ASOS Weather Observations",
        ],
    },
]

STEP_NAMES = [s["name"] for s in STEPS]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hline(char="-", width=60):
    print(char * width, flush=True)

def elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"

def run_step(step, dry_run=False):
    """Run one pipeline step. Returns (success, duration_seconds)."""
    cmd = step["cmd"]
    if cmd is None:
        print(f"  [SKIP] {step['label']} - script not configured", flush=True)
        return True, 0.0

    print(f"  Command: {' '.join(cmd)}", flush=True)

    if dry_run:
        print("  [DRY RUN] - not executed", flush=True)
        return True, 0.0

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=SCRIPT_DIR)
    duration = time.monotonic() - t0
    return proc.returncode == 0, duration

# ---------------------------------------------------------------------------
# Single pipeline run
# ---------------------------------------------------------------------------

def run_pipeline(steps_to_run, dry_run=False, run_label=""):
    """
    Execute the given steps in order. Returns True if all succeeded.
    Stops immediately on the first failure.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = f"  WxStream Pipeline  |  {now}"
    if run_label:
        label += f"  |  {run_label}"

    print(flush=True)
    hline("=")
    print(label, flush=True)
    hline("=")
    print(f"  Steps ({len(steps_to_run)}): "
          f"{', '.join(s['name'] for s in steps_to_run)}", flush=True)
    if dry_run:
        print("  MODE: DRY RUN - commands will not be executed", flush=True)
    print(flush=True)

    run_start = time.monotonic()
    results = []

    for i, step in enumerate(steps_to_run, 1):
        hline()
        print(f"  Step {i}/{len(steps_to_run)}: {step['label']}", flush=True)
        hline()

        success, duration = run_step(step, dry_run=dry_run)
        results.append((step["name"], success, duration))

        if success:
            print(flush=True)
            print(f"  DONE  ({elapsed(duration)})", flush=True)
        else:
            print(flush=True)
            print(f"  FAILED  ({elapsed(duration)})", flush=True)
            hline("=")
            print(f"  Pipeline stopped at step '{step['name']}'.", flush=True)
            print(f"  Fix the issue and re-run with:  "
                  f"python wxstream_run.py --from {step['name']}", flush=True)
            hline("=")
            return False

        print(flush=True)

    total = time.monotonic() - run_start
    hline("=")
    print(f"  All {len(steps_to_run)} steps completed successfully", flush=True)
    print(f"  Total time: {elapsed(total)}", flush=True)
    hline()
    for name, ok, dur in results:
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}]  {name:<12}  {elapsed(dur):>8}", flush=True)
    hline("=")
    print(flush=True)
    return True

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduled(steps_to_run, count, interval_minutes, dry_run=False):
    """
    Launch the pipeline `count` times, firing every `interval_minutes`.
    Each run is launched in its own thread so runs can overlap if the
    previous one is still in progress.
    """
    interval_s = interval_minutes * 60
    threads = []

    hline("=")
    print(f"  WxStream Scheduler", flush=True)
    hline("=")
    print(f"  Runs          : {count}", flush=True)
    print(f"  Interval      : every {interval_minutes} minutes", flush=True)
    print(f"  Steps         : {', '.join(s['name'] for s in steps_to_run)}", flush=True)
    print(f"  Started at    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Expected end  : ~{elapsed(count * interval_s)} from now "
          f"(plus last run duration)", flush=True)
    print(f"  Stop with     : Ctrl+C (in-progress runs will complete)", flush=True)
    hline("=")
    print(flush=True)

    def fire(run_num):
        label = f"Run {run_num}/{count}"
        run_pipeline(steps_to_run, dry_run=dry_run, run_label=label)

    try:
        for run_num in range(1, count + 1):
            fire_time = datetime.now().strftime("%H:%M:%S")
            print(f"[Scheduler] Firing run {run_num}/{count} at {fire_time}", flush=True)

            t = Thread(target=fire, args=(run_num,), daemon=False)
            t.start()
            threads.append(t)

            if run_num < count:
                next_fire = datetime.now().strftime  # placeholder
                print(f"[Scheduler] Next run in {interval_minutes} min  "
                      f"(Ctrl+C to stop between runs)", flush=True)
                print(flush=True)
                try:
                    time.sleep(interval_s)
                except KeyboardInterrupt:
                    print(flush=True)
                    print("[Scheduler] Interrupted — waiting for in-progress "
                          "runs to finish...", flush=True)
                    break

    except KeyboardInterrupt:
        print(flush=True)
        print("[Scheduler] Interrupted — waiting for in-progress "
              "runs to finish...", flush=True)

    # Wait for all launched runs to complete
    for t in threads:
        t.join()

    hline("=")
    print(f"  Scheduler complete  |  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    hline("=")
    print(flush=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WxStream full pipeline orchestrator"
    )
    parser.add_argument(
        "--from", dest="from_step", default=None, metavar="STEP",
        help=f"Start from this step (skip earlier ones). "
             f"One of: {', '.join(STEP_NAMES)}"
    )
    parser.add_argument(
        "--skip", dest="skip_steps", action="append", default=[], metavar="STEP",
        help=f"Skip a specific step (repeatable). One of: {', '.join(STEP_NAMES)}"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--schedule", metavar="N", type=int, default=None,
        help="Run the pipeline N times, once every 30 minutes (use with --interval to change)"
    )
    parser.add_argument(
        "--interval", metavar="MINUTES", type=int, default=30,
        help="Minutes between scheduled runs (default: 30, requires --schedule)"
    )
    args = parser.parse_args()

    # Validate step names
    for name in ([args.from_step] if args.from_step else []) + args.skip_steps:
        if name not in STEP_NAMES:
            print(f"Error: unknown step '{name}'. "
                  f"Valid steps: {', '.join(STEP_NAMES)}")
            sys.exit(1)

    # Determine which steps to run
    steps_to_run = list(STEPS)
    if args.from_step:
        idx = STEP_NAMES.index(args.from_step)
        steps_to_run = steps_to_run[idx:]
    for skip in args.skip_steps:
        steps_to_run = [s for s in steps_to_run if s["name"] != skip]

    if args.schedule:
        if args.schedule < 1:
            print("Error: --schedule N must be >= 1")
            sys.exit(1)
        run_scheduled(
            steps_to_run,
            count=args.schedule,
            interval_minutes=args.interval,
            dry_run=args.dry_run,
        )
    else:
        success = run_pipeline(steps_to_run, dry_run=args.dry_run)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
