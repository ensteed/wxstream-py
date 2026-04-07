"""
silence_strip.py
----------------
Strip trailing silence from all MP3 files in the recordings/ directory.

Run between the calls step (awos_test_run.py) and transcription
(transcribe_whisper.py) to prevent Whisper hallucination loops caused
by long silent tails at the end of AWOS recordings.

Usage:
    python silence_strip.py                 # process recordings/ next to this script
    python silence_strip.py --dir /path     # specify recordings directory
    python silence_strip.py --threshold -35 # silence threshold in dBFS (default: -40)
    python silence_strip.py --duration 1.0  # min silence duration to strip (default: 1.5s)

Requirements: ffmpeg in PATH
"""

import os
import sys
import argparse
import subprocess
import shutil
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
_OUTPUT_DIR  = os.path.join(_PROJECT_DIR, "output")


def strip_file(mp3_path, threshold_db=-40, min_silence_s=1.5):
    """
    Strip trailing silence from a single MP3 file in place.
    Returns (success, orig_kb, new_kb, skipped_reason).
    skipped_reason is None on success, or a string explaining why it was skipped.
    """
    if not os.path.isfile(mp3_path):
        return False, 0, 0, "file not found"

    orig_kb = os.path.getsize(mp3_path) / 1024
    tmp_path = mp3_path + ".silstrip.tmp.mp3"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", mp3_path,
                "-af", (
                    f"silenceremove="
                    f"stop_periods=-1:"
                    f"stop_duration={min_silence_s}:"
                    f"stop_threshold={threshold_db}dB"
                ),
                "-c:a", "libmp3lame", "-q:a", "2",
                tmp_path,
            ],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False, orig_kb, 0, f"ffmpeg error: {result.stderr[:200].strip()}"

        new_kb = os.path.getsize(tmp_path) / 1024

        # Sanity check: output should be at least 5 KB and not larger than input
        if new_kb < 5:
            os.remove(tmp_path)
            return False, orig_kb, new_kb, f"output too small ({new_kb:.0f} KB) — skipped"

        shutil.move(tmp_path, mp3_path)
        return True, orig_kb, new_kb, None

    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, orig_kb, 0, "ffmpeg timed out"

    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False, orig_kb, 0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Strip trailing silence from AWOS recordings before transcription"
    )
    parser.add_argument(
        "--dir", default=os.path.join(_OUTPUT_DIR, "recordings"),
        help="Directory containing MP3 files (default: output/recordings/ relative to project root)"
    )
    parser.add_argument(
        "--threshold", type=float, default=-40.0, metavar="dBFS",
        help="Silence threshold in dBFS (default: -40). Lower = more aggressive."
    )
    parser.add_argument(
        "--duration", type=float, default=1.5, metavar="SECONDS",
        help="Minimum silence duration to strip in seconds (default: 1.5)"
    )
    args = parser.parse_args()

    recordings_dir = args.dir

    if not os.path.isdir(recordings_dir):
        print(f"ERROR: recordings directory not found: {recordings_dir}")
        sys.exit(1)

    mp3_files = sorted(
        f for f in os.listdir(recordings_dir) if f.lower().endswith(".mp3")
    )

    if not mp3_files:
        print(f"No MP3 files found in {recordings_dir}")
        sys.exit(0)

    print(f"Silence strip  |  {datetime.now().strftime('%H:%M:%S')}")
    print(f"Directory      : {recordings_dir}")
    print(f"Files          : {len(mp3_files)}")
    print(f"Threshold      : {args.threshold} dBFS")
    print(f"Min duration   : {args.duration}s")
    print("-" * 60)

    ok_count   = 0
    skip_count = 0
    fail_count = 0
    total_saved_kb = 0.0

    for filename in mp3_files:
        mp3_path = os.path.join(recordings_dir, filename)
        success, orig_kb, new_kb, reason = strip_file(
            mp3_path, args.threshold, args.duration
        )

        if success:
            saved = orig_kb - new_kb
            total_saved_kb += saved
            ok_count += 1
            print(f"  OK      {filename:<38}  {orig_kb:6.0f} -> {new_kb:6.0f} KB  (-{saved:.0f} KB)")
        elif reason and "skipped" in reason:
            skip_count += 1
            print(f"  SKIP    {filename:<38}  {reason}")
        else:
            fail_count += 1
            print(f"  FAIL    {filename:<38}  {reason}")

    print("-" * 60)
    print(f"Done: {ok_count} stripped, {skip_count} skipped, {fail_count} failed")
    print(f"Total silence removed: {total_saved_kb:.0f} KB")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
