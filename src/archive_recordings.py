#!/usr/bin/env python3
"""
archive_recordings.py
---------------------
Archives MP3 recordings from WxStream output directories into dated zip files.

Groups recordings within a 5-minute time window into a single zip, named for
the recording session time. Zips are stored in a day folder named for the
recording date.

Directory structure produced:
    output/
    └── archive/
        ├── 2026_0407/
        │   ├── 2026_0407_1400.zip   <- all recordings from ~14:00 session
        │   ├── 2026_0407_1430.zip
        │   └── ...
        └── 2026_0406/
            └── ...

Sources scanned (all MP3s in these directories):
    output/recordings/
    output/stripped_recordings/
    output/trimmed_recordings/

Each zip preserves the source subdirectory name as a prefix so files from
different source folders don't collide:
    recordings/KAIZ_20260407_140034.mp3
    stripped_recordings/KAIZ_20260407_140034.mp3
    trimmed_recordings/KAIZ_20260407_140034.mp3

Usage:
    # Archive all unarchived recordings
    python src/archive_recordings.py

    # Dry run — show what would be created without writing
    python src/archive_recordings.py --dry-run

    # Archive a specific date only
    python src/archive_recordings.py --date 2026-04-07

    # Delete source files after archiving (use with caution)
    python src/archive_recordings.py --delete-source

    # Override the 5-minute grouping window
    python src/archive_recordings.py --window 10
"""

import os
import sys
import argparse
import zipfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")

SOURCE_DIRS = [
    "recordings",
    "stripped_recordings",
    "trimmed_recordings",
]

ARCHIVE_DIR = os.path.join(OUTPUT_DIR, "archive")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_mp3_datetime(filename: str) -> datetime | None:
    """
    Extract datetime from an MP3 filename of the form:
        KAIZ_20260407_140034.mp3   -> 2026-04-07 14:00:34 UTC
    Returns None if the filename doesn't match the expected pattern.
    """
    stem = Path(filename).stem          # e.g. KAIZ_20260407_140034
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    date_part = parts[-2]               # 20260407
    time_part = parts[-1]               # 140034
    if len(date_part) != 8 or len(time_part) != 6:
        return None
    try:
        return datetime(
            int(date_part[:4]),
            int(date_part[4:6]),
            int(date_part[6:8]),
            int(time_part[:2]),
            int(time_part[2:4]),
            int(time_part[4:6]),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def collect_mp3s(source_dirs: list[str],
                 filter_date: str | None = None,
                 before_date: str | None = None,
                 ) -> list[tuple[str, str, datetime]]:
    """
    Scan source directories for MP3 files with parseable timestamps.
    Returns list of (source_subdir, full_path, datetime).

    filter_date : only include files from this exact date (YYYY-MM-DD)
    before_date : only include files strictly before this date (YYYY-MM-DD)
                  used by --archive-old to select everything prior to today
    """
    cutoff = None
    if before_date:
        cutoff = datetime.strptime(before_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    found = []
    for subdir in source_dirs:
        dirpath = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.lower().endswith(".mp3"):
                continue
            dt = parse_mp3_datetime(fname)
            if dt is None:
                print(f"  SKIP  {subdir}/{fname}  (unparseable timestamp)")
                continue
            if filter_date:
                if dt.strftime("%Y-%m-%d") != filter_date:
                    continue
            if cutoff:
                if dt.replace(tzinfo=timezone.utc) >= cutoff:
                    continue
            found.append((subdir, os.path.join(dirpath, fname), dt))
    return found


def group_by_session(recordings: list[tuple[str, str, datetime]],
                     window_minutes: int = 5
                     ) -> dict[tuple[str, str], list[tuple[str, str, datetime]]]:
    """
    Group recordings into sessions where all recordings fall within
    window_minutes of each other.

    Uses a greedy forward-scan: the first file in a group anchors the session
    time, and any subsequent file within window_minutes of that anchor joins
    the session. When a file falls outside the window a new session starts.

    Returns a dict keyed by (day_label, session_label) ->
        list of (subdir, path, datetime)
    e.g. ("2026_0407", "2026_0407_1400") -> [...]
    """
    if not recordings:
        return {}

    # Sort by datetime across all source dirs
    sorted_recs = sorted(recordings, key=lambda x: x[2])
    window = timedelta(minutes=window_minutes)

    sessions: dict[tuple[str, str], list] = {}
    session_anchor: datetime | None = None
    current_key: tuple[str, str] | None = None

    for subdir, path, dt in sorted_recs:
        if session_anchor is None or dt - session_anchor > window:
            # Start a new session anchored at this file's time
            session_anchor = dt
            day_label     = dt.strftime("%Y_%m%d")
            session_label = dt.strftime("%Y_%m%d_%H%M")
            current_key   = (day_label, session_label)

        sessions.setdefault(current_key, []).append((subdir, path, dt))

    return sessions


def zip_label_exists(day_dir: str, session_label: str) -> bool:
    """Return True if a zip for this session already exists."""
    return os.path.isfile(os.path.join(day_dir, f"{session_label}.zip"))


def create_zip(day_dir: str, session_label: str,
               recordings: list[tuple[str, str, datetime]],
               dry_run: bool = False) -> int:
    """
    Create a zip file for a session. Files are stored with their source
    subdirectory as an in-zip prefix to avoid name collisions.
    Returns the number of files added.
    """
    zip_path = os.path.join(day_dir, f"{session_label}.zip")

    if dry_run:
        print(f"  [dry-run] Would create {zip_path} ({len(recordings)} files)")
        for subdir, path, dt in sorted(recordings, key=lambda x: x[2]):
            arcname = f"{subdir}/{os.path.basename(path)}"
            print(f"    {arcname}")
        return len(recordings)

    os.makedirs(day_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for subdir, path, dt in sorted(recordings, key=lambda x: x[2]):
            arcname = f"{subdir}/{os.path.basename(path)}"
            zf.write(path, arcname)

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"  Created  {zip_path}  ({len(recordings)} files, {size_kb:.0f} KB)")
    return len(recordings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive WxStream recordings into dated session zips"
    )
    parser.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Only archive recordings from this date (default: all dates)"
    )
    parser.add_argument(
        "--window", type=int, default=5, metavar="MINUTES",
        help="Session grouping window in minutes (default: 5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be archived without creating any files"
    )
    parser.add_argument(
        "--delete-source", action="store_true", default=True,
        help="Delete source MP3s after successfully archiving them (default: True)"
    )
    parser.add_argument(
        "--no-delete", action="store_true",
        help="Keep source MP3s after archiving (overrides --delete-source default)"
    )
    parser.add_argument(
        "--archive-old", action="store_true",
        help="Archive and delete all recordings strictly before today's date"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip sessions whose zip already exists (default: True)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-archive sessions even if zip already exists"
    )
    args = parser.parse_args()

    skip_existing = not args.force
    delete_source = args.delete_source and not args.no_delete

    # --archive-old: everything strictly before today, always deletes source
    before_date = None
    if args.archive_old:
        before_date   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        delete_source = True
        if args.date:
            print("WARNING: --date is ignored when --archive-old is set")

    print("WxStream Recording Archiver")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"Archive dir  : {ARCHIVE_DIR}")
    print(f"Sources      : {', '.join(SOURCE_DIRS)}")
    print(f"Window       : {args.window} minutes")
    if args.archive_old:
        print(f"Mode         : ARCHIVE OLD (before {before_date})")
    elif args.date:
        print(f"Date filter  : {args.date}")
    if args.dry_run:
        print("Mode         : DRY RUN")
    if delete_source:
        print("Mode         : DELETE SOURCE after archive")
    print("-" * 60)

    # Collect all MP3s
    recordings = collect_mp3s(
        SOURCE_DIRS,
        filter_date=None if args.archive_old else args.date,
        before_date=before_date,
    )
    if not recordings:
        print("No MP3 files found.")
        return

    print(f"Found {len(recordings)} MP3 file(s) across {len(SOURCE_DIRS)} directories")
    print()

    # Group into sessions
    sessions = group_by_session(recordings, window_minutes=args.window)
    print(f"Grouped into {len(sessions)} session(s)")
    print()

    total_zipped   = 0
    total_skipped  = 0
    total_deleted  = 0
    archived_paths = []

    for (day_label, session_label), session_recs in sorted(sessions.items()):
        day_dir = os.path.join(ARCHIVE_DIR, day_label)

        if skip_existing and zip_label_exists(day_dir, session_label):
            print(f"  SKIP     {session_label}.zip  (already exists)")
            total_skipped += 1
            continue

        print(f"  Session  {session_label}  ({len(session_recs)} files)")
        n = create_zip(day_dir, session_label, session_recs, dry_run=args.dry_run)
        total_zipped += n

        if delete_source and not args.dry_run:
            for _, path, _ in session_recs:
                try:
                    os.remove(path)
                    total_deleted += 1
                except OSError as e:
                    print(f"    WARN  Could not delete {path}: {e}")

        archived_paths.extend(path for _, path, _ in session_recs)

    print()
    print("-" * 60)
    if args.dry_run:
        print(f"Dry run complete. {total_zipped} file(s) would be archived "
              f"into {len(sessions) - total_skipped} zip(s).")
    else:
        print(f"Done. {total_zipped} file(s) archived into "
              f"{len(sessions) - total_skipped} zip(s).")
        if total_skipped:
            print(f"Skipped {total_skipped} existing zip(s).")
        if total_deleted:
            print(f"Deleted {total_deleted} source file(s) from {len(SOURCE_DIRS)} directories.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
