#!/usr/bin/env python3
"""
s3_upload.py — WxStream S3 Recording Uploader

Uploads raw, stripped, and trimmed recordings from a run directory to AWS S3.

S3 key structure:
  wxstream/recordings/{run_id}/{station_id}/raw/{filename}
  wxstream/recordings/{run_id}/{station_id}/stripped/{filename}
  wxstream/recordings/{run_id}/{station_id}/trimmed/{filename}

Usage:
  # Upload a specific run
  python s3_upload.py --run runs/20250408_120000

  # Upload the latest run
  python s3_upload.py --run runs/latest

  # Upload all runs not yet in S3
  python s3_upload.py --all

  # Dry run (no uploads, just show what would be uploaded)
  python s3_upload.py --run runs/latest --dry-run
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ---------------------------------------------------------------------------
# Configuration — override via environment variables or edit defaults below
# ---------------------------------------------------------------------------

BUCKET_NAME    = os.environ.get("WXSTREAM_S3_BUCKET", "wxstream-recordings")
S3_PREFIX      = os.environ.get("WXSTREAM_S3_PREFIX", "wxstream/recordings")
AWS_REGION     = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

RUNS_DIR       = Path(os.environ.get("WXSTREAM_RUNS_DIR", "/root/WxStream/runs"))
STRIPPED_DIR   = Path(os.environ.get("WXSTREAM_STRIPPED_DIR", "/root/WxStream/stripped_recordings"))

# Max parallel uploads
MAX_WORKERS    = int(os.environ.get("WXSTREAM_S3_WORKERS", "8"))

# Audio extensions to consider
AUDIO_EXTS     = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("s3_upload")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def get_s3_client():
    """Return a boto3 S3 client, raising a clear error if credentials are missing."""
    try:
        client = boto3.client("s3", region_name=AWS_REGION)
        # Quick credential check
        client.list_buckets()
        return client
    except NoCredentialsError:
        log.error(
            "AWS credentials not found. Configure via:\n"
            "  export AWS_ACCESS_KEY_ID=...\n"
            "  export AWS_SECRET_ACCESS_KEY=...\n"
            "or use an IAM role / AWS credentials file (~/.aws/credentials)."
        )
        sys.exit(1)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("InvalidClientTokenId", "AccessDenied"):
            log.error("AWS credential error: %s", e)
            sys.exit(1)
        # Other errors (e.g. network) — still return the client; bucket ops will surface issues
        return boto3.client("s3", region_name=AWS_REGION)


def s3_key_exists(s3, key: str) -> bool:
    """Return True if the S3 key already exists in the bucket."""
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_file(s3, local_path: Path, s3_key: str, dry_run: bool = False) -> dict:
    """
    Upload a single file to S3.
    Returns a result dict with keys: path, key, status, bytes.
    """
    result = {"path": str(local_path), "key": s3_key, "status": None, "bytes": 0}

    if not local_path.exists():
        result["status"] = "missing"
        return result

    size = local_path.stat().st_size
    result["bytes"] = size

    if dry_run:
        result["status"] = "dry_run"
        log.info("[DRY RUN] Would upload %s → s3://%s/%s (%s)",
                 local_path.name, BUCKET_NAME, s3_key, _fmt_bytes(size))
        return result

    if s3_key_exists(s3, s3_key):
        result["status"] = "skipped"
        log.debug("Skipping (already exists): %s", s3_key)
        return result

    try:
        s3.upload_file(
            str(local_path),
            BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "audio/wav"},
        )
        result["status"] = "uploaded"
        log.info("Uploaded %s → s3://%s/%s (%s)",
                 local_path.name, BUCKET_NAME, s3_key, _fmt_bytes(size))
    except ClientError as e:
        result["status"] = "error"
        result["error"] = str(e)
        log.error("Failed to upload %s: %s", local_path, e)

    return result


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------

def resolve_run_dir(run_arg: str) -> Path:
    """Resolve --run argument to an absolute Path."""
    p = Path(run_arg)
    if not p.is_absolute():
        p = RUNS_DIR / run_arg
    # Follow symlink (e.g. runs/latest)
    p = p.resolve()
    if not p.exists():
        log.error("Run directory not found: %s", p)
        sys.exit(1)
    return p


def discover_run_recordings(run_dir: Path) -> list[dict]:
    """
    Walk a run directory and return a list of upload task dicts.

    Expected run layout (matches wxstream_pipeline.py):
      run_dir/
        recordings/          ← raw recordings per station
          KXXX_<timestamp>.wav
        trimmed/             ← trimmed single-loop recordings
          KXXX_trimmed.wav
        ...

    Stripped recordings live in a shared stripped_recordings/ dir
    with filenames matching: KXXX_<timestamp>_stripped.wav
    """
    run_id = run_dir.name  # e.g. "20250408_120000"
    tasks = []

    # --- Raw recordings ---
    raw_dir = run_dir / "recordings"
    if raw_dir.exists():
        for f in sorted(raw_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                station_id = _station_from_filename(f.name)
                tasks.append({
                    "local_path": f,
                    "s3_key": f"{S3_PREFIX}/{run_id}/{station_id}/raw/{f.name}",
                    "type": "raw",
                })
    else:
        log.warning("No recordings/ directory found in %s", run_dir)

    # --- Stripped recordings ---
    # Stripped files are named KXXX_<timestamp>_stripped.wav and live in
    # STRIPPED_DIR.  Match them to this run by timestamp in the filename.
    if STRIPPED_DIR.exists():
        for f in sorted(STRIPPED_DIR.iterdir()):
            if f.suffix.lower() not in AUDIO_EXTS:
                continue
            if run_id not in f.name and not _run_id_in_name(run_id, f.name):
                continue
            station_id = _station_from_filename(f.name)
            tasks.append({
                "local_path": f,
                "s3_key": f"{S3_PREFIX}/{run_id}/{station_id}/stripped/{f.name}",
                "type": "stripped",
            })

    # --- Trimmed recordings ---
    trimmed_dir = run_dir / "trimmed"
    if trimmed_dir.exists():
        for f in sorted(trimmed_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                station_id = _station_from_filename(f.name)
                tasks.append({
                    "local_path": f,
                    "s3_key": f"{S3_PREFIX}/{run_id}/{station_id}/trimmed/{f.name}",
                    "type": "trimmed",
                })
    else:
        log.debug("No trimmed/ directory in %s", run_dir)

    return tasks


def discover_all_runs() -> list[tuple[Path, list[dict]]]:
    """Return (run_dir, tasks) for every timestamped run in RUNS_DIR."""
    if not RUNS_DIR.exists():
        log.error("Runs directory not found: %s", RUNS_DIR)
        sys.exit(1)

    results = []
    for entry in sorted(RUNS_DIR.iterdir()):
        # Only process timestamped dirs (YYYYMMDD_HHMMSS), skip symlinks like 'latest'
        if entry.is_symlink() or not entry.is_dir():
            continue
        if len(entry.name) == 15 and entry.name[8] == "_":
            tasks = discover_run_recordings(entry)
            if tasks:
                results.append((entry, tasks))
    return results


# ---------------------------------------------------------------------------
# Upload orchestration
# ---------------------------------------------------------------------------

def upload_tasks(s3, tasks: list[dict], dry_run: bool = False) -> dict:
    """Run all upload tasks in a thread pool and return summary stats."""
    stats = {"uploaded": 0, "skipped": 0, "error": 0, "missing": 0,
             "dry_run": 0, "bytes_uploaded": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(upload_file, s3, t["local_path"], t["s3_key"], dry_run): t
            for t in tasks
        }
        for future in as_completed(futures):
            res = future.result()
            status = res.get("status", "error")
            stats[status] = stats.get(status, 0) + 1
            if status == "uploaded":
                stats["bytes_uploaded"] += res.get("bytes", 0)

    return stats


def print_summary(stats: dict, run_label: str = ""):
    label = f" [{run_label}]" if run_label else ""
    log.info("─" * 60)
    log.info("Upload summary%s:", label)
    log.info("  Uploaded : %d  (%s)", stats["uploaded"], _fmt_bytes(stats["bytes_uploaded"]))
    log.info("  Skipped  : %d  (already in S3)", stats["skipped"])
    log.info("  Dry run  : %d", stats["dry_run"])
    log.info("  Missing  : %d  (file not found locally)", stats["missing"])
    log.info("  Errors   : %d", stats["error"])
    log.info("─" * 60)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _station_from_filename(name: str) -> str:
    """
    Extract the ICAO station ID from a filename like:
      KSTJ_20250408_120000.wav  → KSTJ
      KSTJ_trimmed.wav          → KSTJ
      KSTJ_20250408_stripped.wav → KSTJ
    Falls back to 'UNKNOWN' if not parseable.
    """
    stem = Path(name).stem  # strip extension
    parts = stem.split("_")
    if parts and len(parts[0]) in (3, 4) and parts[0].isupper():
        return parts[0]
    return "UNKNOWN"


def _run_id_in_name(run_id: str, filename: str) -> bool:
    """Check whether the run's date portion appears in the filename."""
    date_part = run_id[:8]  # YYYYMMDD
    return date_part in filename


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Upload WxStream recordings (raw/stripped/trimmed) to AWS S3."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run",  metavar="RUN_DIR",
                       help="Path to a specific run directory (or 'runs/latest')")
    group.add_argument("--all",  action="store_true",
                       help="Upload all timestamped run directories")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded without actually uploading")
    p.add_argument("--bucket",  default=BUCKET_NAME,
                   help=f"S3 bucket name (default: {BUCKET_NAME})")
    p.add_argument("--prefix",  default=S3_PREFIX,
                   help=f"S3 key prefix (default: {S3_PREFIX})")
    p.add_argument("--workers", type=int, default=MAX_WORKERS,
                   help=f"Parallel upload threads (default: {MAX_WORKERS})")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging")
    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Apply CLI overrides
    global BUCKET_NAME, S3_PREFIX, MAX_WORKERS
    BUCKET_NAME = args.bucket
    S3_PREFIX   = args.prefix
    MAX_WORKERS = args.workers

    log.info("WxStream S3 Uploader")
    log.info("  Bucket : s3://%s/%s", BUCKET_NAME, S3_PREFIX)
    log.info("  Workers: %d%s", MAX_WORKERS, "  [DRY RUN]" if args.dry_run else "")

    s3 = get_s3_client()

    if args.run:
        run_dir = resolve_run_dir(args.run)
        log.info("Run: %s", run_dir)
        tasks = discover_run_recordings(run_dir)
        if not tasks:
            log.warning("No recordings found in %s", run_dir)
            sys.exit(0)
        log.info("Found %d recording(s) to process.", len(tasks))
        stats = upload_tasks(s3, tasks, dry_run=args.dry_run)
        print_summary(stats, run_label=run_dir.name)

    elif args.all:
        all_runs = discover_all_runs()
        if not all_runs:
            log.warning("No run directories found in %s", RUNS_DIR)
            sys.exit(0)
        log.info("Found %d run(s) to process.", len(all_runs))
        total_stats = {"uploaded": 0, "skipped": 0, "error": 0,
                       "missing": 0, "dry_run": 0, "bytes_uploaded": 0}
        for run_dir, tasks in all_runs:
            log.info("Processing run: %s (%d files)", run_dir.name, len(tasks))
            stats = upload_tasks(s3, tasks, dry_run=args.dry_run)
            print_summary(stats, run_label=run_dir.name)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)
        log.info("═" * 60)
        log.info("TOTAL across all runs:")
        print_summary(total_stats)


if __name__ == "__main__":
    main()
