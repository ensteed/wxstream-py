#!/usr/bin/env python3
"""
s3_upload.py — WxStream S3 Recording Uploader

Uploads raw, stripped, and trimmed recordings to AWS S3.

Source folders:
  /root/WxStream/output/recordings/           → raw
  /root/WxStream/output/stripped_recordings/  → stripped
  /root/WxStream/output/trimmed_recordings/   → trimmed

S3 key structure:
  wxstream/recordings/{station_id}/raw/{filename}
  wxstream/recordings/{station_id}/stripped/{filename}
  wxstream/recordings/{station_id}/trimmed/{filename}

Usage:
  # Upload all new recordings
  python s3_upload.py

  # Upload a specific type only
  python s3_upload.py --type raw
  python s3_upload.py --type stripped
  python s3_upload.py --type trimmed

  # Dry run (show what would be uploaded without uploading)
  python s3_upload.py --dry-run
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# ---------------------------------------------------------------------------
# Configuration — override via environment variables or edit defaults below
# ---------------------------------------------------------------------------

BUCKET_NAME  = os.environ.get("WXSTREAM_S3_BUCKET", "wxstream-recordings")
S3_PREFIX    = os.environ.get("WXSTREAM_S3_PREFIX", "wxstream/recordings")
AWS_REGION   = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
OUTPUT_DIR   = Path(os.environ.get("WXSTREAM_OUTPUT_DIR", "/root/WxStream/output"))
MAX_WORKERS  = int(os.environ.get("WXSTREAM_S3_WORKERS", "8"))

AUDIO_EXTS   = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# Maps folder name → S3 type label
RECORDING_FOLDERS = {
    "recordings":          "raw",
    "stripped_recordings": "stripped",
    "trimmed_recordings":  "trimmed",
}

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
        client.list_buckets()
        return client
    except NoCredentialsError:
        log.error(
            "AWS credentials not found. Configure via:\n"
            "  export AWS_ACCESS_KEY_ID=...\n"
            "  export AWS_SECRET_ACCESS_KEY=...\n"
            "or run: aws configure"
        )
        sys.exit(1)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("InvalidClientTokenId", "AccessDenied"):
            log.error("AWS credential error: %s", e)
            sys.exit(1)
        return boto3.client("s3", region_name=AWS_REGION)


def s3_key_exists(s3, key: str) -> bool:
    """Return True if the S3 key already exists in the bucket."""
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "403"):
            return False
        raise


def upload_file(s3, local_path: Path, s3_key: str, dry_run: bool = False) -> dict:
    """Upload a single file to S3. Returns a result dict."""
    result = {"path": str(local_path), "key": s3_key, "status": None, "bytes": 0}

    if not local_path.exists():
        result["status"] = "missing"
        return result

    size = local_path.stat().st_size
    result["bytes"] = size

    if dry_run:
        result["status"] = "dry_run"
        log.info("[DRY RUN] %s → s3://%s/%s (%s)",
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

def discover_recordings(type_filter: str = None) -> list[dict]:
    """
    Scan output folders and return upload task dicts.

    Each task has:
      local_path — Path to the audio file
      s3_key     — Destination S3 key
      type       — raw | stripped | trimmed
    """
    tasks = []

    for folder_name, rec_type in RECORDING_FOLDERS.items():
        if type_filter and rec_type != type_filter:
            continue

        folder = OUTPUT_DIR / folder_name
        if not folder.exists():
            log.warning("Folder not found, skipping: %s", folder)
            continue

        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        if not files:
            log.info("No audio files found in %s", folder)
            continue

        log.info("Found %d file(s) in %s", len(files), folder)
        for f in files:
            station_id = _station_from_filename(f.name)
            s3_key = f"{S3_PREFIX}/{station_id}/{rec_type}/{f.name}"
            tasks.append({
                "local_path": f,
                "s3_key": s3_key,
                "type": rec_type,
            })

    return tasks


# ---------------------------------------------------------------------------
# Upload orchestration
# ---------------------------------------------------------------------------

def upload_tasks(s3, tasks: list[dict], dry_run: bool = False) -> dict:
    """Run all upload tasks in a thread pool and return summary stats."""
    stats = {"uploaded": 0, "skipped": 0, "error": 0,
             "missing": 0, "dry_run": 0, "bytes_uploaded": 0}

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


def print_summary(stats: dict):
    log.info("─" * 60)
    log.info("Upload summary:")
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
      KSTJ_20250408_120000.wav       → KSTJ
      KSTJ_trimmed.wav               → KSTJ
      KSTJ_20250408_stripped.wav     → KSTJ
    Falls back to 'UNKNOWN' if not parseable.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if parts and len(parts[0]) in (3, 4) and parts[0].isupper():
        return parts[0]
    return "UNKNOWN"


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
    p.add_argument("--type", choices=["raw", "stripped", "trimmed"],
                   help="Upload only this recording type (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded without actually uploading")
    p.add_argument("--bucket", default=BUCKET_NAME,
                   help=f"S3 bucket name (default: {BUCKET_NAME})")
    p.add_argument("--prefix", default=S3_PREFIX,
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

    global BUCKET_NAME, S3_PREFIX, MAX_WORKERS
    BUCKET_NAME = args.bucket
    S3_PREFIX   = args.prefix
    MAX_WORKERS = args.workers

    log.info("WxStream S3 Uploader")
    log.info("  Bucket    : s3://%s/%s", BUCKET_NAME, S3_PREFIX)
    log.info("  Output dir: %s", OUTPUT_DIR)
    log.info("  Workers   : %d%s", MAX_WORKERS, "  [DRY RUN]" if args.dry_run else "")

    s3 = get_s3_client()

    tasks = discover_recordings(type_filter=args.type)
    if not tasks:
        log.warning("No recordings found to upload.")
        sys.exit(0)

    log.info("Found %d recording(s) to process.", len(tasks))
    stats = upload_tasks(s3, tasks, dry_run=args.dry_run)
    print_summary(stats)


if __name__ == "__main__":
    main()
