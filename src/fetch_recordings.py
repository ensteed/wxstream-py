"""
Fetch Recordings from Twilio
Pulls all recordings created during the test run, matches them to stations
using the call SIDs from test_run_results.json, and saves a recordings log.

Can be run standalone or imported by awos_test_run.py to auto-fetch
recordings as soon as calls complete.

Usage:
    export TWILIO_SID=ACxxxx
    export TWILIO_AUTH=your_auth_token

    # Fetch recording metadata only
    python fetch_recordings.py

    # Also download audio files to ./recordings/
    python fetch_recordings.py --download

    # Fetch recordings from a specific date (default: today)
    python fetch_recordings.py --date 2026-03-17
"""

import json
import os
import time
import argparse
import requests
from datetime import datetime, timezone, date
from pathlib import Path

from twilio.rest import Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TWILIO_SID     = os.getenv("TWILIO_SID",    "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
TWILIO_AUTH    = os.getenv("TWILIO_AUTH",   "your_auth_token")

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR   = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR    = os.path.join(_PROJECT_DIR, "output")

RESULTS_FILE   = os.path.join(_OUTPUT_DIR, "logs", "test_run_results.json")
RECORDINGS_LOG = os.path.join(_OUTPUT_DIR, "logs", "recordings_log.json")
DOWNLOAD_DIR   = Path(_OUTPUT_DIR) / "recordings"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sid_map(results_file: str) -> dict:
    """Build a call SID -> station info map from the test run results."""
    try:
        with open(results_file) as f:
            results = json.load(f)
        sid_map = {}
        for r in results.get("results", []):
            if r.get("sid"):
                sid_map[r["sid"]] = {
                    "station":  r["station"],
                    "location": r["location"],
                    "phone":    r["phone"],
                    "type":     r["type"],
                }
        print(f"Loaded {len(sid_map)} call SIDs from {results_file}")
        return sid_map
    except FileNotFoundError:
        print(f"WARNING: {results_file} not found - recordings won't be matched to stations")
        return {}


def format_duration(seconds) -> str:
    if seconds is None:
        return "unknown"
    return f"{int(seconds) // 60}m {int(seconds) % 60}s"


def download_recording(recording_sid: str, dest_path: Path) -> bool:
    """Download a recording MP3 to dest_path."""
    try:
        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{TWILIO_SID}/Recordings/{recording_sid}.mp3"
        )
        resp = requests.get(url, auth=(TWILIO_SID, TWILIO_AUTH), stream=True, timeout=30)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ERROR downloading {recording_sid}: {e}")
        return False


def build_log_entry(rec, station_info: dict, download: bool = False) -> dict:
    """Build a recordings log entry from a Twilio recording object."""
    station_id = station_info.get("station", "UNKNOWN")
    location   = station_info.get("location", "Unknown")
    duration_s = int(rec.duration) if rec.duration else None
    exceeded   = duration_s and duration_s >= 118

    entry = {
        "recording_sid":  rec.sid,
        "call_sid":       rec.call_sid,
        "station":        station_id,
        "location":       location,
        "phone":          station_info.get("phone", ""),
        "type":           station_info.get("type", ""),
        "duration_s":     duration_s,
        "duration_fmt":   format_duration(duration_s),
        "status":         rec.status,
        "date_created":   str(rec.date_created),
        "exceeded_limit": exceeded,
        "local_file":     None,
    }

    if download and rec.status == "completed":
        filename = f"{station_id}_{rec.date_created.strftime('%Y%m%d_%H%M%S')}.mp3"
        dest = DOWNLOAD_DIR / filename
        print(f"  Downloading {station_id} ({format_duration(duration_s)}) -> {filename}")
        if download_recording(rec.sid, dest):
            entry["local_file"] = str(dest)
            time.sleep(0.1)

    return entry


def save_log(log: list) -> None:
    exceeded_list = [r["station"] for r in log if r["exceeded_limit"]]
    output = {
        "fetched_at":     datetime.now(timezone.utc).isoformat(),
        "total":          len(log),
        "exceeded_limit": exceeded_list,
        "recordings":     log,
    }
    with open(RECORDINGS_LOG, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Recordings log saved -> {RECORDINGS_LOG}")


def print_summary(log: list) -> None:
    matched       = sum(1 for r in log if r["station"] != "UNKNOWN")
    unmatched     = len(log) - matched
    exceeded_list = [r["station"] for r in log if r["exceeded_limit"]]
    short_list    = [r for r in log if r.get("duration_s") and r["duration_s"] < 10]

    print("-" * 60)
    print(f"Total recordings  : {len(log)}")
    print(f"Matched to station: {matched}")
    print(f"Unmatched         : {unmatched}")
    if exceeded_list:
        print(f"Exceeded 120s limit ({len(exceeded_list)}): {', '.join(exceeded_list)}")
    if short_list:
        print(f"Very short (<10s) ({len(short_list)}): {', '.join(r['station'] for r in short_list)}")
        print("  -> Short recordings may indicate disconnected numbers or wrong lines")

# ---------------------------------------------------------------------------
# Fetch by call SID list — called directly from awos_test_run.py
# ---------------------------------------------------------------------------

def fetch_recordings_for_sids(
    client: Client,
    sid_map: dict,
    download: bool = False,
) -> list:
    """
    Fetch recordings matched to a specific set of call SIDs.
    Used when awos_test_run.py calls this directly after confirming
    all calls have completed.
    """
    print(f"\nFetching recordings for {len(sid_map)} calls...")
    print("-" * 60)

    log = []
    for call_sid, station_info in sid_map.items():
        try:
            recs = client.recordings.list(call_sid=call_sid)
            if not recs:
                print(f"  {station_info['station']:<8} NO RECORDING")
                continue
            for rec in recs:
                entry = build_log_entry(rec, station_info, download)
                exceeded_flag = " *** EXCEEDED LIMIT" if entry["exceeded_limit"] else ""
                print(
                    f"  {entry['station']:<8} {entry['duration_fmt']:<10} "
                    f"{rec.status:<12} {rec.sid}{exceeded_flag}"
                )
                log.append(entry)
        except Exception as e:
            print(f"  {station_info.get('station', call_sid):<8} ERROR - {e}")

    log.sort(key=lambda x: x["station"])
    print_summary(log)
    save_log(log)
    if download:
        print(f"Audio files saved -> {DOWNLOAD_DIR}/")
    return log

# ---------------------------------------------------------------------------
# Fetch by date — used when running standalone
# ---------------------------------------------------------------------------

def fetch_recordings(target_date: date, download: bool = False) -> None:
    if TWILIO_SID.startswith("AC" + "x"):
        print("ERROR: TWILIO_SID not set.")
        return

    client  = Client(TWILIO_SID, TWILIO_AUTH)
    sid_map = load_sid_map(RESULTS_FILE)

    print(f"Fetching recordings for {target_date}...")

    date_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=timezone.utc)
    date_end   = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=timezone.utc)

    recordings = client.recordings.list(
        date_created_after=date_start,
        date_created_before=date_end,
    )

    print(f"Found {len(recordings)} recording(s)")
    print("-" * 60)

    log = []
    for rec in recordings:
        station_info = sid_map.get(rec.call_sid, {})
        entry        = build_log_entry(rec, station_info, download)
        exceeded_flag = " *** EXCEEDED LIMIT" if entry["exceeded_limit"] else ""
        matched_flag  = "" if station_info else " [unmatched]"
        print(
            f"  {entry['station']:<8} {entry['duration_fmt']:<10} "
            f"{rec.status:<12} {rec.sid}{exceeded_flag}{matched_flag}"
        )
        log.append(entry)

    log.sort(key=lambda x: x["station"])
    print_summary(log)
    save_log(log)
    if download:
        print(f"Audio files saved -> {DOWNLOAD_DIR}/")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Twilio recordings from a test run")
    parser.add_argument(
        "--date", type=str, default=str(date.today()),
        help="Date to fetch recordings for (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download MP3 audio files to ./recordings/"
    )
    args = parser.parse_args()

    try:
        target = date.fromisoformat(args.date)
    except ValueError:
        print(f"ERROR: Invalid date format '{args.date}' - use YYYY-MM-DD")
        exit(1)

    fetch_recordings(target_date=target, download=args.download)
