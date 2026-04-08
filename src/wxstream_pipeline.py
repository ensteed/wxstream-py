"""
wxstream_pipeline.py
--------------------
Pipelined WxStream orchestrator.

Pipeline flow per run:
  Phase 1  - Place Twilio calls (or load local recordings with --local)
  Phase 2  - Collect transcription worker results
  Phase 3  - Save transcripts.json  (into run directory)
  Phase 4  - parse_transcripts.py   (-> parsed_results.json)
  Phase 4b - audio_trim.py          (-> trim_manifest.json + trimmed audio)
  Phase 5  - generate_report.py     (-> awos_report.html)
  Phase 6  - atlas_import.py        (-> MongoDB Atlas, if configured)
  Phase 7  - s3_upload.py           (-> AWS S3, if WXSTREAM_S3_BUCKET set)

Each run gets its own timestamped directory under runs/. A runs/latest
symlink always points to the most recent run.

Usage:
    python wxstream_pipeline.py                        # full pipeline
    python wxstream_pipeline.py --local                # skip calls, use local recordings
    python wxstream_pipeline.py --workers 10           # concurrent Whisper workers
    python wxstream_pipeline.py --dry-run              # simulate without API calls
    python wxstream_pipeline.py --schedule 6           # run 6 times every 30 min
    python wxstream_pipeline.py --schedule 6 --interval 60
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)   # WxStream/
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "output")
sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

log = logging.getLogger("wxstream_pipeline")
log.setLevel(logging.DEBUG)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"
))
_file = logging.FileHandler(
    os.path.join(OUTPUT_DIR, "logs", "wxstream_pipeline.log"), encoding="utf-8"
)
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
if not log.handlers:
    log.addHandler(_console)
    log.addHandler(_file)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRANSCRIBE_WORKERS = 10
STATIONS_FILE      = os.path.join(PROJECT_DIR, "missouri_awos_asos_stations.json")
RECORDINGS_DIR     = os.path.join(OUTPUT_DIR, "recordings")
STRIPPED_DIR       = os.path.join(OUTPUT_DIR, "stripped_recordings")
TRIMMED_DIR        = os.path.join(OUTPUT_DIR, "trimmed_recordings")
RUNS_DIR           = os.path.join(OUTPUT_DIR, "runs")

# Active run paths — overwritten by _init_run_dir() at pipeline start
TRANSCRIPTS_FILE   = os.path.join(OUTPUT_DIR, "transcripts.json")
PARSED_FILE        = os.path.join(OUTPUT_DIR, "parsed_results.json")

# ---------------------------------------------------------------------------
# Module-level pre-loads (shared across all worker threads)
# ---------------------------------------------------------------------------

_STATIONS_DICT = {}
if os.path.isfile(STATIONS_FILE):
    with open(STATIONS_FILE) as _f:
        _STATIONS_DICT = {s["id"]: s for s in json.load(_f)}

from transcribe import build_transcription_prompt

_OPENAI_CLIENT = None

def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-x"):
            raise RuntimeError("OPENAI_API_KEY not set")
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
    return _OPENAI_CLIENT

# ---------------------------------------------------------------------------
# Hallucination stripping
# ---------------------------------------------------------------------------

def _find_timestamp_freeze_point(words, min_advance=0.05, freeze_run=8):
    """
    Find the index in word timestamps where Whisper's alignment freezes —
    i.e. where timestamps stop advancing meaningfully. This marks the onset
    of a hallucination loop regardless of what the text looks like.

    Returns the index of the first frozen word, or len(words) if no freeze
    is detected (meaning all timestamps are usable).

    A freeze is declared when `freeze_run` or more consecutive words all have
    start times within `min_advance` seconds of each other.
    """
    if not words:
        return 0
    frozen_since = None
    for i in range(1, len(words)):
        if words[i]["start"] - words[i - 1]["start"] < min_advance:
            if frozen_since is None:
                frozen_since = i - 1
            if i - frozen_since >= freeze_run - 1:
                return frozen_since
        else:
            frozen_since = None
    return len(words)


def strip_hallucinations(text, words=None, min_phrase_words=4, min_repeats=3):
    """
    Detect and remove repetitive Whisper hallucination blocks from both the
    transcript text and word timestamps.

    For word timestamps, uses freeze-point detection rather than token
    re-matching: finds where timestamps stop advancing (the hallucination
    onset) and truncates there. This is robust against cases where the
    hallucination dominates the timestamp list and token matching fails.

    Returns (cleaned_text, cleaned_words, was_cleaned, removed_chars).
    """
    original_len = len(text)
    token_list   = text.split()

    if len(token_list) < min_phrase_words * min_repeats:
        return text, words, False, 0

    max_phrase = min(len(token_list) // min_repeats, 60)
    for phrase_len in range(max_phrase, min_phrase_words - 1, -1):
        i = 0
        while i <= len(token_list) - phrase_len:
            phrase = token_list[i:i + phrase_len]
            count  = 1
            j      = i + phrase_len
            while j + phrase_len <= len(token_list) and \
                    token_list[j:j + phrase_len] == phrase:
                count += 1
                j     += phrase_len
            if count >= min_repeats:
                del token_list[i + phrase_len:j]
            else:
                i += 1

    cleaned_text  = " ".join(token_list)
    was_cleaned   = cleaned_text != text
    removed_chars = original_len - len(cleaned_text)

    cleaned_words = words
    if words:
        # Truncate word timestamps at the freeze point — the moment Whisper's
        # alignment locks up, regardless of whether text stripping fired.
        freeze_idx = _find_timestamp_freeze_point(words)
        if freeze_idx < len(words):
            cleaned_words = words[:freeze_idx]
            # If text was also cleaned, was_cleaned is already True.
            # If only timestamps were frozen (text passed the phrase check),
            # flag it so the caller knows something was trimmed.
            was_cleaned   = True
            removed_chars = max(removed_chars, original_len - len(cleaned_text))
        else:
            cleaned_words = words

    return cleaned_text, cleaned_words, was_cleaned, removed_chars

# ---------------------------------------------------------------------------
# Per-station worker
# ---------------------------------------------------------------------------

def process_station(station_id, local_mp3_path, station_meta, dry_run=False):
    """
    Transcribe a single station recording using the Whisper API.
    Parse and trim happen after all stations complete.
    """
    log.info("[%s] Transcribing...", station_id)
    t0 = time.monotonic()

    if dry_run:
        log.info("[%s] DRY RUN - skipping", station_id)
        return {"station": station_id, "dry_run": True}

    from datetime import datetime as _dt

    client      = _get_openai_client()
    station_ctx = _STATIONS_DICT.get(station_id, {
        "id":       station_id,
        "location": station_meta.get("location", station_id),
        "type":     station_meta.get("type", "AWOS"),
    })

    with open(local_mp3_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            prompt=build_transcription_prompt(station_ctx),
            language="en",
            temperature=0.0,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    raw = result.text.strip()
    word_count = len(raw.split())

    # Always capture both timestamp types from the single API response
    raw_word_timestamps = [
        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
        for w in (result.words or [])
    ]
    raw_segment_timestamps = [
        {"text": s.text.strip(), "start": round(s.start, 3), "end": round(s.end, 3)}
        for s in (result.segments or [])
    ]

    words = raw_word_timestamps

    # --- Detect word-timestamp collapse and fall back to segment timestamps ---
    # Whisper's word-level alignment can collapse on repetitive audio, producing
    # only a handful of timestamps for hundreds of words. Segment timestamps are
    # Whisper's native unit and are significantly more reliable. Since we request
    # both granularities in the same call, no second API call is needed.
    timestamp_collapsed = len(words) < max(10, word_count * 0.10)
    if timestamp_collapsed:
        log.warning(
            "[%s] Word timestamp collapse detected (%d timestamps for %d words) "
            "— falling back to segment timestamps from same response",
            station_id, len(words), word_count
        )
        # Convert segment timestamps to word-compatible format.
        # Each token in the segment gets an evenly-distributed timestamp
        # within the segment's start/end window.
        words = []
        for seg in (result.segments or []):
            seg_words = seg.text.strip().split()
            if not seg_words:
                continue
            seg_start = round(seg.start, 3)
            seg_end   = round(seg.end,   3)
            duration  = seg_end - seg_start
            step      = duration / len(seg_words)
            for k, w in enumerate(seg_words):
                words.append({
                    "word":  w,
                    "start": round(seg_start + k * step, 3),
                    "end":   round(seg_start + (k + 1) * step, 3),
                })
        log.info(
            "[%s] Segment fallback: %d segments -> %d word-equivalent timestamps",
            station_id, len(result.segments or []), len(words)
        )

    log.info("[%s] Transcribed in %.1fs (%d words, %d timestamps%s)", station_id,
             time.monotonic() - t0, word_count, len(words),
             ", seg-fallback" if timestamp_collapsed else "")

    cleaned_raw, cleaned_words, was_cleaned, removed_chars = \
        strip_hallucinations(raw, words=words)
    if was_cleaned:
        log.warning("[%s] Hallucination stripped: %d chars removed (%d -> %d)",
                    station_id, removed_chars, len(raw), len(cleaned_raw))
    else:
        log.info("[%s] No hallucinations detected", station_id)

    return {
        "station":             station_id,
        "location":            station_meta.get("location", ""),
        "type":                station_meta.get("type", "AWOS"),
        "recording_sid":       station_meta.get("sid", ""),
        "call_sid":            station_meta.get("call_sid", ""),
        "duration_s":          station_meta.get("duration_s", 0),
        "date_created":        station_meta.get("date_created",
                                   _dt.now(timezone.utc).isoformat()),
        "raw_transcript":      raw,
        "cleaned_transcript":  cleaned_raw if was_cleaned else None,
        "hallucination_chars": removed_chars if was_cleaned else 0,
        "word_timestamps":          cleaned_words,
        "raw_word_timestamps":      raw_word_timestamps,
        "segment_timestamps":       raw_segment_timestamps,
        "timestamp_source":         "segment" if timestamp_collapsed else "word",
        "processed_at":             _dt.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Silence stripping
# ---------------------------------------------------------------------------

def _strip_trailing_silence(mp3_path, station_id, threshold_db=-40, min_silence_s=1.5):
    """
    Remove trailing silence from a downloaded MP3 before Whisper transcription.
    Writes to STRIPPED_DIR, leaving the original in RECORDINGS_DIR untouched.
    Returns the stripped file path on success, or original path on failure.
    """
    os.makedirs(STRIPPED_DIR, exist_ok=True)
    filename = os.path.basename(mp3_path)
    out_path = os.path.join(STRIPPED_DIR, filename)

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
                out_path,
            ],
            capture_output=True, text=True
        )
        if result.returncode == 0 and os.path.getsize(out_path) > 1024:
            orig_kb = os.path.getsize(mp3_path) / 1024
            new_kb  = os.path.getsize(out_path) / 1024
            log.info("[%s] Silence strip: %.0f KB -> %.0f KB -> %s",
                     station_id, orig_kb, new_kb, out_path)
            return out_path
        else:
            if os.path.exists(out_path):
                os.remove(out_path)
            if result.returncode != 0:
                log.warning("[%s] Silence strip failed (ffmpeg rc=%d): %s",
                            station_id, result.returncode, result.stderr[:200])
    except Exception as e:
        log.warning("[%s] Silence strip error: %s", station_id, e)
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass

    return mp3_path

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_recording(client, call_sid, station_meta):
    """
    Download the recording for a completed call.
    Looks up the RE... recording SID from the CA... call SID,
    downloads the MP3, then strips trailing silence.
    """
    import time as _time
    import requests
    from datetime import datetime as _dt

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    station_id = station_meta.get("id") or station_meta.get("station", "UNKN")

    recording_sid = None
    for attempt in range(6):
        recordings = client.recordings.list(call_sid=call_sid, limit=1)
        if recordings:
            recording_sid = recordings[0].sid
            break
        log.info("[%s] Recording not ready yet (attempt %d/6), waiting 5s...",
                 station_id, attempt + 1)
        _time.sleep(5)

    if not recording_sid:
        raise RuntimeError(f"No recording found for call {call_sid} after 30s")

    date_str   = _dt.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename   = f"{station_id}_{date_str}.mp3"
    local_path = os.path.join(RECORDINGS_DIR, filename)

    url  = (f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{client.account_sid}/Recordings/{recording_sid}.mp3")
    resp = requests.get(url, auth=(client.username, client.password), timeout=60)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)

    log.info("[%s] Downloaded %s (%.1f KB) [call=%s recording=%s]",
             station_id, filename, len(resp.content) / 1024,
             call_sid, recording_sid)

    return _strip_trailing_silence(local_path, station_id)

# ---------------------------------------------------------------------------
# Run directory initialisation
# ---------------------------------------------------------------------------

def _init_run_dir():
    """
    Create a timestamped directory under runs/ for this pipeline run.
    Updates TRANSCRIPTS_FILE and PARSED_FILE globals to point into it.
    Attaches a per-run log handler and updates the runs/latest symlink.
    Returns (run_dir, run_id).
    """
    global TRANSCRIPTS_FILE, PARSED_FILE

    run_id  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    TRANSCRIPTS_FILE = os.path.join(run_dir, "transcripts.json")
    PARSED_FILE      = os.path.join(run_dir, "parsed_results.json")

    run_handler = logging.FileHandler(
        os.path.join(run_dir, "pipeline.log"), encoding="utf-8")
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(run_handler)

    latest = os.path.join(RUNS_DIR, "latest")
    tmp    = latest + ".tmp"
    try:
        if os.path.islink(tmp):
            os.unlink(tmp)
        os.symlink(run_dir, tmp)
        os.replace(tmp, latest)
    except OSError as e:
        log.warning("Could not update runs/latest symlink: %s", e)

    log.info("Run directory: %s", run_dir)
    return run_dir, run_id

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(dry_run=False, workers=TRANSCRIBE_WORKERS, local_only=False):
    """
    Full pipelined run. Returns list of completed transcript records.
    """
    run_dir, run_id = _init_run_dir()

    client = None
    if not dry_run and not local_only:
        from twilio.rest import Client
        twilio_sid  = os.getenv("TWILIO_SID",  "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        twilio_auth = os.getenv("TWILIO_AUTH", "your_auth_token")
        if twilio_sid.startswith("AC" + "x"):
            log.error("TWILIO_SID not set.")
            sys.exit(1)
        client = Client(twilio_sid, twilio_auth)

    transcript_records = []
    failed_stations    = []
    lock               = threading.Lock()
    executor           = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="wx-worker")
    futures            = {}

    def on_station_complete(sid, station_id, details):
        if dry_run:
            log.info("[%s] DRY RUN - would download and process", station_id)
            return
        log.info("[%s] Call completed (%s) - downloading...",
                 station_id, details.get("final_status"))
        try:
            station_meta = pending_stations.get(station_id, {})
            station_meta["sid"]          = sid
            station_meta["date_created"] = details.get("start_time") or \
                                           datetime.now(timezone.utc).isoformat()
            local_path = download_recording(client, sid, station_meta)
            future = executor.submit(
                process_station, station_id, local_path, station_meta, dry_run
            )
            with lock:
                futures[future] = station_id
        except Exception as e:
            log.error("[%s] Download failed: %s", station_id, e)
            with lock:
                failed_stations.append(station_id)

    pending_stations = {s["id"]: s for s in _STATIONS_DICT.values()}

    # --- Phase 1: Place calls or load local recordings ---
    log.info("=" * 60)
    log.info("WxStream Pipeline  |  %s",
             "LOCAL MODE (no calls)" if local_only else "FULL MODE")
    log.info("=" * 60)
    log.info("Workers: %d concurrent transcriptions", workers)

    if dry_run:
        log.info("DRY RUN - simulating pipeline")
        for sid in ["RE_test1", "RE_test2"]:
            on_station_complete(sid, "KAIZ", {"final_status": "completed"})

    elif local_only:
        src_dir = (STRIPPED_DIR
                   if os.path.isdir(STRIPPED_DIR)
                   and any(f.lower().endswith(".mp3")
                           for f in os.listdir(STRIPPED_DIR))
                   else RECORDINGS_DIR)
        log.info("Local mode: reading recordings from %s", src_dir)

        # Build per-station file list, keyed by modification time so we can
        # select only files from the most recent recording session.
        # MP3 filenames are STATION_YYYYMMDD_HHMMSS.mp3 — use mtime as the
        # source of truth since filename timestamps reflect call time.
        by_station = defaultdict(list)  # station -> [(mtime, path)]
        for fname in os.listdir(src_dir):
            if fname.lower().endswith(".mp3"):
                full = os.path.join(src_dir, fname)
                mtime = os.path.getmtime(full)
                by_station[fname[:4].upper()].append((mtime, full))

        if not by_station:
            log.error("No MP3 files found in %s", src_dir)
            sys.exit(1)

        # Find the most recent file across all stations — that anchors the
        # current session. Only include files within 2 hours of that anchor
        # so stale recordings from previous days are excluded.
        all_mtimes = [mtime for paths in by_station.values() for mtime, _ in paths]
        latest_mtime = max(all_mtimes)
        SESSION_WINDOW_S = 2 * 3600  # 2 hours

        session_by_station = {}
        stale_stations = []
        for station_id, entries in by_station.items():
            # Among files within the session window, pick the most recent
            session_entries = [(mt, p) for mt, p in entries
                               if latest_mtime - mt <= SESSION_WINDOW_S]
            if session_entries:
                _, mp3_path = max(session_entries)
                session_by_station[station_id] = mp3_path
            else:
                stale_stations.append(station_id)

        if stale_stations:
            log.warning("Local mode: skipping %d station(s) with only stale "
                        "recordings (>2h old): %s",
                        len(stale_stations), ", ".join(sorted(stale_stations)))

        log.info("Found %d station(s) in current session: %s",
                 len(session_by_station), ", ".join(sorted(session_by_station)))

        for station_id, mp3_path in sorted(session_by_station.items()):
            station_meta = pending_stations.get(station_id, {
                "id": station_id, "location": station_id, "type": "AWOS",
            })
            station_meta["date_created"] = datetime.now(timezone.utc).isoformat()

            if src_dir == RECORDINGS_DIR:
                mp3_path = _strip_trailing_silence(mp3_path, station_id)

            log.info("[%s] Queuing local recording: %s",
                     station_id, os.path.basename(mp3_path))
            future = executor.submit(
                process_station, station_id, mp3_path, station_meta, dry_run
            )
            with lock:
                futures[future] = station_id

    else:
        from awos_test_run import run_test
        run_test(fetch=True, download=False,
                 per_station_callback=on_station_complete)

    log.info("All calls settled. Waiting for %d worker(s) to finish...",
             len(futures))

    # --- Phase 2: Collect worker results ---
    for future in as_completed(futures):
        station_id = futures[future]
        try:
            record = future.result()
            with lock:
                transcript_records.append(record)
            log.info("[%s] Worker complete", station_id)
        except Exception as e:
            log.error("[%s] Worker failed: %s", station_id, e)
            with lock:
                failed_stations.append(station_id)

    executor.shutdown(wait=False)

    # --- Phase 3: Save transcripts.json ---
    sorted_records = sorted(transcript_records, key=lambda r: r.get("station", ""))
    output = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "run_id":       run_id,
        "total":        len(sorted_records),
        "errors":       [{"station": s} for s in failed_stations],
        "timings":      [],
        "transcripts":  sorted_records,
    }
    if not dry_run:
        with open(TRANSCRIPTS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Transcripts saved -> %s", TRANSCRIPTS_FILE)

    # --- Phase 4: Parse all transcripts ---
    log.info("Parsing transcripts...")
    if not dry_run:
        r = subprocess.run(
            [sys.executable,
             os.path.join(SCRIPT_DIR, "parse_transcripts.py"),
             TRANSCRIPTS_FILE,
             "--output", PARSED_FILE],
            cwd=PROJECT_DIR
        )
        if r.returncode != 0:
            log.error("parse_transcripts.py failed")
            sys.exit(1)

    # --- Phase 4b: Trim audio ---
    log.info("Trimming recordings...")
    if not dry_run:
        r = subprocess.run(
            [sys.executable,
             os.path.join(SCRIPT_DIR, "audio_trim.py"),
             "--run-dir", run_dir],
            cwd=PROJECT_DIR
        )
        if r.returncode != 0:
            log.error("audio_trim.py failed")

    # --- Phase 5: Generate report ---
    log.info("Generating report...")
    if not dry_run:
        report_path = os.path.join(run_dir, "awos_report.html")
        r = subprocess.run(
            [sys.executable,
             os.path.join(SCRIPT_DIR, "generate_report.py"),
             "--input",  PARSED_FILE,
             "--output", report_path,
             "--audio",  TRIMMED_DIR,
             "--title",  "Missouri AWOS / ASOS Weather Observations"],
            cwd=PROJECT_DIR
        )
        if r.returncode != 0:
            log.error("generate_report.py failed")

    # --- Phase 6: Import to MongoDB Atlas ---
    atlas_script = os.path.join(SCRIPT_DIR, "atlas_import.py")
    if os.path.isfile(atlas_script) and not dry_run:
        mongo_pw = os.getenv("MONGO_DB_PASSWORD", "")
        if mongo_pw:
            log.info("Importing to MongoDB Atlas...")
            r = subprocess.run(
                [sys.executable, atlas_script,
                 "--transcripts", TRANSCRIPTS_FILE,
                 "--parsed",      PARSED_FILE],
                cwd=PROJECT_DIR,
                env={**os.environ},
            )
            if r.returncode != 0:
                log.error("atlas_import.py failed (rc=%d)", r.returncode)
            else:
                log.info("MongoDB Atlas import complete")
        else:
            log.info("Skipping Atlas import - MONGO_DB_PASSWORD not set")
    elif not os.path.isfile(atlas_script):
        log.debug("atlas_import.py not found - skipping MongoDB import")

    # --- Phase 7: Upload recordings to S3 ---
    s3_script = os.path.join(SCRIPT_DIR, "s3_upload.py")
    if os.path.isfile(s3_script) and not dry_run:
        s3_bucket = os.getenv("WXSTREAM_S3_BUCKET", "")
        if s3_bucket:
            log.info("Uploading recordings to S3...")
            r = subprocess.run(
                [sys.executable, s3_script],
                cwd=PROJECT_DIR,
                env={**os.environ},
            )
            if r.returncode != 0:
                log.error("s3_upload.py failed (rc=%d)", r.returncode)
            else:
                log.info("S3 upload complete")
        else:
            log.info("Skipping S3 upload - WXSTREAM_S3_BUCKET not set")
    elif not os.path.isfile(s3_script):
        log.debug("s3_upload.py not found - skipping S3 upload")

    log.info("=" * 60)
    log.info("Run directory: %s", run_dir)
    log.info("Pipeline complete | %d stations | %d failed",
             len(sorted_records), len(failed_stations))
    if failed_stations:
        log.warning("Failed: %s", ", ".join(failed_stations))
    log.info("=" * 60)

    return sorted_records

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduled(count, interval_minutes, dry_run=False, workers=TRANSCRIBE_WORKERS):
    log.info("=" * 60)
    log.info("WxStream Scheduler  |  %d runs  |  every %d min",
             count, interval_minutes)
    log.info("=" * 60)

    threads    = []
    interval_s = interval_minutes * 60

    def fire(run_num):
        log.info("[Scheduler] Run %d/%d starting", run_num, count)
        run_pipeline(dry_run=dry_run, workers=workers)
        log.info("[Scheduler] Run %d/%d complete", run_num, count)

    try:
        for run_num in range(1, count + 1):
            t = threading.Thread(target=fire, args=(run_num,), daemon=False)
            t.start()
            threads.append(t)
            if run_num < count:
                log.info("[Scheduler] Next run in %d min  (Ctrl+C to stop)",
                         interval_minutes)
                try:
                    time.sleep(interval_s)
                except KeyboardInterrupt:
                    log.info("[Scheduler] Interrupted - waiting for in-progress runs...")
                    break
    except KeyboardInterrupt:
        log.info("[Scheduler] Interrupted - waiting for in-progress runs...")

    for t in threads:
        t.join()
    log.info("[Scheduler] All scheduled runs complete")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WxStream pipelined orchestrator"
    )
    parser.add_argument("--workers", type=int, default=TRANSCRIBE_WORKERS,
                        help=f"Concurrent Whisper workers (default: {TRANSCRIBE_WORKERS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate pipeline without making calls or API requests")
    parser.add_argument("--local", action="store_true",
                        help="Skip Twilio calls; process recordings already in "
                             "stripped_recordings/ (or recordings/ if empty)")
    parser.add_argument("--schedule", metavar="N", type=int, default=None,
                        help="Run N times, once per --interval minutes")
    parser.add_argument("--interval", metavar="MINUTES", type=int, default=30,
                        help="Minutes between scheduled runs (default: 30)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.schedule:
        if args.local:
            log.warning("--local has no effect with --schedule")
        run_scheduled(
            count=args.schedule,
            interval_minutes=args.interval,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    else:
        run_pipeline(
            dry_run=args.dry_run,
            workers=args.workers,
            local_only=args.local,
        )
