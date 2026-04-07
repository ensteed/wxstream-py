"""
AWOS Transcription and Cleanup
Downloads recordings from Twilio, transcribes with Whisper,
and extracts a single clean AWOS/ASOS report from each.

Requires:
    pip install openai requests
    recordings_log.json  (from fetch_recordings.py)

Usage:
    export OPENAI_API_KEY=sk-xxxx
    export TWILIO_SID=ACxxxx
    export TWILIO_AUTH=your_auth_token

    # Transcribe all recordings
    python transcribe.py

    # Transcribe a specific station only
    python transcribe.py --station KSTL

    # Skip already-transcribed stations (resume a partial run)
    python transcribe.py --resume
"""

import json
import os
import tempfile
import argparse
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-xxxx")
TWILIO_SID      = os.getenv("TWILIO_SID",     "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
TWILIO_AUTH     = os.getenv("TWILIO_AUTH",     "your_auth_token")

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR    = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR     = os.path.join(_PROJECT_DIR, "output")

RECORDINGS_LOG  = os.getenv("RECORDINGS_LOG", os.path.join(_OUTPUT_DIR, "logs", "recordings_log.json"))
STATIONS_FILE   = os.getenv("STATIONS_FILE",  os.path.join(_PROJECT_DIR, "missouri_awos_asos_stations.json"))
OUTPUT_FILE     = os.getenv("OUTPUT_FILE",    os.path.join(_OUTPUT_DIR, "transcripts.json"))

MIN_DURATION_S  = 10   # Skip recordings shorter than this
MAX_WORKERS     = 10    # Concurrent API requests - raise if you hit rate limits rarely

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_recordings(path: str) -> list:
    try:
        with open(path) as f:
            return json.load(f).get("recordings", [])
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run fetch_recordings.py first.")
        exit(1)


def load_stations(path: str) -> dict:
    """Load station metadata keyed by ID."""
    try:
        with open(path) as f:
            return {s["id"]: s for s in json.load(f)}
    except FileNotFoundError:
        print(f"WARNING: {path} not found - prompts will use generic context only")
        return {}


def build_transcription_prompt(station: dict) -> str:
    """
    Build a station-specific prompt that primes the model with the exact
    identifier and location name it should expect to hear in the audio.
    The prompt is written to look like the start of an AWOS broadcast so
    the model treats it as preceding context, not instructions.
    """
    station_id = station.get("id", "")
    location   = station.get("location", "")
    stn_type   = station.get("type", "AWOS")

    return (
        # Station identity - tells the model exactly what identifier to expect
        f"{station_id} {location} automated weather observation. "

        # Canonical AWOS phrasing patterns
        "One four three niner zulu."
        "Wind: two seven zero at one five knots. Peak Gusts two three"
        "Visibility: one zero. Visibility: two and one half."
        "Haze. Light Rain."
        "Sky condition: clear below one two thousand."
        "Sky condition: scattered niner hundred. Broken one thousand niner hundred. Overcast one one thousand." 
        "Temperature: two two celcius. Dewpoint: one eight celcius."
        "Altimeter: two niner niner two inches of mercury."

        # Spoken numbers that commonly get mangled
        "Niner, two, niner, three, zero, one, zero, two, four, zero."

        # Words that sound like other words to the model
        "Knots, ceiling, dewpoint, altimeter, overcast, "
        "broken, scattered, few, clear, calm, visibility, variable. "

        # Sky condition codes
        

        # Remarks phrasing
        "Remarks: Density Altitude one thousand one hundred."
        "Thunderstorm information not available." 
    )


def load_existing_transcripts(path: str) -> dict:
    """Load previously transcribed stations for --resume mode."""
    try:
        with open(path) as f:
            data = json.load(f)
        existing = {t["station"]: t for t in data.get("transcripts", [])}
        print(f"Loaded {len(existing)} existing transcripts from {path}")
        return existing
    except FileNotFoundError:
        return {}

# ---------------------------------------------------------------------------
# Audio retrieval - local file first, Twilio fallback
# ---------------------------------------------------------------------------

def get_audio(rec: dict, use_local: bool) -> tuple[bytes, str]:
    """
    Return (audio_bytes, source) where source is 'local' or 'twilio'.
    If use_local=True and a local_file path exists in the recording log,
    read from disk instead of downloading from Twilio.
    """
    local_path = rec.get("local_file")

    if use_local and local_path and os.path.exists(local_path):
        with open(local_path, "rb") as f:
            return f.read(), "local"

    if use_local and local_path and not os.path.exists(local_path):
        print(f"  WARNING: local file not found at {local_path} - falling back to Twilio")

    # Fall back to downloading from Twilio
    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{TWILIO_SID}/Recordings/{rec['recording_sid']}.mp3"
    )
    resp = requests.get(url, auth=(TWILIO_SID, TWILIO_AUTH), timeout=30)
    resp.raise_for_status()
    return resp.content, "twilio"

# ---------------------------------------------------------------------------
# Transcribe
# ---------------------------------------------------------------------------

def transcribe(client: OpenAI, audio_bytes: bytes, station: dict) -> dict:
    """
    Transcribe audio using whisper-1 with word-level timestamps.
    Returns:
        text            - full raw transcript string
        word_timestamps - list of {word, start, end} in seconds
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=build_transcription_prompt(station),
                language="en",
                temperature=0.0,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        words = [
            {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
            for w in (result.words or [])
        ]
        return {"text": result.text, "word_timestamps": words}
    finally:
        os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Extract single clean report
# ---------------------------------------------------------------------------

def extract_single_report(client: OpenAI, raw_transcript: str, station: dict) -> str:
    """
    AWOS/ASOS broadcasts loop continuously. A 90s recording often contains
    1-2 full readings plus a partial. Extract one clean complete report
    formatted in standard AWOS broadcast order.
    """
    station_id = station.get("id", "")
    location   = station.get("location", "")
    stn_type   = station.get("type", "AWOS")

    prompt = (
        f"The following is a raw transcript of an automated {stn_type} weather "
        f"broadcast from station {station_id} ({location}). "
        f"The broadcast loops continuously so the transcript may contain "
        f"partial or multiple readings.\n\n"

        f"Extract exactly ONE complete reading and format it in this order:\n\n"

        f"1. Station name: use '{location}' with the addition of airport or regional airport or regional business airport according to the raw transcript  \n"
        f"2. automated weather observation\n"
        f"3. Zulu time (e.g. 'one four five five zulu')\n"
        f"4. Wind (e.g. 'wind two seven zero at one five knots') this field may also contain variability in direction (eg variable one five zero to two zero zero) or peak gusts (eg peak gusts two six)\n"
        f"5. Visibility (e.g. 'visibility one zero')\n"
        f"6. Sky condition (e.g. 'sky condition ceiling one two thousand broken')\n"
        f"7. Temperature and dewpoint (e.g. 'temperature two two celcius, dewpoint one eight')\n"
        f"8. Altimeter (e.g. 'altimeter two niner niner two'), sometimes this may be followed by the phrase inches of mercury. If multiple altimeter readings are in the transcript use the last one recorded. The extracted altimeter reading should directly match the raw transcript. Do not change any numbers. \n"
        f"9. Remarks (e.g. 'remarks, density altitude one five hundred, freezing rain, lightning observed')\n\n"
        f"10. Additional Info - automated ASOS stations may have additional broadcast after the automated weather observation which gives local airport information (hours of operation, local frequency informaition, etc). Do not omit this. Take this information directly from the raw transcript on a new line after any remarks. \n\n"


        f"Rules:\n"
        f"- Output each element on its own line\n"
        f"- If a field is missing from the transcript, omit it entirely - do not guess\n"
        f"- NEVER invent or add any content that does not appear in the raw transcript\n"
        f"- Remarks must only contain what is literally spoken in the raw transcript - "
        f"do not add 'thunderstorm began', 'lightning observed', 'freezing fog', or any "
        f"other weather event unless those exact words appear in the raw transcript\n"
        f"10. automated ASOS stations may have an ATIS broadcast after the automated weather observation remarks which gives local airport information. Do not omit this information and provide all of it on a new line after the remarks.\n"
        f"- Fix transcription errors: 'sealing' -> 'ceiling', 'knows'/'nots' -> 'knots', "
        f"'to niner' -> 'two niner', 'all timer' -> 'altimeter', 'due point' -> 'dewpoint'\n"
        f"- Output only the formatted report, nothing else\n\n"
        
        f"NUMBER RULES - critical, follow exactly:\n"
        f"- ALWAYS preserve the exact digits spoken in the raw transcript - never substitute or guess\n"
        f"- Convert numeric digits to spoken words: '0' -> 'zero', '1' -> 'one', '2' -> 'two', "
        f"'3' -> 'three', '4' -> 'four', '5' -> 'five', '6' -> 'six', '7' -> 'seven', "
        f"'8' -> 'eight', '9' -> 'niner'\n"
        f" multidigit numbers should be converted to individual digits:'150' ->  'one five zero'\n"
        f"- Hyphenated digits like '3-0-0-2' mean each digit is spoken separately: "
        f"'3-0-0-2' -> 'three zero zero two'\n"
        f"- Only use 'niner' where the digit 9 actually appears in the raw transcript\n"
        f"- Never replace a digit with 'niner' unless the source is 9\n"
        f"- Example: raw '3-0-0-2' -> 'three zero zero two' NOT 'three niner niner two'\n"

        f"Example output:\n"
        f"Cape Girardeau Regional Airport automated weather observation.\n"
        f"One four five five zulu.\n"
        f"Wind two seven zero at one five.\n"
        f"Visibility one zero.\n"
        f"Sky condition ceiling one two thousand broken.\n"
        f"Temperature two two celcius, dewpoint one eight celcius.\n"
        f"Altimeter two niner niner two.\n"
        f"Remarks, density altitude one thousand five hundred,  thunderstorm information not available.\n\n"

        f"Raw transcript:\n{raw_transcript}"
    )

    response = client.chat.completions.create(
        model="gpt-5.4-nano",  # Cheapest option - plenty for simple text cleanup
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(station_filter: str | None = None, resume: bool = False, use_local: bool = False) -> None:
    if OPENAI_API_KEY.startswith("sk-x"):
        print("ERROR: OPENAI_API_KEY not set.")
        return
    if not use_local and TWILIO_SID.startswith("AC" + "x"):
        print("ERROR: TWILIO_SID not set. Use --local if you have downloaded recordings.")
        return

    client     = OpenAI(api_key=OPENAI_API_KEY)
    recordings = load_recordings(RECORDINGS_LOG)
    stations   = load_stations(STATIONS_FILE)
    existing   = load_existing_transcripts(OUTPUT_FILE) if resume else {}

    # Filter to processable recordings
    to_process = [
        r for r in recordings
        if r.get("status") == "completed"
        and r.get("station") not in (None, "UNKNOWN")
        and r.get("recording_sid")
        and (r.get("duration_s") or 0) >= MIN_DURATION_S
    ]

    # Apply optional filters
    if station_filter:
        to_process = [r for r in to_process if r["station"] == station_filter.upper()]
        if not to_process:
            print(f"No recordings found for station {station_filter.upper()}")
            return

    if resume:
        skipped    = [r for r in to_process if r["station"] in existing]
        to_process = [r for r in to_process if r["station"] not in existing]
        if skipped:
            print(f"Skipping {len(skipped)} already transcribed: {', '.join(r['station'] for r in skipped)}")

    local_count  = sum(1 for r in to_process if r.get("local_file") and os.path.exists(r.get("local_file", "")))
    if use_local:
        print(f"  Local files available: {local_count}/{len(to_process)} - remainder will download from Twilio")
    print(f"Transcribing {len(to_process)} recording(s) with {MAX_WORKERS} concurrent workers...")
    print("=" * 60)

    transcripts = dict(existing)
    errors      = []
    timings     = []
    print_lock  = Lock()
    result_lock = Lock()
    run_start   = time.monotonic()

    def process_one(args):
        i, rec = args
        station      = rec["station"]
        dur          = rec.get("duration_fmt", f"{rec.get('duration_s', '?')}s")
        station_meta = stations.get(station, {"id": station, "location": rec["location"], "type": rec["type"]})

        try:
            t0 = time.monotonic()
            audio, source = get_audio(rec, use_local)
            t_download = time.monotonic() - t0

            t0 = time.monotonic()
            result          = transcribe(client, audio, station_meta)
            raw             = result["text"]
            word_timestamps = result.get("word_timestamps", [])
            t_transcribe    = time.monotonic() - t0

            t0 = time.monotonic()
            clean = extract_single_report(client, raw, station_meta)
            t_cleanup = time.monotonic() - t0

            t_total = t_download + t_transcribe + t_cleanup

            with print_lock:
                print(f"[{i}/{len(to_process)}] {station} ({dur}) [{source}]")
                print(f"  download {t_download:.1f}s  |  transcribe {t_transcribe:.1f}s  |  cleanup {t_cleanup:.1f}s  |  total {t_total:.1f}s")

            return {
                "ok": True,
                "station": station,
                "timing": {
                    "station":      station,
                    "download_s":   round(t_download,   2),
                    "transcribe_s": round(t_transcribe, 2),
                    "cleanup_s":    round(t_cleanup,    2),
                    "total_s":      round(t_total,      2),
                },
                "transcript": {
                    "station":        station,
                    "location":       rec["location"],
                    "type":           rec["type"],
                    "recording_sid":  rec["recording_sid"],
                    "call_sid":       rec.get("call_sid", ""),
                    "duration_s":     rec["duration_s"],
                    "date_created":   rec["date_created"],
                    "raw_transcript":   raw,
                    "word_timestamps":  word_timestamps,
                    "clean_report":     clean,
                    "processed_at":     datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            with print_lock:
                print(f"[{i}/{len(to_process)}] {station} FAIL - {e}")
            return {"ok": False, "station": station, "error": str(e)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, (i + 1, rec)): rec for i, rec in enumerate(to_process)}
        for future in as_completed(futures):
            result = future.result()
            with result_lock:
                if result["ok"]:
                    transcripts[result["station"]] = result["transcript"]
                    timings.append(result["timing"])
                else:
                    errors.append({"station": result["station"], "error": result["error"]})

    # Sort by station ID and save
    sorted_transcripts = sorted(transcripts.values(), key=lambda x: x["station"])

    output = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total":        len(sorted_transcripts),
        "errors":       errors,
        "timings":      timings,
        "transcripts":  sorted_transcripts,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Timing summary
    run_total = time.monotonic() - run_start
    print("=" * 60)
    print(f"Done  - {len(to_process) - len(errors)} transcribed, {len(errors)} errors")
    print(f"Total wall time: {run_total:.1f}s")
    if timings:
        avg_dl  = sum(t["download_s"]   for t in timings) / len(timings)
        avg_tr  = sum(t["transcribe_s"] for t in timings) / len(timings)
        avg_cl  = sum(t["cleanup_s"]    for t in timings) / len(timings)
        slow_tr = max(timings, key=lambda t: t["transcribe_s"])
        slow_cl = max(timings, key=lambda t: t["cleanup_s"])
        print(f"")
        print(f"Average per station:")
        print(f"  Download    {avg_dl:.1f}s")
        print(f"  Transcribe  {avg_tr:.1f}s")
        print(f"  Cleanup     {avg_cl:.1f}s")
        print(f"  Total       {avg_dl + avg_tr + avg_cl:.1f}s")
        print(f"")
        print(f"Slowest transcription : {slow_tr['station']} ({slow_tr['transcribe_s']}s)")
        print(f"Slowest cleanup       : {slow_cl['station']} ({slow_cl['cleanup_s']}s)")
    if errors:
        print(f"Errors: {', '.join(e['station'] for e in errors)}")
    print(f"Saved -> {OUTPUT_FILE}")

    # Print sample of first new transcript
    new = [t for t in sorted_transcripts if t["station"] not in existing]
    if new:
        sample = new[0]
        print(f"\nSample - {sample['station']} ({sample['location']}):")
        print("-" * 60)
        print(sample["clean_report"])
        print("-" * 60)


def reclean(client: OpenAI, stations: dict, station_filter: str | None = None) -> None:
    """
    Re-run the cleanup step on existing raw transcripts in transcripts.json.
    Does not download or re-transcribe audio - only rewrites clean_report.
    """
    try:
        with open(OUTPUT_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {OUTPUT_FILE} not found. Run transcribe first.")
        return

    existing = data.get("transcripts", [])
    if not existing:
        print("No existing transcripts found.")
        return

    to_reclean = existing
    if station_filter:
        to_reclean = [t for t in existing if t["station"] == station_filter.upper()]
        if not to_reclean:
            print(f"No transcript found for {station_filter.upper()}")
            return

    print(f"Re-cleaning {len(to_reclean)} transcript(s) with {MAX_WORKERS} concurrent workers...")
    print("=" * 60)

    errors     = []
    print_lock = Lock()
    result_lock = Lock()
    updated    = {t["station"]: t for t in existing}
    run_start  = time.monotonic()

    def reclean_one(args):
        i, transcript = args
        station      = transcript["station"]
        raw          = transcript.get("raw_transcript", "")
        station_meta = stations.get(station, {
            "id":       station,
            "location": transcript.get("location", station),
            "type":     transcript.get("type", "AWOS"),
        })

        if not raw:
            with print_lock:
                print(f"[{i}/{len(to_reclean)}] {station} SKIP - no raw transcript")
            return {"ok": False, "station": station, "error": "no raw transcript"}

        try:
            t0    = time.monotonic()
            clean = extract_single_report(client, raw, station_meta)
            elapsed = time.monotonic() - t0

            with print_lock:
                print(f"[{i}/{len(to_reclean)}] {station} OK ({elapsed:.1f}s)")

            return {"ok": True, "station": station, "clean": clean}

        except Exception as e:
            with print_lock:
                print(f"[{i}/{len(to_reclean)}] {station} FAIL - {e}")
            return {"ok": False, "station": station, "error": str(e)}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(reclean_one, (i + 1, t)): t
            for i, t in enumerate(to_reclean)
        }
        for future in as_completed(futures):
            result = future.result()
            with result_lock:
                if result["ok"]:
                    updated[result["station"]]["clean_report"] = result["clean"]
                    updated[result["station"]]["recleaned_at"] = datetime.now(timezone.utc).isoformat()
                else:
                    errors.append({"station": result["station"], "error": result.get("error")})

    elapsed = time.monotonic() - run_start
    sorted_transcripts = sorted(updated.values(), key=lambda x: x["station"])

    data["transcripts"]  = sorted_transcripts
    data["recleaned_at"] = datetime.now(timezone.utc).isoformat()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print("=" * 60)
    success = len(to_reclean) - len(errors)
    print(f"Done - {success} re-cleaned, {len(errors)} errors ({elapsed:.1f}s)")
    if errors:
        print(f"Errors: {', '.join(e['station'] for e in errors)}")
    print(f"Saved -> {OUTPUT_FILE}")

    if sorted_transcripts:
        sample = next((t for t in sorted_transcripts if t["station"] == to_reclean[0]["station"]), None)
        if sample:
            print(f"\nSample - {sample['station']} ({sample.get('location', '')}):")
            print("-" * 60)
            print(sample["clean_report"])
            print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe AWOS recordings")
    parser.add_argument("--station", type=str, default=None,
                        help="Only process a specific station (e.g. KSTL)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip stations already in transcripts.json")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent API workers (default: {MAX_WORKERS})")
    parser.add_argument("--local",   action="store_true",
                        help="Use pre-downloaded MP3s instead of re-downloading from Twilio")
    parser.add_argument("--reclean", action="store_true",
                        help="Re-run cleanup on existing raw transcripts without re-transcribing")
    args = parser.parse_args()
    MAX_WORKERS = args.workers

    if args.reclean:
        client   = OpenAI(api_key=OPENAI_API_KEY)
        stations = load_stations(STATIONS_FILE)
        reclean(client, stations, station_filter=args.station)
    else:
        main(station_filter=args.station, resume=args.resume, use_local=args.local)
