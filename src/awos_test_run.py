"""
AWOS/ASOS One-Shot Test Run
Calls all stations once with max 5 concurrent calls.
Uses inline TwiML — no external server or TwiML Bin needed.
Recordings are stored automatically in your Twilio console.

Usage:
    export TWILIO_SID=ACxxxx
    export TWILIO_AUTH=your_auth_token
    export TWILIO_FROM=+15555550100
    python awos_test_run.py

Optional: test a subset of stations
    python awos_test_run.py --limit 5

Wait for final statuses and fetch recordings
    python awos_test_run.py --fetch --download

Retry stations whose final status was busy / no-answer
    python awos_test_run.py --fetch --retry-unanswered

After the run, find recordings at:
    console.twilio.com → Monitor → Calls → click any call → Recordings tab
"""

import json
import logging
import time
import re
import os
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.voice_response import VoiceResponse, Pause, Record
from fetch_recordings import fetch_recordings_for_sids

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TWILIO_SID    = os.getenv("TWILIO_SID",    "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
TWILIO_AUTH   = os.getenv("TWILIO_AUTH",   "your_auth_token")
TWILIO_FROM   = os.getenv("TWILIO_FROM",   "+15555550100")

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR  = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR   = os.path.join(_PROJECT_DIR, "output")

STATIONS_FILE = os.getenv("STATIONS_FILE", os.path.join(_PROJECT_DIR, "missouri_awos_asos_stations.json"))

MAX_CONCURRENT  = 5
CALL_TIMEOUT    = 30   # Seconds to wait for answer before hanging up
CALL_TIME_LIMIT = 120  # Max call duration in seconds
RETRY_DELAY     = 30   # Seconds before retrying a busy / no-answer result
RESULTS_FILE    = os.path.join(_OUTPUT_DIR, "logs", "test_run_results.json")

RETRYABLE_FINAL_STATUSES = {"busy", "no-answer"}

# ---------------------------------------------------------------------------
# Inline TwiML — no external server needed
# Twilio accepts TwiML XML directly via the twiml= parameter.
# This tells the call to: pause 2s (let AWOS start) → record up to 120s.
# ---------------------------------------------------------------------------

def build_twiml() -> str:
    response = VoiceResponse()
    response.append(Pause(length=2))
    response.append(Record(
        timeout=60,        # Stop recording after 60s of silence
        max_length=120,    # Hard cap at 120s
        play_beep=False,
    ))
    return str(response)

# ---------------------------------------------------------------------------
# Logging — INFO to console, full DEBUG to file
# ---------------------------------------------------------------------------

os.makedirs(os.path.join(_OUTPUT_DIR, "logs"), exist_ok=True)

log = logging.getLogger("awos_test")
log.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S"))

file_handler = logging.FileHandler(os.path.join(_OUTPUT_DIR, "logs", "awos_test_run.log"), encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

if not log.handlers:
    log.addHandler(console)
    log.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_phone(raw: str) -> str:
    """Convert (573) 348-0847 → +15733480847 for Twilio."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    raise ValueError(f"Cannot normalize: {raw!r}")


def load_stations(path: str, limit: int | None = None) -> tuple[list, list]:
    with open(path) as f:
        all_stations = json.load(f)

    valid, skipped = [], []
    for s in all_stations:
        if not s.get("phone"):
            skipped.append({"id": s["id"], "reason": "no phone number"})
            continue
        try:
            s["e164"] = normalize_phone(s["phone"])
            valid.append(s)
        except ValueError:
            skipped.append({"id": s["id"], "reason": f"unparseable number: {s['phone']}"})

    if limit:
        log.info("--limit %d: testing first %d of %d callable stations", limit, limit, len(valid))
        valid = valid[:limit]

    return valid, skipped


def summarize_results(results: list[dict]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for r in results:
        key = r.get("final_status") or r.get("status") or "unknown"
        summary[key] = summary.get(key, 0) + 1
    return summary


def persist_report(
    *,
    started_at: datetime,
    elapsed_s: float,
    total: int,
    skipped: list,
    results: list[dict],
    results_file: str = RESULTS_FILE,
) -> None:
    report = {
        "run_at": started_at.isoformat(),
        "elapsed_s": round(elapsed_s, 1),
        "total": total,
        "skipped": skipped,
        "summary": summarize_results(results),
        "results": results,
    }
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Full report saved -> %s", results_file)

# ---------------------------------------------------------------------------
# Single call
# ---------------------------------------------------------------------------

def place_call(client: Client, station: dict, semaphore: Semaphore, index: int, total: int, attempt: int = 1) -> dict:
    station_id = station["id"]
    result = {
        "station": station_id,
        "location": station["location"],
        "phone": station["phone"],
        "type": station["type"],
        "attempt": attempt,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    with semaphore:
        log.info("[%d/%d] >> %s  %s  %s (attempt %d)", index, total, station_id, station["location"], station["e164"], attempt)
        t0 = time.monotonic()
        try:
            call = client.calls.create(
                to=station["e164"],
                from_=TWILIO_FROM,
                twiml=build_twiml(),   # Inline XML — no server needed
                timeout=CALL_TIMEOUT,
                time_limit=CALL_TIME_LIMIT,
            )
            elapsed = time.monotonic() - t0
            result.update({
                "status": "placed",
                "sid": call.sid,
                "api_ms": round(elapsed * 1000),
                "created_status": getattr(call, "status", None),
                "queue_time": getattr(call, "queue_time", None),
            })
            log.info("[%d/%d] OK %s - SID %s (%.0fms)", index, total, station_id, call.sid, elapsed * 1000)

        except TwilioRestException as e:
            elapsed = time.monotonic() - t0
            if e.code in (13225, 13226):
                # Keeping legacy handling if Twilio rejects immediately, but in most
                # cases final busy / no-answer will arrive later via call fetch.
                result.update({
                    "status": "busy",
                    "final_status": "busy",
                    "twilio_code": e.code,
                    "error": e.msg,
                    "api_ms": round(elapsed * 1000),
                })
                log.warning("[%d/%d] BUSY %s - busy/no answer (code %s)", index, total, station_id, e.code)
            elif e.code == 21215:
                result.update({
                    "status": "invalid_number",
                    "final_status": "invalid_number",
                    "twilio_code": e.code,
                    "error": e.msg,
                    "api_ms": round(elapsed * 1000),
                })
                log.error("[%d/%d] FAIL %s - invalid number %s", index, total, station_id, station["phone"])
            else:
                result.update({
                    "status": "error",
                    "final_status": "error",
                    "twilio_code": e.code,
                    "error": e.msg,
                    "api_ms": round(elapsed * 1000),
                })
                log.error("[%d/%d] FAIL %s - Twilio error %s: %s", index, total, station_id, e.code, e.msg)

        except Exception as e:
            result.update({"status": "error", "final_status": "error", "error": str(e)})
            log.error("[%d/%d] FAIL %s - %s", index, total, station_id, e)

    return result


def fetch_call_details(client: Client, sid: str) -> dict:
    """
    Pull a fuller final snapshot from Twilio so the JSON report explains
    why a call did not complete instead of leaving it as unknown.
    """
    call = client.calls(sid).fetch()
    details = {
        "final_status": getattr(call, "status", None),
        "answered_by": getattr(call, "answered_by", None),
        "duration": getattr(call, "duration", None),
        "start_time": call.start_time.isoformat() if getattr(call, "start_time", None) else None,
        "end_time": call.end_time.isoformat() if getattr(call, "end_time", None) else None,
        "price": getattr(call, "price", None),
        "price_unit": getattr(call, "price_unit", None),
        "direction": getattr(call, "direction", None),
    }

    error_code = getattr(call, "error_code", None)
    error_message = getattr(call, "error_message", None)
    if error_code is not None:
        details["error_code"] = error_code
    if error_message:
        details["error_message"] = error_message

    # Human-readable explanation for common non-completed statuses.
    final_status = details.get("final_status")
    if final_status == "busy":
        details["outcome_detail"] = "Destination line reported busy."
    elif final_status == "no-answer":
        details["outcome_detail"] = f"No answer before Twilio timeout ({CALL_TIMEOUT}s)."
    elif final_status == "failed":
        details["outcome_detail"] = error_message or "Call failed before completion."
    elif final_status == "canceled":
        details["outcome_detail"] = "Call was canceled before completion."
    elif final_status == "completed":
        details["outcome_detail"] = "Call connected and reached a completed state."

    return details


def retry_station_calls(
    client: Client,
    stations_by_id: dict[str, dict],
    results: list[dict],
    semaphore: Semaphore,
) -> list[dict]:
    retry_candidates = [
        r for r in results
        if (r.get("final_status") or r.get("status")) in RETRYABLE_FINAL_STATUSES
    ]
    if not retry_candidates:
        log.info("No busy / no-answer calls to retry.")
        return []

    log.info("-" * 60)
    log.info("Retrying %d busy / no-answer station(s)...", len(retry_candidates))
    log.info("-" * 60)

    if RETRY_DELAY > 0:
        log.info("Waiting %ds before retry batch...", RETRY_DELAY)
        time.sleep(RETRY_DELAY)

    total = len(retry_candidates)
    retry_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=min(total, MAX_CONCURRENT)) as executor:
        futures = {
            executor.submit(
                place_call,
                client,
                stations_by_id[r["station"]],
                semaphore,
                i + 1,
                total,
                r.get("attempt", 1) + 1,
            ): r
            for i, r in enumerate(retry_candidates)
        }
        for future in as_completed(futures):
            retry_result = future.result()
            previous = futures[future]
            retry_result["retry_of_sid"] = previous.get("sid")
            retry_result["retry_reason"] = previous.get("final_status") or previous.get("status")
            retry_results.append(retry_result)

    return retry_results

# ---------------------------------------------------------------------------
# Wait for calls to complete
# ---------------------------------------------------------------------------

def wait_for_calls(
    client: Client,
    results: list,
    poll_interval: int = 10,
    timeout: int = 300,
    per_station_callback=None,
) -> dict:
    """
    Poll Twilio until all placed calls reach a terminal state.
    Returns a dict of call_sid -> detailed final outcome.
    """
    terminal = {"completed", "failed", "busy", "no-answer", "canceled"}

    # Only wait on calls that were successfully placed
    pending = {r["sid"]: r["station"] for r in results if r.get("status") == "placed" and r.get("sid")}
    final: dict[str, dict] = {}
    elapsed = 0

    log.info("-" * 60)
    log.info("Waiting for %d calls to complete...", len(pending))

    while pending and elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        for sid in list(pending):
            try:
                call = client.calls(sid).fetch()
                if call.status in terminal:
                    station = pending.pop(sid)
                    details = fetch_call_details(client, sid)
                    final[sid] = details
                    log.info("  %s (%s) -> %s", station, sid, details.get("final_status"))
                    if per_station_callback and details.get("final_status") == "completed":
                        try:
                            per_station_callback(sid, station, details)
                        except Exception as _cb_err:
                            log.warning("  per_station_callback error for %s: %s", station, _cb_err)
            except Exception as e:
                log.warning("  Could not fetch status for %s: %s", sid, e)

        if pending:
            log.info("  Still active: %d calls (%ds elapsed)", len(pending), elapsed)

    # Anything still pending after timeout is marked as timeout instead of unknown
    for sid, station in pending.items():
        log.warning("  %s timed out waiting for completion", station)
        final[sid] = {
            "final_status": "timeout",
            "outcome_detail": f"Did not reach a terminal Twilio status within {timeout}s of polling.",
        }

    completed = sum(1 for d in final.values() if d.get("final_status") == "completed")
    log.info("All calls settled - %d completed, %d other", completed, len(final) - completed)
    log.info("-" * 60)
    return final

# ---------------------------------------------------------------------------
# Main test run
# ---------------------------------------------------------------------------

def run_test(
    limit: int | None = None,
    fetch: bool = False,
    download: bool = False,
    retry_unanswered: bool = False,
    per_station_callback=None,
    client: "Client | None" = None,
) -> None:
    log.info("=" * 60)
    log.info("AWOS/ASOS ONE-SHOT TEST RUN")
    log.info("=" * 60)

    if client is None:
        # Validate credentials aren't placeholders
        if TWILIO_SID.startswith("AC" + "x"):
            log.error("TWILIO_SID not set. Export your credentials and re-run.")
            return
        client = Client(TWILIO_SID, TWILIO_AUTH)
    stations, skipped = load_stations(STATIONS_FILE, limit)
    stations_by_id = {s["id"]: s for s in stations}
    total = len(stations)

    if not stations:
        log.error("No callable stations found.")
        return

    log.info("Stations to call : %d", total)
    log.info("Skipped          : %d", len(skipped))
    log.info("Max concurrent   : %d", MAX_CONCURRENT)
    log.info("Estimated time   : ~%.0f minutes", (total / MAX_CONCURRENT * 90) / 60)
    log.info("-" * 60)

    semaphore = Semaphore(MAX_CONCURRENT)
    results: list[dict] = []
    start = datetime.now(timezone.utc)

    with ThreadPoolExecutor(max_workers=total) as executor:
        futures = {
            executor.submit(place_call, client, s, semaphore, i + 1, total, 1): s
            for i, s in enumerate(stations)
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results back into station / attempt order for the report
    order = {s["id"]: i for i, s in enumerate(stations)}
    results.sort(key=lambda r: (order.get(r["station"], 999), r.get("attempt", 1)))

    # Optionally wait for calls to complete and annotate the report with final outcome.
    if fetch:
        final_outcomes = wait_for_calls(client, results, per_station_callback=per_station_callback)
        for result in results:
            sid = result.get("sid")
            if sid and sid in final_outcomes:
                result.update(final_outcomes[sid])

        if retry_unanswered:
            retry_results = retry_station_calls(client, stations_by_id, results, semaphore)
            results.extend(retry_results)

            if retry_results:
                retry_outcomes = wait_for_calls(client, retry_results)
                for result in retry_results:
                    sid = result.get("sid")
                    if sid and sid in retry_outcomes:
                        result.update(retry_outcomes[sid])

        # Build sid_map from all completed placed attempts for fetch_recordings_for_sids
        sid_map = {
            r["sid"]: {
                "station": r["station"],
                "location": r["location"],
                "phone": r["phone"],
                "type": r["type"],
                "attempt": r.get("attempt", 1),
                "final_status": r.get("final_status"),
            }
            for r in results
            if r.get("status") == "placed" and r.get("sid")
        }
        fetch_recordings_for_sids(client, sid_map, download=download)
    elif retry_unanswered:
        log.warning("--retry-unanswered is most useful together with --fetch; without --fetch there is no final busy/no-answer status to act on.")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    results.sort(key=lambda r: (order.get(r["station"], 999), r.get("attempt", 1)))

    # Summary
    summary = summarize_results(results)

    log.info("=" * 60)
    log.info("TEST RUN COMPLETE - %.0fs elapsed", elapsed)
    log.info("=" * 60)
    for status, count in sorted(summary.items()):
        icon = {
            "completed": "OK",
            "placed": "OK",
            "busy": "BUSY",
            "no-answer": "BUSY",
            "timeout": "WARN",
            "invalid_number": "FAIL",
            "failed": "FAIL",
            "error": "FAIL",
            "canceled": "WARN",
        }.get(status, "?")
        log.info("  %s %-16s %d call(s)", icon, status, count)
    log.info("-" * 60)

    persist_report(
        started_at=start,
        elapsed_s=elapsed,
        total=total,
        skipped=skipped,
        results=results,
    )
    log.info("Full log saved    -> %s", os.path.join(_OUTPUT_DIR, "logs", "awos_test_run.log"))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot AWOS test run")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only call the first N stations (useful for a quick smoke test)"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Wait for all calls to complete, annotate final statuses, then auto-fetch recordings"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download MP3 audio files when fetching (requires --fetch)"
    )
    parser.add_argument(
        "--retry-unanswered", action="store_true",
        help="After final statuses are known, retry stations whose final status was busy or no-answer and append retry attempts to the JSON report"
    )
    args = parser.parse_args()
    run_test(
        limit=args.limit,
        fetch=args.fetch,
        download=args.download,
        retry_unanswered=args.retry_unanswered,
    )
