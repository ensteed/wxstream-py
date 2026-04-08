"""
audio_trim.py
-------------
Trims raw looped AWOS/ASOS station recordings to a single clean broadcast
using word-level timestamps from transcripts.json.

PRIMARY METHOD (timestamp-based):
  Reads word_timestamps from transcripts.json, finds all "automated weather
  observation" anchors, selects the best complete loop (closest to median
  duration), locates the station name start via silence gaps, and cuts
  precisely with ffmpeg.

FALLBACK METHOD (energy-based):
  Used when no timestamps exist for a station. Detects loop boundaries via
  RMS silence analysis.

Usage:
    python audio_trim.py

Expects:
    stripped_recordings/         <- silence-stripped MP3s (preferred)
    recordings/                  <- raw MP3s (fallback if stripped/ empty)
    transcripts.json             <- with word_timestamps from transcribe.py
    missouri_awos_asos_stations.json

Produces:
    trimmed_recordings/          <- one trimmed MP3 per station

Requirements: Python 3.6+, numpy, ffmpeg in PATH.
"""

import os
import json
import subprocess
import statistics
import sys
import numpy as np

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
# Support scripts living in a subdirectory (e.g. src/) by searching
# upward for the stations JSON and using that directory as the project root.
_PROJECT_DIR     = _SCRIPT_DIR
for _candidate in [_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)]:
    if os.path.isfile(os.path.join(_candidate, "missouri_awos_asos_stations.json")):
        _PROJECT_DIR = _candidate
        break
_STRIPPED_DIR    = os.path.join(_PROJECT_DIR, "stripped_recordings")
_RECORDINGS_DIR  = os.path.join(_PROJECT_DIR, "recordings")
# Prefer stripped_recordings/ (silence removed) when it has MP3s,
# otherwise fall back to recordings/ (raw downloads)
INPUT_DIR        = (_STRIPPED_DIR
                    if os.path.isdir(_STRIPPED_DIR)
                    and any(f.lower().endswith(".mp3")
                            for f in os.listdir(_STRIPPED_DIR))
                    else _RECORDINGS_DIR)
OUTPUT_DIR       = os.path.join(_PROJECT_DIR, "trimmed_recordings")
JSON_FILE        = os.path.join(_PROJECT_DIR, "missouri_awos_asos_stations.json")
TRANSCRIPTS_FILE  = os.path.join(_PROJECT_DIR, "transcripts.json")
PARSED_FILE       = os.path.join(_PROJECT_DIR, "parsed_results.json")
MANIFEST_FILE     = os.path.join(_PROJECT_DIR, "trim_manifest.json")

PREROLL_S        = 0.15   # seconds before station name to avoid clipping
TRAILING_DB      = -42.0  # dBFS threshold for trailing silence removal
MIN_LOOP_S       = 15     # sanity check: loops shorter than this are rejected
MAX_LOOP_S       = 120    # sanity check: loops longer than this are rejected

INVALID_STATION_WORDS = {
    # Words that can appear at a silence gap but are NOT station name starters.
    # Prevents AWOS remark words (Density, Altitude, etc.) being mistaken for
    # the start of a station name when Whisper compresses timestamps.
    'density', 'altitude', 'remarks', 'remark', 'temperature', 'dewpoint',
    'dew', 'altimeter', 'visibility', 'wind', 'sky', 'ceiling', 'scattered',
    'broken', 'overcast', 'clear', 'few', 'haze', 'fog', 'snow', 'rain',
    'peak', 'gust', 'gusts', 'knots', 'celcius', 'celsius', 'inches',
    'mercury', 'automated', 'weather', 'observation', 'zulu',
    # Spoken digits
    'zero', 'one', 'two', 'three', 'four', 'five',
    'six', 'seven', 'eight', 'nine', 'niner',
    # Number magnitude words that end remarks
    'hundred', 'thousand',
    # Remarks phrases that immediately precede the station name
    'thunderstorm', 'thunderstorms', 'information', 'available',
    'distant', 'lightning', 'present', 'vicinity',
    'precipitation', 'drizzle', 'mist', 'unknown', 'airborne', 'missing',
    # Lightning direction remark words (e.g. 'Lightning distance east through south')
    'distance', 'through', 'east',
}

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
# Optional --run-dir argument: write manifest into run directory
import argparse as _ap
_args = _ap.ArgumentParser(add_help=False)
_args.add_argument('--run-dir', default=None)
_parsed_args, _ = _args.parse_known_args()
_RUN_DIR = _parsed_args.run_dir
if _RUN_DIR:
    MANIFEST_FILE = os.path.join(_RUN_DIR, 'trim_manifest.json')
    TRANSCRIPTS_FILE = os.path.join(_RUN_DIR, 'transcripts.json')
    PARSED_FILE      = os.path.join(_RUN_DIR, 'parsed_results.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(JSON_FILE) as f:
    stations = {s["id"]: s for s in json.load(f)}

word_timestamps    = {}
current_run_files  = {}  # station -> expected filename from this run
try:
    with open(TRANSCRIPTS_FILE) as f:
        tx = json.load(f)
    for t in tx.get("transcripts", []):
        wt = t.get("word_timestamps")
        if wt:
            word_timestamps[t["station"]] = wt
    # Build station->filename map from date_created for run-scoped filtering
    for t in tx.get("transcripts", []):
        dc = t.get("date_created", "")
        if dc and t.get("station"):
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(dc.replace('Z', '+00:00'))
                current_run_files[t["station"]] = \
                    f"{t['station']}_{dt.strftime('%Y%m%d_%H%M%S')}.mp3"
            except (ValueError, TypeError):
                pass
    print(f"Loaded word timestamps for {len(word_timestamps)} station(s)")
    if current_run_files:
        print(f"Current run: {len(current_run_files)} station file(s) identified")
except FileNotFoundError:
    print(f"Note: {TRANSCRIPTS_FILE} not found - energy fallback for all files")

selected_loop_times = {}  # station -> 4-digit obs time chosen by parser
try:
    with open(PARSED_FILE) as f:
        for r in json.load(f):
            lt = r.get("selected_loop_time")
            if lt:
                selected_loop_times[r["station"]] = lt
    if selected_loop_times:
        print(f"Loaded selected loop times for {len(selected_loop_times)} station(s)")
except FileNotFoundError:
    pass  # parsed_results.json optional - falls back to median logic

print()

# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def decode_to_pcm(path, sr=22050):
    """Decode MP3 to float32 mono PCM via ffmpeg."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True
    )
    raw = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)
    if len(raw):
        raw /= 32768.0
    return raw, sr


def trim_mp3(input_path, output_path, start_sec, end_sec):
    """Cut clip with ffmpeg stream copy - no re-encode."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{start_sec:.3f}",
         "-t",  f"{end_sec - start_sec:.3f}",
         "-i",  input_path,
         "-c",  "copy",
         output_path],
        check=True
    )


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def find_trailing_silence_end(samples, sr, threshold_db=TRAILING_DB):
    """Return seconds into samples where speech ends (trailing silence begins)."""
    threshold = 10 ** (threshold_db / 20.0)
    chunk     = int(sr * 0.1)
    end       = len(samples)
    while end > chunk:
        if np.sqrt(np.mean(samples[end - chunk:end] ** 2)) > threshold:
            break
        end -= chunk
    return end / sr


# ---------------------------------------------------------------------------
# Timestamp-based loop detection
# ---------------------------------------------------------------------------

def _anchor_obs_time(words, anchor_idx):
    """Decode the 4-digit obs time spoken after an anchor.
    e.g. words after anchor: 'one','five','five','four','zulu' -> '1554'
    Returns a 4-digit string or None.
    """
    WD = {'zero':'0','one':'1','two':'2','three':'3','four':'4',
          'five':'5','six':'6','seven':'7','eight':'8','nine':'9','niner':'9'}
    digits = []
    for w in words[anchor_idx + 3 : anchor_idx + 12]:
        word = w['word'].lower().strip('.,;')
        if word in WD:
            digits.append(WD[word])
        elif word in ('zulu', 'z'):
            break
    return ''.join(digits[:4]) if len(digits) >= 4 else None


def find_loop_from_timestamps(words, obs_time=None, station_first_word=None):
    """
    Identify one complete broadcast loop from word-level timestamps.

    Algorithm:
    1. Find all "automated weather observation" anchors.
    2. Build pairs from ALL adjacent anchors — no longer requires a valid
       station name before the start anchor. Most AWOS recordings start
       mid-broadcast so anchor0 rarely has a preceding station name.
    3. Select the pair whose obs_time matches the parser's selected loop,
       or fall back to median duration.
    4. Determine trim boundaries: use station_name_start if valid, else
       fall back to the anchor's own position (starts right at the
       "automated weather observation" header which is acceptable).

    Returns (start_sec, end_sec) or None if a clean loop cannot be found.
    """
    TRIGGER       = ["automated", "weather", "observation"]
    FILTER_BEFORE = {"for", "the", "is", "an"}

    # Find all valid broadcast anchors
    anchors = []
    i = 0
    while i <= len(words) - len(TRIGGER):
        if all(words[i + j]["word"].lower().strip(".,") == TRIGGER[j]
               for j in range(len(TRIGGER))):
            pre = words[i - 1]["word"].lower().strip(".,") if i > 0 else ""
            if pre not in FILTER_BEFORE:
                anchors.append(i)
            i += len(TRIGGER)
        else:
            i += 1

    if len(anchors) < 2:
        # Single anchor fallback
        if len(anchors) == 1:
            sw, valid = _station_name_start(words, anchors[0], prev_anchor_idx=None)
            start_sec = (max(0.0, words[sw]["start"] - PREROLL_S)
                         if valid
                         else max(0.0, words[anchors[0]]["start"] - PREROLL_S))
            end_sec = words[-1]["end"]
            if MIN_LOOP_S < end_sec - start_sec < MAX_LOOP_S:
                return start_sec, end_sec

        # Zero-anchor fallback: Whisper missed "automated weather observation"
        # but the station name may repeat at consistent intervals.
        if station_first_word:
            sfw = station_first_word.lower().strip(".")
            name_hits = [
                k for k, w in enumerate(words)
                if w["word"].lower().strip(".,") == sfw
            ]
            for j in range(len(name_hits) - 1):
                dur = words[name_hits[j + 1]]["start"] - words[name_hits[j]]["start"]
                if MIN_LOOP_S < dur < MAX_LOOP_S:
                    start_sec = max(0.0, words[name_hits[j]]["start"] - PREROLL_S)
                    end_sec   = words[name_hits[j + 1]]["start"]
                    return start_sec, end_sec

        return None

    # Build pairs from ALL adjacent anchors — pair duration uses station name
    # start when valid, anchor position otherwise
    pairs = []
    for k in range(len(anchors) - 1):
        prev = anchors[k - 1] if k > 0 else None
        sw, valid = _station_name_start(words, anchors[k], prev_anchor_idx=prev)
        # Use station name start for duration when valid, else anchor position
        pair_start_t = words[sw]["start"] if valid else words[anchors[k]]["start"]
        dur = words[anchors[k + 1]]["start"] - pair_start_t
        if MIN_LOOP_S < dur < MAX_LOOP_S:
            pairs.append((k, sw, valid, dur))

    if not pairs:
        return None

    # Select best pair: prefer obs_time match with clean station name start
    def _has_clean_start(pair):
        """True when station name start is genuinely before the anchor,
        not Whisper-compressed to the same timestamp."""
        sw_t = words[pair[1]]["start"]
        a_t  = words[anchors[pair[0]]]["start"]
        return sw_t < a_t - 0.5

    best = None
    if obs_time:
        # Try to find a pair that STARTS at the parser-selected obs time
        # Prefer a pair with a genuine (non-compressed) station name start
        clean_match = None
        any_match   = None
        for pair in pairs:
            if _anchor_obs_time(words, anchors[pair[0]]) == obs_time:
                if any_match is None:
                    any_match = pair
                if _has_clean_start(pair) and clean_match is None:
                    clean_match = pair
        if clean_match:
            best = clean_match
        elif any_match:
            # Use the obs_time match even if timestamps are compressed.
            # Compressed start just means we trim from the anchor position
            # ('automated weather observation') which is acceptable.
            best = any_match
        # No pair starts at obs_time — check if any anchor has that time.
        # Only use this single-anchor path when there are no valid pairs at all;
        # if pairs exist, fall through to median selection instead.
        if best is None and not pairs:
            for a_idx, anchor in enumerate(anchors):
                if _anchor_obs_time(words, anchor) == obs_time:
                    prev = anchors[a_idx - 1] if a_idx > 0 else None
                    sw, valid = _station_name_start(words, anchor, prev_anchor_idx=prev)
                    start_sec = (max(0.0, words[sw]["start"] - PREROLL_S)
                                 if valid
                                 else max(0.0, words[anchor]["start"] - PREROLL_S))
                    end_sec = words[-1]["end"]
                    # Only use single-anchor path if start lands on a reasonable word
                    start_word = next(
                        (w["word"].lower().strip(".,") for w in words
                         if abs(w["start"] - (start_sec + PREROLL_S)) < 0.3), "")
                    if start_word in INVALID_STATION_WORDS and a_idx > 0:
                        break  # bad start - fall through to median
                    if MIN_LOOP_S < end_sec - start_sec < MAX_LOOP_S:
                        return start_sec, end_sec
                    break

    # Fall back to median-duration logic
    if best is None:
        med  = statistics.median(d for _, _, _, d in pairs)
        best = min(pairs, key=lambda p: abs(p[3] - med))

    best_k, best_sw, best_valid, _ = best

    # If the selected pair starts at or near t=0, the recording caught the
    # station name mid-word (Twilio connected mid-broadcast). Prefer the next
    # valid pair so we start at a clean station name occurrence.
    if best_valid and words[best_sw]["start"] < 0.3 and len(pairs) > 1:
        alt = [p for p in pairs if p[0] != best_k]
        if alt:
            best_k, best_sw, best_valid, _ = alt[0]

    # Trim start: use station name if found, else fall back to anchor position
    start_sec = (max(0.0, words[best_sw]["start"] - PREROLL_S)
                 if best_valid
                 else max(0.0, words[anchors[best_k]]["start"] - PREROLL_S))

    # Trim end: if best_k+1 is the last anchor there is no following loop
    # to define a clean end boundary — run to the end of the recording.
    # Otherwise use the station name start of the next anchor.
    if best_k + 1 == len(anchors) - 1 and best_k >= 1:
        # Last anchor with no following pair: selected loop runs to end
        # of recording. Only applies when we're past the first loop (best_k>=1)
        # so pair0 still uses the normal station-name end boundary.
        end_sec = words[-1]["end"]
    else:
        ew, ew_valid = _station_name_start(words, anchors[best_k + 1],
                                           prev_anchor_idx=anchors[best_k])
        if ew_valid and words[ew]["start"] > words[anchors[best_k]]["start"]:
            end_sec = words[ew]["start"]
        else:
            end_sec = words[anchors[best_k + 1]]["start"]

    # FIX 2: if end lands on the last word of the recording AND an earlier
    # complete loop exists (end is a proper station-name boundary, not the
    # last word), prefer that loop. This avoids clipping when Twilio's 90s
    # limit cuts off the selected loop mid-broadcast.
    last_word_end = words[-1]["end"]
    if abs(end_sec - last_word_end) < 0.5 and best_k >= 1:
        for pair in pairs:
            if pair[0] >= best_k:
                continue
            # Compute this earlier pair's end boundary
            _ew2, _ewv2 = _station_name_start(
                words, anchors[pair[0] + 1],
                prev_anchor_idx=anchors[pair[0]])
            _pair_end = (words[_ew2]["start"] if _ewv2
                         else words[anchors[pair[0] + 1]]["start"])
            if abs(_pair_end - last_word_end) > 1.0:
                # Found a complete earlier loop — use it
                _sw2 = pair[1]
                start_sec = (max(0.0, words[_sw2]["start"] - PREROLL_S)
                             if pair[2]
                             else max(0.0, words[anchors[pair[0]]]["start"] - PREROLL_S))
                end_sec   = _pair_end
                best_k    = pair[0]
                best_sw   = _sw2
                break

    if end_sec - start_sec < MIN_LOOP_S:
        return None

    # Final guard: if the selected station name word is an AWOS content word,
    # detection failed — return None and let energy fallback handle it.
    # Use best_sw index directly (not timestamp proximity, which is unreliable
    # when Whisper compresses many words to the same timestamp).
    _sw_word = words[best_sw]["word"].lower().strip(".,")
    if _sw_word in INVALID_STATION_WORDS and _sw_word not in \
            ("automated", "weather", "observation"):
        return None

    return start_sec, end_sec



def _station_name_start(words, trigger_idx, prev_anchor_idx=None):
    """
    Find the start of the station name that precedes an anchor.

    Primary method: content-based.
      Walk backward from the anchor; the station name begins immediately
      after the last AWOS data word (any word in INVALID_STATION_WORDS).
      Station names are capped at STATION_NAME_MAX_WORDS words.

    Fallback: gap-based (>0.5s silence).

    prev_anchor_idx bounds the search so we never cross into a previous loop.
    Returns (word_index, is_valid).
    """
    STATION_NAME_MAX_WORDS = 8

    min_idx = (prev_anchor_idx + 3) if prev_anchor_idx is not None else 0

    # --- Primary: content-based ---
    _DIRECTION_WORDS = {
        'north', 'south', 'east', 'west',
        'northeast', 'northwest', 'southeast', 'southwest',
    }
    last_invalid = min_idx - 1
    for k in range(trigger_idx - 1, min_idx - 1, -1):
        if words[k]["word"].lower().strip(".,;") in INVALID_STATION_WORDS:
            last_invalid = k
            break

    station_start = last_invalid + 1
    # Cap to max station name length (trim excess words from the left)
    if trigger_idx - station_start > STATION_NAME_MAX_WORDS:
        station_start = trigger_idx - STATION_NAME_MAX_WORDS

    # Skip direction words that are remark tail words, not city name starters.
    # A direction word is a remark tail when:
    #   - preceded by 'distant' (lightning remark: 'Lightning Distant South')
    #   - at recording start (pos 0 or 1)
    #   - immediately follows a conjunction ('and','or') that itself follows a
    #     skipped direction (e.g. 'distant southwest and west' -> skip all three)
    _just_skipped_direction = False
    while station_start < trigger_idx:
        w      = words[station_start]["word"].lower().strip(".,;")
        prev_w = (words[station_start - 1]["word"].lower().strip(".,;")
                  if station_start > 0 else "")
        is_remark_direction = (
            w in _DIRECTION_WORDS and (
                prev_w == 'distant'              # lightning remark
                or station_start <= 1            # recording start
                or _just_skipped_direction       # follows another skipped direction
            )
        )
        # Skip bare conjunctions that join two remark directions
        is_remark_connector = (
            w in ('and', 'or', 'through')
            and _just_skipped_direction
        )
        if is_remark_direction or is_remark_connector:
            _just_skipped_direction = True  # keep propagating through connectors
            station_start += 1
        else:
            _just_skipped_direction = False
            break

    if station_start < trigger_idx:
        start_word = words[station_start]["word"].lower().strip(".,;")
        if start_word not in INVALID_STATION_WORDS:
            return station_start, True

    # --- Fallback: gap-based ---
    for k in range(trigger_idx - 1, min_idx, -1):
        if words[k]["start"] - words[k - 1]["end"] > 0.5:
            if words[k]["word"].lower().strip(".,") not in INVALID_STATION_WORDS:
                return k, True

    # --- Last resort: position 0 ---
    if min_idx == 0:
        first = words[0]["word"].lower().strip(".,")
        if first not in ("automated", "weather", "observation")                 and first not in INVALID_STATION_WORDS:
            return 0, True

    return min_idx, False



# ---------------------------------------------------------------------------
# Trim using word timestamps
# ---------------------------------------------------------------------------

def trim_by_timestamps(input_path, output_path, words, obs_time=None, station_first_word=None):
    """
    Trim using word timestamps. Removes trailing silence from the clip.
    Returns (True, duration) on success, (False, reason) on failure.
    """
    result = find_loop_from_timestamps(words, obs_time=obs_time, station_first_word=station_first_word)
    if result is None:
        return False, "could not identify a complete loop in timestamps"

    start_sec, end_sec = result

    # Remove trailing silence from the selected region
    try:
        samples, sr = decode_to_pcm(input_path)
        s0      = int(start_sec * sr)
        s1      = int(end_sec   * sr)
        region  = samples[s0:s1]
        tail    = find_trailing_silence_end(region, sr)
        end_sec = start_sec + tail
    except Exception:
        pass  # skip trailing silence trim if decode fails

    duration = end_sec - start_sec
    if duration < MIN_LOOP_S:
        # Accept if the original recording itself is shorter than MIN_LOOP_S
        original_dur = get_duration(input_path)
        if original_dur >= MIN_LOOP_S:
            return False, f"trimmed duration too short ({duration:.1f}s)"

    duration = end_sec - start_sec
    if duration < MIN_LOOP_S:
        original_dur = get_duration(input_path)
        if original_dur >= MIN_LOOP_S:
            return False, f"trimmed duration too short ({duration:.1f}s)"

    try:
        trim_mp3(input_path, output_path, start_sec, end_sec)
        return True, end_sec - start_sec
    except subprocess.CalledProcessError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Energy-based fallback
# ---------------------------------------------------------------------------

def trim_by_energy(input_path, output_path):
    """
    Fallback: detect loop boundaries via RMS silence analysis.
    Returns (True, duration) on success, (False, reason) on failure.
    """
    try:
        samples, sr = decode_to_pcm(input_path)
    except Exception as e:
        return False, f"decode error: {e}"

    frame = int(sr * 0.05)
    hop   = int(sr * 0.02)
    rms   = np.array([
        np.sqrt(np.mean(samples[i:i + frame] ** 2))
        for i in range(0, len(samples) - frame, hop)
    ])
    threshold     = np.percentile(rms, 20) * 1.5
    silent        = rms < threshold
    starts        = []
    silence_start = None
    for i, is_silent in enumerate(silent):
        if is_silent and silence_start is None:
            silence_start = i
        elif not is_silent and silence_start is not None:
            if (i - silence_start) * (hop / sr) > 0.7:
                starts.append(i * (hop / sr))
            silence_start = None

    if len(starts) < 2:
        return False, "insufficient silence boundaries detected"

    # Pick best pair
    start_sec, end_sec = starts[0], starts[1]
    if len(starts) >= 3 and MIN_LOOP_S < starts[2] - starts[0] < MAX_LOOP_S:
        start_sec, end_sec = starts[0], starts[2]

    start_sec = max(0.0, start_sec - 0.04)
    region    = samples[int(start_sec * sr):int(end_sec * sr)]
    tail      = find_trailing_silence_end(region, sr)
    end_sec   = start_sec + tail

    duration = end_sec - start_sec
    if duration < MIN_LOOP_S:
        original_dur = get_duration(input_path)
        if original_dur >= MIN_LOOP_S:
            return False, f"trimmed duration too short ({duration:.1f}s)"

    try:
        trim_mp3(input_path, output_path, start_sec, end_sec)
        return True, end_sec - start_sec
    except subprocess.CalledProcessError as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Build candidate file list — if we know which files belong to this run
# (from transcripts.json date_created), restrict to those only.
# This prevents stale files from previous runs being re-trimmed.
_all_mp3s = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp3"))
if current_run_files:
    # For each station, prefer the exact filename from transcripts.json.
    # Fall back to the most recent file for that station if exact name missing.
    from collections import defaultdict as _dd
    _by_station = _dd(list)
    for f in _all_mp3s:
        _by_station[f[:4].upper()].append(f)
    files = []
    for station, expected in current_run_files.items():
        if expected in _by_station.get(station, []):
            files.append(expected)
        elif _by_station.get(station):
            # Exact file not found — use most recent for this station
            files.append(sorted(_by_station[station])[-1])
    files = sorted(files)
    print(f"Filtered to {len(files)} file(s) from current run")
else:
    # No run info available — process all files (standalone / manual use)
    files = _all_mp3s

if not files:
    print(f"No MP3 files found in '{INPUT_DIR}/'")
else:
    print(f"Processing {len(files)} file(s)\n")

results  = {"ok": [], "fallback": [], "failed": []}
manifest = {}  # station_id -> trimmed filename for exact report matching

for filename in files:
    input_path  = os.path.join(INPUT_DIR,  filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    station_id  = filename[:4].upper()
    words       = word_timestamps.get(station_id)

    obs_time = selected_loop_times.get(station_id)
    station_meta = stations.get(station_id, {})
    # First word of the station name — used as repeat-anchor fallback
    _loc = station_meta.get("location", "") or station_meta.get("name", "")
    station_first_word = _loc.split()[0] if _loc else None
    if words:
        ok, detail = trim_by_timestamps(input_path, output_path, words,
                                        obs_time=obs_time,
                                        station_first_word=station_first_word)
        if ok:
            print(f"OK       {filename:<38} {detail:.1f}s  [timestamps]")
            results["ok"].append(station_id)
            manifest[station_id] = filename
            continue
        print(f"  Timestamp trim failed ({detail}) - trying energy fallback")

    # Energy fallback
    ok, detail = trim_by_energy(input_path, output_path)
    if ok:
        method = "energy(fb)" if words else "energy"
        print(f"OK       {filename:<38} {detail:.1f}s  [{method}]")
        results["ok" if not words else "fallback"].append(station_id)
        manifest[station_id] = filename
    else:
        print(f"FAILED   {filename:<38} {detail}")
        results["failed"].append(station_id)

# Write manifest: station -> trimmed filename for exact lookup by generate_report.py
with open(MANIFEST_FILE, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest written: {len(manifest)} station(s) -> {MANIFEST_FILE}")

# Summary
print()
print(f"Done: {len(results['ok'])} timestamp, "
      f"{len(results['fallback'])} energy fallback, "
      f"{len(results['failed'])} failed")
if results["failed"]:
    print(f"Failed: {', '.join(results['failed'])}")
