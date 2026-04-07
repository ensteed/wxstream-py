# wxstream_pipeline.py

Pipelined WxStream orchestrator. Calls all Missouri AWOS/ASOS stations via
Twilio, transcribes recordings with Whisper, parses structured weather data,
trims audio to a single broadcast loop, generates an HTML report, and imports
to MongoDB Atlas.

---

## Directory Structure

```
WxStream/
├── missouri_awos_asos_stations.json   # Station registry (authoritative source)
├── src/
│   ├── wxstream_pipeline.py           # This script
│   ├── audio_trim.py
│   ├── parse_transcripts.py
│   ├── generate_report.py
│   ├── atlas_import.py
│   ├── awos_test_run.py
│   ├── fetch_recordings.py
│   ├── silence_strip.py
│   └── transcribe.py
└── output/
    ├── recordings/                    # Raw MP3 downloads from Twilio
    ├── stripped_recordings/           # Silence-stripped MP3s (originals preserved)
    ├── trimmed_recordings/            # Single-loop trimmed MP3s
    ├── runs/                          # Timestamped run directories
    │   ├── latest -> runs/YYYYMMDD_HHMMSS/   # Symlink to most recent run
    │   └── YYYYMMDD_HHMMSS/
    │       ├── transcripts.json
    │       ├── parsed_results.json
    │       ├── trim_manifest.json
    │       ├── awos_report.html
    │       └── pipeline.log
    ├── manifests/                     # trim_manifest.json (standalone runs)
    └── logs/
        └── wxstream_pipeline.log      # Persistent log across all runs
```

Always run from the project root:

```bash
cd /root/WxStream
python src/wxstream_pipeline.py
```

---

## Pipeline Phases

| Phase | Script | Output |
|-------|--------|--------|
| 1 | Twilio calls via `awos_test_run.py` | MP3s in `output/recordings/` |
| 1b | Silence stripping (ffmpeg) | MP3s in `output/stripped_recordings/` |
| 2 | Whisper transcription (parallel workers) | In-memory transcript records |
| 3 | Save transcripts | `runs/YYYYMMDD_HHMMSS/transcripts.json` |
| 4 | `parse_transcripts.py` | `runs/YYYYMMDD_HHMMSS/parsed_results.json` |
| 4b | `audio_trim.py` | `output/trimmed_recordings/` + `trim_manifest.json` |
| 5 | `generate_report.py` | `runs/YYYYMMDD_HHMMSS/awos_report.html` |
| 6 | `atlas_import.py` | MongoDB Atlas (if `MONGO_DB_PASSWORD` is set) |

---

## Usage

```bash
# Full pipeline — place calls, transcribe, parse, trim, report, import
python src/wxstream_pipeline.py

# Skip Twilio calls; process recordings already in output/stripped_recordings/
# (falls back to output/recordings/ if stripped/ is empty)
python src/wxstream_pipeline.py --local

# Control concurrent Whisper workers (default: 10)
python src/wxstream_pipeline.py --workers 5

# Simulate pipeline without making any API calls
python src/wxstream_pipeline.py --dry-run

# Run N times on a repeating schedule (default interval: 30 min)
python src/wxstream_pipeline.py --schedule 6
python src/wxstream_pipeline.py --schedule 6 --interval 60

# Re-trim only, using an existing run's transcripts (no API calls)
python src/audio_trim.py --run-dir output/runs/latest
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TWILIO_SID` | Yes (full mode) | Twilio account SID (`ACxxxx...`) |
| `TWILIO_AUTH` | Yes (full mode) | Twilio auth token |
| `TWILIO_FROM` | Yes (full mode) | Outbound caller ID in E.164 format |
| `OPENAI_API_KEY` | Yes | OpenAI API key for Whisper transcription |
| `MONGO_DB_PASSWORD` | No | Atlas password — Phase 6 skipped if unset |

---

## Transcription

Each station recording is transcribed using the OpenAI Whisper API
(`whisper-1`) with a station-specific prompt. Both word-level and
segment-level timestamps are requested in a single API call.

### Word Timestamp Collapse Detection

Whisper's word-level timestamp alignment can fail silently on repetitive
audio (AWOS broadcasts loop continuously), producing only a handful of
timestamps for hundreds of words. The pipeline detects this automatically:

- If `len(word_timestamps) < max(10, word_count * 10%)`, collapse is detected
- Segment timestamps from the **same API response** are used as a fallback —
  no second call is made
- Segment timestamps are converted to word-equivalent format by distributing
  each segment's time window evenly across its tokens
- The transcript record includes `"timestamp_source": "word"` or `"segment"`
  so you can identify which stations triggered the fallback

### Hallucination Detection

After transcription, repetitive Whisper hallucination loops are detected and
stripped. If stripping occurs, the original text is preserved in
`raw_transcript` and the cleaned version in `cleaned_transcript`.
`hallucination_chars` records how many characters were removed.
`cleaned_transcript` is `null` when no hallucinations were detected —
this is normal; use `raw_transcript` in that case.

---

## Audio Trimming

`audio_trim.py` isolates a single broadcast loop from each recording using
word-level timestamps. It finds all occurrences of the "automated weather
observation" anchor phrase (with 2-of-3 fuzzy matching to tolerate one
mangled word), selects the best complete loop, and cuts precisely with ffmpeg.

If timestamp-based trimming fails, an energy/RMS silence fallback is used.

The `trim_manifest.json` written into the run directory maps each station to
its trimmed filename for exact lookup by `generate_report.py`.

---

## Scheduling

The pipeline can be run on a cron schedule (every 30 minutes is typical):

```bash
# crontab entry
*/30 * * * * cd /root/WxStream && python src/wxstream_pipeline.py >> output/logs/cron.log 2>&1
```

Or use the built-in scheduler for a fixed number of runs:

```bash
python src/wxstream_pipeline.py --schedule 48 --interval 30
```

---

## Station Registry

`missouri_awos_asos_stations.json` in the project root is the sole
authoritative source for station identity (name, type, location, phone).
Transcript content never influences these fields.

---

## MongoDB Atlas

Phase 6 imports each run to Atlas using one database per station
(`kaiz.broadcasts`, `kcgi.broadcasts`, etc.). It is skipped automatically
if `MONGO_DB_PASSWORD` is not set. Each document is upserted on
`(recording_sid, call_datetime)` so re-importing the same run is safe.
