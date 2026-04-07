"""
generate_report.py
------------------
Reads parsed_results.json (output of parse_transcripts.py) and produces
a self-contained HTML weather report with embedded base64 audio.

Usage:
    python generate_report.py
    python generate_report.py --input my_results.json --output report.html --title "My Report"
    python generate_report.py --audio path/to/trimmed_recordings

Audio files are read from a subfolder called 'trimmed_recordings' alongside
the script by default. Each file must start with the ICAO identifier, e.g.:
    KAIZ_recording.mp3  or  KAIZ.mp3  or  KAIZ_something.mp3

Audio is base64-encoded and embedded directly in the HTML so the report
is fully self-contained and works offline.

Requirements: Python 3.6+, no external dependencies.
"""

import json
import sys
import base64
import argparse
import os
from datetime import date, datetime


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_files(audio_dir, results=None, input_path=None):
    """
    Match trimmed MP3 files to stations.

    Priority:
      1. trim_manifest.json (written by audio_trim.py) — exact station->filename
         mapping from the same pipeline run. No guessing, no stale file risk.
      2. Timestamp-based fallback: station prefix + closest datetime to date_created.
         Used when the manifest is absent (e.g. manual runs, older directories).

    Returns {station: base64_data_uri}.
    """
    if not os.path.isdir(audio_dir):
        return {}

    # --- Primary: read manifest written by audio_trim.py ---
    # Prefer manifest in the same directory as the --input parsed_results.json
    # (i.e. the run directory), falling back to the script directory.
    _input_dir    = os.path.dirname(os.path.abspath(audio_dir)) \
                    if audio_dir else os.path.dirname(os.path.abspath(__file__))
    _run_manifest = os.path.join(
        os.path.dirname(os.path.abspath(input_path)) if input_path
        else os.path.dirname(os.path.abspath(__file__)),
        'trim_manifest.json')
    _script_manifest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'trim_manifest.json')
    manifest_path = _run_manifest if os.path.isfile(_run_manifest) \
                    else _script_manifest
    manifest = {}
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Manifest loaded: {len(manifest)} station(s) from {manifest_path}")
    except FileNotFoundError:
        print(f"No manifest at {manifest_path}")

    audio = {}

    if manifest:
        missing = []
        for station, filename in manifest.items():
            filepath = os.path.join(audio_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as fh:
                    b64 = base64.b64encode(fh.read()).decode('ascii')
                audio[station] = f'data:audio/mpeg;base64,{b64}'
            else:
                missing.append(filename)
        if missing:
            print(f"Manifest: {len(audio)} found, {len(missing)} not in audio_dir")
            print(f"  audio_dir : {audio_dir}")
            print(f"  Missing   : {', '.join(missing[:5])}"
                  f"{'...' if len(missing) > 5 else ''}")
            # If manifest references files not in audio_dir, fall through
            # to timestamp-based matching rather than returning empty
            if not audio:
                print("  All manifest entries missing — falling back to "
                      "timestamp-based matching")
                manifest.clear()
        if manifest:
            return audio

    # --- Fallback: timestamp-based matching ---
    print("No manifest found — using timestamp-based audio matching")

    # Build station -> observation datetime lookup
    date_lookup = {}
    if results:
        for r in results:
            dc = r.get('date_created', '')
            if dc:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    date_lookup[r['station']] = _dt.fromisoformat(dc)
                except (ValueError, TypeError):
                    pass

    # Group MP3 files by station prefix
    from collections import defaultdict
    station_files = defaultdict(list)
    for filename in sorted(os.listdir(audio_dir)):
        if not filename.lower().endswith('.mp3'):
            continue
        stem    = os.path.splitext(filename)[0].upper()
        station = stem[:4]
        if not (len(station) == 4 and station[0] == 'K'):
            continue
        import re as _re
        m = _re.search(r'_(\d{8})_(\d{6})\.mp3$', filename, _re.IGNORECASE)
        file_dt = None
        if m:
            try:
                from datetime import datetime as _dt, timezone as _tz
                file_dt = _dt.strptime(m.group(1) + m.group(2),
                                       '%Y%m%d%H%M%S').replace(tzinfo=_tz.utc)
            except ValueError:
                pass
        station_files[station].append((filename, os.path.join(audio_dir, filename), file_dt))

    for station, files in station_files.items():
        obs_time = date_lookup.get(station)

        if obs_time:
            obs_date = obs_time.strftime('%Y%m%d')
            same_day = [(f, p, t) for f, p, t in files if obs_date in f]
            if not same_day:
                same_day = files

            timestamped = [(f, p, t) for f, p, t in same_day if t is not None]
            if timestamped:
                best = min(timestamped,
                           key=lambda x: abs((x[2] - obs_time).total_seconds()))
            else:
                best = sorted(same_day)[0]
        else:
            best = sorted(files)[0]

        with open(best[1], 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode('ascii')
        audio[station] = f'data:audio/mpeg;base64,{b64}'

    return audio



def sky_color(sky):
    """Return a subtle background colour for the sky condition table cell."""
    if not sky or sky in ('N/A', 'Missing'):
        return 'transparent'
    s = sky.upper()
    if 'CLR' in s or 'SKC' in s:
        return '#e8f4fd'   # light blue - clear
    if 'FEW' in s:
        return '#fef9e7'   # pale yellow - few
    if 'SCT' in s:
        return '#fdebd0'   # light orange - scattered
    if 'BKN' in s:
        return '#f2f3f4'   # light grey - broken
    if 'OVC' in s or 'VV' in s:
        return '#e5e7e9'   # grey - overcast/obscured
    return 'transparent'


def na_class(value):
    """Return a CSS class attribute string for N/A or Missing values."""
    if value in ('N/A', 'Missing'):
        return ' class="na-value"'
    return ''


def build_audio_html(station, audio_map):
    """Return an inline audio player HTML string, or empty string if no audio."""
    uri = audio_map.get(station)
    if not uri:
        return ''
    return (
        f'<audio controls preload="none" class="station-audio">'
        f'<source src="{uri}" type="audio/mpeg">'
        f'</audio>'
    )


def build_phenomena_html(phenomena):
    """
    Render weather phenomena list as small badge spans.
    phenomena is a list of (description, code) tuples e.g. [('Haze', 'HZ')].
    """
    if not phenomena:
        return ''
    badges = []
    for item in phenomena:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            desc, code = item[0], item[1]
        else:
            desc, code = str(item), ''
        badges.append(
            f'<span class="wx-badge" title="{desc}">{code or desc}</span>'
        )
    return ' '.join(badges)


def build_card(r, audio_map):
    sky_bg         = sky_color(r['sky'])
    phenomena_html = build_phenomena_html(r.get('phenomena', []))
    audio_html     = build_audio_html(r['station'], audio_map)

    local_info_html = ''
    if r.get('local_info'):
        local_info_html = f'''
      <div class="local-info">
        <span class="local-info-label">&#9432; Local Information</span>
        <span class="local-info-text">{r["local_info"]}</span>
      </div>'''

    return f'''
    <div class="station-card">
      <div class="card-header">
        <span class="icao">{r["station"]}</span>
        <span class="location">{r["airport_name"]}</span>
        <span class="wx-type">{r["type"]}</span>
        <span class="obs-time">Obs: {r["time"]}</span>
        {audio_html}
      </div>
      <div class="metar-string">{r["metar"]}</div>
      <table class="wx-table">
        <tr>
          <th>Wind</th>
          <th>Visibility</th>
          <th>Weather</th>
          <th>Sky Condition</th>
          <th>Temp / Dew Point</th>
          <th>Altimeter</th>
          <th>Remarks</th>
        </tr>
        <tr>
          <td{na_class(r["wind"])}>{r["wind"]}</td>
          <td{na_class(r["visibility"])}>{r["visibility"]}</td>
          <td class="wx-phenomena">{phenomena_html}</td>
          <td style="background:{sky_bg}"{na_class(r["sky"])}>{r["sky"]}</td>
          <td{na_class(r["temp_dp"])}>{r["temp_dp"]}</td>
          <td{na_class(r["altimeter"])}>{r["altimeter"]}</td>
          <td>{r["remarks"]}</td>
        </tr>
      </table>{local_info_html}
    </div>'''


CSS = '''
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #1a1f2e;
    color: #e0e6f0;
    padding: 24px;
  }
  h1 {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #7eb8f7;
    margin-bottom: 6px;
    text-transform: uppercase;
  }
  .subtitle {
    text-align: center;
    font-size: 0.82rem;
    color: #8899aa;
    margin-bottom: 28px;
    letter-spacing: 0.03em;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
    gap: 16px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .station-card {
    background: #242b3d;
    border: 1px solid #2e3a50;
    border-radius: 8px;
    overflow: hidden;
  }
  .card-header {
    background: #1e2840;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid #2e3a50;
    flex-wrap: wrap;
  }
  .icao {
    font-family: 'Courier New', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    color: #7eb8f7;
    min-width: 52px;
  }
  .location {
    font-size: 0.88rem;
    color: #c8d8e8;
    flex: 1;
  }
  .wx-type {
    font-size: 0.75rem;
    color: #556677;
    font-family: monospace;
    background: #161e2e;
    padding: 2px 7px;
    border-radius: 4px;
  }
  .obs-time {
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    color: #aabbcc;
  }
  /* Audio player */
  .audio-player {
    display: flex;
    align-items: center;
  }
  .play-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    white-space: nowrap;
    background: #2a5fa8;
    color: #e0ecff;
    transition: background 0.15s;
  }
  .play-btn:hover { background: #3a72c8; }
  .play-btn.playing { background: #c0392b; color: #ffe0e0; }
  .play-btn.playing:hover { background: #e74c3c; }
  .play-btn--disabled {
    background: #2a3040;
    color: #445566;
    cursor: not-allowed;
  }
  .play-btn--disabled:hover { background: #2a3040; }
  /* METAR */
  .metar-string {
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    padding: 8px 14px;
    background: #161e2e;
    color: #a8d8a8;
    border-bottom: 1px solid #2e3a50;
    letter-spacing: 0.02em;
    word-break: break-all;
  }
  /* Table */
  .wx-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  .wx-table th {
    background: #1e2840;
    color: #7eb8f7;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 6px 10px;
    text-align: left;
    border-right: 1px solid #2e3a50;
  }
  .wx-table th:last-child { border-right: none; }
  .wx-table td {
    padding: 8px 10px;
    border-right: 1px solid #2e3a50;
    color: #ddeeff;
    vertical-align: middle;
  }
  .wx-table td:last-child { border-right: none; }
  .wx-table td[style*="background"] {
    color: #1a1a1a;
    font-weight: 600;
  }
  .wx-table td.na {
    color: #556677;
    font-style: italic;
  }
  .wx-phenomena { vertical-align: middle; }
  .wx-badges { display: flex; flex-wrap: wrap; gap: 4px; }
  .wx-badge {
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }
  /* Local info */
  .local-info {
    background: #1a2535;
    border-top: 1px solid #2e3a50;
    padding: 9px 14px;
  }
  .local-info-label {
    display: block;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #f0c060;
    margin-bottom: 4px;
  }
  .local-info-text {
    display: block;
    font-size: 0.8rem;
    color: #c8d8a0;
    line-height: 1.5;
  }
  @media (max-width: 640px) {
    .grid { grid-template-columns: 1fr; }
    body { padding: 12px; }
  }
'''

# JavaScript: toggle play/pause, reset any other playing audio first
JS = '''
  function togglePlay(id, btn) {
    const audio = document.getElementById(id);
    const allAudios = document.querySelectorAll('audio');
    const allBtns   = document.querySelectorAll('.play-btn:not(.play-btn--disabled)');

    // Stop all other playing audio and reset their buttons
    allAudios.forEach(function(a) {
      if (a.id !== id && !a.paused) {
        a.pause();
        a.currentTime = 0;
      }
    });
    allBtns.forEach(function(b) {
      if (b !== btn) {
        b.classList.remove('playing');
        b.innerHTML = '&#9654; Play Recording';
      }
    });

    // Toggle this one
    if (audio.paused) {
      audio.play();
      btn.classList.add('playing');
      btn.innerHTML = '&#9646;&#9646; Stop';
      audio.onended = function() {
        btn.classList.remove('playing');
        btn.innerHTML = '&#9654; Play Recording';
      };
    } else {
      audio.pause();
      audio.currentTime = 0;
      btn.classList.remove('playing');
      btn.innerHTML = '&#9654; Play Recording';
    }
  }
'''


def build_html(results, title, audio_map):
    today   = date.today().strftime('%d %B %Y')
    cards   = '\n'.join(build_card(r, audio_map) for r in results)
    n_wx    = sum(1 for r in results if r.get('phenomena'))
    n_local = sum(1 for r in results if r.get('local_info'))
    n_audio = sum(1 for r in results if r['station'] in audio_map)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">
  {today} &nbsp;&middot;&nbsp;
  {len(results)} Stations &nbsp;&middot;&nbsp;
  Parsed from raw AWOS/ASOS recordings
  {f"&nbsp;&middot;&nbsp; {n_audio} with audio" if n_audio else ""}
  {f"&nbsp;&middot;&nbsp; {n_wx} with present weather" if n_wx else ""}
  {f"&nbsp;&middot;&nbsp; {n_local} with local info" if n_local else ""}
</p>
<div class="grid">
{cards}
</div>
<script>{JS}</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Build timestamped default output name before arg parsing
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _default_output = f'awos_report_{_ts}.html'
    _reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')

    parser = argparse.ArgumentParser(
        description='Generate a self-contained HTML weather report with embedded audio.'
    )
    parser.add_argument('--input',
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'output', 'parsed_results.json'),
                        help='Input JSON file (default: output/parsed_results.json '
                             'relative to project root)')
    parser.add_argument('--output', default=None,
                        help='Output HTML file '
                             '(default: awos_report_YYYYMMDD_HHMMSS.html)')
    parser.add_argument('--title',  default='AWOS / ASOS Weather Observations',
                        help='Report title shown in the page header')
    parser.add_argument('--audio',  default=None,
                        help='Path to trimmed_recordings folder '
                             '(default: trimmed_recordings/ next to this script)')
    args = parser.parse_args()

    # Resolve paths
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(script_dir)
    _output_dir  = os.path.join(_project_dir, "output")
    audio_dir   = args.audio if args.audio else os.path.join(_output_dir, 'trimmed_recordings')
    os.makedirs(_reports_dir, exist_ok=True)
    output_path = args.output if args.output else os.path.join(_reports_dir, _default_output)

    # Load parsed data first so we can extract the date for audio matching
    with open(args.input, encoding='utf-8') as f:
        results = json.load(f)

    # Derive date filter from parsed data (YYYYMMDD from date_created field)
    # Prevents recordings from a previous day matching today's data
    date_filter = None
    try:
        from datetime import datetime as _dt
        _dc = results[0].get('date_created', '') if results else ''
        if _dc:
            date_filter = _dt.fromisoformat(_dc).strftime('%Y%m%d')
    except Exception:
        pass

    # Load audio — only match files from the same date as the parsed data
    audio_map = load_audio_files(audio_dir, results, input_path=args.input)
    if audio_map:
        print(f"Audio loaded: {len(audio_map)} file(s) - {', '.join(sorted(audio_map))}")
    else:
        print(f"No audio files found in: {audio_dir}")

    html = build_html(results, args.title, audio_map)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report written to: {output_path}  "
          f"({len(results)} stations, {len(audio_map)} with audio)")


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
