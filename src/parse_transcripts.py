"""
parse_transcripts.py
--------------------
Parses raw AWOS/ASOS loop recordings from a transcripts JSON file into
structured weather data, then writes parsed_results.json.

Usage:
    python parse_transcripts.py transcripts.json
    python parse_transcripts.py transcripts.json --output my_results.json

Input JSON structure expected:
    {
        "transcripts": [
            {
                "station": "KAIZ",
                "location": "Kaiser - Lee C Fine",
                "type": "AWOS-3PT",
                "raw_transcript": "..."
            },
            ...
        ]
    }

Requirements: Python 3.6+, no external dependencies.
"""

import json
import os
import re
import argparse
from datetime import datetime
from collections import Counter


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(text):
    """Convert spoken/formatted digits to plain numeric strings."""
    # Pre-split: separate visibility value from adjacent sky altitude
    # e.g. 'Visibility one zero six thousand five hundred' ->
    #      'Visibility one zero. six thousand five hundred'
    text = re.sub(
        r'(visibility[\s.,]+)((?:\w+\s+)*?)(\w+\s+)((?:thousand|hundred)\b)',
        r'\1\2. \3\4', text, flags=re.IGNORECASE)
    # Collapse hyphen-separated single digits: 1-4-5-2 -> 1452
    text = re.sub(r'(?<!\w)(\d)(?:-(\d))+(?!\w)',
                  lambda m: m.group(0).replace('-', ''), text)
    # Insert space at fused AWOS keyword boundaries:
    # 'zuluWind' -> 'zulu Wind', 'WeatherWind' -> 'Weather Wind'
    text = re.sub(r'([Zz]ulu)([A-Za-z])', r'\1 \2', text)
    for _kw in ('Wind', 'Visibility', 'Sky', 'Temperature', 'Dewpoint', 'Altimeter', 'Remarks'):
        text = re.sub(r'([a-z])(' + _kw + r'\b)', r'\1 \2', text)
    # niner / nineer -> 9
    text = re.sub(r'\bninee?r\b', '9', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d)er\b', r'\1', text)  # '9er' -> '9'
    # Spoken tens (for compound numbers like "twenty-one")
    tens = {'twenty': '20', 'thirty': '30'}
    for word, digit in tens.items():
        text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)
    # Spoken word digits
    words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'zero': '0',
    }
    for word, digit in words.items():
        text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)
    # Collapse comma-separated single digits: "1, 4, 5, 2" -> "1452"
    # Collapse comma-separated single digits but NOT when followed by thousand/hundred
    # (those are sky condition altitudes, not single numbers)
    _no_mag = r'(?!\s*,?\s*(?:thousand|hundred))'
    # Space-digit collapse first so '3 0, 2, 3' -> '30, 2, 3' before comma collapse
    text = re.sub(r'\b(\d)(?: (\d))+\b', lambda m: m.group(0).replace(' ', ''), text)
    # Period-separated digits: '3. 0. 2. 5.' -> '3025.' (KPYN-style broadcasts)
    for _ in range(6):
        t2 = re.sub(r'(\d)\.\s*(\d)', r'\1\2', text)
        if t2 == text:
            break
        text = t2
    # Comma-collapse: allow 1-4 digit prefix so '30, 2, 3' -> '3023'
    text = re.sub(r'\b(\d{1,3}),\s*(\d),\s*(\d),\s*(\d)\b' + _no_mag, r'\1\2\3\4', text)
    text = re.sub(r'\b(\d{1,3}),\s*(\d),\s*(\d)\b'           + _no_mag, r'\1\2\3',   text)
    text = re.sub(r'\b(\d{1,3}),\s*(\d)\b'                    + _no_mag, r'\1\2',     text)
    # Second pass: period-digit collapse may have created 2-digit groups
    # e.g. '29, 96' (from 'two, niner, niner. Six') -> '2996'
    text = re.sub(r'\b(\d{1,2}),\s*(\d{2})\b'                 + _no_mag, r'\1\2',     text)
    # Collapse compound tens: "20-1" -> "21", "20 1" -> "21"
    text = re.sub(r'\b(2[0-9])-([1-9])\b',
                  lambda m: str(int(m.group(1)) + int(m.group(2))), text)
    text = re.sub(r'\b(2[0-9]) ([1-9])\b',
                  lambda m: str(int(m.group(1)) + int(m.group(2))), text)
    # Convert spoken magnitudes: "N thousand" -> N*1000, "N hundred" -> N*100
    text = re.sub(r'(\d+),?\s*thousand', lambda m: str(int(m.group(1)) * 1000), text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+),?\s*hundred',  lambda m: str(int(m.group(1)) * 100),  text, flags=re.IGNORECASE)
    # Combine adjacent thousand+hundred: "1000 100" -> "1100"
    text = re.sub(r'\b(\d+)000 (\d+)00\b',
                  lambda m: str(int(m.group(1)) * 1000 + int(m.group(2)) * 100), text)
    # Collapse thousands-separator variants: '12, 000' / '12-000' -> '12000'
    text = re.sub(r'\b(\d{1,2}),\s*000\b', r'\g<1>000', text)
    text = re.sub(r'\b(\d{1,2})-000\b',    r'\g<1>000', text)
    # Spoken decimal point: '128 point 45' -> '128.45' (only between digits)
    text = re.sub(r'(?<=\d)\s+point\s+(?=\d)', '.', text, flags=re.IGNORECASE)
    # Zero-pad 3-digit times before 'local time': '700' -> '0700'
    text = re.sub(r'\b(\d{3})\b(?=\s+local\s+time)',
                  lambda m: m.group(1).zfill(4), text)
    return text


# ---------------------------------------------------------------------------
# Preamble stripping
# ---------------------------------------------------------------------------

def strip_preamble(text):
    """
    Find the last complete broadcast in a looped recording.
    Returns (segment_text, obs_time_4digit) so callers know which loop was selected.
    The obs_time_4digit is used by audio_trim.py to trim the matching loop.

    Prefers the last loop that contains BOTH altimeter and visibility — this
    avoids selecting Whisper-truncated final loops that dropped mid-broadcast
    (e.g. KMAA last loop had wind+altimeter but no visibility keyword).
    Falls back to altimeter-only if no fully complete loop is found.
    """
    pattern = r'[Aa]utomated weather observation[.\s,]+(\d{4})[.\s,]*[Zz]ulu(?:[Ww]eather)?'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text, None
    # First pass: prefer loops with both altimeter and visibility
    for m in reversed(matches):
        segment = text[m.start():]
        if (re.search(r'\baltimeter\b', segment, re.IGNORECASE) and
                re.search(r'\bvisibility\b', segment, re.IGNORECASE)):
            return segment, m.group(1)
    # Second pass: fall back to altimeter-only
    for m in reversed(matches):
        segment = text[m.start():]
        if re.search(r'\baltimeter\b', segment, re.IGNORECASE):
            return segment, m.group(1)
    return text[matches[-1].start():], matches[-1].group(1)


# ---------------------------------------------------------------------------
# Airport name extractor
# ---------------------------------------------------------------------------


def extract_time(text):
    matches = re.findall(r'\b(\d{4})[,.\s]*[Zz]ulu', text)
    return matches[-1] + 'Z' if matches else 'N/A'


def extract_wind(text, full_text=None):
    """Extract wind. full_text (full normalized transcript) used to find
    gusts and variable ranges that may only appear in earlier broadcast loops."""
    ft = full_text or text  # fallback to segment if full text not provided
    if re.search(r'\bwind[\s.,]+(?:missing|information[\s.,]+not[\s.,]+available)\b',
                 text, re.IGNORECASE):
        return 'Missing', 'MIS'
    if re.search(r'\bwind[\s.,]+calm\b', text, re.IGNORECASE):
        return 'Calm', '00000KT'
    # Variable direction with speed: 'wind variable at N'
    mv = re.search(r'\bwind[\s.,]+variable[\s.,]+(?:at[\s.,]+)?(\d+)',
                   text, re.IGNORECASE)
    if mv:
        spd = mv.group(1).zfill(2)
        return f"Variable at {int(spd)} kts", f"VRB{spd}KT"
    # Directional wind
    m = re.search(r'\bwind[\s.,]+(\d{1,3})[\s.,]+(?:at[\s.,]+)?(\d+)',
                  text, re.IGNORECASE)
    if m:
        dir_ = m.group(1).zfill(3)
        spd  = m.group(2).zfill(2)
        # Check for gusts in full text (may be in an earlier broadcast loop)
        mg = re.search(
            r'\bwind[\s.,]+' + dir_ + r'[\s.,]+(?:at[\s.,]+)?\d+[\s.,]+gusts?[\s.,]+(\d+)',
            ft, re.IGNORECASE)
        if not mg:
            mg = re.search(r'\bpeak[\s.,]+gust(?:s)?[\s.,]+(\d+)', ft, re.IGNORECASE)
        gust_part = ''
        gust_metar = ''
        if mg:
            g = mg.group(1).zfill(2)
            gust_part  = f", gusts {int(g)} kts"
            gust_metar = f"G{g}"
        # Check for variable wind range in full text: 'variable between X and Y'
        # Exclude 'visibility variable between' - that's a visibility remark not wind
        mvr = None
        for _mvr_m in re.finditer(
                r'variable[\s,]+between[\s,]+(\d+)[\s,]+and[\s,]+(\d+)', ft, re.IGNORECASE):
            _before = ft[max(0, _mvr_m.start()-25):_mvr_m.start()].lower()
            if 'visibility' not in _before:
                mvr = _mvr_m
                break
        var_part  = ''
        var_metar = ''
        if mvr:
            lo = mvr.group(1).zfill(3)
            hi = mvr.group(2).zfill(3)
            var_part  = f", variable {lo}-{hi}"
            var_metar = f" {lo}V{hi}"
        disp = f"{dir_}\u00b0 at {int(spd)} kts{gust_part}{var_part}"
        metar = f"{dir_}{spd}{gust_metar}KT{var_metar}"
        return disp, metar
    return 'N/A', 'N/A'


def extract_visibility(text):
    # Strip commas used as separators in some AWOS transcripts (e.g. KLBO comma-delimited)
    # so fractional patterns like '2 and 1 half' match across comma-separated tokens.
    text = re.sub(r',\s*', ' ', text)
    # Convert fractional visibility phrases (must run on raw/normalized text
    # before pattern matching, but AFTER normalize to avoid period-collapse)
    # Pre-convert spoken whole numbers adjacent to fractions so downstream
    # lambda patterns (which use \d+) can match them.
    _word_to_digit = [
        (r'\bone\b', '1'), (r'\btwo\b', '2'), (r'\bthree\b', '3'),
        (r'\bfour\b', '4'), (r'\bfive\b', '5'), (r'\bsix\b', '6'),
        (r'\bseven\b', '7'), (r'\beight\b', '8'), (r'\bnine\b', '9'),
    ]
    # Only apply digit substitution inside visibility phrases to avoid
    # clobbering sky/wind/altimeter parsing that runs on the same text.
    # Converts all spoken digit-words within a visibility phrase to digits,
    # then concatenates them so "one zero" -> "10", "one zero two seven" -> "1027".
    def _vis_word_sub(t):
        # Apply single-word substitutions globally within visibility context
        for _wp, _wd in _word_to_digit:
            t = re.sub(_wp + r'(?=\s+and\s+(?:one\s+half|half|three\s+quarter|quarter))', _wd, t, flags=re.IGNORECASE)

        def _collapse_vis_digits(m):
            """
            Replace a run of spoken digits after 'visibility [more than]' with
            their concatenated numeric form.
            e.g. 'one zero'        -> '10'
                 'one zero two seven' -> '1027'
            Stops at the first non-digit word.
            """
            prefix = m.group(1)   # 'visibility ' or 'visibility more than '
            phrase = m.group(2)   # the digit-word run
            _map = {'zero':'0','one':'1','two':'2','three':'3','four':'4',
                    'five':'5','six':'6','seven':'7','eight':'8','nine':'9','niner':'9'}
            tokens = phrase.strip().split()
            digits = []
            for tok in tokens:
                d = _map.get(tok.lower().strip('.,'))
                if d is not None:
                    digits.append(d)
                else:
                    break
            if digits:
                return prefix + ''.join(digits)
            return m.group(0)

        # Match visibility phrase followed by a run of digit-words
        _digit_words = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)'
        t = re.sub(
            r'(visibility[\s.,]+(?:more[\s.,]+than[\s.,]+)?)'
            r'((?:' + _digit_words + r'[\s.,]+)+' + _digit_words + r'|' + _digit_words + r')',
            _collapse_vis_digits, t, flags=re.IGNORECASE)
        return t
    text = _vis_word_sub(text)
    # Normalize comma-separated tokens (Whisper artifact on some stations)
    # e.g. 'visibility, 2, and, 1, half' -> 'visibility 2 and 1 half'
    text = text.replace(',', ' ')
    _frac_vis = [
        # Combined forms MUST come first so 'one half' isn't consumed before
        # the whole phrase 'X and one half' can match.
        (r'\b(\d+)\s+and\s+(?:one\s+half|1\s+half|half)\b',
         lambda m: str(int(m.group(1)) + 0.5)),
        (r'\b(\d+)\s+and\s+(?:three\s+quarters?|3\s+quarters?)\b',
         lambda m: str(int(m.group(1)) + 0.75)),
        (r'\b(\d+)\s+and\s+(?:one\s+quarter|1\s+quarter|quarter)\b',
         lambda m: str(int(m.group(1)) + 0.25)),
        # Bare fractions (only after combined forms so they don't interfere)
        (r'\b(?:three|3)\s+quarters?\b', '0.75'),
        (r'\b(?:one|1)\s+half\b',        '0.5'),
        (r'\b(?:one|1)\s+quarter\b',     '0.25'),
    ]
    for _pat, _val in _frac_vis:
        text = re.sub(_pat, _val, text, flags=re.IGNORECASE)
    # Match decimal values (e.g. 0.75 from 'three quarters') and integers
    # Search ALL visibility mentions; use the first valid numeric one
    for m in re.finditer(r'visibility[\s.,]+(more[\s.,]+than[\s.,]+)?(\d+\.?\d*)',
                         text, re.IGNORECASE):
        prefix  = '>' if m.group(1) else ''
        val_str = m.group(2)
        val     = float(val_str)
        if val > 9999:  # sky-altitude bled into visibility field
            continue
        # Standard AWOS max visibility is 10 SM; values > 10 are garbled —
        # applies whether or not a 'more than' prefix was present.
        # e.g. 'more than one zero two seven' -> >1027 should be capped to >10 SM
        if val > 10:
            digits = str(m.group(2))
            # Two-digit prefix starts with 10 (e.g. '1027' or '107' -> 10 SM)
            if len(digits) >= 2 and int(digits[:2]) == 10:
                return f"{prefix}10 SM"
            if not prefix:
                # Fall back to first digit for non-prefixed garble (e.g. '71' -> 7 SM)
                _fd = int(digits[0])
                if 1 <= _fd <= 9:
                    return f"{_fd} SM"
            continue
        _frac = {0.25: '1/4', 0.5: '1/2', 0.75: '3/4'}
        if val in _frac:
            return f"{prefix}{_frac[val]} SM"
        return f"{prefix}{int(val) if val == int(val) else val} SM"
    # No numeric visibility found — check for explicit missing keyword
    if re.search(r'visibility[\s.,]+(?:missing|information[\s.,]+not[\s.,]+available)\b',
                 text, re.IGNORECASE):
        return 'Missing'
    return 'N/A'


def extract_sky(text):
    """
    Find ALL sky condition layers in the text and return them sorted by
    altitude (lowest first), matching AWOS broadcast order.

    Handles:
      - Multiple BKN/SCT/OVC/FEW layers in a single broadcast
      - 'broken at NNNN' as well as 'NNNN broken' / 'broken NNNN'
      - 'ceiling NNNN broken/overcast' ordering
      - Whisper-split altitudes: 'Ceiling 1000 9. Hundred.' -> 1900 ft
      - 'NNNN. Overcast.' (altitude sentence, coverage sentence) -> OVC at NNNN
      - 'Overcast. NNNN.' (coverage then altitude) -> OVC at NNNN
      - Truly bare 'overcast' with no adjacent altitude -> OVC000
      - Altitude deduplication (same altitude keeps highest coverage type)
      - CLR, SKC, Missing, VV, N/A fallbacks
    """
    seen_alts = {}   # alt_ft -> (metar_code, disp)

    def _add(alt_str, cover):
        alt_str = re.sub(r'[\s,]+', '', str(alt_str))
        try:
            alt = int(alt_str)
        except ValueError:
            return
        # Discard implausibly low altitudes (<100ft) — almost always a Whisper
        # truncation artifact (e.g. "ceiling 11" instead of "ceiling 11000").
        # VV (vertical visibility) can legitimately be near 0 so skip this check there.
        if cover != 'VV' and alt < 100 and alt != 0:
            return
        if alt > 99900 or alt < 0:
            return
        metar = f"{cover}{alt // 100:03d}"
        disp  = f"{cover} {alt:,} ft"
        _priority = {'OVC': 4, 'BKN': 3, 'SCT': 2, 'FEW': 1}
        existing = seen_alts.get(alt)
        if existing is None or _priority.get(cover, 0) > _priority.get(existing[0][:3], 0):
            seen_alts[alt] = (metar, disp)

    # ── Whisper-split ceiling: 'Ceiling 1000 9. Hundred.' -> 1900 ft ─────────
    # Whisper sometimes breaks 'one thousand nine hundred' into '1000 9. Hundred.'
    for m in re.finditer(
            r'ceiling\s+(\d+)\s+(\d)\s*[.,]\s*hundred', text, re.IGNORECASE):
        alt = int(m.group(1)) + int(m.group(2)) * 100
        _add(str(alt), 'BKN')   # ceiling implies BKN; overcast match below will upgrade

    # ── Overcast: 'ceiling NNNN overcast', 'overcast NNNN', 'overcast at NNNN' ─
    for m in re.finditer(
            r'ceiling[\s.,]+(\d[\d,]+)[\s.,]+overcast', text, re.IGNORECASE):
        _add(m.group(1), 'OVC')
    for m in re.finditer(
            r'(?:sky\s+condition\s+)?overcast[\s.,]+(?:at\s+)?(\d[\d,]+)',
            text, re.IGNORECASE):
        _add(m.group(1), 'OVC')

    # ── Broken: 'ceiling NNNN broken', 'NNNN broken', 'broken NNNN/at NNNN' ──
    for m in re.finditer(
            r'ceiling[\s.,]+(\d[\d,]+)[\s.,]+broken', text, re.IGNORECASE):
        _add(m.group(1), 'BKN')
    for m in re.finditer(r'(\d[\d,]+)\s+broken', text, re.IGNORECASE):
        _add(m.group(1), 'BKN')
    for m in re.finditer(r'broken\s+(?:at\s+)?(\d[\d,]+)', text, re.IGNORECASE):
        _add(m.group(1), 'BKN')

    # ── Scattered: 'scattered NNNN', 'scattered at NNNN', 'NNNN scattered' ───
    for m in re.finditer(r'scattered[\s.,]+(?:at\s+)?(\d[\d,]+)', text, re.IGNORECASE):
        _add(m.group(1), 'SCT')
    for m in re.finditer(r'(\d[\d,]+)[\s.,]+scattered', text, re.IGNORECASE):
        pre = text[max(0, m.start() - 20):m.start()].lower()
        if 'haze' not in pre and 'wind' not in pre and 'altitude' not in pre:
            _add(m.group(1), 'SCT')

    # ── Few ────────────────────────────────────────────────────────────────────
    for m in re.finditer(r'few\s+(?:at\s+)?(\d[\d,]+)', text, re.IGNORECASE):
        _add(m.group(1), 'FEW')

    # ── Indefinite ceiling / VV ────────────────────────────────────────────────
    m = re.search(r'indefinite\s+ceiling[\s.,]+(\d+)', text, re.IGNORECASE)
    if m:
        alt = int(m.group(1))
        seen_alts[alt] = (f"VV{alt // 100:03d}", f"VV {alt:,} ft")

    # ── Handle bare 'overcast' with no digit immediately following ─────────────
    # Three sub-cases:
    #   (a) digit precedes within 30 chars (not after temp/dewpoint/altimeter):
    #       "Broken 3900. Overcast." -> OVC at 3900
    #   (b) digit follows past punctuation: "Overcast. 900." -> OVC at 900
    #   (c) neither: truly bare -> OVC000 (surface obscuration)
    _NONSKY_KW = re.compile(r'temperature|dewpoint|altimeter|celcius|celsius', re.IGNORECASE)
    for ovc_m in re.finditer(r'\bovercast\b', text, re.IGNORECASE):
        post_imm = text[ovc_m.end():ovc_m.end() + 15]
        if re.search(r'^\s*\d', post_imm):
            continue  # already handled by main OVC patterns above
        pre30 = text[max(0, ovc_m.start() - 30):ovc_m.start()]
        pre_digit = re.search(r'(\d[\d,]+)\s*[.,\s]*$', pre30)
        if pre_digit and not _NONSKY_KW.search(pre30):
            alt_val = int(re.sub(r'[\s,]+', '', pre_digit.group(1)))
            # Skip altimeter-range values (2800-3100 as raw 4-digit strings)
            if not (2800 <= alt_val <= 3150):
                _add(str(alt_val), 'OVC')
                continue
        post_far = text[ovc_m.end():ovc_m.end() + 20]
        post_digit = re.search(r'^[\s.,]+(\d[\d,]+)', post_far)
        if post_digit:
            _add(post_digit.group(1), 'OVC')
            continue
        # Truly bare overcast — only add OVC000 if no OVC layer found yet
        if not any(v[0].startswith('OVC') for v in seen_alts.values()):
            seen_alts[0] = ('OVC000', 'OVC (surface)')

    # ── Bare 'scattered' with no altitude and nothing else found ───────────────
    if re.search(r'\bscattered\b', text, re.IGNORECASE) and not seen_alts:
        seen_alts[-1] = ('SCT', 'SCT')

    # ── Return layers sorted lowest to highest ────────────────────────────────
    if seen_alts:
        layers = sorted(seen_alts.items())
        metar_codes = ' '.join(v[0] for _, v in layers)
        disp_parts  = ' / '.join(v[1] for _, v in layers)
        return metar_codes, disp_parts

    # ── CLR / SKC ─────────────────────────────────────────────────────────────
    m = re.search(
        r'\b(?:sky\s+condition[\s,.]+clear|clear)[\s,.]+below[\s,.]+(\d[\d\s,.]*\d|\d)',
        text, re.IGNORECASE)
    if m:
        alt = re.sub(r'[\s,.]+', '', m.group(1).rstrip(', .'))
        if alt.isdigit() and int(alt) <= 12000:
            return 'CLR', f'CLR (below {int(alt):,} ft)'
    if re.search(r'\b(sky\s+condition[\s,]+clear|clear[\s,]+below|clr\b|skc\b)',
                 text, re.IGNORECASE):
        return 'CLR', 'CLR'

    # ── Missing sensor ─────────────────────────────────────────────────────────
    if re.search(r'\b(?:sky\s+condition|ceiling)[\s.,]+missing\b', text, re.IGNORECASE):
        return 'M', 'Missing'

    return 'N/A', 'N/A'



def extract_temp_dp(text):
    # Handles: optional 'minus' prefix, 'celcius' (Whisper) and 'celsius',
    # 'dewpoint' (no space) and 'dew point', flexible separators,
    # optional celsius keyword (some transcripts omit it without punctuation),
    # and abbreviated 'temp' as well as full 'temperature'.
    SEP  = r'[\s,.]+'
    TVAL = r'(minus[\s,.]+)?([.\d]+)'
    CEL  = r'(?:[\s,.]+[Cc]el[sc]ius)?' # optional celsius
    for dp_kw in [r'dew[\s,.]*point', r'dewpoint']:
        pat = r'temp(?:erature)?' + SEP + TVAL + CEL + SEP + dp_kw + SEP + TVAL
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            t_neg, t_val, d_neg, d_val = m.group(1), m.group(2), m.group(3), m.group(4)
            t = ('-' if t_neg else '') + t_val.rstrip('.')
            d = ('-' if d_neg else '') + d_val.rstrip('.')
            def fix_decimal(v):
                bare = v.lstrip('-')
                if len(bare) == 2 and bare.endswith('0') and float(bare) > 35:
                    return v[:-1] + '.' + v[-1]
                return v
            t = fix_decimal(t)
            d = fix_decimal(d)
            tf, df = float(t), float(d)
            t_metar = f"M{abs(int(tf)):02d}" if tf < 0 else f"{int(tf):02d}"
            d_metar = f"M{abs(int(df)):02d}" if df < 0 else f"{int(df):02d}"
            return f"{t}\u00b0C / {d}\u00b0C", f"{t_metar}/{d_metar}"
    # Fallback: temperature without dewpoint
    pat_t = r'temp(?:erature)?' + SEP + TVAL + CEL
    m = re.search(pat_t, text, re.IGNORECASE)
    if m:
        t_neg, t_val = m.group(1), m.group(2)
        t = ('-' if t_neg else '') + t_val.rstrip('.')
        tf = float(t)
        t_metar = f"M{abs(int(tf)):02d}" if tf < 0 else f"{int(tf):02d}"
        return f"{t}\u00b0C / N/A", f"{t_metar}/"
    if re.search(r'temperature[\s.,]+(?:missing|information[\s.,]+not[\s.,]+available)\b',
                 text, re.IGNORECASE):
        return 'Missing', 'MIS/MIS'
    return 'N/A', 'N/A'

def extract_altimeter(text):
    # Special case: "2, 9er, 9er, 9er" spoken altimeter
    if re.search(r'altimeter\s+2\s*,\s*9', text, re.IGNORECASE):
        return "29.99 inHg", "A2999"
    # Collect altimeter readings — find all 'altimeter NNNN' occurrences
    # Use the FIRST valid 4-digit value that starts with 2 or 3 (realistic range)
    # This avoids KFAM-style glitches where repeated digits follow the real value
    if re.search(r'altimeter[\s.,]+(?:missing|information[\s.,]+not[\s.,]+available)\b',
                 text, re.IGNORECASE):
        return 'Missing', 'AMIS'
    for raw in re.findall(r'altimeter[\s.,]+(\d+(?:\.\d+)?)', text, re.IGNORECASE):
        val = raw.replace('.', '')
        # Truncate to 4 digits — the 5th digit is from adjacent text (e.g. local info)
        if len(val) > 4 and val[0] in ('2', '3'):
            val = val[:4]
        if len(val) == 4 and val[0] in ('2', '3'):
            return f"{val[0]}{val[1]}.{val[2]}{val[3]} inHg", f"A{val}"
    return 'N/A', 'N/A'


# ---------------------------------------------------------------------------
# Weather phenomena extractor
# ---------------------------------------------------------------------------

# Order matters: more specific (freezing) variants before bare ones
PHENOMENA = [
    ('Thunderstorm',     'TS',   r'\bthunderstorm\b(?![\s.,]+information)'),
    ('Freezing Rain',    'FZRA', r'\bfreezing\s+rain\b'),
    ('Freezing Drizzle', 'FZDZ', r'\bfreezing\s+drizzle\b'),
    ('Freezing Fog',     'FZFG', r'\bfreezing\s+fog\b'),
    ('Rain',             'RA',   r'\brain\b'),
    ('Drizzle',          'DZ',   r'\bdrizzle\b'),
    ('Snow',             'SN',   r'\bsnow\b(?!\s+grains)'),
    ('Snow Grains',      'SG',   r'\bsnow\s+grains\b'),
    ('Ice Pellets',      'PL',   r'\bice\s+pellets\b'),
    ('Ice Crystals',     'IC',   r'\bice\s+crystals\b'),
    ('Hail',             'GR',   r'\bhail\b'),
    ('Small Hail',       'GS',   r'\bsmall\s+hail\b'),
    ('Fog',              'FG',   r'\bfog\b'),
    ('Mist',             'BR',   r'\bmist\b'),
    ('Haze',             'HZ',   r'\bhaze\b'),
    ('Unknown Precip',   'UP',   r'\bunknown\s+precipitation\b'),
    ('Squall',           'SQ',   r'\bsquall\b'),
    ('Funnel Cloud',     'FC',   r'\bfunnel\s+cloud\b'),
    ('Tornado',          'FC+',  r'\btornado\b|\bwaterspout\b'),
    ('Volcanic Ash',     'VA',   r'\bvolcanic\s+ash\b'),
    ('Blowing Snow',     'BLSN', r'\bblowing\s+snow\b'),
    ('Blowing Dust',     'BLDU', r'\bblowing\s+dust\b'),
    ('Blowing Sand',     'BLSA', r'\bblowing\s+sand\b'),
    ('Smoke',            'FU',   r'\bsmoke\b'),
    ('Dust',             'DU',   r'\bdust\b(?!\s+\d)'),
    ('Sand',             'SA',   r'\bsand\b'),
    ('Dust/Sand Storm',  'SS',   r'\b(?:dust|sand)\s+storm\b'),
]

# Suppress less-specific code if more-specific variant already matched
SUPPRESS_IF_PARENT = {
    'FG': ['FZFG'],
    'RA': ['FZRA'],
    'DZ': ['FZDZ'],
    'SN': ['BLSN'],
    'GR': ['GS'],
}


def extract_phenomena(text):
    """Return list of (display_name, metar_code) for present weather phenomena.
    Captures light (-) and heavy (+) intensity prefixes when present.
    """
    found_codes = set()
    found = []
    text_lower = text.lower()
    for display, code, pattern in PHENOMENA:
        m = re.search(pattern, text_lower)
        if m:
            # Check up to 20 chars before the match for light/heavy qualifier
            pre = text_lower[max(0, m.start() - 20):m.start()]
            if re.search(r'\bheavy\b', pre):
                intensity = '+'
                disp_prefix = 'Heavy '
            elif re.search(r'\blight\b', pre):
                intensity = '-'
                disp_prefix = 'Light '
            else:
                intensity = ''
                disp_prefix = ''
            full_code = intensity + code
            full_disp = disp_prefix + display
            found_codes.add(code)  # use base code for suppression logic
            found.append((full_disp, full_code))
    # Drop less-specific codes superseded by more specific ones
    return [
        (disp, code) for disp, code in found
        if not any(s in found_codes for s in SUPPRESS_IF_PARENT.get(code.lstrip('+-'), []))
    ]



def extract_remarks(text):
    remarks = []
    m = re.search(r'density[\s.,]+alt(?:itude)?[\s.,]+(minus[\s.,]+)?(\d[\d,]+)', text, re.IGNORECASE)
    if m:
        sign = '-' if m.group(1) else ''
        alt  = m.group(2).replace(',', '')
        remarks.append(f"Density Alt {sign}{int(alt):,} ft")
    if re.search(r'thunderstorm.*?(?:information\s+)?not\s+available', text, re.IGNORECASE):
        remarks.append('TSNO')
    if re.search(r'lightning\s+missing', text, re.IGNORECASE):
        remarks.append('Lightning sensor missing')
    # Lightning observed with distance/direction
    # e.g. 'Lightning distance east through south' -> 'Lightning E-S'
    # e.g. 'Lightning distant' -> 'Lightning Distant'
    _DIR_MAP = {
        'north': 'N', 'south': 'S', 'east': 'E', 'west': 'W',
        'northeast': 'NE', 'northwest': 'NW',
        'southeast': 'SE', 'southwest': 'SW',
    }
    _ltg = re.search(
        r'lightning[\s.,]+(?:distance[\s.,]+|distant[\s.,]+|observed[\s.,]+)?(.*?)'
        r'(?:\.|$|temperature|dewpoint|altimeter|remarks|density)',
        text, re.IGNORECASE)
    if _ltg and not re.search(r'lightning\s+(?:missing|sensor|information)', text, re.IGNORECASE):
        _raw_dir = _ltg.group(1).strip().rstrip('., ')
        if _raw_dir:
            # Replace direction words with abbreviations
            _parts = re.split(r'[\s,]+(?:through|and|to)[\s,]+|[\s,]+', _raw_dir)
            _dirs = [_DIR_MAP.get(p.lower(), p) for p in _parts if p]
            _dir_str = '-'.join(_dirs) if _dirs else _raw_dir
            remarks.append(f'Lightning {_dir_str}')
        else:
            remarks.append('Lightning observed')
    # Ceiling variable between X and Y
    mvc = re.search(
        r'ceiling\s+variable\s+between\s+(\d+)\s+and\s+(\d+)',
        text, re.IGNORECASE)
    if mvc:
        lo = int(mvc.group(1))
        hi = int(mvc.group(2))
        remarks.append(f"CIG variable {lo:,}-{hi:,} ft")
    # Wind variable range already captured in wind field; omit from remarks
    return ', '.join(remarks) if remarks else 'AO2'


# ---------------------------------------------------------------------------
# Local / ATIS information extractor
# ---------------------------------------------------------------------------

def extract_local_info(raw_text):
    """
    Extract supplemental local information that some stations broadcast after
    the weather observation (tower hours, frequencies, fuel availability, NOTAMs).
    Operates on the raw (un-normalised) transcript.
    """
    pattern = r'[Aa]utomated weather observation[.\s,]+(\d{4})[.\s,]*[Zz]ulu(?:[Ww]eather)?'
    segments = re.split(pattern, raw_text)
    broadcast_bodies = segments[2::2]
    if not broadcast_bodies:
        return None
    # Use second-to-last body — local info repeats with each cycle
    body = broadcast_bodies[-2] if len(broadcast_bodies) >= 2 else broadcast_bodies[-1]

    # Split into sentences, avoiding breaks on common abbreviations
    sentences = re.split(
        r'(?<![A-Z][a-z])(?<![Ss]t)(?<![Dd]r)(?<!a\.m)(?<!p\.m)\.\s+(?=[A-Z0-9])',
        body
    )

    local_triggers = [
        r'tower\s+(?:is\s+)?(?:of\s+)?(?:hours|operation)',
        r'common\s+traffic\s+advis',
        r'pilot.operated', r'pilot\s+operated',
        r'approach\s+control',
        r'avgas', r'self-serve', r'full\s+service\s+100',
        r'call\s+before\s+landing', r'\d{3}-\d{3}-\d{4}',
        r'contact\s+\w+.*\s+(?:approach|control|center)',
        r'IFR\s+clearance', r'for\s+additional\s+information',
        r'frequency\s+(?:is\s+)?\d', r'on\s+frequency\s+\d',
        r'\d{3,4}\s+local\s+time',
        r'frequency\s+for\s+(?:automated|weather)',
    ]
    combined    = '|'.join(local_triggers)
    airport_pat = re.compile(r'^[A-Z][\w\s\.\-]+(?:Airport|Field|Center)\s*[,\.]*\s*$')

    local_sentences = []
    in_local = False
    for sent in sentences:
        if re.search(combined, sent, re.IGNORECASE):
            in_local = True
        if in_local:
            if re.search(r'automated\s+weather\s+observation[.\s,]+\d{4}',
                         sent, re.IGNORECASE):
                break
            stripped = sent.strip()
            if stripped and not airport_pat.match(stripped):
                local_sentences.append(stripped)

    if not local_sentences:
        return None

    # Join, then strip any trailing bare station-name fragment
    text = '. '.join(local_sentences).rstrip('.,').strip()
    parts = re.split(r'(?<!\b[A-Z][a-z])\.\s+', text)
    while parts:
        last = parts[-1].strip().rstrip('.,').strip()
        if (re.search(r'(?:Airport|Field|Center|Airpark)\s*$', last)
                and not re.search(r'\d', last)):
            parts.pop()
        else:
            break
    result = '. '.join(parts) + '.' if parts else None
    # Strip leading altimeter fragment but keep any overrun digit that belongs
    # to the local info: 'Altimeter 30332 pumps...' -> '2 pumps...'
    if result:
        result = re.sub(r'^[Aa]ltimeter[\s.,]+\d{4}(\d*[\s.,]+)', r'\1', result).strip()
    return result if result else None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def _truncate_digit_storm(text, min_run=8):
    """
    Detect and truncate Whisper digit-storm hallucinations - runs of
    comma-separated single spoken digits that appear after real content.

    e.g. KLBO: '...Altimeter two niner two inches of mercury.Niner, two,
    niner, three, zero, one, zero, one, zero, zero, zero, zero...'

    Truncates at the start of the first run of min_run+ consecutive
    comma-separated spoken digit-words. Returns the truncated text.
    """
    import re as _re
    digit_word = "(?:zero|one|two|three|four|five|six|seven|eight|niner|nine)"
    sep        = "[, ]+"
    pattern    = "(?:" + digit_word + sep + "){" + str(min_run) + ",}" + digit_word
    m = _re.search(pattern, text, _re.IGNORECASE)
    if m:
        return text[:m.start()].rstrip(' ,.')
    return text


def parse_transcript(t):
    raw       = t['raw_transcript']
    raw       = _truncate_digit_storm(raw)
    norm_full = normalize(raw)
    norm, selected_loop_time = strip_preamble(norm_full)

    station  = t['station']
    location = t['location']
    wx_type  = t.get('type', '')

    # Extract UTC day-of-month from the recording's date_created timestamp
    date_created = t.get('date_created', '')
    try:
        rec_day = datetime.fromisoformat(date_created).strftime('%d')
    except (ValueError, TypeError):
        rec_day = '??'

    time_str              = extract_time(norm)
    wind_disp, wind_metar = extract_wind(norm, norm_full)
    # If wind is N/A for any reason (garbled or missing), search full transcript
    if wind_disp == 'N/A':
        _wm = re.search(r'\bwind[\s.,]+(\d{1,3})[\s.,]+(?:at[\s.,]+)?(\d+)', norm_full,
                        re.IGNORECASE)
        if _wm:
            _dir = _wm.group(1).zfill(3)
            _spd = _wm.group(2).zfill(2)
            # Also look for gusts and variable range in full text
            _full_disp, _full_metar = extract_wind(norm_full, norm_full)
            if _full_disp not in ('N/A', 'Missing'):
                wind_disp, wind_metar = _full_disp, _full_metar
            else:
                wind_disp  = f"{_dir}\u00b0 at {int(_spd)} kts"
                wind_metar = f"{_dir}{_spd}KT"
    vis                   = extract_visibility(norm)
    # Fall back to full transcript if last loop missing or has invalid visibility
    # Standard AWOS visibilities are <=10 SM or >10; values like 16 are garbled
    def _vis_invalid(v):
        if v in ('N/A', 'Missing'):
            return True
        import re as _re
        m = _re.match(r'>?([\d.]+)', v)
        if m:
            n = float(m.group(1))
            return n > 10 and not v.startswith('>')
        return False
    if _vis_invalid(vis):
        _full_vis = extract_visibility(norm_full)
        if not _vis_invalid(_full_vis):
            vis = _full_vis
    sky_metar, sky_disp   = extract_sky(norm)
    # Fall back to full transcript if last loop is missing sky
    # Also fall back when sky is bare 'CLR' with no altitude (garbled last loop)
    if sky_metar in ('N/A', 'CLR'):
        _full_sky_metar, _full_sky_disp = extract_sky(norm_full)
        # Upgrade if: N/A -> anything, or bare CLR -> CLR with qualifier, or CLR -> cloud layer
        _upgrade = (
            sky_metar == 'N/A'
            or _full_sky_metar not in ('N/A', 'CLR')
            or ('(' in _full_sky_disp and '(' not in sky_disp)
        )
        if _upgrade:
            sky_metar, sky_disp = _full_sky_metar, _full_sky_disp
    temp_disp, temp_metar = extract_temp_dp(norm)
    # Fall back if temp entirely missing or contains implausible values
    def _temp_implausible(disp):
        if disp == 'N/A':
            return True
        import re as _re
        vals = _re.findall(r'-?[\d.]+', disp)
        return any(abs(float(v)) > 60 for v in vals if v not in ('.', '-'))
    if _temp_implausible(temp_disp):
        _full_temp, _full_metar = extract_temp_dp(norm_full)
        if not _temp_implausible(_full_temp):
            temp_disp, temp_metar = _full_temp, _full_metar
    # If dewpoint missing from last broadcast, search full transcript as fallback
    if temp_disp.endswith('/ N/A'):
        import re as _re
        SEP  = r'[\s,.]+'
        TVAL = r'(minus[\s,.]+)?([.\d]+)'
        CEL  = r'[Cc]el[sc]ius'
        dp_pat = r'dew[\s,.]*point' + SEP + TVAL + SEP + CEL
        dp_m = _re.search(dp_pat, norm_full, _re.IGNORECASE)
        if not dp_m:
            dp_pat2 = r'dewpoint' + SEP + TVAL
            dp_m = _re.search(dp_pat2, norm_full, _re.IGNORECASE)
        if dp_m:
            d_neg = dp_m.group(1)
            d_val = dp_m.group(2).rstrip('.')
            d = ('-' if d_neg else '') + d_val
            df = float(d)
            d_metar = f'M{abs(int(df)):02d}' if df < 0 else f'{int(df):02d}'
            # Rebuild temp_disp and temp_metar with the found dewpoint
            t_part = temp_disp.split(' / ')[0]
            t_metar_part = temp_metar.rstrip('/')
            temp_disp  = f"{t_part} / {d}\u00b0C"
            temp_metar = f"{t_metar_part}/{d_metar}"
    alt_disp,  alt_metar  = extract_altimeter(norm)
    # Fall back to full transcript if last loop is missing altimeter
    if alt_disp == 'N/A':
        alt_disp, alt_metar = extract_altimeter(norm_full)
    remarks               = extract_remarks(norm)
    # Fall back to full transcript if last loop had no remarks
    if not remarks or remarks == 'AO2':
        _full_remarks = extract_remarks(norm_full)
        if _full_remarks and _full_remarks != 'AO2':
            remarks = _full_remarks
    phenomena             = extract_phenomena(norm)
    local_info            = extract_local_info(norm_full)

    vis_metar  = vis.replace(' SM', 'SM').replace('>', '')
    wx_metar   = ' '.join(code for _, code in phenomena)
    metar_str = (
        f"METAR {station} {rec_day}{time_str} AUTO "
        f"{wind_metar} {vis_metar} "
        + (f"{wx_metar} " if wx_metar else "")
        + f"{sky_metar} {temp_metar} {alt_metar} RMK AO2"
    )

    return {
        'station':            station,
        'date_created':       t.get('date_created', ''),
        'selected_loop_time': selected_loop_time,
        'location':           location,
        'airport_name':       location,
        'type':               wx_type,
        'time':         time_str,
        'wind':         wind_disp,
        'visibility':   vis,
        'sky':          sky_disp,
        'temp_dp':      temp_disp,
        'altimeter':    alt_disp,
        'remarks':      remarks,
        'metar':        metar_str,
        'phenomena':    [c for _, c in phenomena],
        'local_info':   local_info,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Parse AWOS/ASOS loop transcripts into structured weather data.'
    )
    parser.add_argument('input',  help='Input JSON file (transcripts.json)')
    parser.add_argument('--output', default='parsed_results.json',
                        help='Output JSON file (default: parsed_results.json)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    # Load ground-truth station metadata — location and type come from here,
    # never from the transcript text.
    stations_file = os.path.join(script_dir, 'missouri_awos_asos_stations.json')
    stations = {}
    try:
        with open(stations_file, encoding='utf-8') as f:
            stations = {s['id']: s for s in json.load(f)}
        print(f"Loaded {len(stations)} stations from {os.path.basename(stations_file)}")
    except FileNotFoundError:
        print(f"Warning: {stations_file} not found — using location from transcripts")

    # Inject authoritative location and type into each transcript record
    transcripts = data['transcripts']
    for t in transcripts:
        stn = stations.get(t['station'])
        if stn:
            t['location'] = stn.get('location', t.get('location', ''))
            t['type']     = stn.get('type',     t.get('type', 'AWOS'))

    results = [parse_transcript(t) for t in transcripts]

    # Print summary table to terminal
    print(f"\n{'STN':<6} {'TIME':<7} {'WIND':<18} {'VIS':<8} {'SKY':<14} "
          f"{'TEMP/DP':<14} {'ALT':<12} REMARKS")
    print('-' * 100)
    for r in results:
        print(f"{r['station']:<6} {r['time']:<7} {r['wind']:<18} "
              f"{r['visibility']:<8} {r['sky']:<14} {r['temp_dp']:<14} "
              f"{r['altimeter']:<12} {r['remarks']}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    local_count = sum(1 for r in results if r.get('local_info'))
    print(f"\nParsed {len(results)} stations ({local_count} with local info).")
    print(f"Results written to: {args.output}")


if __name__ == '__main__':
    main()
