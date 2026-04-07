#!/usr/bin/env python3
"""
import_awos_to_mongodb_per_site_atlas.py

Import AWOS/ASOS transcript + parsed JSON into MongoDB Atlas using one database
per site/station.

Database layout:
  kaiz.broadcasts
  kcgi.broadcasts
  kcou.broadcasts
  ...

Each database contains only that station's documents.
Each document contains:
- station
- call_datetime
- location
- type
- recording_sid
- call_sid
- duration_s
- transcript.raw_transcript
- transcript.cleaned_transcript
- transcript.word_timestamps
- transcript.raw_word_timestamps
- transcript.segment_timestamps
- transcript.timestamp_source
- parsed (the parsed transcript payload)

Authentication:
- Reads the Atlas password from the MONGO_DB_PASSWORD environment variable.
- Builds the connection string from:
    mongodb+srv://remote:<password>@prod.vew9qwt.mongodb.net/?appName=prod

Usage:
  export MONGO_DB_PASSWORD='your_password_here'

  python import_awos_to_mongodb_per_site_atlas.py \
      --transcripts transcripts.json \
      --parsed parsed_results.json

Optional:
  --collection broadcasts      Collection name inside each station DB
  --db-prefix awos_           Store into awos_kaiz, awos_kcgi, etc.
  --drop-existing             Drop each site's collection before import
  --dry-run                   Show what would happen without writing
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


ATLAS_USER = "remote"
ATLAS_HOST = "prod.vew9qwt.mongodb.net"
ATLAS_APP_NAME = "prod"
PASSWORD_ENV_VAR = "MONGO_DB_PASSWORD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import AWOS/ASOS transcript and parsed JSON into MongoDB Atlas with one DB per site."
    )
    parser.add_argument(
        "--transcripts",
        default="transcripts.json",
        help="Path to transcripts.json",
    )
    parser.add_argument(
        "--parsed",
        default="parsed_results.json",
        help="Path to parsed_results.json",
    )
    parser.add_argument(
        "--collection",
        default="broadcasts",
        help="Collection name inside each site DB",
    )
    parser.add_argument(
        "--db-prefix",
        default="",
        help="Optional DB prefix, e.g. awos_ -> awos_kaiz",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the target collection in each site DB before import",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without writing to MongoDB",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def safe_db_name(station: str, prefix: str = "") -> str:
    cleaned = "".join(ch for ch in station.lower() if ch.isalnum() or ch in ("_", "-"))
    if not cleaned:
        raise ValueError(f"Invalid station for DB name: {station!r}")
    return f"{prefix}{cleaned}"


def build_parsed_map(parsed_rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in parsed_rows:
        station = str(row.get("station", "")).upper().strip()
        date_created = str(row.get("date_created", "")).strip()
        if station and date_created:
            out[(station, date_created)] = row
    return out


def build_document(transcript_row: Dict[str, Any], parsed_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    station = str(transcript_row.get("station", "")).upper().strip()
    date_created_raw = transcript_row.get("date_created")
    call_datetime = parse_iso_dt(date_created_raw)

    doc: Dict[str, Any] = {
        "station": station,
        "call_datetime": call_datetime,
        "call_datetime_raw": date_created_raw,
        "location": transcript_row.get("location"),
        "type": transcript_row.get("type"),
        "recording_sid": transcript_row.get("recording_sid"),
        "call_sid": transcript_row.get("call_sid"),
        "duration_s": transcript_row.get("duration_s"),
        "date_created": call_datetime,
        "source": {
            "transcript_file": os.path.basename(args.transcripts) if "args" in globals() else "transcripts.json",
            "parsed_file": os.path.basename(args.parsed) if "args" in globals() else "parsed_results.json",
            "imported_at": datetime.utcnow(),
        },
        "transcript": {
            "raw_transcript":      transcript_row.get("raw_transcript"),
            "cleaned_transcript":  transcript_row.get("cleaned_transcript"),
            "hallucination_chars": transcript_row.get("hallucination_chars"),
            "word_timestamps":     transcript_row.get("word_timestamps", []),
            "raw_word_timestamps": transcript_row.get("raw_word_timestamps", []),
            "segment_timestamps":  transcript_row.get("segment_timestamps", []),
            "timestamp_source":    transcript_row.get("timestamp_source", "word"),
        },
        "parsed": {},
    }

    if parsed_row:
        doc["parsed"] = {
            "selected_loop_time": parsed_row.get("selected_loop_time"),
            "time": parsed_row.get("time"),
            "wind": parsed_row.get("wind"),
            "visibility": parsed_row.get("visibility"),
            "sky": parsed_row.get("sky"),
            "temp_dp": parsed_row.get("temp_dp"),
            "altimeter": parsed_row.get("altimeter"),
            "remarks": parsed_row.get("remarks"),
            "metar": parsed_row.get("metar"),
            "phenomena": parsed_row.get("phenomena"),
            "local_info": parsed_row.get("local_info"),
            "airport_name": parsed_row.get("airport_name"),
            "location": parsed_row.get("location"),
            "type": parsed_row.get("type"),
            "date_created_raw": parsed_row.get("date_created"),
        }

    return doc


def ensure_indexes(collection: Collection) -> None:
    collection.create_index([("call_datetime", -1)])
    collection.create_index([("recording_sid", 1)])
    collection.create_index(
        [("recording_sid", 1), ("call_datetime", 1)],
        unique=True,
        name="uniq_recording_sid_call_datetime",
    )


def unique_station_ids(transcript_rows: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted({
        str(row.get("station", "")).upper().strip()
        for row in transcript_rows
        if row.get("station")
    })


def build_atlas_uri() -> str:
    password = os.getenv(PASSWORD_ENV_VAR)
    if not password:
        raise RuntimeError(
            f"{PASSWORD_ENV_VAR} is not set. Export it before running this script."
        )

    encoded_password = quote_plus(password)
    return (
        f"mongodb+srv://{ATLAS_USER}:{encoded_password}"
        f"@{ATLAS_HOST}/?appName={ATLAS_APP_NAME}"
    )


def connect_to_atlas() -> MongoClient:
    uri = build_atlas_uri()
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000,
        retryWrites=True,
    )
    client.admin.command("ping")
    return client


def main() -> None:
    global args
    args = parse_args()

    transcripts_blob = load_json(args.transcripts)
    parsed_blob = load_json(args.parsed)

    transcript_rows = transcripts_blob.get("transcripts", [])
    if not isinstance(transcript_rows, list):
        raise ValueError("transcripts.json must contain a top-level 'transcripts' list")
    if not isinstance(parsed_blob, list):
        raise ValueError("parsed_results.json must be a top-level list")

    parsed_map = build_parsed_map(parsed_blob)
    stations = unique_station_ids(transcript_rows)

    print(f"Loaded {len(transcript_rows)} transcript rows")
    print(f"Loaded {len(parsed_blob)} parsed rows")
    print(f"Found {len(stations)} station DB(s): {', '.join(s.lower() for s in stations)}")
    print()

    client: Optional[MongoClient] = None
    if not args.dry_run:
        print("Connecting to MongoDB Atlas...")
        client = connect_to_atlas()
        print("Connected to MongoDB Atlas.")
        print()

    if args.drop_existing:
        for station in stations:
            db_name = safe_db_name(station, args.db_prefix)
            if args.dry_run:
                print(f"[dry-run] Would drop {db_name}.{args.collection}")
            else:
                assert client is not None
                db = client[db_name]
                db.drop_collection(args.collection)
                print(f"Dropped {db_name}.{args.collection}")
        print()

    per_station_ops: Dict[str, List[UpdateOne]] = {}
    matched_count = 0
    unmatched_count = 0

    for t in transcript_rows:
        station = str(t.get("station", "")).upper().strip()
        if not station:
            print("Skipping transcript row with missing station")
            continue

        key = (station, str(t.get("date_created", "")).strip())
        parsed_row = parsed_map.get(key)

        if parsed_row is None:
            unmatched_count += 1
            print(f"WARN  No parsed row matched for {station} @ {key[1]}")
        else:
            matched_count += 1

        doc = build_document(t, parsed_row)

        filter_doc = {
            "recording_sid": doc.get("recording_sid"),
            "call_datetime": doc.get("call_datetime"),
        }
        if not filter_doc["recording_sid"] or not filter_doc["call_datetime"]:
            filter_doc = {
                "station": doc["station"],
                "call_datetime_raw": doc["call_datetime_raw"],
            }

        db_name = safe_db_name(station, args.db_prefix)
        per_station_ops.setdefault(db_name, []).append(
            UpdateOne(filter_doc, {"$set": doc}, upsert=True)
        )

    print(f"Matched transcript->parsed rows: {matched_count}")
    print(f"Unmatched transcript rows:       {unmatched_count}")
    print()

    total_upserted = 0
    total_modified = 0

    for db_name in sorted(per_station_ops):
        ops = per_station_ops[db_name]
        print(f"{db_name}.{args.collection}: {len(ops)} document(s)")

        if args.dry_run:
            continue

        assert client is not None
        db = client[db_name]
        collection = db[args.collection]
        ensure_indexes(collection)
        result = collection.bulk_write(ops, ordered=False)

        upserted = len(result.upserted_ids)
        modified = result.modified_count
        matched = result.matched_count

        total_upserted += upserted
        total_modified += modified

        print(f"  upserted={upserted} modified={modified} matched_existing={matched}")

    print()
    if args.dry_run:
        print("Dry run complete. No data written.")
    else:
        print("Import complete.")
        print(f"Total upserted: {total_upserted}")
        print(f"Total modified: {total_modified}")


if __name__ == "__main__":
    try:
        main()
    except PyMongoError as e:
        raise SystemExit(f"MongoDB error: {e}")
    except Exception as e:
        raise SystemExit(f"Error: {e}")
