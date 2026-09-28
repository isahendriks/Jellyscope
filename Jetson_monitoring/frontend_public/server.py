"""
server.py -- Jellyscope PUBLIC livestream backend. Runs on server-lab (this
machine), NOT the Jetson.

WHY THIS EXISTS, AND WHY HERE
The Jetson does the actual monitoring at a remote station on a fragile,
bandwidth-constrained 5G uplink, 500km from the operators. It already scp's
its results into this machine's jellyscope_incoming/ folder (see INCOMING_DIR
below). This script reads that folder and republishes a small, curated,
read-only subset of it to the public internet. Public visitors never touch
the Jetson at all -- server-lab (a stable university-network machine) absorbs
all of that traffic instead. This is a deliberate architecture decision, not
an accident of what was convenient to build.

This is a *separate* thing from Jetson_monitoring/frontend_private/, the
password-gated ops dashboard served directly by the Jetson itself
(Monitor/analyse.py). That one is for the people running the station; this
one is for the public, has no login (confirmed decision -- this site is meant
to be fully open), and physically runs on a different machine reading files
off disk rather than an in-process API.

WHAT'S DELIBERATELY LEFT OUT
jellyscope_incoming/ also holds device-health telemetry (CPU/GPU/disk/queue
depths, upload failure counts...), environmental sensor readings, and full
camera frames for model retraining. None of that is served here. This file
only ever reads image bytes plus a handful of whitelisted JSON fields (see
PUBLIC_CROP_FIELDS below) -- never a whole metadata file, and never anything
from the metadata/ or environmental_metadata/ folders at all. If you're
tempted to add a field here, ask first whether it's actually meant for a
public audience -- device health/queue/upload-failure info also hints at
exploitable weak points, not just "boring internal numbers".

One exception: live_frame.json (see live_frame_boxes() below) is passed
through whole, unlike crop_metadata/*.json. That's not an inconsistency --
the Jetson writes that specific file itself already containing only
crop coordinates + species label/confidence, precisely so this server doesn't
need its own whitelist for it. Everything else in jellyscope_incoming/ is a
general-purpose file this script doesn't control the shape of, which is why
those get filtered here instead.

HOW IT'S SERVED
Mirrors how Monitor/analyse.py serves frontend_private/ on the Jetson side:
Flask's own dev server, run with threaded=True so one slow client can't block
everyone else, and debug=False/use_reloader=False because this runs
unattended as a long-lived service rather than something being edited live.
See README.md for how this process is kept running (Task Scheduler) and how
it's actually reached from the public internet (Tailscale Funnel).
"""

import heapq
import json
import os
import re
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, send_file, send_from_directory

# ===== CONFIG =====

# Overridable via env var -- the folder layout is fixed, but the drive/user
# account it lives under is a detail of this one machine, not something to
# hardcode into the only copy of this logic.
INCOMING_DIR = Path(os.environ.get("JELLYSCOPE_INCOMING", r"C:\Users\jellyfish\jellyscope_incoming"))
CROPS_DIR = INCOMING_DIR / "crops"
CROP_METADATA_DIR = INCOMING_DIR / "crop_metadata"
LIVE_FRAME_PATH = INCOMING_DIR / "live_frame" / "live_frame.jpg"
# The Jetson writes this sidecar itself, already containing only public-safe
# fields (crop coordinates + species label/confidence, scaled to this exact
# image's pixel space) -- unlike crop_metadata/*.json below, there's nothing
# to whitelist here, so live_frame_boxes() just passes the file through as-is.
LIVE_FRAME_JSON_PATH = INCOMING_DIR / "live_frame" / "live_frame.json"

PORT = int(os.environ.get("JELLYSCOPE_PUBLIC_PORT", "8090"))

# How many of the most-recently-arrived crops /recent_crops hands back.
RECENT_CROPS_LIMIT = 30

# Crop images arrive named like:
#   frame_20260812_132607_00049533_crop_000_area_15.53.png
# under a crops/<YYYYMMDD>/ folder, with a same-stem JSON sidecar under
# crop_metadata/<YYYYMMDD>/ (see class docstring / project brief for a real
# example). These two patterns are the only attacker-reachable input this
# server ever turns into a filesystem path -- see crop_image() below.
DATE_DIR_RE = re.compile(r"^\d{8}$")
CROP_FILENAME_RE = re.compile(r"^frame_\d{8}_\d{6}_\d+_crop_\d+_area_[\d.]+\.png$")

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent), static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("index.html")


def _dates_newest_first():
    """Date-named subfolders under crops/, newest first. Plain string sort --
    YYYYMMDD folder names sort chronologically as strings, no date parsing
    needed."""
    if not CROPS_DIR.is_dir():
        return []
    return sorted(
        (p.name for p in CROPS_DIR.iterdir() if p.is_dir() and DATE_DIR_RE.match(p.name)),
        reverse=True,
    )


def _latest_crop_stems(limit):
    """(date_name, stem) pairs for the most recently arrived crops, newest
    first -- today's folder, then spilling into earlier folders if today
    doesn't have `limit` crops yet (e.g. just after local midnight).

    heapq.nlargest instead of a full sort: filenames embed a timestamp so
    "largest filename" == "most recent crop", and this only has to find the
    top `limit`, not order all of them -- matters once a date folder has
    tens of thousands of files in it (~17,000/day seen in early testing)."""
    stems = []
    for date_name in _dates_newest_first():
        date_dir = CROPS_DIR / date_name
        remaining = limit - len(stems)
        names = heapq.nlargest(
            remaining,
            (p.name for p in date_dir.iterdir() if CROP_FILENAME_RE.match(p.name)),
        )
        stems.extend((date_name, name[: -len(".png")]) for name in names)
        if len(stems) >= limit:
            break
    return stems


# Public JSON key -> key read out of the crop's metadata JSON. Everything else
# in that file (classifier_checkpoint, upload_attempts, source_frame_id, crop
# pixel coordinates, ...) is internal and never gets copied into `info` below.
PUBLIC_CROP_FIELDS = {
    "species": "class_label",
    "confidence": "class_confidence",
    "size_mm2": "region_size_mm2",
    "time": "processed_iso",
}


def _public_crop_info(date_name, stem):
    meta_path = CROP_METADATA_DIR / date_name / f"{stem}.json"
    try:
        raw = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        # Sidecar not written yet, or crop was already cleaned up -- skip it
        # rather than erroring the whole /recent_crops response.
        return None
    info = {public_key: raw.get(source_key) for public_key, source_key in PUBLIC_CROP_FIELDS.items()}
    info["image_url"] = f"/crop_image/{date_name}/{stem}.png"
    return info


# Rebuilding the "latest N crops" list means scanning a date folder that can
# hold tens of thousands of files by the end of a day -- cheap per scan (no
# per-file stat calls) but wasteful to redo for every simultaneous public
# viewer's every poll. This cache lets many viewers share one scan every
# CACHE_SECONDS instead of each triggering their own.
CACHE_SECONDS = 2.0
_cache = {"time": 0.0, "crops": []}


@app.route("/recent_crops")
def recent_crops():
    now = time.time()
    if now - _cache["time"] > CACHE_SECONDS:
        crops = []
        for date_name, stem in _latest_crop_stems(RECENT_CROPS_LIMIT):
            info = _public_crop_info(date_name, stem)
            if info is not None:
                crops.append(info)
        _cache["crops"] = crops
        _cache["time"] = now
    return jsonify(_cache["crops"])


@app.route("/crop_image/<date_name>/<filename>")
def crop_image(date_name, filename):
    # date_name and filename come straight from the URL, i.e. from whoever is
    # asking -- validated against the strict patterns above before they ever
    # touch the filesystem, so this can't be used to read anything outside
    # crops/<date_name>/ (e.g. the metadata JSON, or another folder entirely).
    if not DATE_DIR_RE.match(date_name) or not CROP_FILENAME_RE.match(filename):
        abort(404)
    return send_from_directory(CROPS_DIR / date_name, filename)


@app.route("/live_frame.jpg")
def live_frame():
    if not LIVE_FRAME_PATH.is_file():
        abort(404)
    try:
        response = send_file(LIVE_FRAME_PATH, mimetype="image/jpeg", conditional=False)
    except OSError:
        # The Jetson overwrites this file in place roughly every 3 seconds;
        # on the rare request that lands mid-write, skip it rather than error
        # -- the browser will just ask again shortly (see media.js).
        abort(404)
    # A short-lived cache, not no-store: every poll already uses a fresh
    # ?t=... URL (see media.js), so a new poll can never be served a stale
    # frame from cache regardless of this setting -- what this DOES enable is
    # media.js's "probe the image, then point #liveFrame at the same URL"
    # pattern reusing that first fetch for free instead of downloading the
    # same bytes twice.
    response.headers["Cache-Control"] = "private, max-age=10"
    return response


@app.route("/live_frame.json")
def live_frame_boxes():
    # The Jetson writes the JSON sidecar before the JPEG (see this repo's
    # Jetson-side analyse.py) specifically so that by the time live_frame.jpg's
    # mtime changes -- send.py's upload trigger -- the matching JSON is already
    # written and uploaded alongside it in the same scp call. So this doesn't
    # need the same "might catch a mid-write file" handling as live_frame.jpg
    # above, just a plain missing-file check.
    try:
        raw = LIVE_FRAME_JSON_PATH.read_text()
    except OSError:
        abort(404)
    response = Response(raw, mimetype="application/json")
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True)
