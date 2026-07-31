#!/usr/bin/env python3
"""SSH-friendly quick binary labeler: obs vs no_obs.

Serves one image at a time through a tiny Flask page with keyboard-driven
shortcuts, moving each file straight into dest/obs/ or dest/no_obs/ as you go --
no X11 forwarding or local copy of the images needed, just a browser tab (VSCode's
Remote-SSH auto-forwards the port; over a plain terminal SSH session, forward it
yourself: `ssh -L 8090:localhost:8090 <user>@<host>`).

Images are shown brightest-mean-first, not in filename/capture order: most frames
in a raw capture folder are empty water, so this puts the ones actually likely to
contain something at the front of the queue instead of buried among hundreds of
empty ones scanned in order.

Resumable: already-labeled files (present in dest/obs/ or dest/no_obs/ from a
previous run) are skipped when the queue is rebuilt, so killing/restarting this
script mid-folder loses nothing.

Usage:
    python label_images.py <source_folder> [--dest DEST] [--port 8090]

Then open http://<device-ip>:8090/ and press:
    Right arrow / O   -> obs      (contains an organism)
    Left arrow  / N   -> no_obs   (empty)
    Down arrow  / S    -> skip (leave in place, re-queued at the back)
    U                  -> undo the last label (moves the file back)
"""

import argparse
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Downscale before sending over the wire -- these are full 4512px camera frames;
# shipping them full-size over an SSH-forwarded port makes every label feel
# laggy for no benefit, since the call being made is "obs or not", not a
# pixel-level judgment.
DISPLAY_MAX_PX = 900

# Decoding a full 4512px TIFF/PNG is CPU-only work (libtiff/libpng) that no GPU in this
# pipeline accelerates -- that decode, not the mean itself, is almost certainly what's
# slow for large images. Two real fixes, both applied in compute_brightness_scores():
#   1. cv2.imread releases the GIL during decode, so a thread pool actually parallelizes
#      it across CPU cores (same idiom Pipeline_development/BinaryClassification/
#      functions.py's load_tiles_from_paths_fast already uses).
#   2. Shrinking each frame to BRIGHTNESS_SAMPLE_PX right after decode -- a coarse
#      "brightest first" ranking doesn't need full resolution -- then batching those
#      shrunk frames onto the GPU for one vectorized per-batch mean, instead of one
#      Python-level .mean() call per image on the CPU.
BRIGHTNESS_SAMPLE_PX = 256
BRIGHTNESS_BATCH_SIZE = 64
BRIGHTNESS_DECODE_WORKERS = 8

# Persisted so a restart (e.g. after killing the server mid-folder) never re-scores an
# image it already has a fresh score for -- keyed by filename, with (mtime, size) as a
# cheap fingerprint to recompute if a file's content ever actually changes. Lives in
# `dest` alongside obs/ and no_obs/, the other resumable state this script writes.
BRIGHTNESS_CACHE_FILENAME = ".label_images_brightness_cache.json"

lock = threading.Lock()


def _read_and_shrink(path: Path) -> tuple[Path, np.ndarray | None]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, None
    img = cv2.resize(img, (BRIGHTNESS_SAMPLE_PX, BRIGHTNESS_SAMPLE_PX), interpolation=cv2.INTER_AREA)
    return path, img


def compute_brightness_scores(paths: list[Path], device: torch.device) -> dict[Path, float]:
    """Returns {path: mean_brightness}, decoding in parallel across threads (see
    BRIGHTNESS_DECODE_WORKERS above) and reducing each batch of shrunk frames on
    `device` in one vectorized call rather than one .mean() per image."""
    scores: dict[Path, float] = {}
    batch_imgs: list[np.ndarray] = []
    batch_paths: list[Path] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        stacked = np.stack(batch_imgs, axis=0).astype(np.float32)  # (B, H, W) -- uniform shape, all shrunk the same way
        means = torch.from_numpy(stacked).to(device, non_blocking=True).mean(dim=(1, 2)).cpu().numpy()
        for p, m in zip(batch_paths, means):
            scores[p] = float(m)
        batch_imgs.clear()
        batch_paths.clear()

    done = 0
    with ThreadPoolExecutor(max_workers=BRIGHTNESS_DECODE_WORKERS) as pool:
        for path, img in pool.map(_read_and_shrink, paths):
            done += 1
            if done % 50 == 0 or done == len(paths):
                print(f"  {done}/{len(paths)}", end="\r")
            if img is None:
                scores[path] = -1.0
                continue
            batch_imgs.append(img)
            batch_paths.append(path)
            if len(batch_imgs) >= BRIGHTNESS_BATCH_SIZE:
                flush_batch()
        flush_batch()
    print()
    return scores


def _fingerprint(path: Path) -> list:
    st = path.stat()
    return [st.st_mtime, st.st_size]


def load_brightness_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_brightness_cache(cache_path: Path, cache: dict) -> None:
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(cache_path)


def build_queue(source: Path, obs_dir: Path, no_obs_dir: Path, device: torch.device, cache_path: Path) -> list[Path]:
    already_labeled = {p.name for p in obs_dir.glob("*")} | {p.name for p in no_obs_dir.glob("*")}
    files = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.name not in already_labeled
    )
    print(f"Found {len(files)} unlabeled image(s) "
          f"({len(already_labeled)} already labeled from a previous run, skipped)")

    cache = load_brightness_cache(cache_path)
    scores: dict[Path, float] = {}
    to_compute = []
    for path in files:
        cached = cache.get(path.name)
        if cached is not None and cached[1:] == _fingerprint(path):
            scores[path] = cached[0]
        else:
            to_compute.append(path)

    if to_compute:
        print(f"Computing brightness on {device} for {len(to_compute)} new/changed image(s) "
              f"({len(files) - len(to_compute)} reused from cache) -- brightest first...")
        new_scores = compute_brightness_scores(to_compute, device)
        scores.update(new_scores)
        for path in to_compute:
            cache[path.name] = [new_scores[path], *_fingerprint(path)]
        save_brightness_cache(cache_path, cache)
    else:
        print(f"All {len(files)} brightness score(s) reused from {cache_path.name} -- nothing to compute.")

    files.sort(key=lambda p: scores[p], reverse=True)
    return files


def make_app(source: Path, dest: Path, device: torch.device) -> Flask:
    obs_dir = dest / "obs"
    no_obs_dir = dest / "no_obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    no_obs_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dest / BRIGHTNESS_CACHE_FILENAME

    queue = build_queue(source, obs_dir, no_obs_dir, device, cache_path)
    total_remaining_at_start = len(queue)
    counts = {
        "obs": sum(1 for _ in obs_dir.glob("*")),
        "no_obs": sum(1 for _ in no_obs_dir.glob("*")),
    }
    # (original_path, label, moved_to_path) per label, most recent last -- "skip" isn't
    # pushed here since nothing moved, so undo only ever reverses an actual obs/no_obs move.
    history = []

    app = Flask(__name__)

    def status_dict() -> dict:
        return {
            "remaining": len(queue),
            "obs_count": counts["obs"],
            "no_obs_count": counts["no_obs"],
            "current_name": queue[0].name if queue else None,
            "can_undo": bool(history),
            "all_done": not queue,
        }

    @app.route("/")
    def index():
        return PAGE_HTML

    @app.route("/status")
    def status():
        with lock:
            return jsonify(status_dict())

    @app.route("/current.jpg")
    def current_jpg():
        while True:
            with lock:
                if not queue:
                    return Response(status=404)
                path = queue[0]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                break
            # Unreadable file -- drop it and try the next one instead of getting stuck.
            with lock:
                if queue and queue[0] == path:
                    queue.pop(0)

        h, w = img.shape[:2]
        scale = DISPLAY_MAX_PX / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", img)
        resp = Response(encoded.tobytes(), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.route("/label", methods=["POST"])
    def label():
        choice = (request.get_json(force=True) or {}).get("label")
        if choice not in ("obs", "no_obs", "skip"):
            return jsonify({"error": "label must be 'obs', 'no_obs', or 'skip'"}), 400
        with lock:
            if not queue:
                return jsonify(status_dict())
            path = queue.pop(0)
            if choice == "skip":
                queue.append(path)
            else:
                dest_dir = obs_dir if choice == "obs" else no_obs_dir
                dest_path = dest_dir / path.name
                shutil.move(str(path), str(dest_path))
                history.append((path, choice, dest_path))
                counts[choice] += 1
            return jsonify(status_dict())

    @app.route("/undo", methods=["POST"])
    def undo():
        with lock:
            if not history:
                return jsonify(status_dict())
            original_path, choice, moved_path = history.pop()
            shutil.move(str(moved_path), str(original_path))
            counts[choice] -= 1
            queue.insert(0, original_path)
            return jsonify(status_dict())

    print(f"{total_remaining_at_start} image(s) queued to label.")
    return app


PAGE_HTML = """<!doctype html>
<html><head><title>obs / no_obs labeler</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:-apple-system,Menlo,Consolas,monospace;
         display:flex; flex-direction:column; align-items:center; height:100vh; overflow:hidden; }
  #bar { width:100%; padding:10px 16px; box-sizing:border-box; background:#000; border-bottom:2px solid #0f0;
         display:flex; justify-content:space-between; font-size:14px; }
  #imgWrap { flex:1; display:flex; align-items:center; justify-content:center; width:100%; min-height:0; }
  #img { max-width:96vw; max-height:calc(100vh - 140px); border:2px solid #0f0; border-radius:4px; }
  #done { font-size:28px; }
  #controls { padding:14px; display:flex; gap:14px; }
  button { font-size:16px; padding:10px 18px; border-radius:6px; border:2px solid #444; cursor:pointer;
           background:#222; color:#eee; font-family:inherit; }
  button:active { transform:scale(0.97); }
  #noObsBtn { border-color:#c0392b; }
  #obsBtn { border-color:#0f0; }
  #name { font-size:12px; color:#aaa; text-align:center; padding:4px; }
</style></head>
<body>
  <div id="bar">
    <span>Remaining: <b id="remaining">-</b></span>
    <span>obs: <b id="obsCount" style="color:#0f0">-</b></span>
    <span>no_obs: <b id="noObsCount" style="color:#e74c3c">-</b></span>
  </div>
  <div id="imgWrap"><img id="img"><div id="done" style="display:none">All done!</div></div>
  <div id="name"></div>
  <div id="controls">
    <button id="noObsBtn" onclick="sendLabel('no_obs')">&larr; no_obs (N)</button>
    <button onclick="sendLabel('skip')">skip (S) &darr;</button>
    <button onclick="undo()">undo (U)</button>
    <button id="obsBtn" onclick="sendLabel('obs')">obs (O) &rarr;</button>
  </div>
<script>
let busy = false;

function refreshImage() {
  document.getElementById('img').src = '/current.jpg?t=' + Date.now();
}

function applyStatus(s) {
  document.getElementById('remaining').textContent = s.remaining;
  document.getElementById('obsCount').textContent = s.obs_count;
  document.getElementById('noObsCount').textContent = s.no_obs_count;
  document.getElementById('name').textContent = s.current_name || '';
  if (s.all_done) {
    document.getElementById('img').style.display = 'none';
    document.getElementById('done').style.display = 'block';
  } else {
    document.getElementById('img').style.display = 'block';
    document.getElementById('done').style.display = 'none';
  }
}

async function refreshStatus() {
  const s = await (await fetch('/status')).json();
  applyStatus(s);
  return s;
}

async function sendLabel(choice) {
  if (busy) return;
  busy = true;
  try {
    const resp = await fetch('/label', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({label: choice}),
    });
    applyStatus(await resp.json());
    refreshImage();
  } finally {
    busy = false;
  }
}

async function undo() {
  if (busy) return;
  busy = true;
  try {
    const resp = await fetch('/undo', {method: 'POST'});
    applyStatus(await resp.json());
    refreshImage();
  } finally {
    busy = false;
  }
}

document.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  if (key === 'arrowright' || key === 'o') { e.preventDefault(); sendLabel('obs'); }
  else if (key === 'arrowleft' || key === 'n') { e.preventDefault(); sendLabel('no_obs'); }
  else if (key === 'arrowdown' || key === 's') { e.preventDefault(); sendLabel('skip'); }
  else if (key === 'u') { e.preventDefault(); undo(); }
});

refreshImage();
refreshStatus();
setInterval(refreshStatus, 2000);
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Folder of images to label")
    parser.add_argument("--dest", type=Path, default=None,
                         help="Where to create obs/ and no_obs/ subfolders (default: inside source itself)")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Not a directory: {args.source}")
    dest = args.dest if args.dest is not None else args.source

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    app = make_app(args.source, dest, device)

    print(f"\nOpen http://<this-device-ip>:{args.port}/ in a browser.")
    print(f"Over a plain SSH session (no VSCode auto-forwarding), first run: "
          f"ssh -L {args.port}:localhost:{args.port} <user>@<host>, then open http://localhost:{args.port}/")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
