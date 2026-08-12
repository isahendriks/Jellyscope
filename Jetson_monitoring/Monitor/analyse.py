"""Stage 2: preprocess (GPU) -> segment -> classify, one process. Three
sections below, each styled after the specific existing script it's ported
from, not a generic rewrite:

  PREPROCESS -> control/monitoring/record.py's CPU chain, ported to GPU (gpu_preprocess.py)
  SEGMENT    -> Pipeline_development/BinaryClassification/segment_labeled_images.py's working
                reference loop (not segment_per_image.py, which has a broken import)
  CLASSIFY   -> Pipeline_development/ClassClassification/Train_ViT.py's ViT model

INT8-only: SEGMENT and CLASSIFY both run as TensorRT INT8 engines
(models/segmentation_trt.py, models/vit_classifier_trt.py) -- no FP32 PyTorch
inference happens in this file. Those engines are trained/exported/calibrated
by train/ and engines/build_trt_int8.py; nothing here ever trains anything.
The FP32 loaders (models/segmentation.py, models/vit_classifier.py's
load_classifier) still exist, but only for engines/export_onnx.py's tracing
step -- this file no longer imports them.

One exception: SEGMENT's scorer stage can be either a "binary" TwoClassScorer
(INT8 TensorRT, same as everything else) or a "mahalanobis" distance-based
scorer, which models/segmentation_trt.py runs directly as FP32 PyTorch instead
(cheap linear algebra, and its chi-squared CDF can't be traced to ONNX/TensorRT
in the first place). Whichever mode config.SEGMENTATION_SCORER_MODEL_PATH's
checkpoint was saved in is auto-detected and used -- see
models/segmentation.py's build_scorer()/detect_scorer_type().
"""

import gc
import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

import config
import queue_io
import gpu_preprocess
import segment_core
import leak_alert  # send_slack_alert only -- generic webhook helper, not leak-specific despite the module name
from models.segmentation_trt import load_segmentation_engines
from models.vit_classifier_trt import load_classifier_engine, pad_batch
from models import vit_classifier  # preprocess_crop_for_classifier / pad_to_square only -- no FP32 model

### ==========================
### Values derived from config.py's tuning parameters -- not tunable inputs
### themselves, so they stay here rather than in config.py (see its docstring).
### ==========================
PXL_TO_MM = config.IMAGE_W_MM / config.IMAGE_SIZE_PX
PXL_TO_MM2 = PXL_TO_MM ** 2
tile_size = config.IMAGE_SIZE_PX // config.SEG_TILE_GRID_SIZE
offsets = [int(norm * tile_size) for norm in config.SEG_OFFSETS_NORM]

HEARTBEAT_PATH = config.HEARTBEAT_DIR / "analyse.heartbeat.json"

### ==========================
### Setup: device, models, dark frame
### ==========================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

queue_io.ensure_queue_dirs(config.QUE_FULLFRAMES)
queue_io.ensure_queue_dirs(config.QUE_CROPS)
queue_io.ensure_queue_dirs(config.QUE_TRAINING_FRAMES)
config.HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading segmentation engines (encoder/decoder always INT8; scorer is INT8 or FP32 Mahalanobis, auto-detected)...")
encoder_engine, decoder_engine, scorer_engine, scorer_threshold, seg_grid_size, seg_image_size = load_segmentation_engines(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH, config.ENGINE_DIR, device,
)

if seg_grid_size != config.SEG_TILE_GRID_SIZE:
    raise ValueError(f"Segmentation checkpoint grid_size={seg_grid_size} does not match "
                     f"config.SEG_TILE_GRID_SIZE={config.SEG_TILE_GRID_SIZE}")
if config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE is not None:
    print(f"Overriding checkpoint's threshold ({scorer_threshold:.4f}) with "
          f"config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE={config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE:.4f}")
    scorer_threshold = config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE
peak_threshold = scorer_threshold
secondary_threshold = peak_threshold * 0.99
print(f"Loaded segmentation engines. Scorer threshold: {scorer_threshold:.4f}")

classifier_engine, idx_to_class = None, {}
if config.CLASSIFY:
    print("Loading INT8 ViT classifier engine...")
    classifier_engine, idx_to_class = load_classifier_engine(
        config.VIT_CHECKPOINT_PATH, config.ENGINE_DIR / "vit_classifier_int8.engine", device,
        class_names_fallback=config.VIT_CLASS_NAMES_FALLBACK,
    )
    print(f"Loaded classifier with {len(idx_to_class)} classes (fixed batch {config.VIT_ENGINE_BATCH}).")
else:
    print("CLASSIFY=False -- skipping ViT classifier engine load, crops will be produced unlabeled.")

dark_frame = None
if config.DARK_FRAME_PATH.exists():
    dark_frame = gpu_preprocess.load_dark_frame(str(config.DARK_FRAME_PATH), device, rotate_angle_deg=config.ROTATE_FRAME)
    print(f"Loaded dark frame from {config.DARK_FRAME_PATH}")
else:
    print(f"No dark frame found at {config.DARK_FRAME_PATH}, skipping dark-frame subtraction.")

grid_size_small = config.SEG_TILE_GRID_SIZE * len(offsets)
upscale_factor = config.IMAGE_SIZE_PX / grid_size_small

frames_analysed_total = 0
crops_produced_total = 0
total_processing_time_sum = 0.0  # sum of each frame's "total" (record->stream); /status divides by frames_analysed_total for the running average
process_start_unix = time.time()  # this process's own lifetime, for the "fully live" ETA
# below -- frames_analysed_total/(now-process_start_unix) gives a stable rate estimate
# that naturally resets on every restart, same reasoning as record.py's own field of the
# same name.

# "backlog" or "live" -- see update_system_mode(). Starts in "live" since, as of
# 2026-08-11, fullframe_queues is just the live queue -- sdb1's old backlog is
# deliberately left out (see config.py's QUE_FULLFRAMES_BACKLOG_RESTING), so there's
# nothing to be "in backlog mode" about until/unless the live queue itself grows past
# BACKLOG_MODE_ENTER_THRESHOLD.
system_mode = "live"

# Live-stream summary stats -- deliberately NOT lifetime running totals like
# frames_analysed_total/crops_produced_total above. Over a 3-month unattended
# deployment a growing counter tells an operator glancing at the stream nothing
# about whether the system is *currently* healthy; a timestamp + a bounded
# 24h window do. last_analysed_unix/last_analysed_crops are just the most recent
# process_frame() call's numbers -- the frame's own capture timestamp, not when
# analysis happened, so it's directly comparable to record.py's "last recorded"
# heartbeat field (read_record_heartbeat()) to see how far analysis is lagging
# live capture. crops_last_24h_window is a rolling (timestamp, crop_count) deque
# pruned to the last 24h on every frame, with crops_last_24h kept as a running
# sum over just that window (not recomputed by summing the deque each time) so
# a poll of /status stays O(1) regardless of frame rate.
last_analysed_unix: float | None = None
last_analysed_crops = 0
crops_last_24h = 0
crops_last_24h_window: deque = deque()

# Fullframe backlog-drain state (oldest data first -- see config.py's docstrings on the
# three QUE_FULLFRAMES* paths), declared here rather than only inside the __main__ block
# below so /status can read it for progress reporting even in the narrow window right
# after a restart, before __main__ has populated it for real -- an empty list here just
# means _stream_status() reports no backlog info yet instead of raising NameError.
fullframe_queues: list[tuple] = []
active_queue_idx = 0
fullframe_backlog: list[str] = []

# Backlog-drain rate, for the live-stream's BACKLOG section. Tracked as a plain
# processed-count + start-time pair (reset whenever active_queue_idx changes), NOT
# derived from watching fullframe_backlog's length shrink -- that works fine for the
# two one-shot backlogs (nothing refills them), but the live queue (last stage) gets
# refilled by record.py's own ongoing production, so its list length isn't a shrinking
# subset of any fixed starting count. A plain "how many has this stage actually
# processed, and over how long" counter stays meaningful either way.
active_queue_started_unix: float | None = None
active_queue_processed_count = 0

# Daily Slack report counters -- reset whenever the local calendar date rolls over
# (see maybe_send_daily_report()), independent of crops_produced_total above (which
# is a lifetime total, never reset).
daily_crops_count = 0
daily_mnemiopsis_count = 0
daily_report_date = time.strftime("%Y-%m-%d", time.localtime())

last_training_frame_unix = 0.0  # 0 (not time.time()) so the very first frame after a
# fresh start samples immediately instead of waiting a full TRAINING_FRAME_INTERVAL_S

latest_frame_lock = threading.Lock()
latest_frame_jpeg = None

# Most-recent confidently-labeled crops (oldest evicted automatically past
# RECENT_CROPS_COUNT) plus the rendered thumbnail-strip JPEG built from them --
# same lock/publish idiom as latest_frame_jpeg above, just a second image.
recent_crops_lock = threading.Lock()
recent_labeled_crops = deque(maxlen=config.RECENT_CROPS_COUNT)
latest_crops_strip_jpeg = None


def update_live_frame(enhanced, results) -> None:
    """Draws a green box around each accepted crop (+ its label once confidence
    clears config.CONFIDENCE_THRESHOLD) onto a downscaled copy of the post-PREPROCESS
    frame, and publishes it for the live-stream endpoint below. Downscale
    happens *before* drawing, not after -- boxes/text drawn at full 4512px res
    then shrunk 10x would be illegible."""
    global latest_frame_jpeg
    scale = config.LIVESTREAM_DISP_SCALE / 100
    small = cv2.resize(enhanced, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    display = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    for _crop_stem, _encoded_bytes, sidecar in results:
        x0, y0 = int(sidecar["crop_x0"] * scale), int(sidecar["crop_y0"] * scale)
        x1, y1 = int(sidecar["crop_x1"] * scale), int(sidecar["crop_y1"] * scale)
        cv2.rectangle(display, (x0, y0), (x1, y1), config.CROP_BOX_COLOR, 1)
        if sidecar["class_confidence"] is not None and sidecar["class_confidence"] >= config.CONFIDENCE_THRESHOLD:
            cv2.putText(display, sidecar["class_label"], (x0, max(10, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.CROP_BOX_COLOR, 1, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".jpg", display)
    if ok:
        with latest_frame_lock:
            latest_frame_jpeg = encoded.tobytes()


def update_recent_crops_strip(results) -> None:
    """Appends this frame's confidently-labeled crops (same CONFIDENCE_THRESHOLD
    gate as the box labels drawn in update_live_frame above) to recent_labeled_crops,
    then re-renders the whole deque into a single side-by-side thumbnail-strip JPEG
    for the live-stream window. Decodes crop_image back out of results' already-encoded
    PNG bytes rather than re-slicing `enhanced`, so this has no dependency on the
    full-resolution frame surviving past this call."""
    global latest_crops_strip_jpeg
    with recent_crops_lock:
        for _crop_stem, encoded_bytes, sidecar in results:
            if (sidecar["class_confidence"] is not None
                    and sidecar["class_confidence"] >= config.CONFIDENCE_THRESHOLD
                    and sidecar["peak_val"] > config.RECENT_CROPS_MIN_PEAK_VAL):
                recent_labeled_crops.append((encoded_bytes, sidecar["class_label"]))

        if not recent_labeled_crops:
            return

        # Each tile: the crop image, boxed in the same green used for the main frame's
        # crop boxes (config.CROP_BOX_COLOR), with its class label captioned in a black
        # band underneath -- centered, white-on-black so it stays legible regardless of
        # how bright/dark the crop itself is (green-on-crop, tried first, washed out on
        # brighter crops). Tiles are separated by a gap column/row, not butted edge-to-edge,
        # so each one's border reads as its own box rather than one continuous grid line.
        thumb_px = config.RECENT_CROPS_THUMB_PX
        caption_px = config.RECENT_CROPS_CAPTION_PX
        gap_px = 6
        tile_h = thumb_px + caption_px
        tiles = []
        for encoded_bytes, class_label in recent_labeled_crops:
            crop_image = cv2.imdecode(np.frombuffer(encoded_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            body = cv2.resize(crop_image, (thumb_px, thumb_px), interpolation=cv2.INTER_AREA)
            body = cv2.cvtColor(body, cv2.COLOR_GRAY2BGR)

            tile = np.zeros((tile_h, thumb_px, 3), dtype=np.uint8)
            tile[:thumb_px, :] = body
            cv2.rectangle(tile, (0, 0), (thumb_px - 1, thumb_px - 1), config.CROP_BOX_COLOR, 1)

            label_text = class_label or "?"
            (text_w, _text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = max(2, (thumb_px - text_w) // 2)
            cv2.putText(tile, label_text, (text_x, thumb_px + caption_px - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)

        # Wraps into a grid up to RECENT_CROPS_COLUMNS wide instead of one ever-widening
        # row -- at RECENT_CROPS_COUNT=40 a single row would be ~4000px wide, well past
        # the browser viewport, forcing the page to scroll/clip it. Padded with blank
        # tiles up to a full last row so every row hstacks to the same width (a ragged
        # last row would fail vstack's shape check, not just look uneven).
        #
        # Deque rarely holds a clean multiple of RECENT_CROPS_COLUMNS (it fills up
        # gradually, and RECENT_CROPS_MIN_PEAK_VAL makes qualifying crops sparse), so
        # always using exactly RECENT_CROPS_COLUMNS columns would routinely leave a
        # chunk of the last row visibly blank. Instead pick, from a small window of
        # column counts near RECENT_CROPS_COLUMNS, whichever leaves the least padding --
        # ties broken toward more columns, to keep the grid wide rather than tall. Never
        # narrower than 4 columns even if that means accepting some padding -- a tall
        # 1-2 column strip is a worse look than a partly-blank last row.
        n_tiles = len(tiles)
        max_cols = min(config.RECENT_CROPS_COLUMNS, n_tiles)
        min_cols = min(max_cols, 4)
        n_cols = min(range(min_cols, max_cols + 1), key=lambda c: ((-n_tiles) % c, -c))
        blank_tile = np.zeros((tile_h, thumb_px, 3), dtype=np.uint8)
        tiles += [blank_tile] * (-len(tiles) % n_cols)

        gap_col = np.zeros((tile_h, gap_px, 3), dtype=np.uint8)
        row_imgs = []
        for i in range(0, len(tiles), n_cols):
            row_tiles = tiles[i:i + n_cols]
            row_img = row_tiles[0]
            for tile in row_tiles[1:]:
                row_img = np.hstack([row_img, gap_col, tile])
            row_imgs.append(row_img)

        gap_row = np.zeros((gap_px, row_imgs[0].shape[1], 3), dtype=np.uint8)
        grid = row_imgs[0]
        for row_img in row_imgs[1:]:
            grid = np.vstack([grid, gap_row, row_img])

        ok, encoded_strip = cv2.imencode(".jpg", grid)
        if ok:
            latest_crops_strip_jpeg = encoded_strip.tobytes()


def read_live_metadata() -> dict:
    """Reads metadata.py's live snapshot (metadata_live.json) -- analyse.py
    never touches the M0 serial port or Modbus client directly, metadata.py is
    the sole owner of both; this just reads what it already wrote out."""
    path = config.HEARTBEAT_DIR / "metadata_live.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_send_heartbeat() -> dict:
    """Reads send.py's own heartbeat.json directly -- no exclusive-ownership concern
    here (unlike the M0/strobe reads, which go through metadata.py), it's just a plain
    JSON file, same pattern metadata.py's own heartbeat_status() already uses for
    record/analyse/send. Used for the live-stream's Connection chapter."""
    path = config.HEARTBEAT_DIR / "send.heartbeat.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_record_heartbeat() -> dict:
    """Reads record.py's own heartbeat.json, same pattern as read_send_heartbeat()
    above -- used for the live-stream's "last recorded" timer. last_frame_timestamp_unix
    is only set there when a frame was actually captured and kept (not on every
    heartbeat write), so this reflects real camera activity, not just process liveness."""
    path = config.HEARTBEAT_DIR / "record.heartbeat.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def compute_stage_remaining() -> list[int]:
    """Current remaining count for each stage in fullframe_queues, in order -- 0 for
    stages already finished, fullframe_backlog's length for the active stage. As of
    2026-08-11, fullframe_queues is just the live queue (sdb1's old backlog is
    deliberately left out -- see config.py's QUE_FULLFRAMES_BACKLOG_RESTING), so this is
    just [len(fullframe_backlog)] in practice; kept list-shaped rather than hardcoded to
    one stage in case a backlog stage is ever reintroduced. Shared by
    estimate_camera_live_eta() and the backlog/live mode switch below."""
    return [len(fullframe_backlog) if i == active_queue_idx else 0 for i in range(len(fullframe_queues))]


def compute_total_backlog() -> int:
    return sum(compute_stage_remaining())


def update_system_mode() -> None:
    """Hysteresis-based backlog-mode/live-mode switch -- see config.py's
    BACKLOG_MODE_ENTER_THRESHOLD/EXIT_THRESHOLD docstring for why two thresholds, not
    one. Cheap enough to call every loop iteration: compute_total_backlog() only touches
    already-in-memory/cached state, never lists a directory itself."""
    global system_mode
    total_backlog = compute_total_backlog()
    if system_mode == "live" and total_backlog > config.BACKLOG_MODE_ENTER_THRESHOLD:
        system_mode = "backlog"
        print(f"Total backlog ({total_backlog}) exceeded BACKLOG_MODE_ENTER_THRESHOLD "
              f"({config.BACKLOG_MODE_ENTER_THRESHOLD}) -- entering backlog mode, video stream off.")
    elif system_mode == "backlog" and total_backlog < config.BACKLOG_MODE_EXIT_THRESHOLD:
        system_mode = "live"
        print(f"Total backlog ({total_backlog}) dropped below BACKLOG_MODE_EXIT_THRESHOLD "
              f"({config.BACKLOG_MODE_EXIT_THRESHOLD}) -- entering live mode, video stream on.")


def estimate_camera_live_eta(record_hb: dict) -> dict:
    """Projects when every currently-queued frame will be processed AND analyse.py will
    have caught up to record.py's live production -- i.e. 'no images waiting on disk
    anywhere.' Simplified 2026-08-11: with sdb1's old backlog deliberately left out of
    fullframe_queues (see config.py's QUE_FULLFRAMES_BACKLOG_RESTING), there's no
    separate backlog phase to account for anymore, just a straight chase -- the total
    queued (compute_total_backlog()) closing at (analyse_rate - record_rate) if that's
    positive. If it isn't, there's no finite ETA (returns catching_up=False,
    eta_hours=None) rather than a nonsense negative/infinite number.

    Both rates are lifetime averages since each process's own last restart (see
    process_start_unix in both analyse.py and record.py) -- noisy right after either
    restarts, converging to a stable estimate as more frames accumulate."""
    now = time.time()
    analyse_elapsed = now - process_start_unix
    analyse_rate = frames_analysed_total / analyse_elapsed if analyse_elapsed > 0 and frames_analysed_total else None

    record_start = record_hb.get("process_start_unix")
    record_total = record_hb.get("frames_recorded_total")
    record_rate = (
        record_total / (now - record_start)
        if record_start and record_total and now > record_start else None
    )

    if not analyse_rate or record_rate is None:
        return {"eta_hours": None, "analyse_rate_per_hour": None, "record_rate_per_hour": None, "catching_up": None}

    net_rate = analyse_rate - record_rate
    catching_up = net_rate > 0
    eta_hours = (compute_total_backlog() / net_rate) / 3600 if catching_up else None

    return {
        "eta_hours": eta_hours,
        "analyse_rate_per_hour": analyse_rate * 3600,
        "record_rate_per_hour": record_rate * 3600,
        "catching_up": catching_up,
    }


if config.ENABLE_LIVE_STREAM:
    from flask import Flask, Response, jsonify

    flask_app = Flask(__name__)

    @flask_app.route("/")
    def _stream_index():
        # Video image + crops-thumbnail panel, and the JS that polls/renders them --
        # entirely skipped in backlog mode (2026-08-10, see update_system_mode() /
        # config.py's BACKLOG_MODE_ENTER_THRESHOLD) to save the per-frame render cost
        # while draining the fullframe backlogs, so the page just shows a plain note
        # instead of a permanently-broken image, and doesn't poll two endpoints that
        # would 404 forever. The METADATA status panel below is unaffected either way --
        # /status keeps reporting everything (temps, disk, connection, last-recorded/
        # analysed timers, backlog progress) regardless of mode.
        if system_mode == "live":
            video_html = (
                # width/height (not max-width/max-height) so a small source image -- e.g. at a low
                # LIVESTREAM_DISP_SCALE, chosen for faster transmit/decode -- gets scaled UP to fill
                # the viewport instead of rendering at its tiny native pixel size. min(100vw,100vh)
                # instead of object-fit:contain -- the source frame is always square (IMAGE_SIZE_PX x
                # IMAGE_SIZE_PX, downscaled with equal fx/fy in update_live_frame), so sizing both
                # width/height to the smaller viewport dimension reproduces contain's letterboxing
                # exactly, but as the img element's own box -- not a transparent box around it -- so
                # the border below hugs the visible video, not empty space out to the viewport edge.
                # position:fixed;bottom:0;right:0 replaces object-position for the same bottom-right pin.
                '<img id="videoFrame" style="position:fixed;bottom:0;right:0;'
                'width:min(100vw,100vh);height:min(100vw,100vh);display:block;'
                'border:2px solid #00ff00;box-sizing:border-box">'
                '<div style="position:fixed;top:16px;right:16px;padding:4px 10px;'
                'font-size:11px;font-weight:bold;letter-spacing:0.6px;color:#fff;'
                'background:rgba(0,0,0,0.6);border:1px solid #00ff00;border-radius:4px;'
                'font-family:-apple-system,Menlo,Consolas,monospace">LIVESTREAM</div>'
            )
            crops_panel_html = (
                # Crops panel (border + header) is always visible, even before any labeled crop
                # exists yet -- only the <img> itself and the placeholder text toggle, so it's
                # always clear from the page whether the feature is present vs. genuinely has
                # nothing to show yet (CLASSIFY=False, or no crop has cleared CONFIDENCE_THRESHOLD).
                # Same background as the metadata panel below it, so the two read as one group.
                '<div style="padding:8px;border:2px solid #00ff00;border-radius:6px;'
                'background:rgba(10,10,10,0.88);box-shadow:0 4px 18px rgba(0,0,0,0.45)">'
                '<div style="font-size:11px;font-weight:bold;letter-spacing:0.6px;color:#fff;'
                f'font-family:-apple-system,Menlo,Consolas,monospace;margin-bottom:6px">LAST {config.RECENT_CROPS_COUNT} CROPS</div>'
                '<img id="cropsStrip" style="display:none;max-width:100%;border-radius:3px">'
                '<span id="cropsPlaceholder" style="display:block;font-size:11px;color:#aaa;'
                'font-family:-apple-system,Menlo,Consolas,monospace">Waiting for labeled crops...</span>'
                "</div>"
            )
            video_js = (
                # Polls a single always-fresh frame instead of holding open a multipart MJPEG
                # stream -- see /latest_frame.jpg's comment for why: a persistent stream can only
                # fall further behind if the browser/network ever briefly lags the production
                # rate, since it must display every buffered frame in order before reaching "now".
                # A fresh request each tick always gets whatever's current, nothing accumulates.
                "function updateFrame(){"
                "document.getElementById('videoFrame').src='/latest_frame.jpg?t='+Date.now();"
                "}"
                "updateFrame();setInterval(updateFrame,400);"
                # Polled slower than the main frame (1s vs 400ms) -- labeled crops accumulate
                # far slower than raw frames, so there's nothing to gain from a faster poll here.
                "const cropsStripImg=document.getElementById('cropsStrip');"
                "const cropsPlaceholder=document.getElementById('cropsPlaceholder');"
                "cropsStripImg.onload=function(){"
                "cropsStripImg.style.display='block';cropsPlaceholder.style.display='none';"
                "};"
                "cropsStripImg.onerror=function(){"
                "cropsStripImg.style.display='none';cropsPlaceholder.style.display='block';"
                "};"
                "function updateCropsStrip(){"
                "cropsStripImg.src='/latest_crops_strip.jpg?t='+Date.now();"
                "}"
                "updateCropsStrip();setInterval(updateCropsStrip,1000);"
            )
        else:
            video_html = ""
            crops_panel_html = (
                '<div style="padding:8px 12px;border:2px solid #00ff00;border-radius:6px;'
                'background:rgba(10,10,10,0.88);box-shadow:0 4px 18px rgba(0,0,0,0.45);'
                'font-size:11px;color:#aaa;font-family:-apple-system,Menlo,Consolas,monospace">'
                "Video + crops feed off (backlog mode) -- draining fullframe backlogs.</div>"
            )
            video_js = ""

        return (
            '<html><head><title>Jellyscope Livestream</title></head>'
            '<body style="margin:0;background:#000;min-height:100vh;overflow:auto">'
            + video_html +
            # Single fixed top-left column, in normal document flow (not each child
            # individually position:fixed like the old status/logtail pair was) -- crops
            # stacks above metadata here, and neither can ever overlap or need a
            # JS-computed offset to avoid each other, unlike the old logtail box that
            # measured status's rendered height on every /status poll to place itself.
            '<div style="position:fixed;top:16px;left:16px;max-width:calc(100vw - 32px);'
            'display:flex;flex-direction:column;gap:10px">'
            + crops_panel_html +
            "<div>"
            '<div style="font-size:11px;font-weight:bold;letter-spacing:0.6px;color:#fff;'
            'font-family:-apple-system,Menlo,Consolas,monospace;margin-bottom:6px">METADATA</div>'
            # <div>, not <pre> -- a multi-column layout needs a block container it can
            # split into columns; individual lines below still get pre's whitespace
            # preservation (double-space field separators, explicit \n line breaks) via
            # white-space:pre-wrap instead, which additionally (unlike plain pre) wraps a
            # too-long line within its column rather than overflowing it horizontally.
            '<div id="status" style="margin:0;padding:16px 20px;font-size:15px;line-height:1.6;'
            'color:#fff;background:rgba(10,10,10,0.88);border:2px solid #00ff00;border-radius:10px;'
            'white-space:pre-wrap;font-family:-apple-system,Menlo,Consolas,monospace;'
            'column-count:3;column-gap:26px;'
            'box-shadow:0 4px 18px rgba(0,0,0,0.45)"></div>'
            "</div>"
            "</div>"
            "<script>"
            + video_js +
            "async function updateStatus(){"
            "try{"
            "const d=await (await fetch('/status')).json();"
            "const fmt=(v,u)=>(v===null||v===undefined)?'N/A':v.toFixed(1)+u;"
            "const overMax=(v,m)=>v!==null&&v!==undefined&&m!==null&&m!==undefined&&v>=m;"
            "const nearMax=(v,m)=>v!==null&&v!==undefined&&m!==null&&m!==undefined&&v>=0.9*m;"
            "const flag=(v,m)=>overMax(v,m)?' [DANGER]':nearMax(v,m)?' [WARN]':'';"
            "const colorize=(text,color)=>color?`<span style=\"color:${color}\">${text}</span>`:text;"
            "const tempColor=(v,m)=>overMax(v,m)?'#ff4d4d':nearMax(v,m)?'#ffa64d':null;"
            "const vsMax=(v,m,u)=>colorize(`${fmt(v,u)} / max ${fmt(m,u)}${flag(v,m)}`,tempColor(v,m));"
            "const hdr=(label)=>`<b style=\"font-size:17px;letter-spacing:0.4px\">${label}</b>\\n`;"
            "const fmtSpeed=(bps)=>{"
            "if(bps===null||bps===undefined)return'N/A';"
            "if(bps>=1e6)return(bps/1e6).toFixed(2)+' MB/s';"
            "if(bps>=1e3)return(bps/1e3).toFixed(1)+' KB/s';"
            "return bps.toFixed(0)+' B/s';"
            "};"
            "const fmtAge=(s)=>{"
            "if(s===null||s===undefined)return'N/A';"
            "if(s<120)return s.toFixed(0)+'s ago';"
            "if(s<7200)return(s/60).toFixed(1)+'m ago';"
            "if(s<172800)return(s/3600).toFixed(1)+'h ago';"
            "return(s/86400).toFixed(1)+'d ago';"
            "};"
            "const fmtTime=(u)=>(u===null||u===undefined)?'N/A':new Date(u*1000).toLocaleTimeString();"
            "const age=(u)=>(u===null||u===undefined)?null:(Date.now()/1000)-u;"
            "const fmtEta=(h)=>{"
            "if(h===null||h===undefined)return'N/A';"
            "if(h<48)return h.toFixed(1)+'h';"
            "return(h/24).toFixed(1)+'d';"
            "};"
            # Each section is its own break-inside:avoid block, so the column layout
            # below only ever breaks *between* sections, never through the middle of
            # one -- margin-bottom (not the old blank "\n \n" line) provides the gap
            # between sections stacked in the same column. Optional `color` tints the
            # whole section (header + lines) -- used for CONNECTION when send.py is
            # down, so only that section goes red instead of the whole box.
            "const section=(title,lines,color)=>`<div style=\"break-inside:avoid;"
            "-webkit-column-break-inside:avoid;margin-bottom:10px${color?`;color:${color}`:''}\">`+"
            "hdr(title)+lines.join('\\n')+`</div>`;"
            "document.getElementById('status').innerHTML="
            "`<div style=\"break-inside:avoid;-webkit-column-break-inside:avoid;margin-bottom:10px\">"
            "<b style=\"font-size:18px\">${d.time}</b>\\n"
            "Mode: ${colorize(d.system_mode==='live'?'LIVE':'BACKLOG',d.system_mode==='live'?null:'#ffa64d')}"
            " (${(d.total_backlog||0).toLocaleString()} pending)\\n"
            "Last recorded: ${fmtTime(d.last_recorded_unix)} (${fmtAge(age(d.last_recorded_unix))})\\n"
            "Last analysed: ${fmtTime(d.last_analysed_unix)} (${fmtAge(age(d.last_analysed_unix))})\\n"
            "Crops in last image: ${d.last_analysed_crops}  Crops in last 24h: ${d.crops_last_24h}\\n"
            "Sampling: every ${d.frame_skip} frame(s)</div>`+"
            # BACKLOG only rendered while there's actually one to report -- once
            # system_mode is 'live' there's nothing left to catch up on, so the whole
            # section (including "Fully live in", which only makes sense as a countdown
            # while still catching up) disappears rather than sitting there showing a
            # stale/meaningless "done" state.
            "(d.system_mode==='live'?'':section('BACKLOG',["
            "`Fully live in: ${d.camera_live_catching_up===true?fmtEta(d.camera_live_eta_hours):"
            "d.camera_live_catching_up===false?`NOT catching up (record ${fmt(d.record_rate_per_hour,'/hr')} "
            "vs analyse ${fmt(d.analyse_rate_per_hour,'/hr')})`:'estimating...'}`,"
            "...(d.backlog_stages||[]).map(s=>"
            "`${s.label}: ${s.status==='done'?'done':s.status==='pending'?'pending':"
            "'active, '+s.remaining.toLocaleString()+' remaining'"
            "+(s.rate_per_hour?' -- '+s.rate_per_hour.toFixed(0)+'/hr, ETA '+fmtEta(s.eta_hours):'')}`)"
            "]))+"
            "section('QUEUE',["
            "`Avg processing time: ${fmt(d.avg_processing_time_s,' s')}`,"
            "`Crops queue: ${(d.que_crops_depth||0).toLocaleString()}  "
            "Full frames queue: ${(d.que_fullframes_depth||0).toLocaleString()}`,"
            "])+"
            "section('LEAK DETECTION',["
            "`Leak: ${d.leak_detected===true?'!!! DETECTED !!!':d.leak_detected===false?'OK':'N/A'}`,"
            "`Enclosure humidity: ${fmt(d.bme280_humidity_pct,'%')}  "
            "Dew point: ${fmt(d.dew_point_c,' C')}`,"
            "])+"
            "section('ENVIRONMENTAL',["
            "`Bar3XT pressure: ${fmt(d.bar3xt_pressure_mbar,' mbar')}  "
            "depth: ${fmt(d.bar3xt_depth_m,' m')}`,"
            "`Bar3XT temp: ${fmt(d.bar3xt_temp_c,' C')}  "
            "DS18B20 temp: ${fmt(d.ds18b20_temp_c,' C')}`,"
            "])+"
            "section('DEVICE',["
            "`Enclosure pressure: ${fmt(d.bme280_pressure_mbar,' mbar')}`,"
            "`Enclosure temp (BME280): ${fmt(d.bme280_temp_c,' C')}`,"
            "`Camera: ${vsMax(d.camera_temp_c,d.camera_max_temp_c,' C')}`,"
            "`Strobe converter: ${vsMax(d.strobe_converter_temp_c,d.strobe_max_temp_c,' C')}`,"
            "`Strobe driver: ${vsMax(d.strobe_driver_temp_c,d.strobe_max_temp_c,' C')}`,"
            "`Jetson: ${vsMax(d.jetson_temp_c_mean,d.jetson_max_temp_c,' C')}`,"
            "`CPU: ${fmt(d.cpu_percent,'%')}  GPU: ${fmt(d.gpu_percent_mean,'%')} (max ${fmt(d.gpu_percent_max,'%')})`,"
            "`Disk free: device: ${fmt(d.disk_root,' GB')}  "
            "ssd1: ${fmt(d.disk_ssd1,' GB')}  ssd2: ${fmt(d.disk_ssd2,' GB')}`,"
            "])+"
            "section('CONNECTION',["
            "`Status: ${!d.send_alive?'PROCESS DOWN':(d.consecutive_upload_failures>0?"
            "`FAILING (${d.consecutive_upload_failures}x)`:'OK')}`,"
            "`Last successful upload: ${fmtAge(d.last_success_age_s)}  "
            "Last upload speed: ${fmtSpeed(d.upload_speed_bps)}`,"
            "`Link speed: ${fmtSpeed(d.link_speed_bps)} "
            "(probed ${fmtAge(d.link_speed_probed_age_s)})`,"
            "],!d.send_alive?'#ff4d4d':null);"
            # Only a positive leak reading turns the whole box red -- temperature
            # readings are flagged in place per-value (tempColor/vsMax above) and a
            # down send.py only tints the CONNECTION section (color arg above), so
            # one bad reading doesn't drown out everything else on the page.
            "const danger=d.leak_detected===true;"
            "document.getElementById('status').style.background="
            "danger?'rgba(192,57,43,0.85)':'rgba(10,10,10,0.88)';"
            "}catch(e){}"
            "}"
            "updateStatus();setInterval(updateStatus,1000);"
            "</script>"
            "</body></html>"
        )

    @flask_app.route("/status")
    def _stream_status():
        live = read_live_metadata()
        disk = live.get("disk_free_gb_by_path", {})
        avg_processing_time_s = (
            total_processing_time_sum / frames_analysed_total if frames_analysed_total else None
        )
        send_hb = read_send_heartbeat()
        send_age_s = (time.time() - send_hb["last_update_unix"]) if send_hb.get("last_update_unix") else None
        last_success_unix = send_hb.get("last_success_unix")
        record_hb = read_record_heartbeat()
        # Cheap by construction -- reads already-in-memory state (which backlog is active,
        # how many stems are left in its cached listing) rather than re-listing any
        # directory, so this costs nothing extra on top of what the main loop already
        # does. See fullframe_queues' module-level declaration above.
        active_queue_elapsed_s = time.time() - active_queue_started_unix if active_queue_started_unix else None
        active_queue_rate_per_hour = (
            active_queue_processed_count / active_queue_elapsed_s * 3600
            if active_queue_elapsed_s and active_queue_processed_count else None
        )
        backlog_stages = [
            {
                "label": label,
                "status": "done" if i < active_queue_idx else ("active" if i == active_queue_idx else "pending"),
                "remaining": (0 if i < active_queue_idx
                              else len(fullframe_backlog) if i == active_queue_idx
                              else None),
                # Only meaningful for the active stage -- rate of frames actually
                # processed since this stage began (not derived from watching remaining
                # shrink, which would be misleading once the live stage keeps refilling
                # from record.py's ongoing production -- see the module-level comment
                # on active_queue_started_unix/active_queue_processed_count).
                "rate_per_hour": active_queue_rate_per_hour if i == active_queue_idx else None,
                "eta_hours": (len(fullframe_backlog) / active_queue_rate_per_hour
                              if i == active_queue_idx and active_queue_rate_per_hour else None),
            }
            for i, (_, label) in enumerate(fullframe_queues)
        ]
        camera_live = estimate_camera_live_eta(record_hb)
        return jsonify({
            "time": time.strftime("%H:%M:%S"),
            "last_recorded_unix": record_hb.get("last_frame_timestamp_unix"),
            "last_analysed_unix": last_analysed_unix,
            "last_analysed_crops": last_analysed_crops,
            "crops_last_24h": crops_last_24h,
            "backlog_stages": backlog_stages,
            "system_mode": system_mode,
            "total_backlog": sum(compute_stage_remaining()),
            "backlog_mode_enter_threshold": config.BACKLOG_MODE_ENTER_THRESHOLD,
            "backlog_mode_exit_threshold": config.BACKLOG_MODE_EXIT_THRESHOLD,
            "camera_live_eta_hours": camera_live["eta_hours"],
            "camera_live_catching_up": camera_live["catching_up"],
            "analyse_rate_per_hour": camera_live["analyse_rate_per_hour"],
            "record_rate_per_hour": camera_live["record_rate_per_hour"],
            "frame_skip": config.FRAME_SKIP,
            "avg_processing_time_s": avg_processing_time_s,
            "bar3xt_pressure_mbar": live.get("bar3xt_pressure_mbar"),
            "bar3xt_depth_m": live.get("bar3xt_depth_m"),
            "bar3xt_temp_c": live.get("bar3xt_temp_c"),
            "bme280_pressure_mbar": live.get("bme280_pressure_mbar"),
            "bme280_humidity_pct": live.get("bme280_humidity_pct"),
            "bme280_temp_c": live.get("bme280_temp_c"),
            "dew_point_c": live.get("dew_point_c"),
            "ds18b20_temp_c": live.get("ds18b20_temp_c"),
            "leak_detected": live.get("leak_detected"),
            "camera_temp_c": live.get("camera_temp_c"),
            "camera_max_temp_c": config.CAMERA_MAX_TEMP_C,
            "strobe_temp_c_mean": live.get("strobe_temp_c_mean"),
            "strobe_converter_temp_c": live.get("strobe_converter_temp_c"),
            "strobe_driver_temp_c": live.get("strobe_driver_temp_c"),
            "strobe_max_temp_c": config.STROBE_MAX_TEMP_C,
            "jetson_temp_c_mean": live.get("jetson_temp_c_mean"),
            "jetson_max_temp_c": config.JETSON_MAX_TEMP_C,
            "cpu_percent": live.get("cpu_percent"),
            "gpu_percent_mean": live.get("gpu_percent_mean"),
            "gpu_percent_max": live.get("gpu_percent_max"),
            "disk_root": disk.get(str(config.ROOT_DISK_PATH)),
            "disk_ssd1": disk.get(str(config.EXTERNAL_SSD_PATHS[0])),
            "disk_ssd2": disk.get(str(config.EXTERNAL_SSD_PATHS[1])),
            # Already computed by metadata.py every METADATA_SAMPLE_INTERVAL_S and read
            # here from its live snapshot -- NOT a fresh queue_io.queue_depth() call on
            # every /status poll (this endpoint's polled once/sec by the live-stream
            # page's setInterval; que_crops has held tens of thousands of files during
            # backlog incidents, and re-listing that every second is exactly the
            # anti-pattern send.py's own crop_backlog caching was built to avoid -- see
            # its module-level comment on the 2026-08-03 incident).
            "que_crops_depth": live.get("que_crops_depth"),
            "que_fullframes_depth": live.get("que_fullframes_depth"),
            "send_alive": (send_age_s is not None and send_age_s < config.HEARTBEAT_STALE_S),
            "send_age_s": send_age_s,
            "upload_speed_bps": send_hb.get("last_speed_bps"),
            "last_upload_ok": send_hb.get("last_attempt_ok"),
            "consecutive_upload_failures": send_hb.get("consecutive_failures"),
            "last_success_age_s": (time.time() - last_success_unix) if last_success_unix else None,
            # Fixed-size probe (transfer.py's probe_link_speed), NOT derived from real
            # crop/json uploads -- upload_speed_bps above goes misleadingly low once real
            # batches shrink to a handful of KB (queue caught up to real-time) and scp's
            # fixed per-call overhead dominates the timing. This is a trustworthy "how
            # fast is the link actually" reading independent of current queue depth.
            "link_speed_bps": send_hb.get("link_speed_bps"),
            "link_speed_probed_age_s": ((time.time() - send_hb["link_speed_probed_unix"])
                                         if send_hb.get("link_speed_probed_unix") else None),
        })

    @flask_app.route("/latest_frame.jpg")
    def _latest_frame():
        # Single-shot fetch of whatever's currently latest, not a persistent multipart
        # stream -- the index page polls this on a timer instead of using /stream, since
        # multipart/x-mixed-replace has no way to skip ahead: if the browser's decode or
        # the network ever falls even briefly behind the ~1fps production rate, it queues
        # every frame in arrival order and can only fall further behind, never catch up,
        # since it must display everything already buffered before it reaches "now". A
        # plain per-request fetch can't accumulate that kind of backlog -- it always
        # returns whatever's current at request time, dropping anything in between.
        with latest_frame_lock:
            frame = latest_frame_jpeg
        if frame is None:
            return Response(status=404)
        resp = Response(frame, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @flask_app.route("/latest_crops_strip.jpg")
    def _latest_crops_strip():
        # Same single-shot-fetch reasoning as /latest_frame.jpg above, and same
        # 404-when-nothing-yet behavior -- the index page's JS hides the <img> on error.
        with recent_crops_lock:
            frame = latest_crops_strip_jpeg
        if frame is None:
            return Response(status=404)
        resp = Response(frame, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @flask_app.route("/stream")
    def _stream_feed():
        def generate():
            while True:
                with latest_frame_lock:
                    frame = latest_frame_jpeg
                if frame is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.2)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def write_heartbeat() -> None:
    now = time.time()
    heartbeat = {
        "last_update": time.strftime("%H:%M:%S", time.localtime(now)),  # human-readable
        "last_update_unix": now,  # what metadata.py's staleness check actually uses
        "frames_analysed_total": frames_analysed_total,
        "crops_produced_total": crops_produced_total,
    }
    tmp = HEARTBEAT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(heartbeat))
    tmp.replace(HEARTBEAT_PATH)


def maybe_send_daily_report() -> None:
    """Checked once per main-loop tick (idle or not, see bottom of file) rather than
    only when a frame is processed -- so the report still fires close to midnight on
    a quiet night with no frames arriving right at the rollover. Fires at most once
    per calendar date change (local time, same time.localtime() basis write_heartbeat()
    already uses), covering whatever date just ended."""
    global daily_report_date, daily_crops_count, daily_mnemiopsis_count
    today = time.strftime("%Y-%m-%d", time.localtime())
    if today == daily_report_date:
        return
    leak_alert.send_slack_alert(
        f":bar_chart: *Daily report for {daily_report_date}*\n"
        f"Crops detected: {daily_crops_count}\n"
        f"Mnemiopsis sightings: {daily_mnemiopsis_count}"
    )
    daily_report_date = today
    daily_crops_count = 0
    daily_mnemiopsis_count = 0


def maybe_queue_training_frame(enhanced, image_name: str, source_frame_id, source_timestamp_unix) -> None:
    """Every TRAINING_FRAME_INTERVAL_S, queues the current post-PREPROCESS (pre-SEGMENT)
    full frame into QUE_TRAINING_FRAMES for send.py to upload -- periodic full-frame
    samples for offline classifier retraining, independent of and much sparser than the
    per-crop que_crops path. Encoded as PNG (lossless, same as crops) rather than TIFF --
    smaller for the same pixels since `enhanced` is already 8-bit, not worth trading
    losslessness away over a per-30-minutes upload."""
    global last_training_frame_unix
    now = time.time()
    if now - last_training_frame_unix < config.TRAINING_FRAME_INTERVAL_S:
        return
    last_training_frame_unix = now

    ok, encoded = cv2.imencode(".png", enhanced)
    if not ok:
        print(f"Failed to encode training frame {image_name}, skipping")
        return
    sidecar = {
        "source_frame_id": source_frame_id,
        "source_timestamp_unix": source_timestamp_unix,
        "image_name": image_name,
        "processed_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        "upload_attempts": 0,
    }
    queue_io.write_item(config.QUE_TRAINING_FRAMES, image_name, encoded.tobytes(), ".png", sidecar)


def segment_and_classify(enhanced, image_name: str, source_frame_id=None, source_timestamp_unix=None) -> tuple[list, dict]:
    """Runs SEGMENT + CLASSIFY on an already-enhanced (post-PREPROCESS) image and
    returns (results, timings) -- results is a list of (crop_stem, encoded_png_bytes,
    crop_sidecar) tuples; timings is {"segment": ..., "inference": ...} (in seconds),
    left for the caller to fold into its own single-line-per-frame summary rather
    than printed here. Doesn't touch any queue itself. Shared by process_frame() below
    (the live, queue-driven production path) and test_analyse_on_folder.py (a
    standalone dev/test harness for already-preprocessed sample images, which has no
    PREPROCESS step to run and no real queue to write into)."""
    
    height, width = enhanced.shape[:2]
    if width != config.IMAGE_SIZE_PX or height != config.IMAGE_SIZE_PX:
        raise ValueError(f"{image_name} is {width}x{height}, expected {config.IMAGE_SIZE_PX}x{config.IMAGE_SIZE_PX} "
                          "(tile_size/upscale_factor are computed assuming this size)")

    t_tiles_start = time.perf_counter()

    ### ==========================
    ### SEGMENT
    ### ==========================

    # Tile extraction/scoring/aggregation/peak-finding/dedup all live in segment_core.py
    # now, shared with test_segmentation_int8.py so the accuracy test exercises the exact
    # same code path production runs (only encode_fn/decode_fn/score_fn differ between the
    # FP32 models here and the INT8 TensorRT engines segmentation_trt.py will provide).

    tiles_tensor, rows_list, cols_list, _tile_bboxes = segment_core.extract_tiles(
        enhanced, device, config.SEG_TILE_GRID_SIZE, offsets, config.SEG_OFFSETS_NORM, tile_size, seg_image_size,
    )
    
    t_tiles_end = time.perf_counter()

    scores_array = segment_core.score_tiles(
        tiles_tensor, rows_list, cols_list, encoder_engine, decoder_engine, scorer_engine,
        config.SEG_ENGINE_BATCH, device,
    )

    t_inference_end = time.perf_counter()

    accepted = []
    if tiles_tensor is not None:
        n_offsets = len(offsets)
        prob_map_small, coverage_map_small = segment_core.aggregate_to_patch_map(
            rows_list, cols_list, scores_array, grid_size_small, n_offsets,
        )
        t_probmap_end = time.perf_counter()

        candidates = segment_core.find_candidates(
            prob_map_small, peak_threshold, secondary_threshold, upscale_factor,
            config.MIN_REGION_SIZE_PATCHES, config.CROP_PADDING_PIXELS, config.IMAGE_SIZE_PX,
        )
        t_peaks_end = time.perf_counter()

        accepted = segment_core.dedup_candidates(candidates, config.IOU_DEDUP_THRESHOLD, config.N_CROPS_PER_IMAGE)
    else:
        t_probmap_end = t_peaks_end = t_inference_end

    t_segment_end = time.perf_counter()

    ### ==========================
    ### CLASSIFY
    ### ==========================

    # One batched ViT forward pass for every crop in this image, instead of one
    # forward pass per crop -- was the dominant real-time cost (ViT-B/16 paying
    # its kernel-launch/sync overhead ~5x per image instead of once).
    results = []
    if accepted:
        crop_images, crop_tensors = [], []
        region_size_pixels_list, region_size_mm2_list, crop_coords_list = [], [], []
        peak_val_list, mean_val_list = [], []

        for cand in accepted:
            crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
            region_size_pixels = cand["region_size_points"] * upscale_factor ** 2
            region_size_mm2 = region_size_pixels * PXL_TO_MM2
            crop_image = enhanced[crop_y0:crop_y1, crop_x0:crop_x1]

            crop_images.append(crop_image)
            region_size_pixels_list.append(region_size_pixels)
            region_size_mm2_list.append(region_size_mm2)
            crop_coords_list.append((crop_y0, crop_x0, crop_y1, crop_x1))
            peak_val_list.append(cand["peak_val"])
            mean_val_list.append(cand["mean_val"])
            if config.CLASSIFY:
                crop_tensors.append(vit_classifier.preprocess_crop_for_classifier(crop_image, image_size=config.CLASSIFIER_IMAGE_SIZE))

        if config.CLASSIFY:
            img_batch = torch.stack(crop_tensors, dim=0).to(device, non_blocking=True)
            size_batch = torch.tensor(region_size_mm2_list, dtype=torch.float32).unsqueeze(1).to(device, non_blocking=True)

            # The ViT INT8 engine has a fixed batch shape (config.VIT_ENGINE_BATCH=16, chosen with
            # headroom over the observed 0-9 crops/image) -- pad each chunk up to it and slice the
            # real rows back out; a frame with more than VIT_ENGINE_BATCH crops (rare) just loops.
            n_crops = img_batch.shape[0]
            logits_chunks = []
            with torch.inference_mode():
                for start_idx in range(0, n_crops, config.VIT_ENGINE_BATCH):
                    end_idx = min(start_idx + config.VIT_ENGINE_BATCH, n_crops)
                    img_chunk = pad_batch(img_batch[start_idx:end_idx], config.VIT_ENGINE_BATCH)
                    size_chunk = pad_batch(size_batch[start_idx:end_idx], config.VIT_ENGINE_BATCH)
                    chunk_logits = classifier_engine(img_chunk, size_chunk)
                    logits_chunks.append(chunk_logits[:end_idx - start_idx])
                logits = torch.cat(logits_chunks, dim=0)
                probs = torch.softmax(logits, dim=1)
                class_indices = torch.argmax(probs, dim=1)
                confidences = probs.gather(1, class_indices.unsqueeze(1)).squeeze(1)

            class_indices = class_indices.cpu().tolist()
            confidences = confidences.cpu().tolist()

        for crop_idx in range(len(accepted)):
            crop_y0, crop_x0, crop_y1, crop_x1 = crop_coords_list[crop_idx]
            region_size_pixels = region_size_pixels_list[crop_idx]
            region_size_mm2 = region_size_mm2_list[crop_idx]
            peak_val = peak_val_list[crop_idx]
            mean_val = mean_val_list[crop_idx]
            crop_image = crop_images[crop_idx]
            if config.CLASSIFY:
                class_idx = class_indices[crop_idx]
                class_confidence = confidences[crop_idx]
                class_label = idx_to_class[class_idx]
            else:
                class_idx = class_confidence = class_label = None

            # Predicted species never goes in the filename -- only in the sidecar JSON
            # (class_label/class_confidence below). CONFIDENCE_THRESHOLD still gates
            # whether the classifier's guess is considered confident enough to trust,
            # just no longer affects what the crop is named.
            crop_stem = f"{image_name}_crop_{crop_idx:03d}_area_{region_size_mm2:.2f}"
            ok, encoded_crop = cv2.imencode(".png", crop_image)
            if not ok:
                print(f"Failed to encode crop {crop_stem}, skipping")
                continue

            crop_sidecar = {
                "source_frame_id": source_frame_id,
                "source_timestamp_unix": source_timestamp_unix,
                "image_name": image_name,
                "crop_id": crop_idx,
                "crop_y0": crop_y0, "crop_x0": crop_x0, "crop_y1": crop_y1, "crop_x1": crop_x1,
                "region_size_pixels": region_size_pixels,
                "region_size_mm2": region_size_mm2,
                "peak_val": peak_val,
                "mean_val": mean_val,
                "class_label": class_label,
                "class_confidence": class_confidence,
                "class_idx": class_idx,
                "classifier_checkpoint": str(config.VIT_CHECKPOINT_PATH) if config.CLASSIFY else None,
                "processed_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                "upload_attempts": 0,
            }
            results.append((crop_stem, encoded_crop.tobytes(), crop_sidecar))

        if config.CLASSIFY:
            del crop_tensors, img_batch, size_batch, logits_chunks, logits, probs, class_indices, confidences
        del crop_images

    t_classify_end = time.perf_counter()
    timings = {
        "segment": t_segment_end - t_tiles_start,
        "inference": t_classify_end - t_segment_end,
    }
    del tiles_tensor, scores_array
    return results, timings


def read_frame_ahead(image_path, json_path) -> dict:
    """Reads one claimed frame's sidecar JSON and TIFF bytes into memory, timed the same
    way process_frame() used to time this inline before the two were split apart (kept
    split -- it's harmless and arguably cleaner -- even though the background-thread
    prefetch this split originally enabled was tried and reverted the same day; see the
    main loop's call site for why)."""
    t_start = time.perf_counter()
    sidecar = queue_io.read_sidecar(json_path)
    t_sidecar_end = time.perf_counter()
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_gray is None:
        raise ValueError(f"Failed to read {image_path}")
    t_read_end = time.perf_counter()
    return {
        "sidecar": sidecar,
        "image_gray": image_gray,
        "sidecar_time": t_sidecar_end - t_start,
        "imread_time": t_read_end - t_sidecar_end,
    }


def process_frame(stem: str, read_result: dict, claim_duration: float = 0.0) -> None:
    """`claim_duration` is however long the caller's queue_io.claim() took; `read_result`
    is whatever read_frame_ahead() produced -- together they fold into the "record"
    timing below, the full cost of retrieving what record.py already produced, as
    opposed to the preprocess/segment/inference/send/stream work analyse.py itself does
    on it."""
    global crops_produced_total, total_processing_time_sum, daily_crops_count, daily_mnemiopsis_count
    global last_analysed_unix, last_analysed_crops, crops_last_24h
    sidecar = read_result["sidecar"]
    image_gray = read_result["image_gray"]
    # TEMP diagnostic breakdown (2026-08-10, investigating the record/send slowdown) --
    # sidecar_time/imread_time split out of record_time below; remove once resolved.
    sidecar_time = read_result["sidecar_time"]
    imread_time = read_result["imread_time"]
    record_time = claim_duration + sidecar_time + imread_time

    height, width = image_gray.shape[:2]
    if width != config.IMAGE_SIZE_PX or height != config.IMAGE_SIZE_PX:
        raise ValueError(f"Frame {stem} is {width}x{height}, expected {config.IMAGE_SIZE_PX}x{config.IMAGE_SIZE_PX}")

    image_name = stem

    ### ==========================
    ### PREPROCESS
    ### ==========================
    t_preprocess_start = time.perf_counter()
    clahe_grid_n = max(1, round(config.IMAGE_SIZE_PX / config.CLAHE_TILE_PX))
    enhanced = gpu_preprocess.gpu_preprocess_frame(
        image_gray, dark_frame, device, config.HDR_MAX, config.MEDIAN_KERNEL_SIZE,
        config.GAMMA, config.POST_GAIN, config.CLAHE_CLIP, (clahe_grid_n, clahe_grid_n),
        rotate_angle_deg=config.ROTATE_FRAME,
    )
    t_preprocess_end = time.perf_counter()

    maybe_queue_training_frame(enhanced, image_name, sidecar["frame_id"], sidecar["timestamp_unix"])

    results, seg_timings = segment_and_classify(
        enhanced, image_name,
        source_frame_id=sidecar["frame_id"], source_timestamp_unix=sidecar["timestamp_unix"],
    )

    t_stream_start = time.perf_counter()
    if config.ENABLE_LIVE_STREAM and system_mode == "live":
        update_live_frame(enhanced, results)
        update_recent_crops_strip(results)
    t_stream_end = time.perf_counter()

    t_send_start = time.perf_counter()
    write_item_times = []  # TEMP diagnostic (2026-08-10) -- remove once resolved
    for crop_stem, encoded_bytes, crop_sidecar in results:
        t_write_start = time.perf_counter()
        queue_io.write_item(config.QUE_CROPS, crop_stem, encoded_bytes, ".png", crop_sidecar,
                             atomic_image=False, atomic_json=False)
        write_item_times.append(time.perf_counter() - t_write_start)
        crops_produced_total += 1
        daily_crops_count += 1
        if crop_sidecar.get("class_label") == config.MNEMIOPSIS_CLASS_LABEL:
            daily_mnemiopsis_count += 1
    t_send_end = time.perf_counter()

    last_analysed_unix = sidecar["timestamp_unix"]
    last_analysed_crops = len(results)
    crops_last_24h_window.append((last_analysed_unix, last_analysed_crops))
    crops_last_24h += last_analysed_crops
    cutoff = last_analysed_unix - 24 * 3600
    while crops_last_24h_window and crops_last_24h_window[0][0] < cutoff:
        _, expired_count = crops_last_24h_window.popleft()
        crops_last_24h -= expired_count

    preprocess_time = t_preprocess_end - t_preprocess_start
    segment_time = seg_timings["segment"]
    inference_time = seg_timings["inference"]
    send_time = t_send_end - t_send_start
    stream_time = t_stream_end - t_stream_start
    total_time = record_time + preprocess_time + segment_time + inference_time + send_time + stream_time
    total_processing_time_sum += total_time

    write_times_str = ",".join(f"{t:.3f}" for t in write_item_times)
    print(f"[{image_name}] record={record_time:.3f}s(claim={claim_duration:.3f} "
          f"sidecar={sidecar_time:.3f} imread={imread_time:.3f}) preprocess={preprocess_time:.3f}s "
          f"segment={segment_time:.3f}s inference={inference_time:.3f}s "
          f"send={send_time:.3f}s[{write_times_str}] stream={stream_time:.3f}s "
          f"total={total_time:.3f}s crops={len(results)}")

    del image_gray, enhanced, results, sidecar


### ==========================
### Main loop
### ==========================
# Guarded behind __main__ so test_analyse_on_folder.py can `from analyse import
# segment_and_classify` (which runs all the setup above -- device, model
# loading, dark frame) without also kicking off the queue-watching loop below --
# and, for the same reason, without starting a real web server on import either.
if __name__ == "__main__":
    if config.ENABLE_LIVE_STREAM:
        # threaded=True matters here -- /stream is a never-ending generator response
        # (one open connection per viewer, for as long as the page is open). Without
        # it, Flask's dev server can end up mostly busy servicing that one connection
        # and let other requests (status polls, new page loads) queue up behind it,
        # which looks like "the frame barely updates" even though update_live_frame()
        # itself runs on every single processed frame, crop or no crop. Confirmed via
        # a real subprocess test: /status stayed ~3ms even with /stream held open.
        #
        # Also override log_request (not just the "werkzeug" logger's level, which
        # doesn't reliably stick -- run_simple() re-touches its own logging setup when
        # the server actually starts, winning the race against a level set beforehand)
        # to drop routine "GET ... 200" per-request access-log noise.
        from werkzeug.serving import WSGIRequestHandler
        WSGIRequestHandler.log_request = lambda self, *args, **kwargs: None

        threading.Thread(
            target=lambda: flask_app.run(
                host="0.0.0.0", port=config.LIVESTREAM_PORT, debug=False, use_reloader=False, threaded=True,
            ),
            daemon=True,
        ).start()
        print(f"Live stream enabled at http://<jetson-ip>:{config.LIVESTREAM_PORT}/")

    # Just the live queue (sda1, where record.py writes) as of 2026-08-11 -- sdb1's old
    # backlog is deliberately left out (see config.py's QUE_FULLFRAMES_BACKLOG_RESTING
    # docstring for why: sdb1 turned out to have a ~38x throughput ceiling versus sda1,
    # a physical USB-negotiation problem, not something worth continuing to fight in
    # software). Kept list-shaped rather than hardcoded to one queue in case a backlog
    # stage is ever reintroduced -- see compute_stage_remaining().
    fullframe_queues = [
        (config.QUE_FULLFRAMES, "live"),
    ]
    for queue_root, label in fullframe_queues:
        for stem in queue_io.recover_processing(queue_root, ".tiff"):
            print(f"Recovering {stem} from a previous crash ({label})...")
            image_path = queue_root / "processing" / f"{stem}.tiff"
            json_path = queue_root / "processing" / f"{stem}.json"
            try:
                process_frame(stem, read_frame_ahead(image_path, json_path))
                queue_io.ack_delete(queue_root, stem, ".tiff")
                frames_analysed_total += 1
            except Exception as exc:
                queue_io.fail_item(queue_root, stem, ".tiff", str(exc))

    print("analyse.py ready, watching the live (sda1) queue...")

    # Stems already listed from disk but not yet claimed -- refilled by list_ready_stems()
    # only once empty, rather than every frame. incoming/ lives on exFAT, where directory
    # listing is ~O(n) in total entries; re-listing the whole directory to pull just one
    # stem at a time made every single frame pay the cost of scanning the *entire* backlog
    # (tens of thousands of entries after an upload outage let one pile up), instead of
    # that cost being paid once per refill. See the 2026-08-03 slow-pipeline incident.
    #
    # active_queue_idx starts at the oldest backlog and only ever moves forward, never
    # back -- once a backlog's list_ready_stems() comes back empty it's done for good
    # (nothing writes there anymore), so there's no reason to keep checking it once the
    # next one takes over. Only the last queue (live) never advances further -- record.py
    # keeps adding to it, so it's checked forever, same as the original single-queue loop.
    active_queue_idx = 0
    fullframe_backlog: list[str] = []
    active_queue_started_unix = time.time()
    active_queue_processed_count = 0

    while True:
        maybe_send_daily_report()

        if frames_analysed_total % config.DISK_CHECK_EVERY_N_FRAMES == 0:
            free = queue_io.disk_free_bytes(config.QUEUE_ROOT)
            if free < config.ANALYSE_MIN_FREE_BYTES:
                print(f"Critically low disk space ({free / 1e9:.2f} GB free), pausing analysis...")
                write_heartbeat()
                time.sleep(5)
                continue

        if not fullframe_backlog:
            fullframe_backlog = queue_io.list_ready_stems(fullframe_queues[active_queue_idx][0])
            while not fullframe_backlog and active_queue_idx < len(fullframe_queues) - 1:
                finished_label = fullframe_queues[active_queue_idx][1]
                active_queue_idx += 1
                next_label = fullframe_queues[active_queue_idx][1]
                print(f"{finished_label} drained -- switching to the {next_label} queue.")
                fullframe_backlog = queue_io.list_ready_stems(fullframe_queues[active_queue_idx][0])
                active_queue_started_unix = time.time()
                active_queue_processed_count = 0

        # After fullframe_backlog is freshly populated above, never before -- calling this
        # any earlier (e.g. right at loop top) would see fullframe_backlog's stale/initial
        # state and could misjudge the backlog as empty for one iteration right at startup
        # (a real bug caught during testing: it flickered into live mode for a single frame
        # before self-correcting). Keep it ahead of the sleep/continue below, though, so
        # mode still updates promptly even on ticks where the queue is (genuinely) empty.
        update_system_mode()

        if not fullframe_backlog:
            time.sleep(0.5)
            continue

        queue_root = fullframe_queues[active_queue_idx][0]
        stem = fullframe_backlog.pop(0)
        t_claim_start = time.perf_counter()
        claimed = queue_io.claim(queue_root, stem, ".tiff")
        if claimed is None:
            continue
        image_path, json_path = claimed
        claim_duration = time.perf_counter() - t_claim_start

        # NOTE: read_frame_ahead() is called synchronously here, not via _read_executor,
        # as of 2026-08-10 -- the background-prefetch version was tried and reverted the
        # same day. The assumption behind it (record's disk read could overlap for free
        # with process_frame's compute) turned out wrong in practice: process_frame's
        # own "send" stage is *also* disk I/O (crop writes), on the same sdb1 filesystem
        # the backlog reads come from, so the background read thread and the main
        # thread's writes ended up contending for the same exFAT lock concurrently --
        # measured 3-4x *slower* (record times of 8-10s vs the previous ~1.2s) than the
        # plain sequential version below. Left read_frame_ahead()/process_frame() split
        # as two functions (harmless, arguably cleaner) but call them back-to-back on the
        # main thread like the original single-function version did.
        try:
            read_result = read_frame_ahead(image_path, json_path)
            process_frame(stem, read_result, claim_duration)
            queue_io.ack_delete(queue_root, stem, ".tiff")
            frames_analysed_total += 1
            active_queue_processed_count += 1
        except Exception as exc:
            print(f"Failed to process {stem}: {exc}")
            queue_io.fail_item(queue_root, stem, ".tiff", str(exc))

        if frames_analysed_total % config.CUDA_CACHE_CLEANUP_EVERY_N_FRAMES == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        write_heartbeat()
