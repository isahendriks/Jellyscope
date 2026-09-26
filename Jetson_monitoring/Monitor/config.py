"""Shared configuration for the monitoring real-time pipeline.

Every stage (record.py, analyse.py, send.py, metadata.py) imports from here for
paths, checkpoint locations, and every tunable parameter -- including stage-specific
tuning knobs (analyse.py's PREPROCESS/SEGMENT/CLASSIFY/live-stream parameters, etc.),
centralized here rather than at the top of each script, for a single place to look
when adjusting any of them. Values derived FROM these parameters (e.g. analyse.py's
tile_size/offsets/PXL_TO_MM, computed from SEG_TILE_GRID_SIZE/IMAGE_SIZE_PX/IMAGE_W_MM)
still live in the script that uses them, not here -- only the tunable inputs do.
"""

import sys
from pathlib import Path

# config.py lives in Monitor/, one level *inside* the pipeline folder now (scripts
# are split into Monitor/ [live pipeline -- be careful], engines/ [INT8 build
# tooling], and Test/ [safe to experiment with], all siblings under
# Jetson_monitoring/). PIPELINE_DIR is that top-level folder -- where models/,
# trt/, queues/, logs/ etc. actually live -- not config.py's own directory.

MONITOR_DIR = Path(__file__).resolve().parent            # Jetson_monitoring/Monitor/
PIPELINE_DIR = MONITOR_DIR.parent                        # Jetson_monitoring/
REPO_ROOT = PIPELINE_DIR.parent                          # Jellyscope/

# So `from models import ...` (models/ is a child of PIPELINE_DIR, a sibling of
# Monitor/) and `from Pipeline_development.BinaryClassification.functions import
# ...` both work from any of the pipeline scripts, regardless of which of
# Monitor/engines/Test they live in, without a per-file sys.path.append hack.
for _p in (PIPELINE_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Self-contained venv living inside Jetson_monitoring/ itself -- no longer
# borrowed from Jetson_control/, now that Jetson_monitoring/ has its own copy
# (the goal being to eventually retire the rest of Jetson_control/ entirely).
VENV_PYTHON = PIPELINE_DIR / ".venv-jellyscope_on_jetson" / "bin" / "python"

### ==========================
### Disk-backed queues
### ==========================
# QUEUE_ROOT lives entirely on sda1 as of 2026-08-11 -- both external SSDs are the same
# "Portable SSD" model on paper, but a raw dd read test found sdb1 negotiating its USB
# link at ~5.7 MB/s versus sda1's ~219 MB/s (confirmed via kernel device paths: sdb sits
# on usb1, the tegra-xusb controller's USB2 companion root hub, while sda sits on usb2,
# its USB3 root hub -- a cable/connection-negotiation problem on that specific port/cable,
# not a "wrong port" one, since this devkit's ports are all USB 3.2 electrically). That
# ~38x gap was the real, dominant cause of the record/imread slowness chased through most
# of the 2026-08-10 investigation -- exFAT contention was real but secondary. Decision
# (2026-08-11): rather than keep fighting sdb1's throughput ceiling, leave its ~36k-frame
# backlog untouched ("let it rest" -- see QUE_FULLFRAMES_BACKLOG_RESTING below) and run
# the entire active pipeline -- que_fullframes (already here since 2026-08-10 for a
# different reason, see git history), que_crops, que_training_frames, oversized_crops --
# on sda1 instead.
QUEUE_ROOT = Path("/mnt/sda1/jellyscope_queues")

QUE_FULLFRAMES = QUEUE_ROOT / "que_fullframes"  # record.py's target

# sdb1's old backlog (merged 2026-08-10 from two separate incidents -- see git history for
# that whole story) -- as of 2026-08-11, deliberately NOT part of analyse.py's active
# queue list anymore (see its main loop) and not referenced by anything else in this
# file. Left in place on sdb1, untouched, since sdb1's throughput problem (see QUEUE_ROOT
# above) makes continuing to drain it impractical for now. Kept as a named constant purely
# so its location doesn't need rediscovering if it's ever revisited.
QUE_FULLFRAMES_BACKLOG_RESTING = Path("/mnt/sdb1/jellyscope_queues/que_fullframes_backlog")

QUE_CROPS = QUEUE_ROOT / "que_crops"
QUE_TRAINING_FRAMES = QUEUE_ROOT / "que_training_frames"  # periodic post-PREPROCESS full
# frames sampled by analyse.py (see TRAINING_FRAME_INTERVAL_S) for send.py to upload --
# separate from QUE_CROPS, which holds individual per-crop classifier images instead

TRAINING_FRAME_INTERVAL_S = 30 * 60  # how often analyse.py samples one full frame (after
# PREPROCESS, before SEGMENT) into QUE_TRAINING_FRAMES, for periodic offline classifier
# retraining -- independent of and much sparser than the per-crop que_crops upload path
TRAINING_COLLECTION_TARGET = 200  # 0 = disabled; set to e.g. 200 for a one-shot collection
TRAINING_COLLECTION_INTERVAL_S = 10  # seconds between frames while collection is active
TRAINING_COLLECTION_STATE_PATH = PIPELINE_DIR / "logs" / "training_collection_state.json"

### ==========================
### Oversized crop handling (send.py) -- see the 2026-07-31 incident: something (camera
### obstruction, lighting glitch, or a SEGMENT false-positive -- still unconfirmed which)
### made SEGMENT briefly treat several whole 4512x4512 frames as one giant "crop" each
### (~7-10MB PNGs, vs ~100KB for a normal crop). scp uploads a batch in one shot, so a
### single one of these made its whole 20-item batch too big to transfer inside
### UPLOAD_TIMEOUT_S over the field link -- which jammed every normal crop queued behind
### it too, since que_crops is strict FIFO. MAX_UPLOAD_ATTEMPTS_TOTAL would eventually
### fail an item like this out on its own, but at ~minutes per attempt that's far too
### slow to be a real mitigation.
### ==========================
MAX_CROP_UPLOAD_BYTES = 2_000_000  # 2MB -- comfortably above any real crop seen so far
# (largest observed ~380KB), well below the ~7-10MB pathological full-frame ones. A
# crop's PNG over this size is archived straight to OVERSIZED_CROPS_DIR instead of ever
# being hand to scp -- see send.py's process_batch().
OVERSIZED_CROPS_DIR = QUEUE_ROOT / "oversized_crops"  # local-only, never uploaded -- kept
# on disk for manual review/retrieval rather than lost, without blocking the rest of the
# queue or burning retry cycles on a transfer that's very unlikely to complete in time

# A sustained obstruction (see this section's docstring above) can archive a new
# multi-MB crop every frame indefinitely -- nothing else bounds how big
# OVERSIZED_CROPS_DIR grows. Once its total size crosses OVERSIZED_CROPS_MAX_BYTES,
# send.py deletes the oldest archived crops (mtime order) down to
# OVERSIZED_CROPS_PRUNE_TARGET_BYTES -- a lower target, not the same number as the
# trigger, so a still-ongoing obstruction doesn't force a full directory rescan on
# every single archive event (same hysteresis idea as BACKLOG_MODE_ENTER/EXIT_THRESHOLD
# above). Safe to lose the oldest ones first: this directory is local-only/never
# uploaded, so pruning it never discards anything server-lab had a copy of.
OVERSIZED_CROPS_MAX_BYTES = 150_000_000_000  # 150GB
OVERSIZED_CROPS_PRUNE_TARGET_BYTES = 100_000_000_000  # 100GB

# The two mounted external SSDs -- sda1 (training data) and sdb1 (this pipeline's
# queues) -- reported alongside the Jetson's own root partition in metadata.py's
# disk-space fields.
ROOT_DISK_PATH = Path("/")
EXTERNAL_SSD_PATHS = [Path("/mnt/sda1"), Path("/mnt/sdb1")]

# Standard Jetson/Tegra sysfs GPU load node -- reports 0-1000 (permille), not 0-100.
GPU_LOAD_PATH = Path("/sys/devices/platform/bus@0/17000000.gpu/load")

MIN_FREE_BYTES = 2 * 1024**3  # record.py pauses acquisition (stops *producing* new raw
# frames into que_fullframes) below this.

# analyse.py *consumes* que_fullframes -- it deletes each raw frame (tens of MB) after
# extracting a handful of much smaller crop PNGs from it, so low free space is a reason
# for it to keep draining the backlog, not a reason to stop. Gating it on MIN_FREE_BYTES
# the same way record.py is caused the 2026-08-10 incident: disk fills up, analyse.py
# pauses right alongside record.py, so nothing ever gets deleted and free space can
# never recover on its own -- record.py stays paused forever too, since its own
# MIN_FREE_BYTES check never sees space come back. This is only a hard floor against
# ENOSPC crashing mid-write on the crop files analyse.py still writes, not a real
# backpressure threshold, so it's kept far below MIN_FREE_BYTES.
ANALYSE_MIN_FREE_BYTES = 200 * 1024**2

### ==========================
### Camera sampling rate -- the hardware trigger itself can't be changed (fixed
### line-trigger rate), so record.py instead keeps only every Nth triggered
### frame and discards the rest. 1 = keep every frame. Reported on the live
### stream (see analyse.py's /status) so it's visible what rate is actually
### being recorded at.
### ==========================
FRAME_SKIP = 3  # 2026-08-10: raised from 2 -- at FRAME_SKIP=2 (~0.48 kept frames/s) record.py
# outpaced analyse.py's measured ~0.39 frames/s, so the live queue would only ever grow
# once the historical backlogs cleared, never reach empty. At 3 (~0.32 kept frames/s),
# analyse.py should net-drain rather than net-fill it. See analyse.py's live-stream ETA
# for whether this is actually holding in practice.

# Optional dark/background reference frame (control/monitoring/record.py's "bkg.png"),
# subtracted during analyse.py's PREPROCESS step. If this path doesn't exist,
# analyse.py skips dark-frame subtraction with a warning rather than crashing.
DARK_FRAME_PATH = PIPELINE_DIR / "bkg.png"

### ==========================
### Segmentation checkpoints (already trained -- never retrained by this pipeline;
### on-board retraining lives in train/, sibling to this repo's Jetson_monitoring/ folder)
### ==========================
SEGMENTATION_ENCODER_TYPE = "AE"  # "AE" or "VAE" -- must match whichever class the checkpoint below was trained with
SEGMENTATION_AE_MODEL_PATH = PIPELINE_DIR / "models" / "Kristineberg_260814_AE_model16_l64_img128.pth"

# Kristineberg_260814 was trained with its positional embeddings zeroed out, so
# inference has to match: False feeds the encoder's fusion layers a zero embedding
# instead of computing a real one from each tile's row/col (see functions.py's
# Encoder/VariationalEncoder -- self.positional_embeddings_used). Flip to True for a
# checkpoint that was actually trained with positional embeddings.
# NOTE: analyse.py runs SEGMENT as a pre-built INT8 TensorRT engine (see
# models/segmentation_trt.py), not this FP32 model directly -- this flag only takes
# effect where the FP32 model is actually constructed (engines/export_onnx.py,
# engines/collect_calibration_data.py, engines/benchmark_int8_trt.py). Changing it
# requires re-running export_onnx.py -> collect_calibration_data.py ->
# build_trt_int8.py to rebuild seg_encoder_int8.engine before it affects production.
POSITIONAL_EMBEDDINGS_USED = False

# Either scorer mode Pipeline_development/BinaryClassification/Train/train_DNN.py
# can save works here directly, no other change needed: a "binary" checkpoint
# (TwoClassScorer, saved under .../NN/TwoClass/) or a "mahalanobis" one (mean_vec/
# cov_inv, saved under .../NN/Mahalanobis/). Which mode this file is in gets
# auto-detected from its own keys (models/segmentation.py's detect_scorer_type()) --
# there's no separate flag here to keep in sync with whichever one you point at.
# A "binary" scorer runs as an INT8 TensorRT engine like everything else; a
# "mahalanobis" one runs directly as FP32 PyTorch (see models/segmentation_trt.py's
# module docstring for why).
SEGMENTATION_SCORER_MODEL_PATH = PIPELINE_DIR / "models" / "Kristineberg_260814_scorer_mahalanobis_model16_l64_img128.pth"

# Overrides the checkpoint's own saved threshold (best_threshold_f3 at training time --
# see train_DNN.py) without hand-editing/re-saving the .pth file. None = use whatever the
# checkpoint stored. Handy for quickly sweeping thresholds against a labeled test folder
# (Test/test_analyse_on_folder.py) when the trained threshold turns out too strict/loose
# for real data -- e.g. a Mahalanobis checkpoint whose saved 0.9224 finds zero crops on
# known-observation frames. Applies uniformly to either scorer mode (analyse.py just does
# peak_threshold = scorer_threshold either way).
SEGMENTATION_SCORER_THRESHOLD_OVERRIDE = None  # reset by update_segmentation_model.sh for 'Kristineberg_260814' -- re-add manually if you want to override its trained threshold

### ==========================
### ViT classifier checkpoint
### ==========================
# CLASSIFY step (ViT inference, ~0.1-0.4s/frame per analyse.log -- the single biggest
# per-frame cost besides SEGMENT) can be switched off entirely for throughput debugging
# or when only crop capture matters, not species labels. SEGMENT still runs either way,
# so crops are still produced/uploaded -- just with class_label/class_confidence/class_idx
# left as None in the sidecar, and the classifier engine isn't even loaded at startup.
# Off as of 2026-08-10 to speed backlog draining -- crops go out unlabeled until this is
# flipped back on; nothing needs reprocessing to relabel them later since re-running
# CLASSIFY only needs the already-uploaded crop images, not the original full frames.
CLASSIFY = False

# Migrated (standardized) checkpoint -- see models/migrate_checkpoint.py. Point this
# at the *_migrated.pth output, not the original raw state_dict, once you've run it.
VIT_CHECKPOINT_PATH = PIPELINE_DIR / "models" / "vit_classifier_F1_0.8697_acc_0.9293_migrated.pth"

# Only consulted if VIT_CHECKPOINT_PATH turns out to be a legacy raw state_dict with no
# embedded class list. Run models/migrate_checkpoint.py once to bake the class list into
# the checkpoint permanently and this fallback stops being needed.
VIT_CLASS_NAMES_FALLBACK = None  # e.g. ["calanus", "clytia_spp1", ...] in the exact training order

# Exact class_label string (see Pipeline_development/ClassClassification/Train_ViT.py's
# Ctenophora list) that analyse.py's daily Slack report counts as a mnemiopsis sighting --
# must match the training label exactly, including case.
MNEMIOPSIS_CLASS_LABEL = "mnemiopsis"

### ==========================
### INT8 TensorRT engines
### ==========================
ENGINE_DIR = PIPELINE_DIR / "trt" / "engines"
# Fixed batch the SEGMENT engines were built for -- matches analyse.py's actual per-image
# tile count (SEG_TILE_GRID_SIZE**2 * len(SEG_OFFSETS_NORM) = 16*16*5 = 1280), so it always
# divides evenly with no padding needed.
SEG_ENGINE_BATCH = 1280
# Fixed batch the CLASSIFY (ViT) engine was built for. Real per-image crop counts vary
# (observed 0-9 in sample data); analyse.py pads up to this and slices back down. 16 gives
# headroom over the observed max at negligible extra INT8 cost -- see engines/build_trt_int8.py.
# NOT a speed knob: this must match vit_classifier_int8.engine's actual compiled batch shape
# exactly, or every classify call becomes a shape mismatch against the engine. To cap crops
# per frame for speed, tune N_CROPS_PER_IMAGE below instead -- it's independent of this.
VIT_ENGINE_BATCH = 16

### ==========================
### analyse.py: PREPROCESS/SEGMENT/CLASSIFY/live-stream tuning -- centralized here
### (rather than at the top of analyse.py itself) for visibility, even though
### only analyse.py reads them.
### ==========================
IMAGE_SIZE_PX = 4512  # camera frame size -- checked against each frame's actual dims in analyse.py
IMAGE_W_MM = 91  # physical width of the sensor's field of view, in mm

GAIN_DB = 20.0  # hardware gain -- set on the camera's nodemap at acquisition start (record.py, livestream.py)

ROTATE_FRAME = 90.0  # degrees, applied once per frame (and once to the dark frame at load time)
# before dark-subtract/median-blur/gamma/CLAHE -- corrects for the camera's physical mounting
# angle, if any. 0 = no rotation. Rotation keeps the frame's original size (it's square), replicate-
# padding whatever corners the rotation reveals.

# PREPROCESS (control/monitoring/record.py's original values)
HDR_MAX = 4094  # should not be changed!!!!
MEDIAN_KERNEL_SIZE = 3
POST_GAIN = 1
GAMMA = 1
CLAHE_CLIP = 1.0  # lower local-contrast amplification for the noisy, turbid-water scene
CLAHE_TILE_PX = 512  # CLAHE grid size = IMAGE_SIZE_PX / CLAHE_TILE_PX, rounded to nearest int

# Biofouling background subtraction (background_model.py) -- an EMA of the raw
# (pre-PREPROCESS) frame, subtracted before dark-frame correction/median blur/gamma/
# CLAHE, to stop slowly-accumulating fouling on the submerged camera window (cleaned
# only ~twice a week) from being picked up as spurious SEGMENT detections. See
# background_model.py's module docstring for why EMA specifically (vs a windowed mean
# or rolling median).
BG_SUB_ENABLED = False  # turbid water makes the EMA absorb persistent scene illumination
# and suspended-particle haze, leaving mostly noise after subtraction; keep this off at
# the new location until a representative background model can be calibrated there.
# (analyse.py falls back to feeding gpu_preprocess_frame the raw frame unchanged, same
# as before this feature existed) and the live-stream's biofouling debug thumbnails
# go dark, without needing a restart-free code change to back this out in the field.
#
# Was disabled 2026-08-14 after SEGMENT started flagging the *entire* frame as one
# giant crop every time in production (live_frame.json's crop box was
# (0,0)-(451,451), the whole downscaled frame). Root-caused via offline prototyping
# against real captured frames run through the actual INT8 TensorRT encoder/decoder +
# Mahalanobis scorer: NOT alpha (tested a ~1.8-day-decay alpha, still 100% coverage)
# and NOT spatial frequency (tested blurring the background before subtracting with
# kernels up to 801px, still 100% coverage) -- both hypotheses ruled out. The actual
# cause was background_model.update_and_subtract()'s old re-centering step (frame -
# bg + offset, offset=HDR_MAX/2): raw frames are naturally dark/near-zero almost
# everywhere, which is the distribution the AE was trained on, and shifting every
# pixel's typical value to mid-grey broke that regardless of alpha or how bg was
# computed -- CLAHE + the AE choked on the unfamiliar distribution, not on losing
# real structure per se. Fixed (see background_model.py) by clipping to 0 instead of
# recentering, the same pattern the static dark_frame subtraction above already
# uses -- confirmed clean (normal small-coverage detections, no 100%-coverage
# frames) across a 15-frame run against the real scorer. Re-enabled 2026-08-14 after
# that fix.
BG_EMA_ALPHA = 0.02  # ghost left behind by a cleaning event (window wiped, EMA still
# carrying the old fouling values) decays to ~5% residual in roughly 90-150 frames at
# this alpha -- fast enough not to matter, so no explicit cleaning-event detection/
# reset is needed.

# SEGMENT
SEG_TILE_GRID_SIZE = 16
SEG_OFFSETS_NORM = [0, 0.2, 0.4, 0.6, 0.8]  # relative offsets for tile grid (e.g. 0.2 = shift by 20% of tile size)
MIN_REGION_SIZE_PATCHES = 3
CROP_PADDING_PIXELS = 0
IOU_DEDUP_THRESHOLD = 0.3  # matches segment_labeled_images.py's working reference
N_CROPS_PER_IMAGE = 5  # cap crops per frame -- freely tunable for speed/completeness tradeoff.
SEGMENT_ENABLED = False  # False skips segmentation/crop creation while training frames collect
# dedup_candidates sorts by peak_val descending before capping, so the strongest peaks survive
# and only the weakest excess get dropped when a frame has more than this many. Keep this <=
# VIT_ENGINE_BATCH: going over it forces a second (third, ...) full classify batch, each paying
# the FULL fixed engine cost again regardless of how many of its slots are real crops -- that
# staircase is what made classify time roughly double/triple on frames with many detections.
# IMPORTANT: VIT_ENGINE_BATCH itself must NOT be changed here to "speed things up" -- it has to
# exactly match the batch shape actually baked into vit_classifier_int8.engine at build time
# (engines/build_trt_int8.py's VIT_BATCH). Setting it to anything else without rebuilding that
# engine file creates a real shape mismatch between what analyse.py sends and what the compiled
# engine expects -- this is NOT a valid way to tune speed, only N_CROPS_PER_IMAGE (above) is.

# CLASSIFY
CLASSIFIER_IMAGE_SIZE = 256  # must match Train_ViT.py's preprocess Resize((256,256))
CONFIDENCE_THRESHOLD = 0.7  # live-stream box only gets a species label at/above this

DISK_CHECK_EVERY_N_FRAMES = 5

# Periodically returns PyTorch's cached-but-unused GPU memory to the CUDA driver
# (torch.cuda.empty_cache() + gc.collect() in analyse.py's main loop) -- without
# this, fragmentation from many frames of varying-shape tensor allocations builds
# up until a native CUDA allocation blocks indefinitely (a silent hang, not a
# crash, so supervisor.sh's restart loop never fires since the process never
# returns). Root cause of the 2026-07-24 overnight freeze at frame 1324.
CUDA_CACHE_CLEANUP_EVERY_N_FRAMES = 100

### ==========================
### Live-stream (post-PREPROCESS frame + crop boxes), matching livestream.py's
### Flask/MJPEG pattern -- but showing the actual production pipeline's output
### and metadata, not a separate dev/test capture.
### ==========================
ENABLE_LIVE_STREAM = True  # the whole monitoring web server -- /status JSON panel (temps,
# disk, connection, last-recorded/analysed timers, backlog progress) plus the video/crops
# UI. Keep this True to keep monitoring available at all; see BACKLOG_MODE_* below for
# what turns off just the camera-image/crops-thumbnail rendering without losing the
# status panel.
LIVESTREAM_PORT = 8080  # view at http://<jetson-ip>:8080/

# 2026-08-13: guardrails against a slow/remote viewer stalling the dashboard's Flask
# server -- it runs threaded (one OS thread per request, no pool, no cap) in the same
# process as the real-time PREPROCESS/SEGMENT/CLASSIFY loop. send.py's own measured
# upload speed to server-lab has been seen as low as ~1.2-2.6 KB/s (see its log), far
# below the ~50-60KB/frame the video image endpoint serves -- someone viewing the
# dashboard over that same flaky link, not the local LAN, could leave a request's
# socket.send() blocked for a very long time otherwise, since Werkzeug's dev server
# has no default per-connection timeout. Both knobs below only affect
# request-handling threads in analyse.py's Flask app; they never touch record.py or
# the actual capture/analysis loop directly.
LIVESTREAM_REQUEST_TIMEOUT_S = 10  # per-connection socket timeout (read AND write) for
# the dashboard's Flask server. Past this, Werkzeug drops the connection (it already
# catches socket.timeout as a clean "connection dropped" case, not a crash -- see
# WSGIRequestHandler.connection_dropped in werkzeug/serving.py) instead of leaving the
# handling thread blocked indefinitely. A poll-driven dashboard (media.js re-requests
# every 400ms-1s regardless) just asks again next tick, so a dropped slow request costs
# nothing real.
LIVESTREAM_MAX_CONCURRENT_REQUESTS = 8  # hard cap on simultaneous in-flight requests
# to the dashboard Flask app (all routes, checked right after the password gate) --
# bounds how many threads can pile up during LIVESTREAM_REQUEST_TIMEOUT_S's window even
# under a burst of slow/stalled clients. Past this, new requests get a plain 503
# immediately rather than spawning yet another thread; same reasoning as the timeout
# above, the dashboard just retries on its own next poll.

### ==========================
### Backlog mode vs live mode (analyse.py's update_system_mode()) -- automatically
### switches the per-frame video-image and crops-thumbnail rendering
### (update_live_frame/update_recent_crops_strip) off while there's a substantial backlog
### of unanalysed raw frames sitting on disk (old backlog + sdb1 backlog + live queue,
### combined), since that rendering costs real per-frame time (resize/draw-boxes/
### JPEG-encode, ~0.15-0.3s/frame observed) that's pure overhead while nobody's watching
### and analyse.py could instead spend on actually draining the backlog. The /status JSON
### panel keeps working in either mode -- only the live video image and crops-thumbnail
### strip go dark in backlog mode (the page shows a placeholder instead of a broken
### image).
###
### Two separate thresholds, not one, deliberately: a single threshold would flap video
### on/off repeatedly whenever the backlog hovers right around it (e.g. draining down to
### 950, ticking back up to 1050, etc.) -- the gap between ENTER and EXIT is a hysteresis
### band, so once in backlog mode it takes dropping meaningfully lower (not just below
### the same number that triggered it) to switch back.
###
### ENTER raised sharply 2026-08-11, after the sdb1->sda1 move: the ~0.15-0.3s/frame
### render cost that motivated the original 1000 barely matters now that analyse.py runs
### at ~1.79 frames/s (vs record.py's ~0.32 frames/s at FRAME_SKIP=3) -- there's enough
### throughput margin either way. Kept as a real (if now much higher) threshold rather
### than removed entirely, so it still trips as a genuine emergency tripwire if the queue
### ever balloons to incident scale again (the 2026-08-03 incident reached ~63k files).
### ==========================
BACKLOG_MODE_ENTER_THRESHOLD = 50_000  # total pending fullframes above this -> backlog mode
BACKLOG_MODE_EXIT_THRESHOLD = 100  # total pending fullframes below this -> live mode
# % downscale, i.e. NxN pixel binning where N = 100/LIVESTREAM_DISP_SCALE (10 ->
# 10x10 bins, 20 -> 5x5 bins, 50 -> 2x2 bins). Was 50 (2256x2256, ~670KB/JPEG) --
# at the pipeline's then-current ~1fps analysis rate, that much data per frame
# added real transmission/decode lag on top of the already-slow update rate,
# making the live view feel even more sluggish than the ~1fps alone would;
# dropped to 20 -> ~902x902 (much faster to send/decode), then later to 10 ->
# ~451x451.
#
# 2026-08-14: raised to 50 to use some of the remote upload link's spare
# "occupied bandwidth" headroom (see send.py's link_speed_bps vs. upload_speed_bps,
# surfaced on the live-stream's CONNECTION panel) -- confirmed live-slow again
# within the same session, the exact same problem the original 50->20 drop was
# for. Settled on 20 (5x5 bins, ~902x902) as the middle ground: a real step up in
# detail from 10, without reintroducing the local transmission/decode lag 50
# caused both times it's been tried.
LIVESTREAM_DISP_SCALE = 10
LIVESTREAM_JPEG_QUALITY = 60
CROP_BOX_COLOR = (0, 255, 0)  # green, BGR (cv2 convention)

# 2026-08-12: analyse.py overwrites this path on every live-mode frame with a
# downscaled (LIVESTREAM_DISP_SCALE) JPEG -- deliberately *without* crop boxes/labels
# drawn on it (unlike /latest_frame.jpg's local dashboard copy, see update_live_frame())
# -- so the future public livestream site's own frontend has full control over how
# crops are rendered, instead of inheriting this pipeline's box color/label styling
# baked into the pixels. The matching crop/label data goes out separately as JSON, see
# LIVE_FRAME_JSON_PATH below. send.py watches this file's mtime and uploads both to
# server-lab together (always overwriting the same remote filenames) whenever it
# changes. Not a queue -- there's deliberately no history here, only the single most
# recent frame ever matters, same "always current" idea as /latest_frame.jpg itself,
# just also pushed off-device.
LIVE_FRAME_PATH = PIPELINE_DIR / "live_frame_outbox" / "live_frame.jpg"
# Crop boxes/labels matching LIVE_FRAME_PATH's frame, in that same JPEG's pixel
# coordinates (i.e. already scaled by LIVESTREAM_DISP_SCALE, not full-res). Written
# *before* LIVE_FRAME_PATH on every frame (see update_live_frame()) so that by the
# time send.py notices the JPEG's mtime change -- its only upload trigger -- this
# file is already there with matching data to go along with it.
LIVE_FRAME_JSON_PATH = PIPELINE_DIR / "live_frame_outbox" / "live_frame.json"
RECENT_CROPS_COUNT = 100  # how many of the most recent crops (unfiltered -- every
# crop SEGMENT accepts, not just confidently classified ones) to keep in the
# live-stream window's thumbnail strip. The strip renders taller than the Crops
# tab's own viewport at this count -- the frontend scrolls it rather than shrinking
# it to fit (see frontend_private/style.css's #liveViewContainer.scrollable).
RECENT_CROPS_COLUMNS = 5  # thumbnail strip wraps into a grid this many columns wide
# (RECENT_CROPS_COUNT=100 -> 20 rows) instead of one long unbounded row
RECENT_CROPS_THUMB_PX = 96  # each strip thumbnail's square size in pixels, post-resize
RECENT_CROPS_CAPTION_PX = 36  # height of the black caption band under each thumbnail --
# two lines (class label, then observation time), not one, hence taller than a
# single line of text would need

### ==========================
### Remote server (Tailscale)
### ==========================
REMOTE_HOST = "server-lab"
REMOTE_BASE = "Faro_260925_ongoing"  # relative to the SSH user's home on the remote host --
# named after the camera's current deployment (site_startdate_status); update this whenever
# the camera moves to a new deployment location so uploads land in a folder matching where
# the data actually came from, instead of mixing multiple deployments into one folder
SSH_CONNECT_TIMEOUT_S = 10
UPLOAD_TIMEOUT_S = 60
TRAINING_FRAME_UPLOAD_TIMEOUT_S = 180  # full frames (4512x4512) are far bigger than crops --
# UPLOAD_TIMEOUT_S=60 is tuned for small crop batches and could cut off a slow scp of one
# of these over a flaky 5G/Tailscale link before it genuinely finishes
MAX_UPLOAD_RETRIES_PER_CYCLE = 6      # exponential-backoff attempts within one polling cycle
MAX_UPLOAD_ATTEMPTS_TOTAL = 50        # escalate an item to failed/ past this many cycle-level attempts
CROP_UPLOAD_ENABLED = False  # False pauses crop uploads without deleting queued crops

# Kill switch for send.py's uploads to server-lab (crops, training frames, live frame
# preview) -- flip to False to pause all of it, e.g. while the deployment location is
# unsettled or the remote folder is being renamed. Capture/segment/classify keep running
# unaffected: crops and training frames just pile up in their local queues (que_crops,
# que_training_frames) instead of being uploaded, and nothing is lost -- send.py drains
# the backlog normally once this goes back to True. Does NOT affect metadata.py's own
# device-health/environmental uploads or heartbeats, which keep going either way so the
# device stays remotely monitorable.
ENABLE_UPLOADS = True

### ==========================
### Heartbeats (metadata.py only ever reads these; each stage writes its own)
### ==========================
HEARTBEAT_DIR = PIPELINE_DIR / "logs"
HEARTBEAT_STALE_S = 30  # a stage is considered "down" if its heartbeat is older than this

PIPELINE_ALERT_STALE_READS = 2  # ~20s at METADATA_SAMPLE_INTERVAL_S=10 -- consecutive stale
# heartbeat_status() reads pipeline_alert.py requires before firing a "stage down" Slack
# alert, so one missed/racy heartbeat-file read doesn't look like a real outage
PIPELINE_ALERT_COOLDOWN_S = 1800  # 30 min -- same spirit as LEAK_ALERT_COOLDOWN_S: throttles
# repeated "down" alerts if a stage flaps rather than staying cleanly down. Recovery alerts
# are never throttled -- see pipeline_alert.py

### ==========================
### Metadata cadence
### ==========================
METADATA_SAMPLE_INTERVAL_S = 10
METADATA_UPLOAD_INTERVAL_S = 60

### ==========================
### Environmental metadata cadence (Bar3XT pressure/depth, via the Metro M0) --
### independent of METADATA_* above, since device health and environmental
### readings are separate uploaded streams (see metadata.py).
### ==========================
ENVIRONMENTAL_SAMPLE_INTERVAL_S = 1
ENVIRONMENTAL_UPLOAD_INTERVAL_S = 60

### ==========================
### Leak alerting (leak_alert.py) -- direct sensor tier is unambiguous. The indirect
### (dew point / pressure / temperature) thresholds below are calibrated from an actual
### overnight pump-down test (Test/leak_test_logs/leak_test_20260727_212454.csv, ~16h),
### via Test/calibrate_leak_thresholds.py -- not hand-guessed placeholders anymore. That
### script detrends the test's own large, benign pump-down-decay/warm-up curves out of
### each signal first (see its module docstring for why a raw calibration off that CSV
### would badly overshoot), then sets each threshold at 2x the residual 99.5th-percentile
### noise floor over the LEAK_WARNING_WINDOW_S window. Re-run it against a fresh overnight
### test any time the enclosure, sensor, or firmware changes enough that this baseline
### might no longer hold.
### LEAK_TEMP_DEVIATION_THRESHOLD_C is calibrated here but NOT currently wired into
### leak_alert.check_for_leak() -- kept as a reference value only, deliberately not
### added to the Tier-2 AND-condition (see leak_alert.py's module docstring for why
### dew point, not raw temperature or humidity, is what that check actually combines
### with pressure).
### ==========================
LEAK_SENSOR_CONSECUTIVE_READS = 2  # ~20s at METADATA_SAMPLE_INTERVAL_S=10 -- rules out one glitchy read
LEAK_WARNING_WINDOW_S = 900  # 15 min -- trend window for the indirect BME280 signals
LEAK_DEW_POINT_RISE_THRESHOLD_C = 0.60
LEAK_PRESSURE_DEVIATION_THRESHOLD_MBAR = 9.74
LEAK_TEMP_DEVIATION_THRESHOLD_C = 0.74
LEAK_ALERT_COOLDOWN_S = 1800  # 30 min -- don't re-spam the same tier while a condition persists

### ==========================
### Arduino Metro M0 serial port (M0_metadata_BME280.cpp) -- sensors: Bar3XT
### (water pressure/depth/temperature, environmental), BME280 (enclosure air
### pressure/humidity/temperature), leak detector, DS18B20 (enclosure
### temperature) -- the latter three sensors are device metadata, not
### environmental (see metadata.py's collect_sample() vs
### collect_environmental_sample()).
### ==========================
METRO_M0_SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
SERIAL_TIMEOUT_S = 2  # seconds to wait for a line before giving up and returning None

### ==========================
### Strobe controller (Opto Engineering LTDVE1CH-40F) -- converter/driver
### temperature over Modbus/TCP, registers 206/207 per its instruction manual
### section 14.2.41-42 (see strobe.py). Same device start_dev.sh auto-detects
### and livestream.py already polls.
### ==========================
STROBE_IP = "192.168.0.32"
STROBE_MODBUS_PORT = 502
STROBE_MODBUS_UNIT = 32

### ==========================
### Hardware thermal limits -- purely informational (not enforced/throttled by
### this software), for the live-stream's "how close to the danger zone"
### display. Checked 2026-07-27:
###   Camera (Opto Engineering ITA204-GM-20C): -25C to +65C ambient operating
###     range per its datasheet.
###   Strobe controller (Opto Engineering LTDVE1CH-40F): instruction manual
###     section on the MEASURED_TEMPERATURE_CONVERTER/_DRIVER Modbus registers
###     (same ones strobe.py reads) -- shuts off all output channels above
###     90C heatsink temp, reactivates below 80C. Same limit applied to both
###     converter and driver readings; the manual doesn't give them separately.
###   Jetson AGX Orin 64GB: NVIDIA's Thermal Design Guide gives 99C as the
###     recommended-operation limit before thermal throttling kicks in, and
###     105C as the hard Tj max. Using 99C (the actionable one) here.
### ==========================
CAMERA_MAX_TEMP_C = 65.0
STROBE_MAX_TEMP_C = 90.0
JETSON_MAX_TEMP_C = 99.0

### ==========================
### Provenance logbook (~/Documents/logbook.txt) -- for people using this device's
### DATA later, not developers: appends a plain-text entry whenever any setting below
### that affects what gets recorded/detected/delivered actually changes. See
### logbook.py's module docstring for the whole mechanism; this is the only hook site
### needed anywhere in the pipeline -- every script that imports config.py checks in
### automatically, so a hand-edit here, a script's sed (e.g.
### update_segmentation_model.sh), or anything else that changes one of these values
### gets picked up the next time anything imports this file, regardless of which
### process that happens to be.
###
### Deliberately NOT everything in this file -- only settings that change what's IN
### the delivered data (which model produced a detection, whether background
### subtraction or classification ran, what got sampled/kept/dropped/archived), not
### operational tuning (poll intervals, timeouts, retry counts, dashboard/UI knobs)
### that only affects how smoothly the pipeline runs, not what the data itself means.
### ==========================
import logbook as _logbook

LOGBOOK_TRACKED_FIELDS = {
    "segmentation_ae_checkpoint": _logbook.checkpoint_fingerprint(SEGMENTATION_AE_MODEL_PATH),
    "segmentation_scorer_checkpoint": _logbook.checkpoint_fingerprint(SEGMENTATION_SCORER_MODEL_PATH),
    "segmentation_scorer_threshold_override": SEGMENTATION_SCORER_THRESHOLD_OVERRIDE,
    "positional_embeddings_used": POSITIONAL_EMBEDDINGS_USED,
    "vit_checkpoint": _logbook.checkpoint_fingerprint(VIT_CHECKPOINT_PATH),
    "classify_enabled": CLASSIFY,
    "mnemiopsis_class_label": MNEMIOPSIS_CLASS_LABEL,
    "bg_sub_enabled": BG_SUB_ENABLED,
    "bg_ema_alpha": BG_EMA_ALPHA,
    "frame_skip": FRAME_SKIP,
    "rotate_frame_deg": ROTATE_FRAME,
    "dark_frame_path": str(DARK_FRAME_PATH) if DARK_FRAME_PATH.exists() else None,
    "hdr_max": HDR_MAX,
    "median_kernel_size": MEDIAN_KERNEL_SIZE,
    "gamma": GAMMA,
    "post_gain": POST_GAIN,
    "clahe_clip": CLAHE_CLIP,
    "clahe_tile_px": CLAHE_TILE_PX,
    "seg_tile_grid_size": SEG_TILE_GRID_SIZE,
    "seg_offsets_norm": SEG_OFFSETS_NORM,
    "min_region_size_patches": MIN_REGION_SIZE_PATCHES,
    "crop_padding_pixels": CROP_PADDING_PIXELS,
    "iou_dedup_threshold": IOU_DEDUP_THRESHOLD,
    "n_crops_per_image": N_CROPS_PER_IMAGE,
    "classifier_image_size": CLASSIFIER_IMAGE_SIZE,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "max_crop_upload_bytes": MAX_CROP_UPLOAD_BYTES,
}
try:
    _logbook.record_config_snapshot(LOGBOOK_TRACKED_FIELDS)
except Exception as _logbook_exc:  # never let logbook bookkeeping take the pipeline down
    print(f"[config] logbook update failed (non-fatal): {_logbook_exc}")