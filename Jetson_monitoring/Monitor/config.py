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
# On the mounted external SSD, not the Jetson's own ~57GB root partition (which sat at
# 98% full / 1.6GB free before this moved here) -- /mnt/sdb1, not /mnt/sda1, since sda1
# holds the training data corpus and this pipeline's queues are unrelated to that.
QUEUE_ROOT = Path("/mnt/sdb1/jellyscope_queues")

QUE_FULLFRAMES = QUEUE_ROOT / "que_fullframes"
QUE_CROPS = QUEUE_ROOT / "que_crops"

# The two mounted external SSDs -- sda1 (training data) and sdb1 (this pipeline's
# queues) -- reported alongside the Jetson's own root partition in metadata.py's
# disk-space fields.
ROOT_DISK_PATH = Path("/")
EXTERNAL_SSD_PATHS = [Path("/mnt/sda1"), Path("/mnt/sdb1")]

# Standard Jetson/Tegra sysfs GPU load node -- reports 0-1000 (permille), not 0-100.
GPU_LOAD_PATH = Path("/sys/devices/platform/bus@0/17000000.gpu/load")

MIN_FREE_BYTES = 2 * 1024**3  # record.py/analyse.py pause producing below this

### ==========================
### Camera sampling rate -- the hardware trigger itself can't be changed (fixed
### line-trigger rate), so record.py instead keeps only every Nth triggered
### frame and discards the rest. 1 = keep every frame. Reported on the live
### stream (see analyse.py's /status) so it's visible what rate is actually
### being recorded at.
### ==========================
FRAME_SKIP = 1

# Optional dark/background reference frame (control/monitoring/record.py's "bkg.png"),
# subtracted during analyse.py's PREPROCESS step. If this path doesn't exist,
# analyse.py skips dark-frame subtraction with a warning rather than crashing.
DARK_FRAME_PATH = PIPELINE_DIR / "bkg.png"

### ==========================
### Segmentation checkpoints (already trained -- never retrained by this pipeline;
### on-board retraining lives in train/, sibling to this repo's Jetson_monitoring/ folder)
### ==========================
SEGMENTATION_ENCODER_TYPE = "AE"  # "AE" or "VAE" -- must match whichever class the checkpoint below was trained with
SEGMENTATION_AE_MODEL_PATH = PIPELINE_DIR / "models" / "Kristineberg_251128_ae_model16_l64_img128.pth"
SEGMENTATION_SCORER_MODEL_PATH = PIPELINE_DIR / "models" / "Kristineberg_251128_scorer_binary_model16_l64_img128.pth"

### ==========================
### ViT classifier checkpoint
### ==========================
# CLASSIFY step (ViT inference, ~0.1-0.4s/frame per analyse.log -- the single biggest
# per-frame cost besides SEGMENT) can be switched off entirely for throughput debugging
# or when only crop capture matters, not species labels. SEGMENT still runs either way,
# so crops are still produced/uploaded -- just with class_label/class_confidence/class_idx
# left as None in the sidecar, and the classifier engine isn't even loaded at startup.
CLASSIFY = False

# Migrated (standardized) checkpoint -- see models/migrate_checkpoint.py. Point this
# at the *_migrated.pth output, not the original raw state_dict, once you've run it.
VIT_CHECKPOINT_PATH = PIPELINE_DIR / "models" / "vit_classifier_F1_0.8697_acc_0.9293_migrated.pth"

# Only consulted if VIT_CHECKPOINT_PATH turns out to be a legacy raw state_dict with no
# embedded class list. Run models/migrate_checkpoint.py once to bake the class list into
# the checkpoint permanently and this fallback stops being needed.
VIT_CLASS_NAMES_FALLBACK = None  # e.g. ["calanus", "clytia_spp1", ...] in the exact training order

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

ROTATE_FRAME = 90.0  # degrees, applied once per frame (and once to the dark frame at load time)
# before dark-subtract/median-blur/gamma/CLAHE -- corrects for the camera's physical mounting
# angle, if any. 0 = no rotation. Rotation keeps the frame's original size (it's square), replicate-
# padding whatever corners the rotation reveals.

# PREPROCESS (control/monitoring/record.py's original values)
HDR_MAX = 4094  # should not be changed!!!!
MEDIAN_KERNEL_SIZE = 3
POST_GAIN = 1
GAMMA = 0.7
CLAHE_CLIP = 0.01
CLAHE_TILE_PX = 512  # CLAHE grid size = IMAGE_SIZE_PX / CLAHE_TILE_PX, rounded to nearest int

# SEGMENT
SEG_TILE_GRID_SIZE = 16
SEG_OFFSETS_NORM = [0, 0.2, 0.4, 0.6, 0.8]  # relative offsets for tile grid (e.g. 0.2 = shift by 20% of tile size)
MIN_REGION_SIZE_PATCHES = 3
CROP_PADDING_PIXELS = 0
IOU_DEDUP_THRESHOLD = 0.3  # matches segment_labeled_images.py's working reference
N_CROPS_PER_IMAGE = 1  # cap crops per frame -- freely tunable for speed/completeness tradeoff.
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
CONFIDENCE_THRESHOLD = 0.9  # live-stream box only gets a species label at/above this

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
ENABLE_LIVE_STREAM = True
LIVESTREAM_PORT = 8080  # view at http://<jetson-ip>:8080/
# % downscale. Was 50 (2256x2256, ~670KB/JPEG) -- at the pipeline's current ~1fps
# analysis rate, that much data per frame added real transmission/decode lag on top
# of the already-slow update rate, making the live view feel even more sluggish than
# the ~1fps alone would. 20 -> ~902x902, much faster to send/decode; still matches
# livestream.py's own downscale-before-draw convention (see update_live_frame()).
LIVESTREAM_DISP_SCALE = 10
CROP_BOX_COLOR = (0, 255, 0)  # green, BGR (cv2 convention)

### ==========================
### Remote server (Tailscale)
### ==========================
REMOTE_HOST = "server-lab"
REMOTE_BASE = "jellyscope_incoming"  # relative to the SSH user's home on the remote host
SSH_CONNECT_TIMEOUT_S = 10
UPLOAD_TIMEOUT_S = 60
MAX_UPLOAD_RETRIES_PER_CYCLE = 6      # exponential-backoff attempts within one polling cycle
MAX_UPLOAD_ATTEMPTS_TOTAL = 50        # escalate an item to failed/ past this many cycle-level attempts

### ==========================
### Heartbeats (metadata.py only ever reads these; each stage writes its own)
### ==========================
HEARTBEAT_DIR = PIPELINE_DIR / "logs"
HEARTBEAT_STALE_S = 30  # a stage is considered "down" if its heartbeat is older than this

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
### Leak alerting (leak_alert.py) -- direct sensor tier is unambiguous, but
### THE INDIRECT (dew point / pressure) THRESHOLDS BELOW ARE PLACEHOLDERS.
### There's no baseline data from this deployment yet to calibrate against --
### watch actual dew_point_c/bme280_pressure_mbar behavior over a few days of
### normal operation before trusting these numbers. See leak_alert.py's
### module docstring for the full reasoning.
### ==========================
LEAK_SENSOR_CONSECUTIVE_READS = 2  # ~20s at METADATA_SAMPLE_INTERVAL_S=10 -- rules out one glitchy read
LEAK_WARNING_WINDOW_S = 900  # 15 min -- trend window for the indirect BME280 signals
LEAK_DEW_POINT_RISE_THRESHOLD_C = 3.0  # PLACEHOLDER
LEAK_PRESSURE_DEVIATION_THRESHOLD_MBAR = 5.0  # PLACEHOLDER
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