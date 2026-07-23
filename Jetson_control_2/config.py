"""Shared configuration for the Jetson_control_2 real-time pipeline.

Every stage (record.py, analyse.py, send.py, metadata.py) imports from here for
paths, checkpoint locations, and cross-cutting tunables. Stage-specific tuning
knobs (e.g. analyse.py's tile_size/tile_grid_size/offsets, or record.py's camera
node settings) live directly in their own script instead, matching how the rest
of this repo keeps per-script constants at the top of each file.
"""

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent          # Jetson_control_2/
REPO_ROOT = PIPELINE_DIR.parent                          # Jellyscope/

# So `from Pipeline_development.BinaryClassification.functions import ...` works from any
# of the pipeline scripts without a per-file sys.path.append hack.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Shared Python environment -- reused from Jetson_control/ rather than a second
# multi-GB venv under Jetson_control_2/, since disk is tight (9.8G free on /).
VENV_PYTHON = REPO_ROOT / "Jetson_control" / ".venv-jellyscope" / "bin" / "python"

### ==========================
### Disk-backed queues
### ==========================
# Point this at the external SSD once mounted (e.g. /media/jellyfish/PortableSSD/pipeline_queues)
QUEUE_ROOT = PIPELINE_DIR / "queues"

QUE_FULLFRAMES = QUEUE_ROOT / "que_fullframes"
QUE_CROPS = QUEUE_ROOT / "que_crops"

MIN_FREE_BYTES = 2 * 1024**3  # record.py/analyse.py pause producing below this

# Optional dark/background reference frame (control/monitoring/record.py's "bkg.png"),
# subtracted during analyse.py's PREPROCESS step. If this path doesn't exist,
# analyse.py skips dark-frame subtraction with a warning rather than crashing.
DARK_FRAME_PATH = PIPELINE_DIR / "bkg.png"

### ==========================
### Segmentation checkpoints (already trained -- never retrained by this pipeline;
### on-board retraining lives in train/, sibling to this repo's monitoring/ folder)
### ==========================
SEGMENTATION_ENCODER_TYPE = "AE"  # "AE" or "VAE" -- must match whichever class the checkpoint below was trained with
SEGMENTATION_AE_MODEL_PATH = PIPELINE_DIR / "models" / "Kristineberg_251128_ae_model16_l64_img128.pth"
SEGMENTATION_SCORER_MODEL_PATH = PIPELINE_DIR / "models" / "Kristineberg_251128_scorer_binary_model16_l64_img128.pth"

### ==========================
### ViT classifier checkpoint
### ==========================
# Migrated (standardized) checkpoint -- see models/migrate_checkpoint.py. Point this
# at the *_migrated.pth output, not the original raw state_dict, once you've run it.
VIT_CHECKPOINT_PATH = PIPELINE_DIR / "models" / "vit_classifier_F1_0.8697_acc_0.9293_migrated.pth"

# Only consulted if VIT_CHECKPOINT_PATH turns out to be a legacy raw state_dict with no
# embedded class list. Run models/migrate_checkpoint.py once to bake the class list into
# the checkpoint permanently and this fallback stops being needed.
VIT_CLASS_NAMES_FALLBACK = None  # e.g. ["calanus", "clytia_spp1", ...] in the exact training order

### ==========================
### INT8 TensorRT engines (production runtime -- no FP32 inference left in analyse.py)
### ==========================
ENGINE_DIR = PIPELINE_DIR / "trt" / "engines"
# Fixed batch the SEGMENT engines were built for -- matches analyse.py's actual per-image
# tile count (tile_grid_size**2 * len(offsets_norm) = 16*16*5 = 1280), so it always divides
# evenly with no padding needed.
SEG_ENGINE_BATCH = 1280
# Fixed batch the CLASSIFY (ViT) engine was built for. Real per-image crop counts vary
# (observed 0-9 in sample data); analyse.py pads up to this and slices back down. 16 gives
# headroom over the observed max at negligible extra INT8 cost -- see engines/build_trt_int8.py.
VIT_ENGINE_BATCH = 16

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
