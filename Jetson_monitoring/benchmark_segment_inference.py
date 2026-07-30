"""Ad-hoc benchmark: does FP16 and/or cross-image batching speed up the SEGMENT
inference step (AE encode -> decode -> scorer), which analyse.py's timing
breakdown showed as the dominant ~0.61s/image cost? Not part of the production
pipeline -- a standalone experiment so we decide based on measured numbers, not
guesses, before changing analyse.py.

Uses synthetic random tiles (correct shape/value-range, not real image content)
since this only measures model compute time, not segmentation accuracy.

Usage: python benchmark_segment_inference.py
"""

import sys
import time
from pathlib import Path

import torch

_PIPELINE_DIR = Path(__file__).resolve().parent  # Jetson_monitoring/
_MONITOR_DIR = _PIPELINE_DIR / "Monitor"
if str(_MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(_MONITOR_DIR))

import config
from models import segmentation as seg_models

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

seg_model, scorer, scorer_threshold, seg_grid_size, seg_image_size = seg_models.load_segmentation_models(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH,
    config.SEGMENTATION_ENCODER_TYPE, device,
)

TILES_PER_IMAGE = 1280  # 5 offsets x 16x16 grid, matches analyse.py's actual load
N_WARMUP = 3
N_TIMED = 10


def make_batch(n_images: int, dtype: torch.dtype):
    n_tiles = TILES_PER_IMAGE * n_images
    tiles = torch.rand(n_tiles, 1, seg_image_size, seg_image_size, dtype=dtype, device=device)
    rows = (torch.rand(n_tiles, dtype=dtype, device=device) * (seg_grid_size - 1))
    cols = (torch.rand(n_tiles, dtype=dtype, device=device) * (seg_grid_size - 1))
    return tiles, rows, cols


def run_inference(tiles, rows, cols):
    with torch.inference_mode():
        mu = seg_models.encode(seg_model, config.SEGMENTATION_ENCODER_TYPE, tiles, rows, cols)
        x_hat = seg_model.decode(mu)
        recon_error = ((tiles - x_hat) ** 2).flatten(1).mean(dim=1)
        scores = scorer.predict_batch(mu, recon_error, rows, cols)
    return scores


def benchmark(n_images: int, dtype: torch.dtype) -> float:
    tiles, rows, cols = make_batch(n_images, dtype)

    for _ in range(N_WARMUP):
        run_inference(tiles, rows, cols)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(N_TIMED):
        run_inference(tiles, rows, cols)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / N_TIMED


print(f"\n{'config':<16} {'total (s)':>12} {'per-image (s)':>16} {'speedup vs fp32 x1':>20}")
baseline_per_image = None
for use_fp16 in (False, True):
    if use_fp16:
        seg_model.half()
        scorer.half()
    dtype = torch.float16 if use_fp16 else torch.float32

    for n_images in (1, 2, 5):
        elapsed = benchmark(n_images, dtype)
        per_image = elapsed / n_images
        if baseline_per_image is None:
            baseline_per_image = per_image
        speedup = baseline_per_image / per_image
        label = f"{'fp16' if use_fp16 else 'fp32'} x{n_images} img"
        print(f"{label:<16} {elapsed:>12.3f} {per_image:>16.3f} {speedup:>19.2f}x")
