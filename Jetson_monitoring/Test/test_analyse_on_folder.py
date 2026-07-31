#%% Imports
"""Ad-hoc test harness: runs analyse.py's SEGMENT + CLASSIFY logic against a
folder of already-preprocessed sample images (e.g. copied from server-lab's
training-data corpus), skipping the PREPROCESS step entirely since those
images have already been enhanced.

Reuses `analyse.segment_and_classify` directly (not a reimplementation) --
importing analyse.py runs its full startup (device setup, loading the AE +
scorer -- binary or Mahalanobis, whichever config.SEGMENTATION_SCORER_MODEL_PATH
points at -- + ViT engines) but not its queue-watching loop, which is guarded
behind `if __name__ == "__main__":` in analyse.py for exactly this reason.
Any tuning change made in Monitor/analyse.py or Monitor/segment_core.py is
automatically picked up here too -- nothing about the SEGMENT/CLASSIFY logic
is duplicated in this file.

Results are written to a local test_output/ folder, NOT the real que_crops
queue -- so test crops never get picked up by send.py and uploaded to the
server as if they were real monitoring data.

Written in #%% cells (run cell-by-cell in VSCode's interactive window) rather
than as a CLI script -- edit FOLDER below directly instead of passing argv.
"""
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless -- this runs over SSH with no display; plots are saved to a file, not plt.show()
import matplotlib.pyplot as plt
import numpy as np

MONITOR_DIR = Path(__file__).resolve().parent.parent / "Monitor"
if str(MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(MONITOR_DIR))

import analyse
import config
import segment_core
from analyse import segment_and_classify

#%% Configuration
FOLDER = Path("/mnt/sda1/TrainingData/Binary_classifier/Kristineberg_260730/train_scorer/obs/OG_images")  # edit this to point at a different sample set
TEST_OUTPUT_DIR = config.PIPELINE_DIR / "test_output"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR = TEST_OUTPUT_DIR / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# This training-data corpus sometimes has a manually-assigned species name tacked onto
# the filename (e.g. "img_20251128_114204_735_narcomedusa.png"). Real camera frames from
# record.py never have this (just "img_<date>_<time>_<ms>"), so strip it here for output
# naming, to avoid mixing up the manual label with the predicted one.
FRAME_NAME_PATTERN = re.compile(r"^(img_\d{8}_\d{6}_\d{3})")

def clean_image_name(stem: str) -> str:
    match = FRAME_NAME_PATTERN.match(stem)
    return match.group(1) if match else stem


if not FOLDER.is_dir():
    raise FileNotFoundError(f"Not a directory: {FOLDER}")

image_paths = sorted(FOLDER.glob("*.png"))
print(f"Found {len(image_paths)} images in {FOLDER}")



#%% Run SEGMENT + CLASSIFY on every image in the folder
total_crops = 0
total_elapsed = 0.0
n_heatmaps = 0

# Raw scorer output collected alongside the real crop-finding pass below (via
# segment_core directly, the same calls segment_and_classify() makes internally but
# doesn't expose) -- lets the plotting cell at the bottom show where these images'
# actual scores sit relative to analyse.peak_threshold, instead of only the final
# crop count. Useful when crops=0 everywhere and it's unclear whether that's a
# too-strict threshold or the scores genuinely not separating obs from empty.
all_tile_scores = []
per_image_max_score = []

for image_path in image_paths:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"  {image_path.name}: failed to read, skipping")
        continue

    image_name = clean_image_name(image_path.stem)

    tiles_tensor, rows_list, cols_list, _tile_bboxes = segment_core.extract_tiles(
        image, analyse.device, config.SEG_TILE_GRID_SIZE, analyse.offsets, config.SEG_OFFSETS_NORM,
        analyse.tile_size, analyse.seg_image_size,
    )
    scores_array = segment_core.score_tiles(
        tiles_tensor, rows_list, cols_list, analyse.encoder_engine, analyse.decoder_engine, analyse.scorer_engine,
        config.SEG_ENGINE_BATCH, analyse.device,
    )
    all_tile_scores.append(scores_array)
    per_image_max_score.append(float(scores_array.max()) if scores_array.size else float("nan"))

    if tiles_tensor is not None:
        prob_map_small, _coverage_map_small = segment_core.aggregate_to_patch_map(
            rows_list, cols_list, scores_array, analyse.grid_size_small, len(analyse.offsets),
        )
        heatmap_full = cv2.resize(
            prob_map_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST,
        )
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        gray_cmap = "gray" if image.ndim == 2 else None

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(display_image, cmap=gray_cmap)
        axes[0].set_title(image_name)
        axes[0].axis("off")

        axes[1].imshow(display_image, cmap=gray_cmap)
        heat = axes[1].imshow(heatmap_full, cmap="inferno", alpha=0.55, vmin=0.0)
        axes[1].contour(heatmap_full, levels=[analyse.peak_threshold], colors="cyan", linewidths=1.2)
        axes[1].set_title(f"Aggregated probability heatmap\n(cyan = peak_threshold {analyse.peak_threshold:.4f})")
        axes[1].axis("off")
        fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04, label="Aggregated probability score")

        plt.tight_layout()
        plt.savefig(HEATMAP_DIR / f"{image_name}_heatmap.png", dpi=150)
        plt.close(fig)
        n_heatmaps += 1

    start = time.perf_counter()
    results, _timings = segment_and_classify(image, image_name)
    elapsed = time.perf_counter() - start
    total_elapsed += elapsed

    print(f"  {image_path.name}: {len(results)} crop(s) in {elapsed:.2f}s "
          f"(max tile score {per_image_max_score[-1]:.4f})")

    for crop_stem, encoded_bytes, sidecar in results:
        out_path = TEST_OUTPUT_DIR / f"{crop_stem}.png"
        out_path.write_bytes(encoded_bytes)
        # class_label/class_confidence are None when config.CLASSIFY is off
        label = sidecar['class_label'] if sidecar['class_label'] is not None else "unclassified"
        confidence = f"{sidecar['class_confidence']:.2f}" if sidecar['class_confidence'] is not None else "N/A"
        print(f"    -> {label} "
              f"(confidence {confidence}, "
              f"area {sidecar['region_size_mm2']:.2f} mm2) -> {out_path.name}")
        total_crops += 1

#%% Summary
n_images = len(image_paths)
avg = total_elapsed / n_images if n_images else 0.0
print(f"\nDone. {total_crops} crop(s) written to {TEST_OUTPUT_DIR}")
print(f"Saved {n_heatmaps} heatmap(s) to {HEATMAP_DIR}")
print(f"Timing: {total_elapsed:.2f}s total, {avg:.2f}s/image average over {n_images} image(s)")

#%% Plot: raw per-tile score distribution + per-image peak score, vs. the deployed threshold
# Answers "is the threshold too strict, or are the scores themselves not separating obs from
# empty" -- NOT the full crop-acceptance story (find_candidates/dedup_candidates also require
# a connected region of MIN_REGION_SIZE_PATCHES tiles averaging above secondary_threshold, and
# N_CROPS_PER_IMAGE/IOU_DEDUP_THRESHOLD can still drop candidates after this point) -- but if
# scores here don't even clear peak_threshold, nothing downstream will produce a crop either.
all_scores_concat = np.concatenate(all_tile_scores) if all_tile_scores else np.array([])
valid_max_scores = [s for s in per_image_max_score if s == s]  # drop NaNs (images with 0 tiles)
n_clearing = sum(1 for s in valid_max_scores if s >= analyse.peak_threshold)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(all_scores_concat, bins=80, color="steelblue", alpha=0.85)
axes[0].axvline(analyse.peak_threshold, color="red", linestyle="--", label=f"peak_threshold={analyse.peak_threshold:.4f}")
axes[0].set_xlabel("Raw per-tile scorer output")
axes[0].set_ylabel("Tile count")
axes[0].set_title(f"Per-tile score distribution\n({n_images} image(s), {len(all_scores_concat)} tiles)")
axes[0].legend()
axes[0].grid(alpha=0.3)

sorted_max = sorted(valid_max_scores, reverse=True)
axes[1].bar(range(len(sorted_max)), sorted_max, color="darkorange")
axes[1].axhline(analyse.peak_threshold, color="red", linestyle="--", label=f"peak_threshold={analyse.peak_threshold:.4f}")
axes[1].set_xlabel("Image (sorted by its own max tile score)")
axes[1].set_ylabel("Max per-tile score in that image")
axes[1].set_title(f"Per-image peak score -- {n_clearing}/{len(valid_max_scores)} clear the threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = TEST_OUTPUT_DIR / "score_diagnostic.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved score diagnostic plot to {plot_path}")

