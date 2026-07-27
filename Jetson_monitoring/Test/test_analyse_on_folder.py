#%% Imports
"""Ad-hoc test harness: runs analyse.py's SEGMENT + CLASSIFY logic against a
folder of already-preprocessed sample images (e.g. copied from server-lab's
training-data corpus), skipping the PREPROCESS step entirely since those
images have already been enhanced.

Reuses `analyse.segment_and_classify` directly (not a reimplementation) --
importing analyse.py runs its full startup (device setup, loading the AE +
BinaryScorer + ViT engines) but not its queue-watching loop, which is guarded
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

MONITOR_DIR = Path(__file__).resolve().parent.parent / "Monitor"
if str(MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(MONITOR_DIR))

import config
from analyse import segment_and_classify

#%% Configuration
FOLDER = Path.home() / "test_images" / "OG_images"  # edit this to point at a different sample set
TEST_OUTPUT_DIR = config.PIPELINE_DIR / "test_output"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

for image_path in image_paths:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"  {image_path.name}: failed to read, skipping")
        continue

    image_name = clean_image_name(image_path.stem)

    start = time.perf_counter()
    results, _timings = segment_and_classify(image, image_name)
    elapsed = time.perf_counter() - start
    total_elapsed += elapsed

    print(f"  {image_path.name}: {len(results)} crop(s) in {elapsed:.2f}s")

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
print(f"Timing: {total_elapsed:.2f}s total, {avg:.2f}s/image average over {n_images} image(s)")
