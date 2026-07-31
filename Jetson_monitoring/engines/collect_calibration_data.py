"""Collect real (not synthetic) tiles/crops for TensorRT INT8 calibration.

INT8 calibration needs activation ranges from data that looks like production
input. benchmark_segment_inference.py's torch.rand() tiles are fine for
*timing* experiments but would give TensorRT unrealistic (uniform-random)
dynamic ranges for INT8 quantization -- which degrades accuracy silently,
with no error to warn you.

Does NOT import analyse.py. It's tempting to (it already holds a live SEGMENT
pipeline via segment_and_classify()), but analyse.py is INT8-only by design
(see its own module docstring) -- it loads seg_encoder_int8.engine/
seg_decoder_int8.engine at import time, unconditionally. Those engines are
exactly what build_trt_int8.py builds FROM the data this script collects, so
importing analyse.py here is a chicken-and-egg dependency: on a fresh model
swap (update_segmentation_model.sh wipes the stale INT8 engines before this
runs, precisely so calibration never silently runs against a previous
checkpoint's weights) there are no engines yet for analyse.py to load, and
`import analyse` crashes before this script gets to do anything.

So this script builds its own FP32 SEGMENT pipeline instead, reusing
segment_core.py's shared extract_tiles/score_tiles/aggregate_to_patch_map/
find_candidates/dedup_candidates -- the exact same functions analyse.py's
segment_and_classify() calls -- just with FP32 encode/decode/score callables
(from models/segmentation.py's load_segmentation_models(), the same call
engines/export_onnx.py makes to get something traceable to ONNX) standing in
for the INT8 TensorRT engines. That also means real *crop locations* for
CLASSIFY calibration no longer depend on any TensorRT engine existing, and it
was never valid to calibrate the INT8 encoder/decoder/scorer against their own
INT8 output anyway -- FP32 is the correct reference here regardless.

Writes, under monitoring/trt/calibration/:
  seg_tiles.npy, seg_rows.npy, seg_cols.npy   -- real SEGMENT input tiles
  seg_mu.npy, seg_recon_err.npy               -- real FP32 encoder/decoder
                                                  output on those tiles, for
                                                  calibrating seg_decoder/
                                                  seg_scorer (which don't see
                                                  raw tiles directly)
  vit_crops.npy, vit_sizes.npy                -- real accepted crops (may be
                                                  empty if no images in the
                                                  sample folder produced any
                                                  accepted crop -- pick a
                                                  folder with the objects you
                                                  actually monitor)

Not yet run against real data -- expect to check that vit_crops.npy actually
comes out non-empty for whatever sample folder you point this at (frames
with zero detections make sense as SEGMENT calibration data but contribute
nothing to CLASSIFY calibration).

Usage: python collect_calibration_data.py [folder_of_images] [--max-images N]
"""

import argparse
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent  # Jetson_monitoring/
for _p in (_PIPELINE_DIR / "Monitor", _PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2
import numpy as np
import torch

import config
import segment_core
from models import segmentation as seg_models
from models import vit_classifier

CAL_DIR = config.PIPELINE_DIR / "trt" / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Same derivation analyse.py uses (config.py's docstring keeps values computed FROM
# tunable parameters out of config.py itself, so this duplication -- not an import --
# is the intended pattern, matching analyse.py's own tile_size/offsets lines).
tile_size = config.IMAGE_SIZE_PX // config.SEG_TILE_GRID_SIZE
offsets = [int(norm * tile_size) for norm in config.SEG_OFFSETS_NORM]
PXL_TO_MM2 = (config.IMAGE_W_MM / config.IMAGE_SIZE_PX) ** 2

print("Loading FP32 AE + scorer (the correct calibration reference -- calibrating INT8 "
      "ranges against the INT8 engines' own output would be circular)...")
seg_model_fp32, scorer_fp32, scorer_threshold, seg_grid_size, seg_image_size = seg_models.load_segmentation_models(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH,
    config.SEGMENTATION_ENCODER_TYPE, device,
)
seg_model_fp32.eval()

if seg_grid_size != config.SEG_TILE_GRID_SIZE:
    raise ValueError(f"Segmentation checkpoint grid_size={seg_grid_size} does not match "
                     f"config.SEG_TILE_GRID_SIZE={config.SEG_TILE_GRID_SIZE}")

peak_threshold = scorer_threshold
if config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE is not None:
    print(f"Overriding checkpoint's threshold ({scorer_threshold:.4f}) with "
          f"config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE={config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE:.4f}")
    peak_threshold = config.SEGMENTATION_SCORER_THRESHOLD_OVERRIDE
secondary_threshold = peak_threshold * 0.99

grid_size_small = config.SEG_TILE_GRID_SIZE * len(offsets)
upscale_factor = config.IMAGE_SIZE_PX / grid_size_small


def encode_fn(tiles_batch, rows_batch, cols_batch):
    return seg_models.encode(seg_model_fp32, config.SEGMENTATION_ENCODER_TYPE, tiles_batch, rows_batch, cols_batch)


decode_fn = seg_model_fp32.decode
score_fn = scorer_fp32.predict_batch


def collect_image_paths(folder: Path, max_images: int) -> list:
    """Picks up to max_images calibration images from `folder`. If it has obs/ and
    no_obs/ subfolders (the standard Binary_classifier/<name>/Test/ layout), samples
    roughly evenly from both -- a flat sorted-glob-then-slice would silently favor
    no_obs over obs whenever max_images is smaller than either subfolder's count,
    since "no_obs" sorts alphabetically before "obs". Falls back to a flat recursive
    glob under `folder` if neither subfolder exists (e.g. a plain folder of images)."""
    # Recursive ("**/*.png", not "*.png") -- on-disk layout nests the actual PNGs
    # under an OG_images/ subfolder (obs/OG_images/*.png, no_obs/OG_images/*.png),
    # not directly in obs/no_obs, and a non-recursive glob silently found 0 images
    # in both, feeding torch.cat() an empty list further down instead of erroring
    # here where the actual cause is obvious.
    obs_dir, no_obs_dir = folder / "obs", folder / "no_obs"
    if obs_dir.is_dir() or no_obs_dir.is_dir():
        obs_paths = sorted(obs_dir.glob("**/*.png")) if obs_dir.is_dir() else []
        no_obs_paths = sorted(no_obs_dir.glob("**/*.png")) if no_obs_dir.is_dir() else []
        per_class = max_images // 2
        selected = obs_paths[:per_class] + no_obs_paths[:per_class]
        remaining = max_images - len(selected)
        if remaining > 0:
            # one class had fewer than its share -- top up from the other's leftovers
            # so max_images is still honored as long as enough images exist overall.
            leftover = obs_paths[per_class:] + no_obs_paths[per_class:]
            selected += leftover[:remaining]
        print(f"Found {len(obs_paths)} obs + {len(no_obs_paths)} no_obs image(s) under {folder}, "
              f"using {len(selected)} ({sum(1 for p in selected if p in obs_paths)} obs + "
              f"{sum(1 for p in selected if p in no_obs_paths)} no_obs)")
        return selected
    image_paths = sorted(folder.glob("**/*.png"))[:max_images]
    print(f"Found {len(image_paths)} image(s) in {folder}, using {len(image_paths)}")
    return image_paths


def main(folder: Path, max_images: int) -> None:
    image_paths = collect_image_paths(folder, max_images)
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found under {folder} -- check the folder layout "
                                 "(expects obs/, no_obs/, or a flat/nested folder of .png files).")

    all_tiles, all_rows, all_cols = [], [], []
    all_crops, all_sizes = [], []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (config.IMAGE_SIZE_PX, config.IMAGE_SIZE_PX):
            print(f"  skipping {image_path.name} (unreadable or wrong size)")
            continue

        tiles_tensor, rows_list, cols_list, _tile_bboxes = segment_core.extract_tiles(
            image, device, config.SEG_TILE_GRID_SIZE, offsets, config.SEG_OFFSETS_NORM, tile_size, seg_image_size,
        )
        if tiles_tensor is None:
            print(f"  skipping {image_path.name} (produced no tiles)")
            continue
        all_tiles.append(tiles_tensor.cpu())
        all_rows.append(torch.tensor(rows_list, dtype=torch.float32))
        all_cols.append(torch.tensor(cols_list, dtype=torch.float32))

        # Mirrors analyse.py's segment_and_classify() SEGMENT stage exactly (same
        # segment_core functions), just with FP32 encode_fn/decode_fn/score_fn above
        # standing in for the INT8 engines it uses in production.
        scores_array = segment_core.score_tiles(
            tiles_tensor, rows_list, cols_list, encode_fn, decode_fn, score_fn, config.SEG_ENGINE_BATCH, device,
        )
        prob_map_small, _coverage_map_small = segment_core.aggregate_to_patch_map(
            rows_list, cols_list, scores_array, grid_size_small, len(offsets),
        )
        candidates = segment_core.find_candidates(
            prob_map_small, peak_threshold, secondary_threshold, upscale_factor,
            config.MIN_REGION_SIZE_PATCHES, config.CROP_PADDING_PIXELS, config.IMAGE_SIZE_PX,
        )
        accepted = segment_core.dedup_candidates(candidates, config.IOU_DEDUP_THRESHOLD, config.N_CROPS_PER_IMAGE)

        for cand in accepted:
            crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
            crop_image = image[crop_y0:crop_y1, crop_x0:crop_x1]
            crop_tensor = vit_classifier.preprocess_crop_for_classifier(crop_image, image_size=config.CLASSIFIER_IMAGE_SIZE)
            region_size_pixels = cand["region_size_points"] * upscale_factor ** 2
            all_crops.append(crop_tensor)
            all_sizes.append(region_size_pixels * PXL_TO_MM2)

        print(f"  {image_path.name}: {tiles_tensor.shape[0]} tiles, {len(accepted)} crops")

    seg_tiles = torch.cat(all_tiles, dim=0)
    seg_rows = torch.cat(all_rows, dim=0)
    seg_cols = torch.cat(all_cols, dim=0)
    print(f"\nTotal SEGMENT calibration tiles: {seg_tiles.shape[0]}")
    np.save(CAL_DIR / "seg_tiles.npy", seg_tiles.numpy())
    np.save(CAL_DIR / "seg_rows.npy", seg_rows.numpy())
    np.save(CAL_DIR / "seg_cols.npy", seg_cols.numpy())

    print("Running real FP32 encoder+decoder over calibration tiles (needed to calibrate seg_decoder/seg_scorer)...")
    mu_list, recon_list = [], []
    with torch.inference_mode():
        for start in range(0, seg_tiles.shape[0], config.SEG_ENGINE_BATCH):
            end = min(start + config.SEG_ENGINE_BATCH, seg_tiles.shape[0])
            tb = seg_tiles[start:end].to(device)
            rb = seg_rows[start:end].to(device)
            cb = seg_cols[start:end].to(device)
            mu = seg_models.encode(seg_model_fp32, config.SEGMENTATION_ENCODER_TYPE, tb, rb, cb)
            x_hat = seg_model_fp32.decode(mu)
            recon_err = ((tb - x_hat) ** 2).flatten(1).mean(dim=1)
            mu_list.append(mu.cpu().numpy())
            recon_list.append(recon_err.cpu().numpy())
    np.save(CAL_DIR / "seg_mu.npy", np.concatenate(mu_list, axis=0))
    np.save(CAL_DIR / "seg_recon_err.npy", np.concatenate(recon_list, axis=0))

    if all_crops:
        vit_crops = torch.stack(all_crops, dim=0)
        vit_sizes = torch.tensor(all_sizes, dtype=torch.float32).unsqueeze(1)
        print(f"Total CLASSIFY calibration crops: {vit_crops.shape[0]}")
        np.save(CAL_DIR / "vit_crops.npy", vit_crops.numpy())
        np.save(CAL_DIR / "vit_sizes.npy", vit_sizes.numpy())
    else:
        print("No accepted crops found across these images -- vit_crops.npy NOT written. "
              "build_trt_int8.py / benchmark_int8_trt.py will skip the CLASSIFY engine. "
              "Point this at a folder that actually contains organisms.")

    print(f"\nDone. Calibration arrays written to {CAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=str(Path.home() / "test_images" / "OG_images"))
    parser.add_argument("--max-images", type=int, default=60)  # 60 -- was 30, bumped now that
    # this can represent two classes (obs+no_obs) rather than one flat pool; ~30 of each by default
    args = parser.parse_args()
    main(Path(args.folder), args.max_images)
