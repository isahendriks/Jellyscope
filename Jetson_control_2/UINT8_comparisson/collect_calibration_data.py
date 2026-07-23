"""Collect real (not synthetic) tiles/crops for TensorRT INT8 calibration.

INT8 calibration needs activation ranges from data that looks like production
input. benchmark_segment_inference.py's torch.rand() tiles are fine for
*timing* experiments but would give TensorRT unrealistic (uniform-random)
dynamic ranges for INT8 quantization -- which degrades accuracy silently,
with no error to warn you.

Reuses analyse.py's own tile-extraction/tiling parameters and its real
segment_and_classify() function by importing analyse as a module (same trick
test_analyse_on_folder.py uses: analyse.py's queue-watching loop is guarded
behind `if __name__ == "__main__":`, so importing it just runs the device +
model-loading setup). That means:
  - SEGMENT calibration tiles are generated with the exact tile_size/grid/
    offsets analyse.py itself uses -- can't drift out of sync.
  - CLASSIFY calibration crops are the real *accepted* crops that come out of
    the real SEGMENT step on these images, decoded back from the PNG bytes
    segment_and_classify() returns, then run through the same
    preprocess_crop_for_classifier() the production path uses.

Run this on the Jetson against a folder of real preprocessed sample images
(same default folder test_analyse_on_folder.py uses).

Writes, under Jetson_control_2/trt/calibration/:
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Jetson_control_2/

import cv2
import numpy as np
import torch

import analyse
import config
from models import vit_classifier

CAL_DIR = config.PIPELINE_DIR / "trt" / "calibration"
CAL_DIR.mkdir(parents=True, exist_ok=True)


def extract_calibration_tiles(image: np.ndarray):
    """Mirrors segment_and_classify()'s tile-extraction+resize block exactly,
    reading tile_size/grid/offsets/seg_image_size from the already-imported
    analyse module so this can never drift out of sync with production
    tiling -- but stops after resize instead of continuing to inference."""
    enhanced_gpu = torch.from_numpy(image.astype(np.float32)).to(analyse.device, non_blocking=True)
    tile_slices, rows_list, cols_list = [], [], []
    for offset, offset_norm in zip(analyse.offsets, analyse.offsets_norm):
        for row in range(analyse.tile_grid_size):
            for col in range(analyse.tile_grid_size):
                y0, x0 = offset + row * analyse.tile_size, offset + col * analyse.tile_size
                y1 = min(y0 + analyse.tile_size, analyse.IMAGE_SIZE_PX)
                x1 = min(x0 + analyse.tile_size, analyse.IMAGE_SIZE_PX)
                tile = enhanced_gpu[y0:y1, x0:x1]
                pad_h, pad_w = analyse.tile_size - tile.shape[0], analyse.tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = torch.nn.functional.pad(
                        tile.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate",
                    ).squeeze(0).squeeze(0)
                tile_slices.append(tile)
                rows_list.append(row + offset_norm)
                cols_list.append(col + offset_norm)

    tiles_stacked = torch.stack(tile_slices, dim=0).unsqueeze(1)
    tiles_tensor = torch.nn.functional.interpolate(
        tiles_stacked, size=(analyse.seg_image_size, analyse.seg_image_size), mode="area",
    ) / 255.0
    rows_tensor = torch.tensor(rows_list, dtype=torch.float32, device=analyse.device)
    cols_tensor = torch.tensor(cols_list, dtype=torch.float32, device=analyse.device)
    return tiles_tensor, rows_tensor, cols_tensor


def main(folder: Path, max_images: int) -> None:
    image_paths = sorted(folder.glob("*.png"))[:max_images]
    print(f"Found {len(image_paths)} images in {folder}, using {len(image_paths)}")

    all_tiles, all_rows, all_cols = [], [], []
    all_crops, all_sizes = [], []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (analyse.IMAGE_SIZE_PX, analyse.IMAGE_SIZE_PX):
            print(f"  skipping {image_path.name} (unreadable or wrong size)")
            continue

        tiles, rows, cols = extract_calibration_tiles(image)
        all_tiles.append(tiles.cpu())
        all_rows.append(rows.cpu())
        all_cols.append(cols.cpu())

        results = analyse.segment_and_classify(image, image_path.stem)
        for _crop_stem, encoded_bytes, sidecar in results:
            crop_arr = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            crop_tensor = vit_classifier.preprocess_crop_for_classifier(
                crop_arr, image_size=analyse.CLASSIFIER_IMAGE_SIZE,
            )
            all_crops.append(crop_tensor)
            all_sizes.append(sidecar["region_size_mm2"])

        print(f"  {image_path.name}: {tiles.shape[0]} tiles, {len(results)} crops")

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
        for start in range(0, seg_tiles.shape[0], analyse.batch_size):
            end = min(start + analyse.batch_size, seg_tiles.shape[0])
            tb = seg_tiles[start:end].to(analyse.device)
            rb = seg_rows[start:end].to(analyse.device)
            cb = seg_cols[start:end].to(analyse.device)
            mu = analyse.seg_models.encode(analyse.seg_model, config.SEGMENTATION_ENCODER_TYPE, tb, rb, cb)
            x_hat = analyse.seg_model.decode(mu)
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
    parser.add_argument("--max-images", type=int, default=30)
    args = parser.parse_args()
    main(Path(args.folder), args.max_images)
