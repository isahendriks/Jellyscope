"""Stage 2: preprocess (GPU) -> segment -> classify, one process. Three
sections below, each styled after the specific existing script it's ported
from, not a generic rewrite:

  PREPROCESS -> control/monitoring/record.py's CPU chain, ported to GPU (gpu_preprocess.py)
  SEGMENT    -> ImagePipeline/BinaryClassification/segment_labeled_images.py's working
                reference loop (not segment_per_image.py, which has a broken import)
  CLASSIFY   -> ImagePipeline/ClassClassification/Train_ViT.py's ViT model

Loads your already-trained AE + BinaryScorer + ViT checkpoints; nothing here
ever trains or fine-tunes anything.
"""

import json
import time

import cv2
import numpy as np
import torch

import config
import queue_io
import gpu_preprocess
from models import segmentation as seg_models
from models import vit_classifier

### ==========================
### Segmentation tuning (kept here, not in config.py -- these are knobs you'll
### likely adjust while looking at this specific script, mirroring exactly how
### segment_labeled_images.py defines them near the top of its own loop)
### ==========================
tile_grid_size = 16
offsets_norm = [0, 0.2, 0.4, 0.6, 0.8]  # relative offsets for tile grid (e.g. 0.2 = shift by 20% of tile size)

IMAGE_SIZE_PX = 4512  # camera frame size -- checked against each frame's actual dims below
IMAGE_W_MM = 91  # physical width of the sensor's field of view, in mm
PXL_TO_MM = IMAGE_W_MM / IMAGE_SIZE_PX
PXL_TO_MM2 = PXL_TO_MM ** 2
tile_size = IMAGE_SIZE_PX // tile_grid_size
offsets = [int(norm * tile_size) for norm in offsets_norm]

min_region_size_patches = 3
crop_padding_pixels = 0
N_CROPS_PER_IMAGE = None  # cap crops extracted per frame; None = no cap
IOU_DEDUP_THRESHOLD = 0.3  # matches segment_labeled_images.py's working reference (not segment_per_image.py's broken 0.5)
batch_size = 8912  # tiles per GPU batch, matches segment_labeled_images.py

### ==========================
### GPU preprocessing tuning (control/monitoring/record.py's original values)
### ==========================
HDR_MAX = 4094  # should not be changed!!!!
MEDIAN_KERNEL_SIZE = 3
POST_GAIN = 1
GAMMA = 0.7
CLAHE_CLIP = 0.01
CLAHE_TILE_PX = 512  # kornia wants grid_size (tile count), converted from this below

CLASSIFIER_IMAGE_SIZE = 256  # matches Train_ViT.py's preprocess Resize((256,256))
CONFIDENCE_THRESHOLD = 0.9  # crop filename only gets the predicted species name at/above this

DISK_CHECK_EVERY_N_FRAMES = 5
HEARTBEAT_PATH = config.HEARTBEAT_DIR / "analyse.heartbeat.json"

### ==========================
### Setup: device, models, dark frame
### ==========================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

queue_io.ensure_queue_dirs(config.QUE_FULLFRAMES)
queue_io.ensure_queue_dirs(config.QUE_CROPS)
config.HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading segmentation models (AE + BinaryScorer)...")
seg_model, scorer, scorer_threshold, seg_grid_size, seg_image_size = seg_models.load_segmentation_models(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH,
    config.SEGMENTATION_ENCODER_TYPE, device,
)
if seg_grid_size != tile_grid_size:
    raise ValueError(f"Segmentation checkpoint grid_size={seg_grid_size} does not match tile_grid_size={tile_grid_size}")
peak_threshold = scorer_threshold
secondary_threshold = peak_threshold * 0.99
print(f"Loaded segmentation models. Scorer threshold: {scorer_threshold:.4f}")

print("Loading ViT classifier...")
classifier, idx_to_class = vit_classifier.load_classifier(
    config.VIT_CHECKPOINT_PATH, device, class_names_fallback=config.VIT_CLASS_NAMES_FALLBACK,
)
print(f"Loaded classifier with {len(idx_to_class)} classes.")

dark_frame_gpu = None
if config.DARK_FRAME_PATH.exists():
    dark_frame_gpu = gpu_preprocess.load_dark_frame(str(config.DARK_FRAME_PATH), device)
    print(f"Loaded dark frame from {config.DARK_FRAME_PATH}")
else:
    print(f"No dark frame found at {config.DARK_FRAME_PATH}, skipping dark-frame subtraction.")

grid_size_small = tile_grid_size * len(offsets)
upscale_factor = IMAGE_SIZE_PX / grid_size_small

frames_analysed_total = 0
crops_produced_total = 0


def write_heartbeat() -> None:
    heartbeat = {
        "last_update": time.time(),
        "frames_analysed_total": frames_analysed_total,
        "crops_produced_total": crops_produced_total,
    }
    tmp = HEARTBEAT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(heartbeat))
    tmp.replace(HEARTBEAT_PATH)


def segment_and_classify(enhanced, image_name: str, source_frame_id=None, source_timestamp_unix=None) -> list:
    """Runs SEGMENT + CLASSIFY on an already-enhanced (post-PREPROCESS) image and
    returns a list of (crop_stem, encoded_png_bytes, crop_sidecar) tuples --
    doesn't touch any queue itself. Shared by process_frame() below (the live,
    queue-driven production path) and test_analyse_on_folder.py (a standalone
    dev/test harness for already-preprocessed sample images, which has no
    PREPROCESS step to run and no real queue to write into)."""
    height, width = enhanced.shape[:2]
    if width != IMAGE_SIZE_PX or height != IMAGE_SIZE_PX:
        raise ValueError(f"{image_name} is {width}x{height}, expected {IMAGE_SIZE_PX}x{IMAGE_SIZE_PX} "
                          "(tile_size/upscale_factor are computed assuming this size)")

    t_tiles_start = time.perf_counter()

    ### ==========================
    ### SEGMENT
    ### ==========================
    # Tile extraction + resize, batched on GPU instead of a Python loop calling
    # cv2.resize per tile (1280 tiles/image with 5 offsets x 16x16 grid) -- the
    # slicing loop below is still Python, but it's cheap indexing with no real
    # compute; the actual resize (the expensive part) happens once, batched,
    # across all tiles together.
    enhanced_gpu = torch.from_numpy(enhanced.astype(np.float32)).to(device, non_blocking=True)

    tile_slices, rows_list, cols_list = [], [], []
    for offset, offset_norm in zip(offsets, offsets_norm):
        for row in range(tile_grid_size):
            for col in range(tile_grid_size):
                y0 = offset + row * tile_size
                x0 = offset + col * tile_size
                y1 = min(y0 + tile_size, IMAGE_SIZE_PX)
                x1 = min(x0 + tile_size, IMAGE_SIZE_PX)
                tile = enhanced_gpu[y0:y1, x0:x1]
                pad_h, pad_w = tile_size - tile.shape[0], tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = torch.nn.functional.pad(
                        tile.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate",
                    ).squeeze(0).squeeze(0)
                tile_slices.append(tile)
                rows_list.append(row + offset_norm)
                cols_list.append(col + offset_norm)

    t_tiles_end = time.perf_counter()
    t_resize_end = t_inference_end = t_probmap_end = t_peaks_end = t_tiles_end

    accepted = []
    if tile_slices:
        # Matches preprocess_tile's cv2.resize(INTER_AREA) + /255.0 normalization,
        # batched: one interpolate call for all 1280 tiles instead of 1280 cv2 calls.
        tiles_stacked = torch.stack(tile_slices, dim=0).unsqueeze(1)  # (N, 1, tile_size, tile_size)
        tiles_tensor = torch.nn.functional.interpolate(
            tiles_stacked, size=(seg_image_size, seg_image_size), mode="area",
        ) / 255.0
        rows_tensor = torch.tensor(rows_list, dtype=torch.float32)
        cols_tensor = torch.tensor(cols_list, dtype=torch.float32)

        t_resize_end = time.perf_counter()

        scores_list = []
        with torch.inference_mode():
            for start_idx in range(0, len(tiles_tensor), batch_size):
                end_idx = min(start_idx + batch_size, len(tiles_tensor))
                tiles_batch = tiles_tensor[start_idx:end_idx].to(device, non_blocking=True)
                rows_batch = rows_tensor[start_idx:end_idx].to(device, non_blocking=True)
                cols_batch = cols_tensor[start_idx:end_idx].to(device, non_blocking=True)

                mu = seg_models.encode(seg_model, config.SEGMENTATION_ENCODER_TYPE, tiles_batch, rows_batch, cols_batch)
                x_hat = seg_model.decode(mu)
                recon_error = ((tiles_batch - x_hat) ** 2).flatten(1).mean(dim=1)
                scores = scorer.predict_batch(mu, recon_error, rows_batch, cols_batch)

                scores_list.append(scores.detach().cpu().numpy())

        scores_array = np.concatenate(scores_list, axis=0)

        t_inference_end = time.perf_counter()

        # Aggregate tile scores into a small-grid probability map (matches
        # functions.py's aggregate_scores_to_map).
        prob_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)
        coverage_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)
        n_offsets = len(offsets)
        for tile_row, tile_col, pred_score in zip(rows_list, cols_list, scores_array):
            small_row = int(round(float(tile_row) * n_offsets))
            small_col = int(round(float(tile_col) * n_offsets))
            small_row = max(0, min(small_row, grid_size_small - 1))
            small_col = max(0, min(small_col, grid_size_small - 1))
            prob_map_small[small_row:small_row + n_offsets, small_col:small_col + n_offsets] += float(pred_score)
            coverage_map_small[small_row:small_row + n_offsets, small_col:small_col + n_offsets] += 1.0
        prob_map_small = np.divide(prob_map_small, coverage_map_small, out=prob_map_small, where=coverage_map_small > 0)

        t_probmap_end = time.perf_counter()

        maxima_small = seg_models.find_local_maxima(prob_map_small, peak_threshold)

        candidates = []
        for peak_y_small, peak_x_small, peak_val in maxima_small:
            region_small = seg_models.flood_fill_region(prob_map_small, peak_y_small, peak_x_small, secondary_threshold)
            if len(region_small) < min_region_size_patches ** 2:
                continue
            region_small_points = np.asarray(list(region_small), dtype=np.int32)

            min_y_small = int(region_small_points[:, 0].min())
            max_y_small = int(region_small_points[:, 0].max())
            min_x_small = int(region_small_points[:, 1].min())
            max_x_small = int(region_small_points[:, 1].max())

            y0 = int(round(min_y_small * upscale_factor))
            y1 = int(round((max_y_small + 1) * upscale_factor))
            x0 = int(round(min_x_small * upscale_factor))
            x1 = int(round((max_x_small + 1) * upscale_factor))

            crop_y0 = max(0, y0 - crop_padding_pixels)
            crop_x0 = max(0, x0 - crop_padding_pixels)
            crop_y1 = min(IMAGE_SIZE_PX, y1 + crop_padding_pixels)
            crop_x1 = min(IMAGE_SIZE_PX, x1 + crop_padding_pixels)

            # Ensure crop is square by taking max dimension and re-centering.
            crop_h = crop_y1 - crop_y0
            crop_w = crop_x1 - crop_x0
            crop_size = max(crop_h, crop_w)
            y_center = (crop_y0 + crop_y1) // 2
            x_center = (crop_x0 + crop_x1) // 2
            crop_y0 = y_center - crop_size // 2
            crop_x0 = x_center - crop_size // 2
            crop_y1 = crop_y0 + crop_size
            crop_x1 = crop_x0 + crop_size

            # Clip to image boundaries.
            if crop_y0 < 0:
                crop_y0 = 0
                crop_y1 = min(crop_size, IMAGE_SIZE_PX)
            if crop_y1 > IMAGE_SIZE_PX:
                crop_y1 = IMAGE_SIZE_PX
                crop_y0 = max(0, crop_y1 - crop_size)
            if crop_x0 < 0:
                crop_x0 = 0
                crop_x1 = min(crop_size, IMAGE_SIZE_PX)
            if crop_x1 > IMAGE_SIZE_PX:
                crop_x1 = IMAGE_SIZE_PX
                crop_x0 = max(0, crop_x1 - crop_size)

            candidates.append({
                "region_size_points": len(region_small_points),
                "peak_val": float(peak_val),
                "crop_coords": (crop_y0, crop_x0, crop_y1, crop_x1),
            })

        t_peaks_end = time.perf_counter()

        # Pick the strongest peaks first so overlapping regions keep the highest score.
        candidates = sorted(candidates, key=lambda c: c["peak_val"], reverse=True)
        if N_CROPS_PER_IMAGE is not None:
            candidates = candidates[:N_CROPS_PER_IMAGE]

        for cand in candidates:
            crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
            crop_h = crop_y1 - crop_y0
            crop_w = crop_x1 - crop_x0
            crop_area = crop_h * crop_w

            is_duplicate = False
            for prev in accepted:
                prev_y0, prev_x0, prev_y1, prev_x1 = prev["crop_coords"]
                prev_area = prev["crop_area"]
                inter_y0 = max(crop_y0, prev_y0)
                inter_x0 = max(crop_x0, prev_x0)
                inter_y1 = min(crop_y1, prev_y1)
                inter_x1 = min(crop_x1, prev_x1)
                if inter_y1 > inter_y0 and inter_x1 > inter_x0:
                    inter_area = (inter_y1 - inter_y0) * (inter_x1 - inter_x0)
                    union_area = crop_area + prev_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > IOU_DEDUP_THRESHOLD:
                        is_duplicate = True
                        break
            if not is_duplicate:
                cand["crop_area"] = crop_area
                accepted.append(cand)

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

        for cand in accepted:
            crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
            region_size_pixels = cand["region_size_points"] * upscale_factor ** 2
            region_size_mm2 = region_size_pixels * PXL_TO_MM2
            crop_image = enhanced[crop_y0:crop_y1, crop_x0:crop_x1]

            crop_images.append(crop_image)
            crop_tensors.append(vit_classifier.preprocess_crop_for_classifier(crop_image, image_size=CLASSIFIER_IMAGE_SIZE))
            region_size_pixels_list.append(region_size_pixels)
            region_size_mm2_list.append(region_size_mm2)
            crop_coords_list.append((crop_y0, crop_x0, crop_y1, crop_x1))

        img_batch = torch.stack(crop_tensors, dim=0).to(device, non_blocking=True)
        size_batch = torch.tensor(region_size_mm2_list, dtype=torch.float32).unsqueeze(1).to(device, non_blocking=True)

        with torch.inference_mode():
            logits = classifier(img_batch, size_batch)
            probs = torch.softmax(logits, dim=1)
            class_indices = torch.argmax(probs, dim=1)
            confidences = probs.gather(1, class_indices.unsqueeze(1)).squeeze(1)

        class_indices = class_indices.cpu().tolist()
        confidences = confidences.cpu().tolist()

        for crop_idx in range(len(accepted)):
            crop_y0, crop_x0, crop_y1, crop_x1 = crop_coords_list[crop_idx]
            region_size_pixels = region_size_pixels_list[crop_idx]
            region_size_mm2 = region_size_mm2_list[crop_idx]
            crop_image = crop_images[crop_idx]
            class_idx = class_indices[crop_idx]
            class_confidence = confidences[crop_idx]
            class_label = idx_to_class[class_idx]

            # Only put the predicted species in the filename once confidence clears
            # the threshold -- below that, the crop is still saved (and still fully
            # labeled in the sidecar JSON), just without a species name in the title.
            if class_confidence >= CONFIDENCE_THRESHOLD:
                crop_stem = f"{image_name}_crop_{crop_idx:03d}_{class_label}_area_{region_size_mm2:.2f}"
            else:
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
                "class_label": class_label,
                "class_confidence": class_confidence,
                "class_idx": class_idx,
                "classifier_checkpoint": str(config.VIT_CHECKPOINT_PATH),
                "processed_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                "upload_attempts": 0,
            }
            results.append((crop_stem, encoded_crop.tobytes(), crop_sidecar))

    t_classify_end = time.perf_counter()
    print(f"[{image_name}] tiles={t_tiles_end - t_tiles_start:.3f}s "
          f"(resize={t_resize_end - t_tiles_end:.3f}s "
          f"inference={t_inference_end - t_resize_end:.3f}s "
          f"probmap={t_probmap_end - t_inference_end:.3f}s "
          f"peaks={t_peaks_end - t_probmap_end:.3f}s "
          f"dedup={t_segment_end - t_peaks_end:.3f}s) "
          f"segment={t_segment_end - t_tiles_end:.3f}s "
          f"classify={t_classify_end - t_segment_end:.3f}s "
          f"total={t_classify_end - t_tiles_start:.3f}s crops={len(results)}")

    return results


def process_frame(image_path, json_path) -> None:
    global crops_produced_total
    sidecar = queue_io.read_sidecar(json_path)
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_gray is None:
        raise ValueError(f"Failed to read {image_path}")

    height, width = image_gray.shape[:2]
    if width != IMAGE_SIZE_PX or height != IMAGE_SIZE_PX:
        raise ValueError(f"Frame {image_path.name} is {width}x{height}, expected {IMAGE_SIZE_PX}x{IMAGE_SIZE_PX}")

    image_name = image_path.stem

    ### ==========================
    ### PREPROCESS
    ### ==========================
    clahe_grid_n = max(1, round(IMAGE_SIZE_PX / CLAHE_TILE_PX))
    enhanced = gpu_preprocess.gpu_preprocess_frame(
        image_gray, dark_frame_gpu, device, HDR_MAX, MEDIAN_KERNEL_SIZE,
        GAMMA, POST_GAIN, CLAHE_CLIP, (clahe_grid_n, clahe_grid_n),
    )

    results = segment_and_classify(
        enhanced, image_name,
        source_frame_id=sidecar["frame_id"], source_timestamp_unix=sidecar["timestamp_unix"],
    )
    for crop_stem, encoded_bytes, crop_sidecar in results:
        queue_io.write_item(config.QUE_CROPS, crop_stem, encoded_bytes, ".png", crop_sidecar)
        crops_produced_total += 1


### ==========================
### Main loop
### ==========================
# Guarded behind __main__ so test_analyse_on_folder.py can `from analyse import
# segment_and_classify` (which runs all the setup above -- device, model
# loading, dark frame) without also kicking off the queue-watching loop below.
if __name__ == "__main__":
    for stem in queue_io.recover_processing(config.QUE_FULLFRAMES, ".tiff"):
        print(f"Recovering {stem} from a previous crash...")
        image_path = config.QUE_FULLFRAMES / "processing" / f"{stem}.tiff"
        json_path = config.QUE_FULLFRAMES / "processing" / f"{stem}.json"
        try:
            process_frame(image_path, json_path)
            queue_io.ack_delete(config.QUE_FULLFRAMES, stem, ".tiff")
            frames_analysed_total += 1
        except Exception as exc:
            queue_io.fail_item(config.QUE_FULLFRAMES, stem, ".tiff", str(exc))

    print("analyse.py ready, watching que_fullframes...")

    while True:
        if frames_analysed_total % DISK_CHECK_EVERY_N_FRAMES == 0:
            free = queue_io.disk_free_bytes(config.QUEUE_ROOT)
            if free < config.MIN_FREE_BYTES:
                print(f"Low disk space ({free / 1e9:.2f} GB free), pausing analysis...")
                write_heartbeat()
                time.sleep(5)
                continue

        stems = queue_io.list_ready_stems(config.QUE_FULLFRAMES)
        if not stems:
            time.sleep(0.5)
            continue

        stem = stems[0]
        claimed = queue_io.claim(config.QUE_FULLFRAMES, stem, ".tiff")
        if claimed is None:
            continue
        image_path, json_path = claimed

        try:
            process_frame(image_path, json_path)
            queue_io.ack_delete(config.QUE_FULLFRAMES, stem, ".tiff")
            frames_analysed_total += 1
        except Exception as exc:
            print(f"Failed to process {stem}: {exc}")
            queue_io.fail_item(config.QUE_FULLFRAMES, stem, ".tiff", str(exc))

        write_heartbeat()
