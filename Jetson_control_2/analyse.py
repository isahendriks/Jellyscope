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
"""

import json
import time

import cv2
import numpy as np
import torch

import config
import queue_io
import gpu_preprocess
import segment_core
from models.segmentation_trt import load_segmentation_engines
from models.vit_classifier_trt import load_classifier_engine, pad_batch
from models import vit_classifier  # preprocess_crop_for_classifier / pad_to_square only -- no FP32 model

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
# Tile batch size (config.SEG_ENGINE_BATCH) lives in config.py, not here -- the SEGMENT INT8
# engine was built at that fixed shape, so analyse.py and the engine must agree on it.

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

print("Loading INT8 segmentation engines (encoder/decoder/scorer)...")
encoder_engine, decoder_engine, scorer_engine, scorer_threshold, seg_grid_size, seg_image_size = load_segmentation_engines(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH, config.ENGINE_DIR, device,
)
if seg_grid_size != tile_grid_size:
    raise ValueError(f"Segmentation checkpoint grid_size={seg_grid_size} does not match tile_grid_size={tile_grid_size}")
peak_threshold = scorer_threshold
secondary_threshold = peak_threshold * 0.99
print(f"Loaded INT8 segmentation engines. Scorer threshold: {scorer_threshold:.4f}")

print("Loading INT8 ViT classifier engine...")
classifier_engine, idx_to_class = load_classifier_engine(
    config.VIT_CHECKPOINT_PATH, config.ENGINE_DIR / "vit_classifier_int8.engine", device,
    class_names_fallback=config.VIT_CLASS_NAMES_FALLBACK,
)
print(f"Loaded classifier with {len(idx_to_class)} classes (fixed batch {config.VIT_ENGINE_BATCH}).")

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
    # Tile extraction/scoring/aggregation/peak-finding/dedup all live in segment_core.py
    # now, shared with test_segmentation_int8.py so the accuracy test exercises the exact
    # same code path production runs (only encode_fn/decode_fn/score_fn differ between the
    # FP32 models here and the INT8 TensorRT engines segmentation_trt.py will provide).
    tiles_tensor, rows_list, cols_list, _tile_bboxes = segment_core.extract_tiles(
        enhanced, device, tile_grid_size, offsets, offsets_norm, tile_size, seg_image_size,
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
            min_region_size_patches, crop_padding_pixels, IMAGE_SIZE_PX,
        )
        t_peaks_end = time.perf_counter()

        accepted = segment_core.dedup_candidates(candidates, IOU_DEDUP_THRESHOLD, N_CROPS_PER_IMAGE)
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
    print(f"[{image_name}] tiles_resize={t_tiles_end - t_tiles_start:.3f}s "
          f"(inference={t_inference_end - t_tiles_end:.3f}s "
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
