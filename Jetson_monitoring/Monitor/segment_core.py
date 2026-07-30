"""Shared segmentation building blocks, extracted from analyse.py's original
monolithic `segment_and_classify` so the ROC/PR accuracy test script
(test_segmentation_int8.py) can score tiles/patches using the *exact* code path
production runs, instead of a reimplementation that could quietly drift from
what's actually deployed.

`encode_fn`/`decode_fn`/`score_fn` are passed in by the caller rather than
imported here, so this file doesn't change when analyse.py switches from the
FP32 PyTorch models (models/segmentation.py) to the INT8 TensorRT engines
(models/segmentation_trt.py) -- only the callables passed in change.
"""

import numpy as np
import torch
import torch.nn.functional as F

from models import segmentation as seg_models

__all__ = [
    "extract_tiles",
    "score_tiles",
    "aggregate_to_patch_map",
    "find_candidates",
    "dedup_candidates",
]


def extract_tiles(enhanced, device, tile_grid_size, offsets, offsets_norm, tile_size, seg_image_size):
    """GPU-batched tile extraction + resize. Returns (tiles_tensor, rows_list, cols_list,
    tile_bboxes) -- tiles_tensor is (N,1,seg_image_size,seg_image_size) normalized to [0,1]
    on `device`; rows_list/cols_list are Python floats (grid coordinate + offset_norm,
    matching functions.py's row/col convention); tile_bboxes is a parallel list of
    (y0, x0, y1, x1) pixel coordinates (pre-padding) for each tile -- kept alongside
    rows_list/cols_list rather than re-derived from them (ambiguous/fragile due to float
    round-trips), needed by test_segmentation_int8.py to test each tile against ground
    truth. tiles_tensor is None if the frame produced no tiles."""
    enhanced_gpu = torch.from_numpy(enhanced.astype(np.float32)).to(device, non_blocking=True)

    tile_slices, rows_list, cols_list, tile_bboxes = [], [], [], []
    for offset, offset_norm in zip(offsets, offsets_norm):
        for row in range(tile_grid_size):
            for col in range(tile_grid_size):
                y0 = offset + row * tile_size
                x0 = offset + col * tile_size
                y1 = min(y0 + tile_size, enhanced_gpu.shape[0])
                x1 = min(x0 + tile_size, enhanced_gpu.shape[1])
                tile = enhanced_gpu[y0:y1, x0:x1]
                pad_h, pad_w = tile_size - tile.shape[0], tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(
                        tile.unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate",
                    ).squeeze(0).squeeze(0)
                tile_slices.append(tile)
                rows_list.append(row + offset_norm)
                cols_list.append(col + offset_norm)
                tile_bboxes.append((y0, x0, y1, x1))

    if not tile_slices:
        return None, rows_list, cols_list, tile_bboxes

    # Matches preprocess_tile's cv2.resize(INTER_AREA) + /255.0 normalization,
    # batched: one interpolate call for all tiles instead of one cv2 call per tile.
    tiles_stacked = torch.stack(tile_slices, dim=0).unsqueeze(1)  # (N, 1, tile_size, tile_size)
    tiles_tensor = torch.nn.functional.interpolate(
        tiles_stacked, size=(seg_image_size, seg_image_size), mode="area",
    ) / 255.0
    return tiles_tensor, rows_list, cols_list, tile_bboxes


def score_tiles(tiles_tensor, rows_list, cols_list, encode_fn, decode_fn, score_fn, batch_size, device):
    """Runs encode -> decode -> score over all tiles in chunks of batch_size. Returns a
    numpy array of one score per tile, in the same order as rows_list/cols_list."""
    if tiles_tensor is None:
        return np.zeros(0, dtype=np.float32)

    rows_tensor = torch.tensor(rows_list, dtype=torch.float32)
    cols_tensor = torch.tensor(cols_list, dtype=torch.float32)

    scores_list = []
    with torch.inference_mode():
        for start_idx in range(0, len(tiles_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(tiles_tensor))
            tiles_batch = tiles_tensor[start_idx:end_idx].to(device, non_blocking=True)
            rows_batch = rows_tensor[start_idx:end_idx].to(device, non_blocking=True)
            cols_batch = cols_tensor[start_idx:end_idx].to(device, non_blocking=True)

            mu = encode_fn(tiles_batch, rows_batch, cols_batch)
            x_hat = decode_fn(mu)
            recon_error = ((tiles_batch - x_hat) ** 2).flatten(1).mean(dim=1)
            scores = score_fn(mu, recon_error, rows_batch, cols_batch)

            scores_list.append(scores.detach().cpu().numpy())

    return np.concatenate(scores_list, axis=0)


def aggregate_to_patch_map(rows_list, cols_list, scores_array, grid_size_small, n_offsets):
    """Aggregates raw tile scores into a small-grid probability map (matches
    functions.py's aggregate_scores_to_map). Returns (prob_map_small, coverage_map_small)."""
    prob_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)
    coverage_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)
    for tile_row, tile_col, pred_score in zip(rows_list, cols_list, scores_array):
        small_row = int(round(float(tile_row) * n_offsets))
        small_col = int(round(float(tile_col) * n_offsets))
        small_row = max(0, min(small_row, grid_size_small - 1))
        small_col = max(0, min(small_col, grid_size_small - 1))
        prob_map_small[small_row:small_row + n_offsets, small_col:small_col + n_offsets] += float(pred_score)
        coverage_map_small[small_row:small_row + n_offsets, small_col:small_col + n_offsets] += 1.0
    prob_map_small = np.divide(prob_map_small, coverage_map_small, out=prob_map_small, where=coverage_map_small > 0)
    return prob_map_small, coverage_map_small


def find_candidates(prob_map_small, peak_threshold, secondary_threshold, upscale_factor,
                     min_region_size_patches, crop_padding_pixels, image_size_px):
    """Peak-find + flood-fill the patch map into candidate crop regions (pre-dedup).
    Returns a list of {"region_size_points", "peak_val", "crop_coords"} dicts."""
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
        crop_y1 = min(image_size_px, y1 + crop_padding_pixels)
        crop_x1 = min(image_size_px, x1 + crop_padding_pixels)

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
            crop_y1 = min(crop_size, image_size_px)
        if crop_y1 > image_size_px:
            crop_y1 = image_size_px
            crop_y0 = max(0, crop_y1 - crop_size)
        if crop_x0 < 0:
            crop_x0 = 0
            crop_x1 = min(crop_size, image_size_px)
        if crop_x1 > image_size_px:
            crop_x1 = image_size_px
            crop_x0 = max(0, crop_x1 - crop_size)

        candidates.append({
            "region_size_points": len(region_small_points),
            "peak_val": float(peak_val),
            "crop_coords": (crop_y0, crop_x0, crop_y1, crop_x1),
        })

    return candidates


def dedup_candidates(candidates, iou_threshold, n_crops_cap=None):
    """Picks strongest peaks first, drops overlapping candidates (IoU > iou_threshold)."""
    candidates = sorted(candidates, key=lambda c: c["peak_val"], reverse=True)
    if n_crops_cap is not None:
        candidates = candidates[:n_crops_cap]

    accepted = []
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
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            cand["crop_area"] = crop_area
            accepted.append(cand)

    return accepted
