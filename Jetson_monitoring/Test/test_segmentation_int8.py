#%% ### PACKAGES ###
"""INT8 TensorRT SEGMENT accuracy validation, styled after
Pipeline_development/BinaryClassification/Test/test_segmentation2.py (same #%%
cell layout, same banner prints, same sklearn ROC/PR/confusion-matrix idioms,
same labelme-mask ground truth via functions.py's
get_manual_crops_from_labelme_json/crops_overlap) -- but run against the INT8
TensorRT engines actually deployed in analyse.py, not the FP32 PyTorch models,
and at three granularities instead of one:

  per-tile  (new)  -- every one of the 1280 raw tiles segment_core.py's
                      extract_tiles/score_tiles produce per image, scored by
                      the INT8 scorer before any aggregation. Ground truth =
                      does this tile's pixel bbox overlap the labelme mask at
                      all. The finest-grained, closest-to-the-network view --
                      no existing script computes this.
  per-patch (new)  -- segment_core.py's aggregate_to_patch_map() output (the
                      grid_size_small x grid_size_small probability map
                      analyse.py already computes internally before peak-
                      finding). One step coarser than per-tile, one step finer
                      than the final crops.
  per-crop (existing style) -- find_candidates+dedup_candidates matched against
                      manual crops via crops_overlap, same TP/FP/FN/TN idea
                      test_segmentation2.py already uses, continuous score =
                      each candidate's peak_val (this cell's equivalent of
                      that script's "peak_intensity").

Uses the exact same tile-extraction/aggregation/peak-finding code
(segment_core.py) analyse.py's production path uses -- the point is to
validate what's actually deployed, not a reimplementation of it that could
quietly drift.
"""

import sys
from pathlib import Path

# Test/ sits directly under Jetson_monitoring/ (a sibling of Monitor/, engines/, models/) --
# PIPELINE_DIR gets `from models import ...` working; Monitor/ additionally gets `import
# config`/`import segment_core`/`from analyse import ...` working, since those now live
# one level deeper than PIPELINE_DIR (in Monitor/, not directly in Jetson_monitoring/).
PIPELINE_DIR = Path(__file__).resolve().parent.parent  # Jetson_monitoring/
for _p in (PIPELINE_DIR / "Monitor", PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score,
    recall_score, f1_score,
)

import config
import segment_core
from models.segmentation_trt import load_segmentation_engines
from Pipeline_development.BinaryClassification.functions import get_manual_crops_from_labelme_json, crops_overlap

print("=" * 70)
print("INT8 SEGMENT accuracy validation -- per-tile / per-patch / per-crop")
print("=" * 70)

#%% Configuration and paths
# Adapted from test_segmentation2.py's ROOT_DIR_C convention -- this runs on the Jetson
# itself (not the Windows training box), so ROOT_DIR points at the mounted training SSD
# instead of a Windows drive letter.
ROOT_DIR = Path("/mnt/sda1/TrainingData/Binary_classifier")
monitoring_effort = "Kristineberg_251128"  # matches config.SEGMENTATION_AE_MODEL_PATH's checkpoint

images_folder = ROOT_DIR / monitoring_effort / "test" / "OG_images"
manual_masks_dir = images_folder.parent / "manual_masks"  # test_segmentation2.py's convention
output_folder = ROOT_DIR / monitoring_effort / "test" / "test_segmentation_int8"
output_folder.mkdir(parents=True, exist_ok=True)

# Same geometry constants analyse.py defines at the top of its own file -- must match
# exactly, since this script scores tiles using the exact same segment_core.py functions
# production uses.
tile_grid_size = 16
offsets_norm = [0, 0.2, 0.4, 0.6, 0.8]
IMAGE_SIZE_PX = 4512
IMAGE_W_MM = 91
PXL_TO_MM = IMAGE_W_MM / IMAGE_SIZE_PX
PXL_TO_MM2 = PXL_TO_MM ** 2
tile_size = IMAGE_SIZE_PX // tile_grid_size
offsets = [int(norm * tile_size) for norm in offsets_norm]
n_offsets = len(offsets)
grid_size_small = tile_grid_size * n_offsets
upscale_factor = IMAGE_SIZE_PX / grid_size_small

min_region_size_patches = 3
crop_padding_pixels = 0
IOU_DEDUP_THRESHOLD = 0.3
MIN_MANUAL_COVERED_PRED_LARGER = 0.70   # matches functions.py's crops_overlap default usage
MIN_MANUAL_COVERED_PRED_SMALLER = 0.90

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

if not images_folder.exists():
    raise FileNotFoundError(
        f"{images_folder} not found -- copy the test split (OG_images/ + manual_masks/) "
        "onto the mounted SSD first"
    )
if not manual_masks_dir.exists():
    raise FileNotFoundError(f"{manual_masks_dir} not found")

#%% Load INT8 segmentation engines
print("Loading INT8 segmentation engines (encoder/decoder/scorer)...")
encoder_engine, decoder_engine, scorer_engine, scorer_threshold, seg_grid_size, seg_image_size = load_segmentation_engines(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH, config.ENGINE_DIR, device,
)
if seg_grid_size != tile_grid_size:
    raise ValueError(f"Checkpoint grid_size={seg_grid_size} does not match tile_grid_size={tile_grid_size}")
print(f"Loaded. Deployed scorer_threshold: {scorer_threshold:.4f}")

#%% Find test images with a matching manual mask
image_paths = sorted(images_folder.glob("*.png"))
mask_paths = {p.stem: p for p in manual_masks_dir.glob("*.json")}
image_paths = [p for p in image_paths if p.stem in mask_paths]
print(f"Found {len(image_paths)} test images with a matching manual_masks/*.json in {manual_masks_dir}")


def rasterize_labelme_mask(json_path, shape_hw) -> np.ndarray:
    """Same rasterization get_manual_crops_from_labelme_json does internally
    (functions.py:1026), reproduced here since that function returns per-region
    crop dicts, not the raw pixel mask this script needs to test tile/patch
    overlap directly against."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for shape in data.get("shapes", []):
        pts = np.array(shape.get("points", []), dtype=np.float32)
        if pts.size == 0:
            continue
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            x0, x1 = sorted([int(round(x0)), int(round(x1))])
            y0, y1 = sorted([int(round(y0)), int(round(y1))])
            cv2.rectangle(mask, (x0, y0), (x1, y1), 1, thickness=-1)
        elif shape_type in {"polygon", "freehand", "linestrip"}:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask

#%% Run SEGMENT (INT8) on every test image, collecting tile / patch / crop scores + labels
tile_rows = []
patch_rows = []
crop_rows = []

for image_path in image_paths:
    image_name = image_path.stem
    print(f"Processing {image_name}...", end=" ")
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image_gray is None:
        print("failed to read, skipping")
        continue
    height, width = image_gray.shape[:2]
    if width != IMAGE_SIZE_PX or height != IMAGE_SIZE_PX:
        print(f"{width}x{height}, expected {IMAGE_SIZE_PX}x{IMAGE_SIZE_PX}, skipping")
        continue

    gt_mask = rasterize_labelme_mask(mask_paths[image_name], (height, width))
    manual_crops = get_manual_crops_from_labelme_json(
        json_path=mask_paths[image_name], image_shape=image_gray.shape, image_path=image_path,
        image_name=image_name, PXL_TO_MM2=PXL_TO_MM2, crop_padding_pixels=crop_padding_pixels,
    )

    ### Per-tile scores (raw scorer output, pre-aggregation)
    tiles_tensor, rows_list, cols_list, tile_bboxes = segment_core.extract_tiles(
        image_gray, device, tile_grid_size, offsets, offsets_norm, tile_size, seg_image_size,
    )
    scores_array = segment_core.score_tiles(
        tiles_tensor, rows_list, cols_list, encoder_engine, decoder_engine, scorer_engine,
        config.SEG_ENGINE_BATCH, device,
    )
    for (y0, x0, y1, x1), score in zip(tile_bboxes, scores_array):
        label_manual = int(gt_mask[y0:y1, x0:x1].any())
        tile_rows.append({
            "image_name": image_name, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
            "score": float(score), "label_manual": label_manual,
        })

    ### Per-patch scores (aggregate_to_patch_map's intermediate probability map)
    prob_map_small, coverage_map_small = segment_core.aggregate_to_patch_map(
        rows_list, cols_list, scores_array, grid_size_small, n_offsets,
    )
    for patch_row in range(grid_size_small):
        for patch_col in range(grid_size_small):
            if coverage_map_small[patch_row, patch_col] <= 0:
                continue  # never covered by any tile -- not a valid patch to score
            y0 = int(round(patch_row * upscale_factor))
            y1 = int(round((patch_row + 1) * upscale_factor))
            x0 = int(round(patch_col * upscale_factor))
            x1 = int(round((patch_col + 1) * upscale_factor))
            label_manual = int(gt_mask[y0:y1, x0:x1].any())
            patch_rows.append({
                "image_name": image_name, "patch_row": patch_row, "patch_col": patch_col,
                "score": float(prob_map_small[patch_row, patch_col]), "label_manual": label_manual,
            })

    ### Per-crop candidates -- a low (not the deployed scorer_threshold) peak/secondary
    ### threshold so weak candidates get caught too, giving a continuous peak_val to sweep
    ### a threshold over (mirrors test_segmentation2.py's peak_intensity-as-score convention).
    ### NOT literally 0.0: the scorer's sigmoid output is always >0, so a secondary_threshold
    ### of exactly 0 never stops flood_fill_region's region growing -- every peak floods the
    ### *entire* probability map into one giant region every time (confirmed: first run with
    ### 0.0 produced exactly 1 "candidate" per image, always overlapping the manual crop,
    ### giving a meaningless all-positive, zero-negative dataset). CANDIDATE_PEAK_THRESHOLD
    ### is chosen well below the deployed scorer_threshold (0.3004) but high enough above 0
    ### that flood-fill stays local to genuinely elevated regions.
    CANDIDATE_PEAK_THRESHOLD = 0.05
    CANDIDATE_SECONDARY_THRESHOLD = 0.02
    candidates = segment_core.find_candidates(
        prob_map_small, peak_threshold=CANDIDATE_PEAK_THRESHOLD, secondary_threshold=CANDIDATE_SECONDARY_THRESHOLD,
        upscale_factor=upscale_factor, min_region_size_patches=min_region_size_patches,
        crop_padding_pixels=crop_padding_pixels, image_size_px=IMAGE_SIZE_PX,
    )
    accepted = segment_core.dedup_candidates(candidates, IOU_DEDUP_THRESHOLD)

    matched_manual_ids = set()
    for crop_id, cand in enumerate(accepted):
        crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
        pred_crop = {"crop_y0": crop_y0, "crop_x0": crop_x0, "crop_y1": crop_y1, "crop_x1": crop_x1}
        label_manual = 0
        for manual_id, manual_crop in enumerate(manual_crops):
            if crops_overlap(pred_crop, manual_crop, MIN_MANUAL_COVERED_PRED_LARGER, MIN_MANUAL_COVERED_PRED_SMALLER):
                label_manual = 1
                matched_manual_ids.add(manual_id)
                break
        region_size_mm2 = cand["region_size_points"] * (upscale_factor ** 2) * PXL_TO_MM2
        crop_rows.append({
            "image_name": image_name, "crop_id": crop_id,
            "crop_y0": crop_y0, "crop_x0": crop_x0, "crop_y1": crop_y1, "crop_x1": crop_x1,
            "peak_intensity": cand["peak_val"], "region_size_mm2": region_size_mm2,
            "label_manual": label_manual, "source": "predicted",
        })

    # Manual crops with no matching predicted candidate at all -- guaranteed misses even at
    # peak_threshold=0, i.e. the deployed model never produced a local maximum there.
    for manual_id, manual_crop in enumerate(manual_crops):
        if manual_id not in matched_manual_ids:
            crop_rows.append({
                "image_name": image_name, "crop_id": f"manual_{manual_id}",
                "crop_y0": manual_crop["crop_y0"], "crop_x0": manual_crop["crop_x0"],
                "crop_y1": manual_crop["crop_y1"], "crop_x1": manual_crop["crop_x1"],
                "peak_intensity": np.nan, "region_size_mm2": manual_crop["region_size_mm2"],
                "label_manual": 1, "source": "manual_unmatched",
            })

    print(f"{len(tile_bboxes)} tiles, {len(accepted)} candidates, {len(manual_crops)} manual crops")

df_tiles = pd.DataFrame(tile_rows)
df_patches = pd.DataFrame(patch_rows)
df_crops = pd.DataFrame(crop_rows)

df_tiles.to_csv(output_folder / "tile_scores.csv", index=False)
df_patches.to_csv(output_folder / "patch_scores.csv", index=False)
df_crops.to_csv(output_folder / "crop_metadata.csv", index=False)
print(f"\nWrote {len(df_tiles)} tile rows, {len(df_patches)} patch rows, {len(df_crops)} crop rows to {output_folder}")


#%% Shared ROC / PR / confusion-matrix helper (matches test_segmentation2.py's metrics block)
def report_roc_pr(df, score_col, label_col, title, deployed_threshold=None):
    df = df[df[score_col].notna()].copy()
    df[label_col] = df[label_col].astype(int)
    y_true = df[label_col].to_numpy()
    y_score = df[score_col].to_numpy()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    # F-beta threshold search, same argmax-over-precision[:-1]/recall[:-1] idiom as
    # test_segmentation2.py/train_DNN.py.
    def f_beta_threshold(beta):
        f_beta = (1 + beta ** 2) * (precision[:-1] * recall[:-1]) / (beta ** 2 * precision[:-1] + recall[:-1] + 1e-8)
        return pr_thresholds[np.argmax(f_beta)]

    threshold_f1 = f_beta_threshold(1.0)
    youden_index = tpr - fpr
    threshold_youden = roc_thresholds[np.argmax(youden_index)]

    threshold = deployed_threshold if deployed_threshold is not None else threshold_youden
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    accuracy_total = accuracy_score(y_true, y_pred)
    precision_total = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_total = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_total = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"n = {len(df)} (positives={int(y_true.sum())}, negatives={int((1 - y_true).sum())})")
    print(f"ROC AUC: {roc_auc:.4f}   Average precision: {average_precision:.4f}")
    print(f"Threshold used: {threshold:.4f} "
          f"(Youden's J: {threshold_youden:.4f}, F1-optimal: {threshold_f1:.4f})")
    print(f"Accuracy: {accuracy_total:.4f}  Precision: {precision_total:.4f}  "
          f"Recall: {recall_total:.4f}  F1: {f1_total:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].scatter(fpr[np.argmax(youden_index)], tpr[np.argmax(youden_index)], color="red", zorder=5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{title} -- ROC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, label=f"AP = {average_precision:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title} -- Precision/Recall")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Obs", "Empty"]).plot(
        ax=axes[2], cmap="Blues", values_format=".2f", colorbar=False,
    )
    axes[2].set_title(f"{title} -- Confusion Matrix @ threshold={threshold:.4f}")

    plt.tight_layout()
    plt.show()
    return {"roc_auc": roc_auc, "average_precision": average_precision, "threshold": threshold}


#%% Per-tile ROC / PR (new granularity -- raw scorer output, no aggregation)
tile_metrics = report_roc_pr(
    df_tiles, "score", "label_manual",
    title="Per-tile (raw INT8 scorer output)", deployed_threshold=scorer_threshold,
)

#%% Per-patch ROC / PR (new granularity -- analyse.py's internal aggregated probability map)
patch_metrics = report_roc_pr(
    df_patches, "score", "label_manual",
    title="Per-patch (aggregated probability map)", deployed_threshold=scorer_threshold,
)

#%% Per-crop ROC / PR (matches test_segmentation2.py's existing crop-level convention)
crop_metrics = report_roc_pr(
    df_crops, "peak_intensity", "label_manual",
    title="Per-crop (post peak-find + dedup, vs. manual crops)", deployed_threshold=scorer_threshold,
)

#%% Summary
print("\n" + "=" * 70)
print("SUMMARY -- INT8 SEGMENT accuracy across granularities")
print("=" * 70)
for name, m in [("Per-tile", tile_metrics), ("Per-patch", patch_metrics), ("Per-crop", crop_metrics)]:
    print(f"{name:<12} ROC AUC={m['roc_auc']:.4f}  AP={m['average_precision']:.4f}")
