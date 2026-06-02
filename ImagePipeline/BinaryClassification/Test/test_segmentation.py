#%% Import cells
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import time
from matplotlib.patches import Rectangle
import gc

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

# matplotlib.use("TkAgg")  # Interactive backend so debug plots can open windows.
import matplotlib.pyplot as plt
# plt.ion()

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import find_local_maxima, flood_fill_region, robust_flood_fill
from segment_per_batch import report_torch_environment, load_vae_model, load_scorer_model, aggregate_scores_to_map, image_to_probability_map, batch_extract_tiles_from_images, batch_run_inference, split_results_by_image, save_crop_metadata_rows


#%% test segmentation on a single image
device = report_torch_environment()

# Define paths and parameters
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
# ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data"
monitoring_effort = "Kristineberg_251128"  # for titles and saved model names, e.g. "kristineberg_251128"
# images_folder = os.path.join(ROOT_DIR_R, monitoring_effort)
images_folder = os.path.join(ROOT_DIR_C, monitoring_effort, "test", "OG_images")
output_folder = Path(os.path.join(ROOT_DIR_C, monitoring_effort, "test", "test_segmentation"))
output_folder.mkdir(parents=True, exist_ok=True)

tile_grid_size = 16
latent_dim = 32

IMAGE_SIZE_PX = 4512  # All images are square 4512x4512
tile_size = IMAGE_SIZE_PX // tile_grid_size

PLOT_DEBUG = True
SAVE_DEBUG_PLOTS = False
SAVE_CROPS = False

model_name = f"../models/VAE/{monitoring_effort}_vae_model{tile_grid_size}_l{latent_dim}.pth"
scorer_mode = "binary"

# Load all images    
# images_path = sorted(str(p) for p in Path(images_folder).glob("*.png"))

# Load images in crop_metadata.csv to ensure we only test on images that have manual crops (and avoid accidentally loading any non-image files in the folder)
manual_crops_csv = os.path.join(images_folder[:-10], f"tiles{tile_grid_size}_offsets{len(offsets_norm)}", "crop_metadata.csv")
manual_crops_metadata = pd.read_csv(manual_crops_csv)
images_in_metadata = manual_crops_metadata["image_path"].unique()
images_path = [p for p in images_in_metadata]

N_TEST_IMAGES = None
if N_TEST_IMAGES is not None and len(images_path) > N_TEST_IMAGES:
    images_path = images_path[:N_TEST_IMAGES]

# Load models and thresholds
model, _, _, grid_size_loaded = load_vae_model(device, model_name, latent_dim)
scorer, scorer_threshold, _ = load_scorer_model(device, model_name, scorer_mode, latent_dim, tile_grid_size)

# set parameters for segmentation
peak_threshold = 0.9
secondary_threshold = 0.9 * peak_threshold
# mean_threshold = 0.6
min_region_size_patches = 3
crop_padding_pixels = 0
offsets_norm = [0, 0.2, 0.4, 0.6, 0.8]  # relative offsets for tile grid (e.g. 0.2 means shift grid by 20% of tile size)

offsets = [int(norm * tile_size) for norm in offsets_norm]

batch_size_images = 1  # Process 32 images per batch

batch_size = 8912  # Process 8192 tiles per batch (adjust based on GPU memory)
vae_image_size = 128

grid_size_small = tile_grid_size * len(offsets)
upscale_factor = IMAGE_SIZE_PX/grid_size_small 

# Set scorer mode (binary or one_class) and load corresponding model and threshold
if grid_size_loaded != tile_grid_size:
    raise ValueError(f"⚠ Warning: VAE model grid_size={grid_size_loaded} does not match expected tile_grid_size={tile_grid_size}. ")

output_folder_crops = os.path.join(output_folder, "crops")
os.makedirs(output_folder_crops, exist_ok=True)

output_folder_prob_maps = os.path.join(output_folder, "probability_maps")
os.makedirs(output_folder_prob_maps, exist_ok=True)

output_folder_debug_plots = os.path.join(output_folder, "debug_plots")
os.makedirs(output_folder_debug_plots, exist_ok=True)

crop_metadata_csv = os.path.join(output_folder_crops, "crop_metadata.csv")

### Print initial summary of processing plan
print("\n" + "=" * 70)
print("Starting Batch Image Processing for Anomaly Detection")
print("=" * 70)
    
print(f"Monitoring effort:      {monitoring_effort}")
print(f"Input images folder:    {images_folder}")
print(f"Number of images:       {len(images_path)}")
print(f"Image size (px):        {IMAGE_SIZE_PX}x{IMAGE_SIZE_PX}")
print(f"Tile grid size:         {tile_grid_size}x{tile_grid_size} (tile size: {tile_size}px)")
print(f"VAE model:              {model_name}")
print(f"Scorer mode:            {scorer_mode}")
print(f"Threshold:              {peak_threshold}" )
    
#%% Start processing images in batches    
print(f"\nStarting processing of {len(images_path)} images with {len(offsets)} offsets...\n")

# timing accumulators
total_images_processed = 0
total_processing_time = 0.0
total_image_load_time = 0.0

crop_counter = 0
images_batch = []
image_paths_batch = []
batch_load_time_accumulator = 0.0

# Start processing images in batches
for image_idx, image_path_str in enumerate(images_path, start=1):
    image_path = Path(image_path_str)
    img_load_start = time.perf_counter()
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
    images_batch.append(image_gray)
    image_paths_batch.append(image_path)
        
    # Process batch when full or at end of list
    if len(images_batch) == batch_size_images or image_idx == len(images_path):
        batch_start_time = time.perf_counter()
        tiles_tensor, rows_tensor, cols_tensor, image_tile_counts, (rows_list, cols_list) = batch_extract_tiles_from_images(images_batch, tile_size, tile_grid_size, offsets, offsets_norm, vae_image_size) # type: ignore
            
        # run inference on all tiles in batch and aggregate results back to image-level maps
        scores_array, preds_array, mu_array, recon_array = batch_run_inference(
            tiles_tensor, rows_tensor, cols_tensor, model, scorer, scorer_threshold, device, batch_size, image_tile_counts
        )
            
        results_per_image = split_results_by_image(scores_array, rows_list, cols_list, tile_grid_size, offsets, image_tile_counts)         

        deferred_crops = []
        batch_crop_rows = []
                
        # Process each image's results
        for batch_img_idx, (image_gray, image_path) in enumerate(zip(images_batch, image_paths_batch)):
            image_name = image_path.stem

            result = results_per_image[batch_img_idx]
            if result is None:
                # print("no tiles")
                continue
                    
            prob_map_small = result["prob_map_small"]    
                        
            maxima_small = find_local_maxima(prob_map_small, peak_threshold)
            if len(maxima_small) == 0:
                continue

            # Convert each small-grid flood-filled region into a pixel crop bbox
            candidates = []
            for peak_y_small, peak_x_small, peak_val in maxima_small:
                region_small = flood_fill_region(prob_map_small, peak_y_small, peak_x_small, secondary_threshold)

                if len(region_small) < min_region_size_patches ** 2:
                    continue

                region_small_points = np.asarray(list(region_small), dtype=np.int32)
                region_mean = prob_map_small[region_small_points[:, 0], region_small_points[:, 1]].mean() if len(region_small_points) > 0 else 0.0
                if region_mean < mean_threshold:
                    continue

                min_y_small = int(region_small_points[:, 0].min())
                max_y_small = int(region_small_points[:, 0].max())
                min_x_small = int(region_small_points[:, 1].min())
                max_x_small = int(region_small_points[:, 1].max())

                # Map small-grid bounds to full-resolution pixel bbox
                y0 = int(round(min_y_small * upscale_factor))
                y1 = int(round((max_y_small + 1) * upscale_factor))
                x0 = int(round(min_x_small * upscale_factor))
                x1 = int(round((max_x_small + 1) * upscale_factor))

                crop_y0 = max(0, y0 - crop_padding_pixels)
                crop_x0 = max(0, x0 - crop_padding_pixels)
                crop_y1 = min(IMAGE_SIZE_PX, y1 + crop_padding_pixels)
                crop_x1 = min(IMAGE_SIZE_PX, x1 + crop_padding_pixels)

                # Ensure crop is square by taking max dimension and re-centering
                crop_h = crop_y1 - crop_y0
                crop_w = crop_x1 - crop_x0
                crop_size = max(crop_h, crop_w)
                        
                y_center = (crop_y0 + crop_y1) // 2
                x_center = (crop_x0 + crop_x1) // 2
                        
                crop_y0 = y_center - crop_size // 2
                crop_x0 = x_center - crop_size // 2
                crop_y1 = crop_y0 + crop_size
                crop_x1 = crop_x0 + crop_size
                        
                # Clip to image boundaries
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

                candidates.append({"region_small_points": region_small_points,
                                    "peak_val": float(peak_val),
                                    "crop_coords": (crop_y0, crop_x0, crop_y1, crop_x1),
                                    "region_mean_intensity": float(region_mean),
                                    "region_size": len(region_small_points)
                                    })

            if len(candidates) == 0:
                continue

            # Pick the strongest peaks first
            candidates = sorted(candidates, key=lambda c: c["peak_val"], reverse=True)

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
                                
                        if iou > 0.5:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    cand["crop_area"] = crop_area
                    accepted.append(cand)
              
            # Save crops and metadata
            crop_metadata = []
            for crop_idx, cand in enumerate(accepted):
                region_points = cand["region_small_points"]
                peak_val = cand["peak_val"]
                crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
                crop_h = crop_y1 - crop_y0
                crop_w = crop_x1 - crop_x0

                region_values = prob_map_small[region_points[:, 0], region_points[:, 1]] if len(region_points) > 0 else np.array([], dtype=np.float32)
                region_mean = region_values.mean() if len(region_values) > 0 else 0.0
                region_max = region_values.max() if len(region_values) > 0 else 0.0
                    
                crop_image = image_gray[crop_y0:crop_y1, crop_x0:crop_x1]
                crop_name = f"{image_name}_crop_{crop_idx:03d}.png"
                crop_path = os.path.join(output_folder_crops, crop_name)

                crop_metadata.append({  "OG_image_path": image_path,
                                    "image_name": image_name,
                                    "crop_id": crop_idx,
                                    "crop_width": crop_w,
                                    "crop_height": crop_h,
                                    "crop_y0": crop_y0,
                                    "crop_x0": crop_x0,
                                    "crop_y1": crop_y1,
                                    "crop_x1": crop_x1,
                                    "region_size_pixels": len(region_points),
                                    "peak_intensity": peak_val,
                                    "region_mean_intensity": region_mean,
                                    "region_max_intensity": region_max,
                                    "label_manual": 0,  # to be updated later based on overlap with manual crops
                                    })

                deferred_crops.append((crop_image.copy(), crop_path))

                crop_counter += 1
            
            # Print summary of results for this image
            print(f"Processed image {total_images_processed}/{len(images_path)}: {image_name} | #crops: {len(accepted)} | Total crops so far: {crop_counter}")

            batch_crop_rows.extend(crop_metadata)
            
        ### Compare IoU of manual crops to see if crop is TP or FP
        
        manual_crops = manual_crops_metadata[manual_crops_metadata["image_path"] == str(image_path)]
        if len(manual_crops) > 0 and len(batch_crop_rows) > 0:
            
            ### Add manual crop to batch_crop_rows with label_manual=1 for easy comparison in debug plots (even though we won't save it as a crop since it's not generated by the model)
            crop_metadata_manual = []
            for manual_idx, manual_row in manual_crops.iterrows():
                manual_crop_row = manual_row.to_dict()
                crop_w = manual_crop_row["bbox_x1"] - manual_crop_row["bbox_x0"]
                crop_h = manual_crop_row["bbox_y1"] - manual_crop_row["bbox_y0"]
                crop_metadata_manual.append({  "OG_image_path": image_path,
                                    "image_name": image_name,
                                    "crop_id": len(batch_crop_rows) + manual_idx,  # assign next crop_id
                                    "crop_width": manual_crop_row["bbox_x1"] - manual_crop_row["bbox_x0"],
                                    "crop_height": manual_crop_row["bbox_y1"] - manual_crop_row["bbox_y0"],
                                    "crop_y0": manual_crop_row["bbox_y0"],
                                    "crop_x0": manual_crop_row["bbox_x0"],
                                    "crop_y1": manual_crop_row["bbox_y1"],
                                    "crop_x1": manual_crop_row["bbox_x1"],
                                    "region_size_pixels": manual_crop_row['selected_hr_cells'],
                                    "peak_intensity": None,
                                    "region_mean_intensity": None,
                                    "region_max_intensity": None,
                                    "label_manual": 1,  # to be updated later based on overlap with manual crops
                                    })
            batch_crop_rows.extend(crop_metadata_manual)
            min_overlap = 0.1 # minimum IoU overlap between predicted crop and manual crop to be considered a correct prediction (you can adjust this threshold based on how precise you want the crops to be)
            
            # determine labels based on overlap with manual crops (if a predicted crop overlaps with any manual crop, label it as 1, otherwise 0)
            for pred_idx, pred_row in enumerate(batch_crop_rows):
                OG_image_path = pred_row["OG_image_path"]

                pred_y0, pred_x0, pred_y1, pred_x1 = pred_row["crop_y0"], pred_row["crop_x0"], pred_row["crop_y1"], pred_row["crop_x1"]
                for _, manual_row in manual_crops.iterrows():
                    manual_y0, manual_x0, manual_y1, manual_x1 = manual_row["bbox_y0"], manual_row["bbox_x0"], manual_row["bbox_y1"], manual_row["bbox_x1"]
                    inter_y0 = max(pred_y0, manual_y0)
                    inter_x0 = max(pred_x0, manual_x0)
                    inter_y1 = min(pred_y1, manual_y1)
                    inter_x1 = min(pred_x1, manual_x1)
                    if inter_y1 > inter_y0 and inter_x1 > inter_x0:
                        iou = (inter_y1 - inter_y0) * (inter_x1 - inter_x0) / ((pred_y1 - pred_y0) * (pred_x1 - pred_x0) + (manual_y1 - manual_y0) * (manual_x1 - manual_x0) - (inter_y1 - inter_y0) * (inter_x1 - inter_x0))
                        if iou >= min_overlap:
                            batch_crop_rows[pred_idx]["label_manual"] = 1  # label as 1 if it overlaps with a manual crop

                            # remove manual crop from batch_crop_rows to avoid double counting if multiple predicted crops overlap with the same manual crop (since we only want to count one TP per manual crop)
                            batch_crop_rows = [row for row in batch_crop_rows if not (row["label_manual"] == 1 and row["crop_y0"] == manual_y0 and row["crop_x0"] == manual_x0 and row["crop_y1"] == manual_y1 and row["crop_x1"] == manual_x1)]
                            

        # Make subplot with all extracted crops (save-only by default)
        print(len(batch_crop_rows), "crops extracted for this image, comparing to", len(manual_crops), "manual crops")
        if len(batch_crop_rows) > 0 and (PLOT_DEBUG or SAVE_DEBUG_PLOTS):
            ### Plot subplot with heatmaps and flood filled regions with red contour and final crop with green box for debugging
            # print(f"plotting image: {image_name} with {len(candidates)} candidates, {len(accepted)} accepted crops")
            fig, ax = plt.subplots(1,3, figsize=(16, 8))
                
            # Manual crop overlay
            ax[0].imshow(image_gray, cmap="gray", vmin=0, vmax=255)
            for _, manual_row in manual_crops.iterrows():   
                manual_y0, manual_x0, manual_y1, manual_x1 = manual_row["bbox_y0"], manual_row["bbox_x0"], manual_row["bbox_y1"], manual_row["bbox_x1"]
                rect = Rectangle((manual_x0, manual_y0), manual_x1 - manual_x0, manual_y1 - manual_y0,
                                    edgecolor="cyan", facecolor="none", linewidth=2, label="Manual Crop")
                ax[0].add_patch(rect)
                
            # Heatmap           
            heatmap_full = cv2.resize(  prob_map_small.astype(np.float32),
                                        (image_gray.shape[1], image_gray.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

            ax[1].imshow(image_gray, cmap="gray", vmin=0, vmax=255)
            im = ax[1].imshow(heatmap_full, cmap="jet", alpha=0.3, vmin=0, vmax=1)
            ax[1].set_title("Probability Map Overlay")
            plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Anomaly Probability")
            ax[1].axis("off")   
                        
            # Segments 
            ax[2].imshow(image_gray, cmap="gray", vmin=0, vmax=255)
                                               
            # Draw contour for each flood-filled region and box for each accepted crop
            for crop in batch_crop_rows:
                crop_y0, crop_x0, crop_y1, crop_x1 = crop["crop_y0"], crop["crop_x0"], crop["crop_y1"], crop["crop_x1"]
                if crop["label_manual"] == 1:
                    col = "lime"
                    label = "True Positive Crop"
                else:
                    col = "magenta"
                    label = "False Positive Crop"
                        
                rect = Rectangle((crop_x0, crop_y0), crop_x1 - crop_x0, crop_y1 - crop_y0,
                                edgecolor=col, facecolor="none", linewidth=2, label=label)
                ax[2].add_patch(rect)
                ax[2].set_title("Detected Regions and Crops")
            handles, labels = ax[2].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax[2].legend(by_label.values(), by_label.keys())
            ax[2].axis("off")
                
            plt.suptitle(f"Image: {image_name[13:23]}")
            plot_path = os.path.join(output_folder_debug_plots, f"{image_name}.png")
            if SAVE_DEBUG_PLOTS:
                plt.savefig(plot_path, bbox_inches="tight")
                
            if PLOT_DEBUG:
                plt.show()
                 
            plt.close('all')
            gc.collect()
                    
            # plot the accepted crops in separate subplots
            fig, axes = plt.subplots(1, len(accepted), figsize = (len(accepted)* 8, 8))
            if len(accepted) == 1:
                axes = [axes]
            for idx, cand in enumerate(accepted):
                crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
                crop_image = image_gray[crop_y0:crop_y1, crop_x0:crop_x1]
                axes[idx].imshow(crop_image, cmap="gray", vmin=0, vmax=255)
                axes[idx].set_title(f"Crop {idx+1}, peak={cand['peak_val']:.2f}, region_mean={cand['region_mean_intensity']:.2f}, region size={cand['region_size']} patches")
                axes[idx].axis("off")
            plt.suptitle(f"Extracted Crops from {image_name[13:23]} with species: {image_name[24:]}", fontsize=16)
            plt.show()
            
        # If the image has already been saved before, remove all entries for that image from the metadata CSV and re-save with the new crops (to avoid duplicates if we re-run the script)
        if os.path.exists(crop_metadata_csv):
            existing_metadata = pd.read_csv(crop_metadata_csv)
            if image_path in existing_metadata["OG_image_path"].values:
                # print(f"Image {image_name} already has entries in crop metadata CSV, removing old entries and re-saving with new crops to avoid duplicates")
                existing_metadata = existing_metadata[existing_metadata["OG_image_path"] != str(image_path)]
                existing_metadata.to_csv(crop_metadata_csv, index=False)
        
        # If the image alrady has crops saved, remove the old crops before saving the new ones (to avoid duplicates if we re-run the script)
        for crop_row in batch_crop_rows:
            crop_name = f"{crop_row['image_name']}_crop_{crop_row['crop_id']:03d}.png"
            crop_path = os.path.join(output_folder_crops, crop_name)
            if os.path.exists(crop_path):
                # print(f"Crop {crop_name} already exists, removing old crop before saving new one to avoid duplicates")
                os.remove(crop_path)
        
        # Now save the new crops and metadata for this batch of images
        for crop_image, crop_path in deferred_crops:
            cv2.imwrite(crop_path, crop_image)
        
        # Save the metadata rows for this batch of crops to the CSV (appending if the file already exists)
        save_crop_metadata_rows(crop_metadata_csv, batch_crop_rows)        
        
        # Update timing and print batch summary
        total_images_processed += len(images_batch)
        batch_process_time = time.perf_counter() - batch_start_time
        total_processing_time += batch_process_time
        print(f"Batch complete: {len(images_batch)} images | process: {batch_process_time:.2f}s | total images: {total_images_processed}/{len(images_path)} | crops: {crop_counter}")

        # Clean up batch
        del tiles_tensor, rows_tensor, cols_tensor, scores_array, preds_array, mu_array, recon_array, results_per_image
        del deferred_crops, batch_crop_rows
        del prob_map_small, maxima_small, candidates, accepted, crop_metadata
        
        images_batch = []
        image_paths_batch = []
        batch_load_time_accumulator = 0.0

if total_images_processed > 0:
    total_time = total_image_load_time + total_processing_time
    load_pct = 100*total_image_load_time/total_time
    process_pct = 100*total_processing_time/total_time
    print(f"\nOverall timing:")
    print(f"  Image loading: {total_image_load_time:.2f}s ({load_pct:.1f}% of total)")
    print(f"  Batch processing: {total_processing_time:.2f}s ({process_pct:.1f}% of total)")
    print(f"  Total elapsed: {total_time:.2f}s")
    
print(f"\n✓ Complete! Total crops extracted: {crop_counter}")

# %% Plot ROC curve for the extracted crops if labels are available (requires 'label' column in crop_metadata.csv)



predicted_crops_csv = os.path.join(output_folder_crops, "crop_metadata.csv")


predicted_crops_metadata = pd.read_csv(predicted_crops_csv)

segmentation_performance_summary = []

for image_path in images_path:

    image_name = Path(image_path).stem
    manual_crops = manual_crops_metadata[manual_crops_metadata["image_path"] == image_path]
    predicted_crops = predicted_crops_metadata[predicted_crops_metadata["OG_image_path"] == image_path]
    
    # Set variable that is threshold for calculating ROC curve based on the peak intensity of the region (or you could use mean intensity or another metric)
    min_overlap = 0.1 # minimum IoU overlap between predicted crop and manual crop to be considered a correct prediction (you can adjust this threshold based on how precise you want the crops to be)
    
    predicted_crops["prediction_correct"] = 0
    
    # determine labels based on overlap with manual crops (if a predicted crop overlaps with any manual crop, label it as 1, otherwise 0)
    for pred_idx, pred_row in predicted_crops.iterrows():
        pred_y0, pred_x0, pred_y1, pred_x1 = pred_row["crop_y0"], pred_row["crop_x0"], pred_row["crop_y1"], pred_row["crop_x1"]
        for _, manual_row in manual_crops.iterrows():
            manual_y0, manual_x0, manual_y1, manual_x1 = manual_row["bbox_y0"], manual_row["bbox_x0"], manual_row["bbox_y1"], manual_row["bbox_x1"]
            inter_y0 = max(pred_y0, manual_y0)
            inter_x0 = max(pred_x0, manual_x0)
            inter_y1 = min(pred_y1, manual_y1)
            inter_x1 = min(pred_x1, manual_x1)
            if inter_y1 > inter_y0 and inter_x1 > inter_x0:
                iou = (inter_y1 - inter_y0) * (inter_x1 - inter_x0) / ((pred_y1 - pred_y0) * (pred_x1 - pred_x0) + (manual_y1 - manual_y0) * (manual_x1 - manual_x0) - (inter_y1 - inter_y0) * (inter_x1 - inter_x0))
                if iou >= min_overlap:
                    label_manual = 1
                else:                    
                    label_manual = 0

        # append to ROC summary
        segmentation_performance_summary.append({
            "image_name": image_name,
            "predicted_crop_id": pred_row["crop_id"],
            "label_manual": label_manual,
            "peak_intensity": pred_row["peak_intensity"],
            "region_mean_intensity": pred_row["region_mean_intensity"],
            "region_size_pixels": pred_row["region_size_pixels"],
        })
  
df_summary = pd.DataFrame(segmentation_performance_summary)  
      
# print summary
print(f"\nSummary :")
print(f"Total predicted crops: {len(df_summary)}")
print(f"Correct predictions: {len(df_summary[df_summary['label_manual'] == 1])}")
print(f"Incorrect predictions: {len(df_summary) - len(df_summary[df_summary['label_manual'] == 1])}")

#%% Treshold analysis
# Calculate the ROC curve and precision recall
threshold_metric = "region_mean_intensity"  # you can change this to "region_mean_intensity" or another metric if you want to see how it performs as a predictor

fpr, tpr, roc_thresholds = roc_curve(df_summary["label_manual"], df_summary[threshold_metric])
roc_auc = auc(fpr, tpr)

precision, recall, pr_thresholds = precision_recall_curve(df_summary["label_manual"], df_summary[threshold_metric])
average_precision = average_precision_score(df_summary["label_manual"], df_summary[threshold_metric])

threshold_f05 = pr_thresholds[np.argmax((1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall + 1e-8))]  # F0.5-score optimal threshold
threshold_f1 = pr_thresholds[np.argmax(2 * (precision * recall) / (precision + recall + 1e-8))]  # F1-score optimal threshold
threshold_f2 = pr_thresholds[np.argmax((1 + 2**2) * (precision * recall) / (2**2 * precision + recall + 1e-8))]  # F2-score optimal threshold
threshold_f3 = pr_thresholds[np.argmax((1 + 3**2) * (precision * recall) / (3**2 * precision + recall + 1e-8))]  # F3-score optimal threshold

youden_index = tpr - fpr
threshold_youden = roc_thresholds[np.argmax(youden_index)]  # Youden's J statistic optimal threshold

threshold = threshold_youden  # you can choose which threshold to use based on the precision-recall tradeoff you want (e.g. F1 for balanced, F0.5 for more precision, F2 for more recall)
df_summary["label_predicted"] = (df_summary[threshold_metric] >= threshold).astype(int)
cm = confusion_matrix(df_summary["label_manual"], df_summary["label_predicted"], labels=[1, 0])

roc_thrsh_idx = np.argmin(np.abs(roc_thresholds - threshold))
pr_thrsh_idx = np.argmin(np.abs(pr_thresholds - threshold))

print(f"Average Precision: {average_precision:.4f}")
print(f"AUC: {roc_auc:.4f}")
   
# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.scatter(fpr[roc_thrsh_idx], tpr[roc_thrsh_idx], color="red", label=f"Chosen threshold = {threshold:.4f}", zorder=5)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve for Predicted Crops vs Manual Crops")
plt.legend(loc="lower right")
plt.show()
    
# Plot precision-recall curve
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color="blue", label=f"Precision-Recall curve (AP = {average_precision:.4f})")
plt.scatter(recall[pr_thrsh_idx], precision[pr_thrsh_idx], color="red", label=f"Chosen threshold = {threshold:.4f}", zorder=5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve for Predicted Crops vs Manual Crops")
plt.legend(loc="lower left")

# Plot confusion matrix of correct vs incorrect predictions
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Observation", "Empty"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix for Predicted Crops vs Manual Crops")
plt.show()
    
    