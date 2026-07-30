#%% Import cells
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import torch
import time
from matplotlib.patches import Rectangle
import gc

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

# matplotlib.use("TkAgg")  # Interactive backend so debug plots can open windows.
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from functions import find_local_maxima, flood_fill_region, get_manual_crops_from_labelme_json, Autoencoder, VariationalAutoencoder, OneClassScorer, TwoClassScorer, batch_run_inference, split_results_by_image, batch_extract_tiles_from_images, MahalanobisScorer, crops_overlap

#%% Activate GPU
print("="*40)
print("PyTorch Environment Report")
print("="*40)

print(f"PyTorch version:        {torch.__version__}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"GPU available:          {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_index = 0
    print(f"GPU name:               {torch.cuda.get_device_name(gpu_index)}")
    print(f"Compute capability:     {torch.cuda.get_device_capability(gpu_index)}")
    print(f"Current device:         {torch.cuda.current_device()}")
    print(f"Total memory (GB):      {torch.cuda.get_device_properties(gpu_index).total_memory / 1e9:.2f}")

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print("="*40)
print(f"Using device:      {device}")

#%% Define paths and parameters
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data"
monitoring_effort = "Kristineberg_250915"  # for titles and saved model names, e.g. "kristineberg_251128"
# images_folder = os.path.join(ROOT_DIR_R, monitoring_effort)
# images_folder = os.path.join(ROOT_DIR_R, monitoring_effort)
images_folder = os.path.join(ROOT_DIR_C, monitoring_effort, "test", "OG_images")
output_folder = Path(os.path.join(ROOT_DIR_C, monitoring_effort, "test", "test_segmentation"))
output_folder.mkdir(parents=True, exist_ok=True)

tile_grid_size = 16
latent_dim = 64
image_size = 128
encoder = "AE"  # options: "VAE", "AE"
scorer_mode = "binary"  # options: "one_class", "binary", "mahalanobis"
offsets_norm = [0, 0.2, 0.4, 0.6, 0.8]  # relative offsets for tile grid (e.g. 0.2 means shift grid by 20% of tile size)
best_threshold = 0.17

IMAGE_SIZE_PX = 4512  # All images are square 4512x4512
IMAGE_W = 91 # width of the image in mm
PXL_TO_MM = IMAGE_W / IMAGE_SIZE_PX  # conversion factor from pixels to mm
PXL_TO_MM2 = PXL_TO_MM ** 2  # conversion factor from pixels^2 to mm^2
tile_size = IMAGE_SIZE_PX // tile_grid_size

MANUAL_ANNOTATIONS_AVAILABLE = True
PLOT_DEBUG = True
SAVE_DEBUG_PLOTS = False
SAVE_CROPS = True

# set parameters for segmentation
min_region_size_patches = 3
crop_padding_pixels = 0

offsets = [int(norm * tile_size) for norm in offsets_norm]

# Load all images    
# images_path = sorted(str(p) for p in Path(images_folder).glob("*.png"))

# Load images in crop_metadata.csv to ensure we only test on images that have manual crops (and avoid accidentally loading any non-image files in the folder)
if MANUAL_ANNOTATIONS_AVAILABLE:
    manual_crops_csv = os.path.join(images_folder[:-10], f"tiles{tile_grid_size}_offsets{len(offsets_norm)}_labelme", "crops_metadata.csv")
    manual_crops_metadata = pd.read_csv(manual_crops_csv)
    images_in_metadata = manual_crops_metadata["image_path"].unique()
    images_path = [p for p in images_in_metadata]
else:
    print("loading all images...")
    images_path = sorted(str(p) for p in Path(images_folder).glob("*.png"))
    print("loading all images... Done!", end='\r')

N_TEST_IMAGES = None
if N_TEST_IMAGES is not None and len(images_path) > N_TEST_IMAGES:
    images_path = images_path[:N_TEST_IMAGES]

### Load VAE model
model_name = f"../models/{encoder}/{monitoring_effort}_{encoder}_model{tile_grid_size}_l{latent_dim}_img{image_size}.pth"
checkpoint_encoder = torch.load(model_name, map_location=device, weights_only=False)

latent_dims = checkpoint_encoder.get("latent_dim")
image_size = checkpoint_encoder.get("image_size")
hidden_channels = checkpoint_encoder.get("hidden_channels")
grid_size = checkpoint_encoder.get("grid_size")

if encoder == "VAE":
    model = VariationalAutoencoder(latent_dims=latent_dims, image_size=image_size, hidden_channels=hidden_channels, grid_size=grid_size)
elif encoder == "AE":
    model = Autoencoder(latent_dims=latent_dims, image_size=image_size, hidden_channels=hidden_channels, grid_size=grid_size)
else: 
    raise ValueError(f"Invalid encoder type: {encoder}. Must be 'VAE' or 'AE'.")

model.load_state_dict(checkpoint_encoder["model_state_dict"])
model.to(device)
model.eval()

print(f"Loaded {encoder} model with latent_dim={latent_dims}, grid_size={grid_size}")

### Load scorer model and threshold
if scorer_mode == "one_class":
    scorer_name = model_name.replace(f"{encoder}/", "NN/OneClass/").replace(f"{encoder}_model", "scorer_oneclass_model")
    checkpoint_scorer = torch.load(scorer_name, map_location=device, weights_only=False)
    
    scorer = OneClassScorer(
        latent_dim=latent_dim,
        hidden_dim=256,
        emb_dim=16,
        grid_size=tile_grid_size,
    )
        
    scorer.load_state_dict(checkpoint_scorer["model_state_dict"])
    scorer_threshold = float(checkpoint_scorer.get("threshold", 0.5))
    scorer.to(device)
    scorer.eval()
    
elif scorer_mode == "binary":
    scorer_name = model_name.replace(f"{encoder}/", "NN/TwoClass/").replace(f"{encoder}_model", "scorer_binary_model")
    checkpoint_scorer = torch.load(scorer_name, map_location=device, weights_only=False)
    
    scorer = TwoClassScorer(
        latent_dim=latent_dim,
        hidden_dim=256,
        dropout=0.3,
        grid_size=tile_grid_size,
   )

    scorer.load_state_dict(checkpoint_scorer["model_state_dict"])
    scorer_threshold = float(checkpoint_scorer.get("threshold", 0.5))
    scorer.to(device)
    scorer.eval()

elif scorer_mode == "mahalanobis":
    scorer_name = model_name.replace(f"{encoder}/", "NN/Mahalanobis/").replace(f"{encoder}_model", "scorer_mahalanobis_model")
    checkpoint_scorer = torch.load(scorer_name, map_location=device, weights_only=False)
    mean_vec = checkpoint_scorer["mean_vec"].to(device)
    cov_inv = checkpoint_scorer["cov_inv"].to(device)
    dist_calib = checkpoint_scorer["dist_calib"].to(device)
    recon_calib = checkpoint_scorer["recon_calib"].to(device)
    scorer_threshold = float(checkpoint_scorer.get("threshold", 0.5))

    scorer = MahalanobisScorer(
            latent_dim=latent_dim,
            grid_size=tile_grid_size,
            mean_vec=mean_vec,
            cov_inv=cov_inv,
            dist_calib=dist_calib,
            recon_calib=recon_calib,
        ).to(device)

 
print(f"✓ Loaded scorer checkpoint: {scorer_name}")
print(f"✓ Loaded scorer threshold: {scorer_threshold:.4f}")   

peak_threshold = scorer_threshold # minimum score for a tile to be considered a peak candidate (you can adjust this threshold based on how many crops you want to extract and how strong the scores are)
secondary_threshold = peak_threshold*0.99
mean_threshold = scorer_threshold*0.8 # minimum mean score of the region for it to be considered a valid crop (you can adjust this threshold based on how strong the scores are in the regions you want to extract)

# Thresholds for overlap between manual and predicted area 
min_manual_covered_pred_larger = 0.70   # predicted crop is larger: must cover 70% of manual crop
min_manual_covered_pred_smaller = 0.90  # predicted crop is smaller: must cover 90% of manual crop (we want to be stricter about small predicted crops since they could easily fit inside a manual crop without really covering the important parts of it, whereas if the predicted crop is larger we are more lenient since it at least covers the entire manual crop even if it also includes a lot of extra area)
    
### Set batch sizes for processing (adjust based on GPU memory and image size)
batch_size_images = 1  # Process 32 images per batch

batch_size = 8912  # Process 8192 tiles per batch (adjust based on GPU memory)
vae_image_size = 128

grid_size_small = tile_grid_size * len(offsets)
upscale_factor = IMAGE_SIZE_PX/grid_size_small 

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
        scores_list = []
        mu_list = []
        recon_list = []

        with torch.inference_mode():
            for start_idx in range(0, len(tiles_tensor), batch_size):
                end_idx = min(start_idx + batch_size, len(tiles_tensor))

                tiles_batch = tiles_tensor[start_idx:end_idx].to(device, non_blocking=True)
                rows_batch = rows_tensor[start_idx:end_idx].to(device, non_blocking=True)
                cols_batch = cols_tensor[start_idx:end_idx].to(device, non_blocking=True)
                    
                torch.cuda.synchronize()  # Ensure GPU is ready
                    
                if encoder=='VAE':
                    mu, _ = model.encode(tiles_batch, row=rows_batch, col=cols_batch)
                elif encoder == 'AE':
                    mu = model.encode(tiles_batch, row=rows_batch, col=cols_batch)
                else:
                    raise ValueError(f"Invalid encoder type: {encoder}. Must be 'VAE' or 'AE'.")
                
                x_hat = model.decode(mu)
                recon_error = ((tiles_batch - x_hat) ** 2).flatten(1).mean(dim=1)
                
                scores = scorer.predict_batch(mu, recon_error, rows_batch, cols_batch)
                torch.cuda.synchronize()  # Ensure all ops complete
                    
                mu_list.append(mu.detach().cpu().numpy())
                recon_list.append(recon_error.detach().cpu().numpy())
                scores_list.append(scores.detach().cpu().numpy())
                    
                # free temporary GPU tensors and clear cache (guarded)
                try:
                    del tiles_batch, rows_batch, cols_batch, mu, x_hat, recon_error, scores
                except NameError:
                    pass
                torch.cuda.empty_cache()

        scores_array = np.concatenate(scores_list, axis=0)
        mu_array = np.concatenate(mu_list, axis=0)
        recon_array = np.concatenate(recon_list, axis=0)
        
        # Seperate batch results back into individual images and aggregate tile scores into image-level probability maps
        results_per_image = split_results_by_image(scores_array, rows_list, cols_list, tile_grid_size, offsets, image_tile_counts)         

        batch_crop_metadata = []
                
        # Process each image's results
        for batch_img_idx, (image_gray, image_path) in enumerate(zip(images_batch, image_paths_batch)):
            image_name = image_path.stem

            predicted_crop_metadata = []
            manual_crop_metadata = []
            accepted = []
            candidates = []
            image_crop_metadata = []

            result = results_per_image[batch_img_idx]
            if result is None:
                # print("no tiles")
                continue
                    
            prob_map_small = result["prob_map_small"] # type: ignore  
            
            # Find local maxima in the small-grid probability map to identify candidate regions
            maxima_small = find_local_maxima(prob_map_small, peak_threshold/1.5)

            # Convert each small-grid flood-filled region into a pixel crop bbox
            candidates = []
            for peak_y_small, peak_x_small, peak_val in maxima_small:
                region_small = flood_fill_region(prob_map_small, peak_y_small, peak_x_small, secondary_threshold)

                if len(region_small) < min_region_size_patches ** 2:
                    continue

                region_small_points = np.asarray(list(region_small), dtype=np.int32)
                region_mean = prob_map_small[region_small_points[:, 0], region_small_points[:, 1]].mean() if len(region_small_points) > 0 else 0.0
                
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
                                
                        if iou > 0.3:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    cand["crop_area"] = crop_area
                    accepted.append(cand)
                          
            # Save crops and metadata
            predicted_crop_metadata = []
            
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

                predicted_crop_metadata.append({  "OG_image_path": image_path,
                                    "image_name": image_name,
                                    "crop_id": crop_idx,
                                    "crop_width": crop_w,
                                    "crop_height": crop_h,
                                    "crop_y0": crop_y0,
                                    "crop_x0": crop_x0,
                                    "crop_y1": crop_y1,
                                    "crop_x1": crop_x1,
                                    "region_size_pixels": int(len(region_points)*upscale_factor**2),  # approximate region size in pixels by multiplying number of small-grid points by area of each small grid cell
                                    "region_size_mm2": int(len(region_points)*upscale_factor**2 * PXL_TO_MM2),  # approximate region size in mm^2
                                    "peak_intensity": peak_val,
                                    "region_mean_intensity": region_mean,
                                    "label_manual": False,  # to be updated later based on overlap with manual crops
                                    "label_predicted": np.nan,  # to be updated later based on overlap with predicted crops (for now we label all predicted crops as 1 since they are all considered positive predictions until we compare to manual crops)
                                    "label": np.nan,  # to be updated later with TP/FP/FN/TN after comparing with manual crops
                                    "source": "predicted",
                                    "matched_by_pred": None,
                                    })

                crop_counter += 1
                
            # --- Load manual crops for this image ---
            manual_crop_metadata = []

            if MANUAL_ANNOTATIONS_AVAILABLE:

                mask_path = image_path.parent.parent / "manual_masks" / Path(image_name + ".json")
                manual_crop_metadata = get_manual_crops_from_labelme_json(
                    json_path=mask_path,
                    image_shape=image_gray.shape,
                    image_path=image_path,
                    image_name=image_name,
                    PXL_TO_MM2=PXL_TO_MM2
                )

                    
                for i, crop in enumerate(manual_crop_metadata):
                    crop["crop_id"] = len(predicted_crop_metadata) + i
                    crop["source"] = crop.get("source", "manual")
                    crop["matched_by_pred"] = False
                    crop["label_manual"] = True
                    crop["label_predicted"] = None

                # initialize predicted crops
                for crop in predicted_crop_metadata:
                    crop["matched_by_pred"] = False
                    crop["label_manual"] = False
                    crop["label_predicted"] = None

                # Match predicted and manual crops 
                if MANUAL_ANNOTATIONS_AVAILABLE and predicted_crop_metadata and manual_crop_metadata:
                    for pred_crop in predicted_crop_metadata:
                        for manual_crop in manual_crop_metadata:
                            if crops_overlap(pred_crop, manual_crop,min_manual_covered_pred_larger,min_manual_covered_pred_smaller):
                                pred_crop["matched_by_pred"] = True
                                pred_crop["label_manual"] = True
                                manual_crop["matched_by_pred"] = True

                #  Combine metadata 
                image_crop_metadata = predicted_crop_metadata + manual_crop_metadata

                # Final labels 
                if MANUAL_ANNOTATIONS_AVAILABLE:
                    for crop in image_crop_metadata:
                        if crop["source"] == "predicted":
                            crop["label_predicted"] = crop["peak_intensity"] >= peak_threshold
                            if crop["label_manual"] and crop["label_predicted"]:
                                crop["label"] = "TP"
                            elif crop["label_manual"] and not crop["label_predicted"]:
                                crop["label"] = "FN"
                            elif not crop["label_manual"] and crop["label_predicted"]:
                                crop["label"] = "FP"
                            else:
                                crop["label"] = "TN"

                        elif crop["source"] == "manual":
                            crop["label_predicted"] = None
                            crop["label"] = "TP" if crop["matched_by_pred"] else "FN"                    
            
            
            # Make subplot with all extracted crops
            if len(image_crop_metadata) > 0 and (PLOT_DEBUG or SAVE_DEBUG_PLOTS):
                ### Plot subplot with heatmaps and flood filled regions with red contour and final crop with green box for debugging
                fig, ax = plt.subplots(1,2, figsize=(16, 8))
                    
                # crop overlay
                ax[0].imshow(image_gray, cmap="gray", vmin=0, vmax=255)
                for crop in image_crop_metadata:   
                    crop_y0, crop_x0, crop_y1, crop_x1 = crop["crop_y0"], crop["crop_x0"], crop["crop_y1"], crop["crop_x1"]
                    label_manual = crop["label_manual"]
                    label_predicted = crop["label_predicted"]
                    source = crop["source"]
                    
                    if source == "predicted" and MANUAL_ANNOTATIONS_AVAILABLE:
                        if crop['label'] == "TP":
                            col = "green"
                            linestyle = "solid"
                            alpha = 1
                        elif crop['label'] == "FP":
                            col = "red"
                            linestyle = "solid"
                            alpha = 1
                        elif crop['label'] == "FN":
                            col = "red"
                            linestyle = "dashed"
                            alpha = 1
                            crop['label'] = "FN"
                        elif crop['label'] == "TN":
                            col = "green"
                            linestyle = "dashed"
                            alpha = 1
                                        
                    rect = Rectangle((crop_x0, crop_y0), crop_x1 - crop_x0, crop_y1 - crop_y0,
                                        edgecolor=col, facecolor="none", linewidth=2, alpha = alpha, linestyle=linestyle, label=crop['label'])
                    
                    ax[0].add_patch(rect)
                    
                ax[0].legend(handles=[Rectangle((0,0),1,1, edgecolor="green", facecolor="none", linestyle="solid", label="TP"),
                                     Rectangle((0,0),1,1, edgecolor="red", facecolor="none", linestyle="solid", label="FP"),
                                     Rectangle((0,0),1,1, edgecolor="red", facecolor="none", linestyle="dashed", label="FN"),
                                     Rectangle((0,0),1,1, edgecolor="green", facecolor="none", linestyle="dashed", label="TN"),
                                     Rectangle((0,0),1,1, edgecolor="cyan", facecolor="none", linestyle="solid", label="Manual crop")])
                
                
                # Heatmap                      
                cnorm = colors.TwoSlopeNorm(vmin=prob_map_small.min(), vcenter=peak_threshold, vmax=prob_map_small.max())    
                heatmap_full = cv2.resize(prob_map_small.astype(np.float32), (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

                ax[1].imshow(image_gray, cmap="gray", vmin=0, vmax=255)
                im = ax[1].imshow(heatmap_full, cmap="jet", alpha=0.3, norm = cnorm)
                ax[1].set_title("Probability Map Overlay")
                plt.colorbar(im,ticks=[0, peak_threshold, 1], ax=ax[1], fraction=0.046, pad=0.04, label="Anomaly Probability")
                ax[1].axis("off")   
                            
                plt.suptitle(f"Image: {image_name[13:23]}")
                plot_path = os.path.join(output_folder_debug_plots, f"{image_name}.png")
                if SAVE_DEBUG_PLOTS:
                    plt.savefig(plot_path, bbox_inches="tight")
                    
                if PLOT_DEBUG:
                    plt.show()
                    
                plt.close('all')
                gc.collect()
                        
                # plot the true predictions
                TPs = [crop for crop in image_crop_metadata if crop['label_manual'] == True and crop['label_predicted'] == True and crop['source'] == "predicted"]
                FPs = [crop for crop in image_crop_metadata if crop['label_manual'] == False and crop['label_predicted'] == True and crop['source'] == "predicted"]
                
                Ps = TPs + FPs
                if MANUAL_ANNOTATIONS_AVAILABLE and (len(Ps) > 0):
                    fig, axes = plt.subplots(1, len(Ps), figsize = (len(Ps)* 8, 8))
                    if len(Ps) == 1:
                        axes = [axes]
                    for idx, crop in enumerate(Ps):
                        crop_y0, crop_x0, crop_y1, crop_x1 = crop["crop_y0"], crop["crop_x0"], crop["crop_y1"], crop["crop_x1"]
                        crop_image = image_gray[crop_y0:crop_y1, crop_x0:crop_x1]
                        axes[idx].imshow(crop_image, cmap="gray", vmin=0, vmax=255)
                        axes[idx].set_title(f"Crop {idx+1}, peak={crop['peak_intensity']:.2f}, region_mean={crop['region_mean_intensity']:.2f}, label = {crop['label']}")
                        axes[idx].axis("off")
                    plt.suptitle(f"Extracted Crops from {image_name[13:23]} with species: {image_name[24:]}", fontsize=16)
                    plt.show()

            # save the new crops for this image
            batch_crop_metadata.extend(image_crop_metadata)
            
            if SAVE_CROPS:
                # Create four folders, TN, TP, FN, FP, and save crops in corresponding folders based on their labels (for easier review of results)
                if MANUAL_ANNOTATIONS_AVAILABLE:
                    for folder in ["TN", "TP", "FN", "FP"]:
                        folder_path = os.path.join(output_folder_crops, folder)
                        os.makedirs(folder_path, exist_ok=True)                
                else:
                    folder_path = os.path.join(output_folder_crops, "predicted")
                    os.makedirs(folder_path, exist_ok=True)
                    
                for crop in image_crop_metadata:
                    if MANUAL_ANNOTATIONS_AVAILABLE:
                        crop_name = f"{crop['label']}\\{crop['image_name']}_crop_{crop['crop_id']:03d}_area_{crop['region_size_mm2']}.png"
                    else: 
                        crop_name = f"predicted\\{crop['image_name']}_crop_{crop['crop_id']:03d}_area_{crop['region_size_mm2']}.png"
                    
                    crop_path = os.path.join(output_folder_crops, crop_name)
                    crop['crop_path'] = crop_path  # add crop path to metadata for later reference
                    
                    crop_y0, crop_x0, crop_y1, crop_x1 = crop["crop_y0"], crop["crop_x0"], crop["crop_y1"], crop["crop_x1"]
                    crop_image = image_gray[crop_y0:crop_y1, crop_x0:crop_x1]
                    
                    cv2.imwrite(crop_path, crop_image)
                       
        # Save batch metadata to CSV (appending if file already exists)
        if len(batch_crop_metadata) > 0:
            batch_crop_metadata_df = pd.DataFrame(batch_crop_metadata)

            if os.path.exists(crop_metadata_csv):
                existing_metadata = pd.read_csv(crop_metadata_csv, on_bad_lines="skip")

                # Remove previous rows for images in this batch
                batch_image_paths = batch_crop_metadata_df["OG_image_path"].astype(str).unique()
                existing_metadata = existing_metadata[
                    ~existing_metadata["OG_image_path"].astype(str).isin(batch_image_paths)
                ]

                # Align columns safely
                all_columns = sorted(set(existing_metadata.columns) | set(batch_crop_metadata_df.columns))
                existing_metadata = existing_metadata.reindex(columns=all_columns)
                batch_crop_metadata_df = batch_crop_metadata_df.reindex(columns=all_columns)

                combined = pd.concat([existing_metadata, batch_crop_metadata_df], ignore_index=True)
            else:
                combined = batch_crop_metadata_df

            combined.to_csv(crop_metadata_csv, index=False)
                
        # Update timing and print batch summary
        total_images_processed += len(images_batch)
        batch_process_time = time.perf_counter() - batch_start_time
        total_processing_time += batch_process_time
        predicted_time_remaining = (len(images_path) - total_images_processed) * (total_processing_time / total_images_processed) if total_images_processed > 0 else 0
        print(f"Batch complete: {len(images_batch)} images | process: {batch_process_time:.2f}s | total images: {total_images_processed}/{len(images_path)} | crops: {crop_counter} | est. time remaining: {predicted_time_remaining/60:.2f} min")

        # Clean up batch
        del tiles_tensor, rows_tensor, cols_tensor, scores_array, mu_array, recon_array, results_per_image
        del batch_crop_metadata
        
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

# Load metadata from CSV
predicted_crops_csv = os.path.join(output_folder_crops, "crop_metadata.csv")
df_predictions = pd.read_csv(predicted_crops_csv)
segmentation_performance_summary = []

# Calculate the ROC curve and precision recall
threshold_metric = "peak_intensity"  # you can change this to "region_mean_intensity" or another metric if you want to see how it performs as a predictor

fpr, tpr, roc_thresholds = roc_curve(df_predictions["label_manual"], df_predictions[threshold_metric])
roc_auc = auc(fpr, tpr)

precision, recall, pr_thresholds = precision_recall_curve(df_predictions["label_manual"], df_predictions[threshold_metric])
average_precision = average_precision_score(df_predictions["label_manual"], df_predictions[threshold_metric])

threshold_f05 = pr_thresholds[np.argmax((1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall + 1e-8))]  # F0.5-score optimal threshold
threshold_f1 = pr_thresholds[np.argmax(2 * (precision * recall) / (precision + recall + 1e-8))]  # F1-score optimal threshold
threshold_f2 = pr_thresholds[np.argmax((1 + 2**2) * (precision * recall) / (2**2 * precision + recall + 1e-8))]  # F2-score optimal threshold
threshold_f3 = pr_thresholds[np.argmax((1 + 3**2) * (precision * recall) / (3**2 * precision + recall + 1e-8))]  # F3-score optimal threshold

youden_index = tpr - fpr
threshold_youden = roc_thresholds[np.argmax(youden_index)]  # Youden's J statistic optimal threshold

threshold = threshold_f2  # you can choose which threshold to use based on the precision-recall tradeoff you want (e.g. F1 for balanced, F0.5 for more precision, F2 for more recall)
df_predictions["label_predicted"] = (df_predictions[threshold_metric] >= threshold).astype(int)
cm = confusion_matrix(df_predictions["label_manual"], df_predictions["label_predicted"], labels=[1,0])

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
plt.grid()
plt.xlabel("False Positive Rate", fontweight="bold", fontsize = 14)
plt.ylabel("True Positive Rate", fontweight="bold", fontsize = 14)
plt.title(f"ROC Curve for Predicted Crops vs Manual Crops", fontweight="bold", fontsize=18)
plt.legend(loc="lower right", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
    
# Plot precision-recall curve
plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color="blue", label=f"Precision-Recall curve (AP = {average_precision:.4f})")
plt.scatter(recall[pr_thrsh_idx], precision[pr_thrsh_idx], color="red", label=f"Chosen threshold = {threshold:.4f}", zorder=5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve for Predicted Crops vs Manual Crops")
plt.legend(loc="lower left")

# Plot confusion matrix of correct vs incorrect predictions
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Obs", "Empty"])
disp.plot(cmap="Blues", values_format="d")
# increase size of numbers in confusion matrix
for text in plt.gca().texts:
    text.set_fontsize(14)
plt.gca().images[-1].colorbar.remove()
plt.title(f"Confusion Matrix for Predicted Crops vs Manual Crops")

plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12, rotation=90)
plt.xlabel("Predicted Label", fontweight="bold", fontsize=14)
plt.ylabel("True Label", fontweight="bold", fontsize=14)
plt.title("")
plt.show()    

#%% Plot False positives
FPs = df_predictions[(df_predictions["label_manual"] == 0) & (df_predictions["label_predicted"] == 1)]
FNs = df_predictions[(df_predictions["label_manual"] == 1) & (df_predictions["label_predicted"] == 0)]
TPs = df_predictions[(df_predictions["label_manual"] == 1) & (df_predictions["label_predicted"] == 1)]
TNs = df_predictions[(df_predictions["label_manual"] == 0) & (df_predictions["label_predicted"] == 0)]

max_to_show = 10
max_per_row = 5

if len(FNs) > 0:
    FNs = FNs.head(max_to_show)
    rows = (len(FNs) - 1) // max_per_row + 1
    cols = min(len(FNs), max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))
    axes = axes.flatten() if len(FNs) > 1 else [axes]
    for ax, crop in zip(axes, FNs.itertuples()):
        crop_image = cv2.imread(crop.crop_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(crop_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"PI: {crop.peak_intensity:.2f}, RMI: {crop.region_mean_intensity:.2f}")
        ax.axis("off")
    plt.suptitle(f"False Negative Crops (Predicted negative but labeled positive) with {threshold_metric} < {threshold:.4f}", fontsize=16)
    plt.show()

if len(TPs) > 0:
    TPs = TPs.head(max_to_show)
    rows = (len(TPs) - 1) // max_per_row + 1
    cols = min(len(TPs), max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))
    axes = axes.flatten() if len(TPs) > 1 else [axes]
    for ax, crop in zip(axes, TPs.itertuples()):
        crop_image = cv2.imread(crop.crop_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(crop_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"PI: {crop.peak_intensity:.2f}, RMI: {crop.region_mean_intensity:.2f}")
        ax.axis("off")
    plt.suptitle(f"True Positive Crops (Predicted positive and labeled positive) with {threshold_metric} >= {threshold:.4f}", fontsize=16)
    plt.show()

if len(FPs) > 0:
    FPs = FPs.head(max_to_show)
    rows = (len(FPs) - 1) // max_per_row + 1
    cols = min(len(FPs), max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))
    axes = axes.flatten() if len(FPs) > 1 else [axes]
    for ax, crop in zip(axes, FPs.itertuples()):
        crop_image = cv2.imread(crop.crop_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(crop_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"PI: {crop.peak_intensity:.2f}, RMI: {crop.region_mean_intensity:.2f}")
        ax.axis("off")
    plt.suptitle(f"False Positive Crops (Predicted positive but labeled negative) with {threshold_metric} >= {threshold:.4f}", fontsize=16)
    plt.show()

if len(TNs) > 0:
    TNs = TNs.head(max_to_show)
    rows = (len(TNs) - 1) // max_per_row + 1
    cols = min(len(TNs), max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))
    axes = axes.flatten() if len(TNs) > 1 else [axes]
    for ax, crop in zip(axes, TNs.itertuples()):
        crop_image = cv2.imread(crop.crop_path, cv2.IMREAD_GRAYSCALE)
        ax.imshow(crop_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"PI: {crop.peak_intensity:.2f}, RMI: {crop.region_mean_intensity:.2f}")
        ax.axis("off")
    plt.suptitle(f"True Negative Crops (Predicted negative and labeled negative) with {threshold_metric} < {threshold:.4f}", fontsize=16)
    plt.show()


# %%
