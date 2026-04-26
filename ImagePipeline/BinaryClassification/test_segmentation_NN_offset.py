#%% Import packages and set up environment
### PACKAGES ###
import os

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
import cv2

from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import matplotlib as mpl

from functions import VariationalAutoencoder, ObservationScorer, test_image_pipeline, extract_tiles_with_offset, extract_tiles_with_offset, find_local_maxima, flood_fill_region

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
# ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
# monitoring_effort = "Kristineberg_251128" 
monitoring_effort = "Kristineberg_260424" 

test_images_folder = os.path.join(ROOT_DIR_C, monitoring_effort, "val", "OG_images")
tile_grid_size = 16 # how many tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = 4512 // tile_grid_size # size of the tiles that the original image is split into
latent_dim = 32 # how many tiles along one side 
model_name = f"models/{monitoring_effort}_vae_model{tile_grid_size}_l{latent_dim}.pth"

# Autoencoder parameters
batch_size = 2048
image_size = 128
hidden_channels = 32

# Other parameters
N_test = None # how many test images to process (set to None to process all)
SAVE_PREDICTION_IMAGES = False

# Offset grid parameters
offsets = [0, tile_size // 5, 2 * tile_size // 5, 3 * tile_size // 5, 4 * tile_size // 5]  # 0, 1/5, 2/5, 3/5, 4/5 offsets

# NN Scorer mode: 'binary' or 'one_class'
# Change this parameter to compare performance between modes
scorer_mode_preference = 'auto'  # 'auto' tries binary first, then one_class

print(f"\n{'='*70}")
print("Loading Neural Network Scorer")
print(f"{'='*70}")
print(f"Mode preference: {scorer_mode_preference}")

# Try to load scorer based on mode preference
scorer = None
scorer_mode_actual = None
scorer_threshold = 0.5

if scorer_mode_preference in ['auto', 'binary']:
    # Try BINARY mode first
    scorer_binary_name = model_name.replace("vae_model", "scorer_binary_model").replace(".pth", ".pth")
    try:
        checkpoint = torch.load(scorer_binary_name, map_location=device, weights_only=False)
        scorer_mode_actual = checkpoint.get('mode', 'binary')
        scorer = ObservationScorer(latent_dim=latent_dim, hidden_dim=256, dropout=0.3, 
                                   grid_size=tile_grid_size, mode=scorer_mode_actual)
        scorer.load_state_dict(checkpoint['model_state_dict'])
        scorer_threshold = checkpoint.get('threshold', 0.5)
        scorer.to(device)
        scorer.eval()
        print(f"✓ Loaded BINARY NN scorer")
        print(f"  Model: {scorer_binary_name}")
        print(f"  Threshold: {scorer_threshold:.4f}")
    except FileNotFoundError:
        pass

if scorer is None and scorer_mode_preference in ['auto', 'one_class']:
    # Try ONE-CLASS mode
    scorer_oneclass_name = model_name.replace("vae_model", "scorer_oneclass_model").replace(".pth", ".pth")
    try:
        checkpoint = torch.load(scorer_oneclass_name, map_location=device, weights_only=False)
        scorer_mode_actual = checkpoint.get('mode', 'one_class')
        scorer = ObservationScorer(latent_dim=latent_dim, hidden_dim=256, dropout=0.3, 
                                   grid_size=tile_grid_size, mode=scorer_mode_actual)
        scorer.load_state_dict(checkpoint['model_state_dict'])
        scorer_threshold = checkpoint.get('threshold', 0.5)
        scorer.to(device)
        scorer.eval()
        print(f"✓ Loaded ONE-CLASS NN scorer")
        print(f"  Model: {scorer_oneclass_name}")
        print(f"  Threshold: {scorer_threshold:.4f}")
    except FileNotFoundError:
        pass

if scorer is None:
    print(f"✗ ERROR: No NN scorer models found!")
    print(f"  Please train a scorer first:")
    print(f"    - python train_scorer_oneclass.py   (one-class mode)")
    print(f"    - python train_scorer_network.py    (binary mode)")
    exit(1)

print(f"Using {scorer_mode_actual.upper()} mode for inference")
    
test_images_path = list(Path(test_images_folder).glob("*.png"))

print(f"Found {len(test_images_path)} test images in: {test_images_folder}")
if N_test is not None and len(test_images_path) > N_test:
    rand_idxs = np.random.choice(len(test_images_path), size=N_test, replace=False)
    test_images_path = [test_images_path[i] for i in rand_idxs]
    print(f"Showing a random subset of {N_test} test images for evaluation:")

print(f"loading VAE model from: {model_name}")


#%% Load model
checkpoint = torch.load(model_name, map_location=device, weights_only=False)

latent_dims = checkpoint.get("latent_dim")

if latent_dims!= latent_dim:
    print(f"⚠ Warning: Loaded model latent_dims={latent_dims} does not match expected latent_dim={latent_dim}. Using loaded value.")
    
image_size = checkpoint.get("image_size")
hidden_channels = checkpoint.get("hidden_channels")
grid_size = checkpoint.get("grid_size")
state_dict = checkpoint["model_state_dict"]

model = VariationalAutoencoder(latent_dims=latent_dims, image_size=image_size, hidden_channels=hidden_channels, grid_size=grid_size)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

#%% Load many images (folder) and process with offset grids
"""
The code below loads all test images from the specified folder and processes them using offset grids.
For each offset (0, 1/5, 2/5, 3/5, 4/5 of tile_size), tiles are extracted and processed through the VAE/NN pipeline.
Results are aggregated by averaging probabilities from overlapping regions.
"""

# Dictionary to store results for each offset
offset_results = {}

for offset_idx, offset in enumerate(offsets):
    print(f"\n{'='*60}")
    print(f"Processing offset {offset_idx + 1}/{len(offsets)}: offset = {offset} pixels")
    print(f"{'='*60}")
    
    ### 1. Load images, extract tiles with offset, and store in list
    tiles_list = []
    rows_list = []
    image_names = []
    cols_list = []

    for image_path in test_images_path:
        print(f"Processing test image: {image_path.stem}, ({test_images_path.index(image_path) + 1}/{len(test_images_path)})", end="\r")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        tiles = extract_tiles_with_offset(image, tile_size, tile_grid_size, offset_x=offset, offset_y=offset)
        
        for tile in tiles:
            tiles_list.append((tile[0], tile[1], tile[2], image_path.stem))  # Store tuple expected by collate_fn

    print(f"\nLoaded and split {len(test_images_path)} test images into {len(tiles_list)} tiles with offset {offset}.")

    ### 2. Create data loader for test tiles
    tile_loader = DataLoader(
        tiles_list,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_image_pipeline,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    print(f"Created tile dataloader with batch size {batch_size}. Total batches: {len(tile_loader)}")
    
    ### 3. Run the tile dataloader through the VAE and collect results
    
    mu_list = []
    image_names_list = []
    recon_list = []
    svm_scores_list = []
    path_label_list = []
    tile_rows = []
    tile_cols = []
    tile_list = []
    tile_tensor_list = []

    batch_counter = 1

    for batch in tile_loader:
        tiles, rows, cols, image_names = batch
        
        tiles = tiles.to(device, non_blocking=True)

        mu, _ = model.encode(tiles, row=rows, col=cols)
        
        mu = mu.cpu().detach().numpy()
                    
        # reconstruction error per tile (mean squared error over pixels)
        x_hat = model(tiles, row=rows, col=cols)
        recon_err = ((tiles - x_hat) ** 2).flatten(1).mean(dim=1)
        recon_list.append(recon_err.cpu().detach())
        
        mu_list.append(mu)
        tile_rows.append(rows.cpu().numpy())
        tile_cols.append(cols.cpu().numpy())
        image_names_list.append(image_names)

        tiles_numpy = tiles.cpu().numpy()
        tile_list.append(tiles_numpy)
        
        # Predict using NN scorer
        rows_numpy = rows.cpu().numpy()
        cols_numpy = cols.cpu().numpy()
        recon_err_np = recon_err.cpu().detach().numpy()

        mu_tensor = torch.from_numpy(mu).float().to(device)
        recon_err_tensor = torch.from_numpy(recon_err_np).float().to(device)
        rows_tensor = torch.from_numpy(rows_numpy).long().to(device)
        cols_tensor = torch.from_numpy(cols_numpy).long().to(device)
        
        svm_scores, label_svm_is_obs = scorer.predict_batch(mu_tensor, recon_err_tensor, rows_tensor, cols_tensor, 
                                                             threshold=scorer_threshold)
        svm_scores = svm_scores.cpu().numpy()
        label_svm_is_obs = label_svm_is_obs.cpu().numpy()

        svm_scores_list.append(svm_scores)
        label_list_svm.append(label_svm_is_obs)
        
        is_obs = label_svm_is_obs

        batch_counter += 1
        print(f"processed batch {batch_counter}/{len(tile_loader)} |Classification: obs={label_svm_is_obs.sum()} empty = {(~label_svm_is_obs).sum()}", end="\r")

    mu_array = np.concatenate(mu_list, axis=0)
    labels_array_svm = np.concatenate(label_list_svm, axis=0)
    labels_array_binary_svm = None
    image_names_array = np.concatenate(image_names_list, axis=0)
    tile_rows_array = np.concatenate(tile_rows, axis=0)
    tile_cols_array = np.concatenate(tile_cols, axis=0)
    recon_array = torch.cat(recon_list, dim=0).cpu().numpy()
    svm_scores_array = np.concatenate(svm_scores_list, axis=0)
    
    # Create dataframe with image names, tile numbers, and latent representations
    latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
    df_latent = pd.DataFrame(mu_array, columns=latent_cols)
    df_latent.insert(0, "image_name", image_names_array)
    df_latent.insert(1, "tile_row", tile_rows_array)
    df_latent.insert(2, "tile_col", tile_cols_array)
    df_latent.insert(3, "recon_error", recon_array)
    df_latent.insert(4, "label_predicted_nn", labels_array_svm)
    df_latent.insert(5, "nn_scores", svm_scores_array)
    
    offset_results[offset] = df_latent
    
    print(f"\nClass distribution (offset={offset}): obs={labels_array_svm.sum()}, empty={(~labels_array_svm).sum()}")

#%% ######### SEGMENTATION AND VISUALIZATION ##########

# Use NN scorer threshold
threshold_method = scorer_threshold
method_name = f"NN ({scorer_mode_actual.upper()})"

print(f"Using {method_name} with threshold: {threshold_method:.4f}")


#%% Aggregate results from all offsets at full resolution, then downsample to 48x48
print(f"\n{'='*60}")
print("Aggregating results from all offsets into 48x48 grid")
print(f"{'='*60}")

# Get unique image names
image_names_unique = offset_results[offsets[0]]["image_name"].unique()

# Grid dimensions: tile_grid_size * number_of_offsets
n_offsets = len(offsets)
grid_size_small = tile_grid_size * n_offsets  # 16 * 3 = 48

# Create probability maps for each image (at 48x48 resolution)
probability_maps = {}  # Maps image_name -> 48x48 probability array
coverage_maps = {}     # Maps image_name -> 48x48 coverage array
image_dimensions = {}  # Maps image_name -> (height, width) of original image
upscale_factors = {}   # Maps image_name -> upscale factor for that image

for image_name in image_names_unique:
    print(f"Aggregating results for image: {image_name}, index {image_names_unique.tolist().index(image_name) + 1}/{len(image_names_unique)}", end="\r")
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_h, image_w = image.shape[:2]
    
    # Store original image dimensions for later upscaling
    image_dimensions[image_name] = (image_h, image_w)
    
    # Aggregate at full resolution first
    prob_map_full = np.zeros((image_h, image_w), dtype=np.float32)
    coverage_map_full = np.zeros((image_h, image_w), dtype=np.float32)
    
    # Aggregate scores from all offsets
    for offset in offsets:
        df_offset = offset_results[offset]
        df_image = df_offset[df_offset["image_name"] == image_name]
        
        for df_tile in df_image.itertuples():
            y0 = offset + int(df_tile.tile_row * tile_size)
            x0 = offset + int(df_tile.tile_col * tile_size)
            y1 = min(y0 + tile_size, image_h)
            x1 = min(x0 + tile_size, image_w)
            
            tile_score = float(getattr(df_tile, score_method))
            
            # Add score to probability map
            prob_map_full[y0:y1, x0:x1] += tile_score
            # Track coverage
            coverage_map_full[y0:y1, x0:x1] += 1
    
    # Average the probabilities by coverage
    prob_map_full = np.divide(prob_map_full, coverage_map_full, where=coverage_map_full > 0, out=prob_map_full.copy())
    
    # Downsample to 48x48
    prob_map_small = cv2.resize(prob_map_full, (grid_size_small, grid_size_small), interpolation=cv2.INTER_AREA)
    coverage_map_small = cv2.resize(coverage_map_full, (grid_size_small, grid_size_small), interpolation=cv2.INTER_NEAREST)
    
    probability_maps[image_name] = prob_map_small
    coverage_maps[image_name] = coverage_map_small
    
    # Store upscale factor for this image (full resolution / 48x48)
    upscale_factors[image_name] = (image_h / grid_size_small, image_w / grid_size_small)

#%% Visualize results with blob contours and heatmaps

output_folder_heatmaps = Path(test_images_folder) / "heatmaps_offset" / "tiles{}".format(tile_grid_size)
output_folder_heatmaps.mkdir(parents=True, exist_ok=True)

output_folder_contours = Path(test_images_folder) / "contours_offset" / "tiles{}".format(tile_grid_size)
output_folder_contours.mkdir(parents=True, exist_ok=True)

# Normalize the observation metric for color mapping across all images
# Compute min/max on-the-fly to avoid memory issues with large images
print("Computing min/max for normalization...")
obs_min = float('inf')
obs_max = float('-inf')

for prob_map in probability_maps.values():
    obs_min = min(obs_min, prob_map.min())
    obs_max = max(obs_max, prob_map.max())

print(f"Score range: [{obs_min:.3f}, {obs_max:.3f}], Loaded threshold: {threshold_method:.3f}")

# Smart threshold handling: if threshold is outside score range, recalculate
if threshold_method < obs_min or threshold_method > obs_max:
    print(f"⚠ WARNING: Threshold {threshold_method:.3f} is outside score range [{obs_min:.3f}, {obs_max:.3f}]")
    print("  Using median of scores as threshold instead.")
    
    # Compute median across all probability maps
    all_scores = []
    for prob_map in probability_maps.values():
        all_scores.extend(prob_map.flatten())
    all_scores = np.array(all_scores)
    threshold_method = np.median(all_scores)
    print(f"  New threshold (median): {threshold_method:.3f}")
else:
    print(f"✓ Threshold {threshold_method:.3f} is within score range")


print(f"Final threshold (normalized): {threshold_method:.3f}")

#%% First: Show contour outlines overlaid on the original image (upscaled from 48x48)
print(f"\n{'='*60}")
print("Creating contour visualizations")
print(f"{'='*60}")

for image_name in image_names_unique:
    print(f"Creating contour outline for image: {image_name}", end="\r")
    
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]
    
    # Work in 48x48 space
    prob_map_small = probability_maps[image_name]
    mask_small = (prob_map_small >= threshold_method).astype(np.uint8) * 255
    
    # Upscale mask to original image size
    mask = cv2.resize(mask_small, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    
    # Find contours from the upscaled binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image_rgb)
    
    # Draw contours as green outlines
    for contour in contours:
        contour_xy = contour.squeeze().astype(int)
        if contour_xy.ndim == 1:  # Single point
            contour_xy = contour_xy.reshape(-1, 2)
        if len(contour_xy) > 2:  # Only draw if contour has more than 2 points
            ax.plot(contour_xy[:, 0], contour_xy[:, 1], color='lime', linewidth=2)
            # Close the contour
            ax.plot([contour_xy[-1, 0], contour_xy[0, 0]], [contour_xy[-1, 1], contour_xy[0, 1]], color='lime', linewidth=2)
    
    ax.axis("off")
    
    if SAVE_PREDICTION_IMAGES:
        save_path = output_folder_contours / f"{image_name}.png"
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


#%% Second: Create heatmap visualizations (from 48x48 grid, upscale for display)
print(f"\nCreating aggregated heatmap visualizations")

# Optional: Allow manual threshold override
MANUAL_THRESHOLD = None  # Set to a value (e.g., 0.5) to override the loaded threshold
if MANUAL_THRESHOLD is not None:
    print(f"⚠ Using manual threshold override: {MANUAL_THRESHOLD:.3f}")
    threshold_method = MANUAL_THRESHOLD

threshold_heatmap = (threshold_method - obs_min) / (obs_max - obs_min)

cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"])
norm = mpl.colors.TwoSlopeNorm(vmin=threshold_heatmap - 0.2, vcenter=threshold_heatmap, vmax=threshold_heatmap + 0.2)

### Loop through each unique image and plot the aggregated heatmap
for i, image_name in enumerate(image_names_unique):
    print(f"Creating aggregated heatmap for image: {image_name} ({i+1}/{len(image_names_unique)})", end="\r")
    
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_h, image_w = image.shape[:2]
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image)
    ax.axis("off")

    prob_map_small = probability_maps[image_name]
    upscale_h, upscale_w = upscale_factors[image_name]
    
    # Work in 48x48 space, upscale visualization
    for row in range(prob_map_small.shape[0]):
        for col in range(prob_map_small.shape[1]):
            tile_prob = prob_map_small[row, col]
            tile_prob_normalized = (tile_prob - obs_min) / (obs_max - obs_min)
            
            color = cmap(norm(tile_prob_normalized))
            
            # Upscale grid cell coordinates to full image space for visualization
            pixel_row = row * upscale_h
            pixel_col = col * upscale_w
            
            rect = Rectangle((pixel_col, pixel_row), upscale_w, upscale_h, linewidth=0, edgecolor="none", facecolor=color, alpha=0.2)
            ax.add_patch(rect)

    if SAVE_PREDICTION_IMAGES:
        save_path = output_folder_heatmaps / f"{image_name}_heatmap_offset.png"
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

#%% Third: Peak Detection and Region Extraction with Flood-Fill, set parameters based on global median and std of probabilities across all images (fast version using concatenation)
print(f"\n{'='*60}")
print("Peak Detection and Region Extraction (on 48x48 grid)")
print(f"{'='*60}")

# Calculate global median for threshold-based approach (fast version using concatenate)
print("Computing global median probability using one sample image...")
probs_sample = probability_maps[image_names_unique[0]].flatten()
# all_probs = np.concatenate([prob_map.flatten() for prob_map in probability_maps.values()])
global_median = np.median(probs_sample)
global_std = np.std(probs_sample)

peak_threshold = threshold_method # Peaks must be this high (median + 1.5*std)
secondary_threshold = threshold_method + 0.1 * abs(global_std)  # Flood-fill stops when below median

# plot histogram of all probabilities with vertical line for global median
plt.figure(figsize=(8, 4))
plt.hist(probs_sample, bins=100, color="blue", alpha=0.7)
plt.axvline(threshold_method, color="green", linestyle="-", label=f"Threshold: {threshold_method:.4f}")
plt.axvline(global_median, color="red", linestyle="-", label=f"Global Median: {global_median:.4f}")
plt.axvline(global_median + global_std, color="red", linestyle="--", label=f"STD: {global_median + 1.5 * global_std:.4f}")
plt.axvline(global_median - global_std, color="red", linestyle="--")
plt.axvline(peak_threshold, color="purple", linestyle="-", label=f"Peak Threshold: {peak_threshold:.4f}")
plt.axvline(secondary_threshold, color="purple", linestyle="--", label=f"Secondary Threshold: {secondary_threshold:.4f}")
plt.title("Histogram of All Probabilities (48x48 grid)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("Probability Value")
plt.ylabel("Frequency")
plt.legend(loc="upper right")
plt.show()

print(f"Global median: {global_median:.4f}, std: {global_std:.4f}")

# Parameters for peak detection - based on MEDIAN instead of max fraction
# Adjust these multipliers to control sensitivity

# External parameters for region extraction
min_region_size_patches = 1  # Minimum number of patches in the 48x48 grid
crop_padding_pixels = 50     # Padding around crop region in full resolution image
output_folder_crops = Path(test_images_folder).parent / "crops_peak_detection"
output_folder_crops.mkdir(parents=True, exist_ok=True)

print(f"Peak threshold: {peak_threshold:.4f}")
print(f"Secondary threshold: {secondary_threshold:.4f}")
print(f"Min region size: {min_region_size_patches} patches in 48x48 grid")
print(f"Crop padding: {crop_padding_pixels} pixels")
print(f"Output folder for crops: {output_folder_crops}")

#%% Carry out peak detection and region extraction for each image, then visualize crops with contours
crop_metadata = []  # Store crop information for CSV export

# Process each image for peak-based cropping
crop_counter = 0
for image_name in image_names_unique:
    print(f"Processing crops for image: {image_name}", end="\r")
    
    probs_sample = probability_maps[image_name].flatten()
    global_std = np.std(probs_sample)

    peak_threshold = threshold_method + 1.2*abs(global_std)  # Peaks must be this high (median + 1.5*std)
    secondary_threshold = threshold_method + 0.5 * abs(global_std)  # Flood-fill stops when below median

    # Work directly on 48x48 probability map
    prob_map_small = probability_maps[image_name]
    image_h, image_w = image_dimensions[image_name]
    upscale_h, upscale_w = upscale_factors[image_name]
    
    # Find local maxima on 48x48 map (no downsampling needed)
    maxima_small = find_local_maxima(prob_map_small, peak_threshold)
    
    if len(maxima_small) == 0:
        print(f"  No peaks found in {image_name}")
        continue
    
    # Load original image
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process each peak on 48x48 map, then upscale results

    regions_found = []

    for peak_y_small, peak_x_small, peak_val in maxima_small:
        # Flood-fill on 48x48 probability map
        region_small = flood_fill_region(prob_map_small, peak_y_small, peak_x_small, secondary_threshold)
        
        # Calculate minimum region size in 48x48 map coordinates
        min_region_size_small = min_region_size_patches ** 2
        if len(region_small) < min_region_size_small:
            continue  # Skip small regions
        
        # Upscale region: each cell (y_small, x_small) expands to its full pixel block
        # Cell at (y_small, x_small) covers pixels [y_small*upscale_h : (y_small+1)*upscale_h]
        region_full = set()
        for y_small, x_small in region_small:
            # Each cell expands to a block of pixels
            y_start = int(round(y_small * upscale_h))
            y_end = int(round((y_small + 1) * upscale_h))
            x_start = int(round(x_small * upscale_w))
            x_end = int(round((x_small + 1) * upscale_w))
            
            # Add all pixels in this block to the region
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if 0 <= y < image_h and 0 <= x < image_w:
                        region_full.add((y, x))
        
        regions_found.append((region_full, peak_val))
    
    # Skip image if no valid regions found after filtering
    if len(regions_found) == 0:
        print(f"  No valid regions found in {image_name} (all filtered out)")
        continue
    
    # Create visualization (only if we have regions to plot)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image_rgb)
    
    # Deduplicate regions: skip overlapping regions (keep the larger one)
    seen_crop_coords = []
    unique_regions = []
    
    for region_idx, (region, peak_val) in enumerate(regions_found):
        region_points = np.array(list(region))
        min_y, min_x = region_points.min(axis=0)
        max_y, max_x = region_points.max(axis=0)
        
        # Add padding for context
        crop_y0 = max(0, min_y - crop_padding_pixels)
        crop_x0 = max(0, min_x - crop_padding_pixels)
        crop_y1 = min(image_h, max_y + crop_padding_pixels)
        crop_x1 = min(image_w, max_x + crop_padding_pixels)
        
        # Make bounding box square: expand to the larger dimension
        crop_h = crop_y1 - crop_y0
        crop_w = crop_x1 - crop_x0
        max_size = max(crop_h, crop_w)
        
        # Expand symmetrically around the center
        center_y = (crop_y0 + crop_y1) / 2.0
        center_x = (crop_x0 + crop_x1) / 2.0
        
        crop_y0 = int(center_y - max_size / 2.0)
        crop_x0 = int(center_x - max_size / 2.0)
        crop_y1 = crop_y0 + max_size
        crop_x1 = crop_x0 + max_size
        
        # Clip to image bounds
        crop_y0 = max(0, crop_y0)
        crop_x0 = max(0, crop_x0)
        crop_y1 = min(image_h, crop_y1)
        crop_x1 = min(image_w, crop_x1)
        
        # Create unique identifier for this crop bounding box
        crop_coords = (crop_y0, crop_x0, crop_y1, crop_x1)
        crop_area = (crop_y1 - crop_y0) * (crop_x1 - crop_x0)
        
        # Check if this region overlaps significantly with any previously seen region
        is_duplicate = False
        for i, (prev_coords, prev_area) in enumerate(seen_crop_coords):
            prev_y0, prev_x0, prev_y1, prev_x1 = prev_coords
            
            # Compute intersection
            inter_y0 = max(crop_y0, prev_y0)
            inter_x0 = max(crop_x0, prev_x0)
            inter_y1 = min(crop_y1, prev_y1)
            inter_x1 = min(crop_x1, prev_x1)
            
            if inter_y1 > inter_y0 and inter_x1 > inter_x0:
                # Regions overlap - compute IoU
                inter_area = (inter_y1 - inter_y0) * (inter_x1 - inter_x0)
                union_area = crop_area + prev_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                
                # If overlap is significant (>50%), skip this region
                if iou > 0.5:
                    print(f"  Skipping region overlapping with previous crop (IoU={iou:.2f})")
                    is_duplicate = True
                    break
        
        if is_duplicate:
            continue
        
        seen_crop_coords.append((crop_coords, crop_area))
        unique_regions.append((region, peak_val, crop_coords, region_points))
    
    # Visualize and crop each unique region
    for crop_idx, (region, peak_val, crop_coords, region_points) in enumerate(unique_regions):
        crop_y0, crop_x0, crop_y1, crop_x1 = crop_coords
        
        crop_h = crop_y1 - crop_y0
        crop_w = crop_x1 - crop_x0
        
        # Draw green outline for the segmentation
        region_points_coords = np.array([(x, y) for y, x in region])  # Convert to (x, y) for plotting
        if len(region_points_coords) > 0:
            # Use convex hull for cleaner outline
            hull = cv2.convexHull(region_points_coords.astype(np.float32))
            hull_points = hull.reshape(-1, 2).astype(int)
            if len(hull_points) > 2:
                ax.plot(hull_points[:, 0], hull_points[:, 1], color='green', linewidth=2)
                ax.plot([hull_points[-1, 0], hull_points[0, 0]], [hull_points[-1, 1], hull_points[0, 1]], 
                       color='green', linewidth=2)
        
        # Draw red bounding box (crop area)
        rect = Rectangle((crop_x0, crop_y0), crop_w, crop_h, linewidth=3, 
                         edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Extract and save crop
        crop_image = image[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_path = output_folder_crops / f"{image_name}_crop_{crop_idx:03d}.png"
        cv2.imwrite(str(crop_path), crop_image)
        
        # Store metadata
        # Get min/max from region_points for metadata
        min_y, min_x = region_points.min(axis=0)
        max_y, max_x = region_points.max(axis=0)
        
        # Convert full-resolution region points back to 48x48 grid coordinates to compute mean
        region_points_small = np.round(region_points / np.array([upscale_h, upscale_w])).astype(int)
        # Clip to valid range to avoid index errors
        region_points_small[:, 0] = np.clip(region_points_small[:, 0], 0, prob_map_small.shape[0] - 1)
        region_points_small[:, 1] = np.clip(region_points_small[:, 1], 0, prob_map_small.shape[1] - 1)
        region_mean = prob_map_small[region_points_small[:, 0], region_points_small[:, 1]].mean() if len(region_points) > 0 else 0.0
            
        crop_metadata.append({
            'image_name': image_name,
            'crop_id': crop_idx,
            'crop_x0': crop_x0,
            'crop_y0': crop_y0,
            'crop_x1': crop_x1,
            'crop_y1': crop_y1,
            'crop_width': crop_w,
            'crop_height': crop_h,
            'region_size_pixels': len(region),
            'peak_intensity': peak_val,
            'region_mean_intensity': region_mean,
            'segmentation_min_y': int(min_y),
            'segmentation_min_x': int(min_x),
            'segmentation_max_y': int(max_y),
            'segmentation_max_x': int(max_x),
        })
        
        crop_counter += 1
    
    # Save visualization with outlines and bounding boxes
    ax.axis("off")
    fig.suptitle(f"Peak Detection Results: {image_name} (Green=Segmentation, Red=Crop)", 
                fontsize=16, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    if SAVE_PREDICTION_IMAGES:
        save_path = output_folder_crops / 'full_images' / f"{image_name}_crop_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    
# Save crop metadata to CSV
if crop_metadata:
    df_crops = pd.DataFrame(crop_metadata)
    csv_path = output_folder_crops / "crop_metadata.csv"
    df_crops.to_csv(csv_path, index=False)
    print(f"\n✓ Extracted {crop_counter} crops and saved metadata to {csv_path}")
else:
    print(f"\n⚠ No crops extracted (check peak_threshold and secondary_threshold parameters)")

print("\nDone!")