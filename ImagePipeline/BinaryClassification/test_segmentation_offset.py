#%% Import packages and set up environment
### PACKAGES ###
import os

import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
import seaborn as sns

import pickle as pkl
from pathlib import Path
import cv2

from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import matplotlib as mpl

from functions import split_image_into_tiles, VariationalAutoencoder, test_image_pipeline, preprocess_tile

def extract_tiles_with_offset(image, tile_size, grid_size, offset_x=0, offset_y=0):
    """
    Extract tiles from image with specified offset.
    
    Args:
        image: input image
        tile_size: size of each tile
        grid_size: number of tiles along one side
        offset_x: horizontal offset (in pixels)
        offset_y: vertical offset (in pixels)
    
    Returns:
        list of (tile, row, col) tuples
    """
    tiles = []
    image_h, image_w = image.shape[:2]
    
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = offset_y + row * tile_size
            x0 = offset_x + col * tile_size
            y1 = min(y0 + tile_size, image_h)
            x1 = min(x0 + tile_size, image_w)
            
            # Handle edge cases by padding if necessary
            tile = image[y0:y1, x0:x1]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = np.pad(tile, ((0, tile_size - tile.shape[0]), (0, tile_size - tile.shape[1])), mode='edge')
            
            tiles.append((tile, row, col))
    
    return tiles

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
monitoring_effort = "Kristineberg_250915" 

test_images_folder = os.path.join(ROOT_DIR_C, monitoring_effort, "test", "OG_images")
tile_grid_size = 32 # how many tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = 4512 // tile_grid_size # size of the tiles that the original image is split into
latent_dim = 32 # how many tiles along one side 
model_name = f"models/{monitoring_effort}_vae_model{tile_grid_size}_l{latent_dim}.pth"

# Autoencoder parameters
batch_size = 2048
image_size = 128
hidden_channels = 32

# Other parameters
N_test = None # how many test images to process (set to None to process all)
SAVE_PREDICTION_IMAGES = True

# Offset grid parameters
offsets = [0, tile_size // 3, 2 * tile_size // 3]  # 0, 1/3, 2/3 offsets

# Load OneClass SVM model and parameters
svm_model_name = model_name.replace("vae_model", "oc_svm_model").replace(".pth", ".pkl")
svm_data = pkl.load(open(svm_model_name, "rb"))
svm_model = svm_data[0]
scaler = svm_data[1]
weight_recon_error = svm_data[2]
nu = svm_data[3] if len(svm_data) > 3 else None
svm_score_threshold_original = svm_data[4] if len(svm_data) > 4 else None  # Store original threshold
svm_score_threshold = svm_score_threshold_original  # Use original by default

# Load Binary SVM model and parameters
binary_svm_model_name = model_name.replace("vae_model", "binary_svm_model").replace(".pth", ".pkl")
try:
    binary_svm_data = pkl.load(open(binary_svm_model_name, "rb"))
    binary_svm_model = binary_svm_data[0]
    binary_scaler = binary_svm_data[1]
    binary_svm_threshold = binary_svm_data[2] if len(binary_svm_data) > 2 else 0.0
    binary_svm_available = True
except FileNotFoundError:
    print(f"⚠ Binary SVM model not found at: {binary_svm_model_name}")
    binary_svm_available = False

test_images_path = list(Path(test_images_folder).glob("*.png"))


print(f"Found {len(test_images_path)} test images in: {test_images_folder}")
if N_test is not None and len(test_images_path) > N_test:
    rand_idxs = np.random.choice(len(test_images_path), size=N_test, replace=False)
    test_images_path = [test_images_path[i] for i in rand_idxs]
    print(f"Showing a random subset of {N_test} test images for evaluation:")

print(f"loading VAE model from: {model_name}")
print(f"loading OneClass SVM model from: {svm_model_name} with parameters:", end="\n")
print(f"weight_recon_error={weight_recon_error}, nu={nu}")
print(f"Original threshold_svm={svm_data[4]:.8f}")

print(f"loading Binary SVM model from: {binary_svm_model_name}")
print(f"Binary SVM threshold: {binary_svm_threshold:.8f}")


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
For each offset (0, 1/3, 2/3 of tile_size), tiles are extracted and processed through the VAE/SVM pipeline.
Results are aggregated by summing probabilities from overlapping regions.
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
    d2_list = []
    recon_list = []
    svm_scores_list = []
    label_list_d2 = []
    label_list_svm = []
    label_list_binary_svm = [] if binary_svm_available else None
    binary_svm_scores_list = [] if binary_svm_available else None
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
        
        # predict using OneClass SVM model    
        svm_features = np.hstack([mu, recon_err.cpu().detach().numpy().reshape(-1, 1)])
        svm_features = scaler.transform(svm_features)
        svm_features[:, -1] *= weight_recon_error
        svm_scores = -svm_model.decision_function(svm_features)
        label_svm_is_obs = (svm_scores >= svm_score_threshold)

        svm_scores_list.append(svm_scores)
        label_list_svm.append(label_svm_is_obs)
        
        is_obs = label_svm_is_obs

        # Predict using Binary SVM if available
        if binary_svm_available:
            binary_svm_features = np.hstack([mu, recon_err.cpu().detach().numpy().reshape(-1, 1)])
            binary_svm_features = binary_scaler.transform(binary_svm_features)
            binary_svm_scores = binary_svm_model.decision_function(binary_svm_features)
            label_binary_svm_is_obs = (binary_svm_scores >= binary_svm_threshold)
            
            if not hasattr(label_list_binary_svm, 'append'):
                label_list_binary_svm = []
                binary_svm_scores_list = []
            label_list_binary_svm.append(label_binary_svm_is_obs)
            binary_svm_scores_list.append(binary_svm_scores)
        
        batch_counter += 1
        print(f"processed batch {batch_counter}/{len(tile_loader)} |SVM classification: obs={label_svm_is_obs.sum()} empty = {(~label_svm_is_obs).sum()}", end="\r")

    mu_array = np.concatenate(mu_list, axis=0)
    labels_array_svm = np.concatenate(label_list_svm, axis=0)
    labels_array_binary_svm = np.concatenate(label_list_binary_svm, axis=0) if binary_svm_available else None
    image_names_array = np.concatenate(image_names_list, axis=0)
    tile_rows_array = np.concatenate(tile_rows, axis=0)
    tile_cols_array = np.concatenate(tile_cols, axis=0)
    recon_array = torch.cat(recon_list, dim=0).cpu().numpy()
    svm_scores_array = np.concatenate(svm_scores_list, axis=0)
    binary_svm_scores_array = np.concatenate(binary_svm_scores_list, axis=0) if binary_svm_available else None
    
    # Create dataframe with image names, tile numbers, and latent representations
    latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
    df_latent = pd.DataFrame(mu_array, columns=latent_cols)
    df_latent.insert(0, "image_name", image_names_array)
    df_latent.insert(1, "tile_row", tile_rows_array)
    df_latent.insert(2, "tile_col", tile_cols_array)
    df_latent.insert(3, "recon_error", recon_array)
    df_latent.insert(4, "label_predicted_svm", labels_array_svm)
    df_latent.insert(5, "label_predicted_bsvm", labels_array_binary_svm)
    df_latent.insert(7, "svm_scores", svm_scores_array)
    df_latent.insert(8, "bsvm_scores", binary_svm_scores_array)
    
    offset_results[offset] = df_latent
    
    print(f"\nClass distribution for Binary SVM method (offset={offset}): {df_latent['label_predicted_bsvm'].value_counts()}")

#%% Aggregate results from all offsets using probability summation
print(f"\n{'='*60}")
print("Aggregating results from all offsets")
print(f"{'='*60}")

method = "bsvm"
score_method = method + "_scores"

# Get unique image names
image_names_unique = offset_results[offsets[0]]["image_name"].unique()

# Create probability maps for each image
probability_maps = {}
coverage_maps = {}

for image_name in image_names_unique:
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_h, image_w = image.shape[:2]
    
    # Initialize probability and coverage maps
    prob_map = np.zeros((image_h, image_w), dtype=np.float32)
    coverage_map = np.zeros((image_h, image_w), dtype=np.float32)
    
    # Aggregate scores from all offsets
    for offset in offsets:
        df_offset = offset_results[offset]
        df_image = df_offset[df_offset["image_name"] == image_name]
        
        for df_tile in df_image.itertuples():
            y0 = offset + int(df_tile.tile_row * tile_size)
            x0 = offset + int(df_tile.tile_col * tile_size)
            y1 = min(y0 + tile_size, image_h)
            x1 = min(x0 + tile_size, image_w)
            
            tile_score = getattr(df_tile, score_method)
            
            # Add score to probability map
            prob_map[y0:y1, x0:x1] += tile_score
            # Track coverage
            coverage_map[y0:y1, x0:x1] += 1
    
    # Average the probabilities by coverage
    prob_map = np.divide(prob_map, coverage_map, where=coverage_map > 0, out=prob_map.copy())
    
    probability_maps[image_name] = prob_map
    coverage_maps[image_name] = coverage_map

#%% Visualize results with blob contours and heatmaps

output_folder_heatmaps = Path(test_images_folder) / "heatmaps_offset" / "tiles{}".format(tile_grid_size)
output_folder_heatmaps.mkdir(parents=True, exist_ok=True)

output_folder_contours = Path(test_images_folder) / "contours_offset" / "tiles{}".format(tile_grid_size)
output_folder_contours.mkdir(parents=True, exist_ok=True)

# Normalize the observation metric for color mapping across all images
all_probs = np.concatenate([prob_map.flatten() for prob_map in probability_maps.values()])
obs_min = all_probs.min()
obs_max = all_probs.max()

threshold_heatmap = (binary_svm_threshold - obs_min) / (obs_max - obs_min)

cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"])
norm = mpl.colors.TwoSlopeNorm(vmin=threshold_heatmap*0.8, vcenter=threshold_heatmap, vmax=threshold_heatmap*1.2)

print(f"Observation likelihood threshold: {threshold_heatmap:.3f}")
print(f"Score range: [{obs_min:.3f}, {obs_max:.3f}], Binary SVM threshold: {binary_svm_threshold:.3f}")

#%% First: Show contour outlines overlaid on the original image
print(f"\n{'='*60}")
print("Creating contour visualizations")
print(f"{'='*60}")

for image_name in image_names_unique:
    print(f"Creating contour outline for image: {image_name}", end="\r")
    
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]
    
    # Create a binary mask from the aggregated probability map
    prob_map = probability_maps[image_name]
    mask = (prob_map >= binary_svm_threshold).astype(np.uint8) * 255
    
    # Find contours from the binary mask
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

#%% Second: Create heatmap visualizations
print(f"\nCreating aggregated heatmap visualizations")

### Loop through each unique image and plot the aggregated heatmap
for i, image_name in enumerate(image_names_unique):
    print(f"Creating aggregated heatmap for image: {image_name} ({i+1}/{len(image_names_unique)})", end="\r")
    
    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image)
    ax.axis("off")

    prob_map = probability_maps[image_name]
    coverage_map = coverage_maps[image_name]
    
    # Create a visualization of the aggregated probabilities with finer granularity based on offsets
    grid_step = tile_size // len(offsets)  # Finer grid due to offset overlaps
    for row in range(0, image.shape[0], grid_step):
        for col in range(0, image.shape[1], grid_step):
            y1 = min(row + grid_step, image.shape[0])
            x1 = min(col + grid_step, image.shape[1])
            
            tile_prob = prob_map[row:y1, col:x1].mean()
            tile_prob_normalized = (tile_prob - obs_min) / (obs_max - obs_min)
            
            color = cmap(norm(tile_prob_normalized))
            rect = Rectangle((col, row), grid_step, grid_step, linewidth=0, edgecolor="none", facecolor=color, alpha=0.2)
            ax.add_patch(rect)

    if SAVE_PREDICTION_IMAGES:
        save_path = output_folder_heatmaps / f"{image_name}_heatmap_offset.png"
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

print("\nDone!")

