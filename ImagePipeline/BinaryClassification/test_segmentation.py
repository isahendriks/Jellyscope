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

def mahalanobis_distance(x, mu, cov_inv):
    diff = x - mu
    d2 = np.sum(diff @ cov_inv * diff, axis=1)  # Mahalanobis distance squared
    return d2

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
tile_grid_size = 32 # how many tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = 4512 // tile_grid_size # size of the tiles that the original image is split into
latent_dim = 32 # how many tiles along one side 
model_name = f"models/{monitoring_effort}_vae_model{tile_grid_size}_l{latent_dim}.pth"

# Autoencoder parameters
batch_size = 2048
image_size = 128
hidden_channels = 32

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

# THRESHOLD ADJUSTMENT: Increase this value to reduce false positives (e.g., 1.2, 1.5, 2.0)
# Higher multiplier = higher threshold = fewer obs predictions = fewer false positives


# tile_size = 4512 // tile_grid_size

test_images_path = list(Path(test_images_folder).glob("*.png"))


print(f"Found {len(test_images_path)} test images in: {test_images_folder}")
# if len(test_images_path) > 50:
#     rand_idxs = np.random.choice(len(test_images_path), size=50, replace=False)
#     test_images_path = [test_images_path[i] for i in rand_idxs]
#     print(f"Showing a random subset of 50 test images for evaluation:")

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

#%% Load many images (folder)
"""
The code below loads all test images from the specified folder, split them into tiles and applies the classification pipeline to each tile.
It creates a dataloader for batch processing on the GPU.
"""
### 1. Load images, split into tiles, and store in a list of tuples (tile, row, col) for all images in the test folder
tiles_list = []
rows_list = []
image_names = []
cols_list = []

for image_path in test_images_path:
    print(f"Processing test image: {image_path.stem}, ({test_images_path.index(image_path) + 1}/{len(test_images_path)})", end="\r")
    tiles = split_image_into_tiles(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE), grid_size=tile_grid_size)  # list of (tile, row, col)
    
    for tile in tiles:
        tiles_list.append((tile[0], tile[1], tile[2], image_path.stem))  # Store tuple expected by collate_fn

print(f"\nLoaded and split {len(test_images_path)} test images into {len(tiles_list)} tiles of size {tile_size}x{tile_size} for evaluation.")

#%% 2. Create data loader for test tiles (with same transformations as during training)
tile_loader = DataLoader(
    tiles_list,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=test_image_pipeline,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

print(f"Created tile dataloader for test images with batch size {batch_size} and image size {image_size}x{image_size}."
      f"\nTotal batches: {len(tile_loader)}"
      f"\nTotal tiles (approx): {len(test_images_path) * tile_grid_size**2}"
)
#%% 3. Run the tile dataloader through the VAE and collect results in a dataframe for analysis and visualization

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

    mu, _ = model.encode(tiles, row=rows, col=cols)  # forward pass with row/col info (currently not used in the model, but can be implemented later)
    
    mu = mu.cpu().detach().numpy()
    
    # reconstruction error per tile (mean squared error over pixels)
    x_hat = model(tiles, row=rows, col=cols)
    recon_err = ((tiles - x_hat) ** 2).flatten(1).mean(dim=1)
    recon_list.append(recon_err.cpu().detach())  # Store the reconstructed tiles for later visualization
    
    mu_list.append(mu)
    tile_rows.append(rows.cpu().numpy())
    tile_cols.append(cols.cpu().numpy())
    image_names_list.append(image_names)

    tiles_numpy = tiles.cpu().numpy()  # Convert tile tensor to numpy array for visualization
    tile_list.append(tiles_numpy)  # Store the original tile image for later visualization
    
    # predict using OneClass SVM model    
    svm_features = np.hstack([mu, recon_err.cpu().detach().numpy().reshape(-1, 1)])  # Combine latent features and reconstruction error for SVM
    svm_features = scaler.transform(svm_features)  # Scale the features using the fitted scaler
    svm_features[:, -1] *= weight_recon_error  # Apply weight to reconstruction error
    svm_scores = -svm_model.decision_function(svm_features)
    label_svm_is_obs = (svm_scores >= svm_score_threshold)

    svm_scores_list.append(svm_scores)
    label_list_svm.append(label_svm_is_obs)
    
    is_obs = label_svm_is_obs  # Use SVM predicted labels for now, can switch to d2 or binary SVM later

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
# tile_array = np.concatenate(tile_list, axis=0)

# Create dataframe with image names, tile numbers, and latent representations
latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
df_latent = pd.DataFrame(mu_array, columns=latent_cols)
df_latent.insert(0, "image_name", image_names_array)  # Add image names as the first column
df_latent.insert(1, "tile_row", tile_rows_array)
df_latent.insert(2, "tile_col", tile_cols_array)
df_latent.insert(3, "recon_error", recon_array)  # Add reconstruction error as a column
df_latent.insert(4, "label_predicted_svm", labels_array_svm)  # Add SVM predicted labels

df_latent.insert(5, "label_predicted_bsvm", labels_array_binary_svm)  # Add Binary SVM predicted labels
df_latent.insert(7, "svm_scores", svm_scores_array)  # Add OneClass SVM scores
df_latent.insert(8, "bsvm_scores", binary_svm_scores_array)  # Add Binary SVM scores


#%% print class distribution for manual labels and predicted labels both methods
print(f"Class distribution for OneClass SVM method: {df_latent['label_predicted_svm'].value_counts()}")
print(f"Class distribution for Binary SVM method: {df_latent['label_predicted_bsvm'].value_counts()}")

#%% chose method for predicted labels (d2 threshold vs SVM)
# labels_predicted = df_latent["label_predicted_d2"].values
labels_predicted = df_latent["label_predicted_bsvm"].values.astype(bool)
observation_metric = "svm_scores"  # Choose which metric to visualize in the heatmap (e.g. "d2" or "svm_scores")

#%% Show the original test image with the predicted obs tiles overlaid as blob outlines

method = "svm"  # Choose which metric to visualize in the heatmap (e.g. "d2" or "svm_scores")
label_method = "label_predicted_" + method  # Choose which predicted label to visualize (e.g. "label_predicted_d2" or "label_predicted_svm")
score_method = method + "_scores"  # Choose which score to visualize in the heatmap (e.g. "d2" or "svm_scores")

# Extract unique image names from the dataframe
image_names_unique = df_latent["image_name"].unique()

# Loop through each unique image and overlay predicted labels as blob outlines on original image
for image_name in image_names_unique:  # Show the first image only for testing
    df_latent_image = df_latent[df_latent["image_name"] == image_name]

    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a binary mask for predicted observations
    image_h, image_w = image.shape[:2]
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    
    for df_tile in df_latent_image.itertuples():
        if df_tile.label_predicted_bsvm == True:  # If predicted as obs
            x0 = int(df_tile.tile_col * tile_size)
            y0 = int(df_tile.tile_row * tile_size)
            x1 = min(x0 + tile_size, image_w)
            y1 = min(y0 + tile_size, image_h)
            mask[y0:y1, x0:x1] = 255
    
    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
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

    ax.set_title(f"Predicted obs blobs in green for image: {image_name}", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.show()

#%% Use heatmap to visualize the predicted obs tiles across the original image
# Global color scale across all images
# treshold_heatmap = bsvm_score_threshold  # Use the same threshold as for classification to set the color scale

SAVE_PREDICTION_IMAGES = True  # Whether to save the heatmap images to disk (will be saved in a "heatmaps" subfolder in the test images folder)
output_folder_heatmaps = Path(test_images_folder) / "heatmaps"
output_folder_heatmaps.mkdir(parents=True, exist_ok=True)

# Normalize the observation metric for color mapping
obs_likelihood = df_latent[score_method] 
obs_min = obs_likelihood.min()
obs_max = obs_likelihood.max()
obs_likelihood_scaled = (obs_likelihood - obs_min) / (obs_max - obs_min)  # Normalize to [0,1] for color mapping

print(f"Score range: [{obs_min:.3f}, {obs_max:.3f}], Loaded threshold: {binary_svm_threshold:.3f}")

# Smart threshold handling: if threshold is outside score range, recalculate
if binary_svm_threshold < obs_min or binary_svm_threshold > obs_max:
    print(f"⚠ WARNING: Threshold {binary_svm_threshold:.3f} is outside score range [{obs_min:.3f}, {obs_max:.3f}]")
    print("  Using median of scores as threshold instead.")
    binary_svm_threshold = obs_likelihood.median()
    print(f"  New threshold (median): {binary_svm_threshold:.3f}")
else:
    print(f"✓ Threshold {binary_svm_threshold:.3f} is within score range")

# Optional: Allow manual threshold override
MANUAL_THRESHOLD = None  # Set to a value (e.g., 0.5) to override the loaded threshold
if MANUAL_THRESHOLD is not None:
    print(f"⚠ Using manual threshold override: {MANUAL_THRESHOLD:.3f}")
    binary_svm_threshold = MANUAL_THRESHOLD

threshold_heatmap = (binary_svm_threshold - obs_min) / (obs_max - obs_min)  # Normalize threshold to [0,1]

cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"])
norm = mpl.colors.TwoSlopeNorm(vmin=threshold_heatmap*0.8, vcenter=threshold_heatmap, vmax=threshold_heatmap*1.2) # for midpoint at threshold

print(f"Final threshold (normalized): {threshold_heatmap:.3f}")

### Loop through each unique image and plot the tiles with a heatmap 

for i, image_name in enumerate(image_names_unique):  # Show heatmap for each image
    print(f"Creating heatmap for image: {image_name} ({i+1}/{len(image_names_unique)})", end="\r")
    df_latent_image = df_latent[df_latent["image_name"] == image_name]

    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.axis("off")
    # ax.set_title(str(image_name) + f" (Image idx: {i})", fontsize=16, fontweight="bold")

    for df_tile in df_latent_image.itertuples():
        x0 = int(df_tile.tile_col * tile_size)
        y0 = int(df_tile.tile_row * tile_size)
        
        # Get the score value for this specific tile and normalize it
        tile_score = float(getattr(df_tile, score_method))
        tile_score_normalized = (tile_score - obs_min) / (obs_max - obs_min)
        
        color = cmap(norm(tile_score_normalized))
        rect = Rectangle((x0, y0), tile_size, tile_size, linewidth=0, edgecolor="none", facecolor=color, alpha=0.2)
        ax.add_patch(rect)
        
    if SAVE_PREDICTION_IMAGES:
        save_path = output_folder_heatmaps / f"{image_name}_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
