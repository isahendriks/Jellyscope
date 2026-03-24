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

from functions import split_image_into_tiles, classify_image, mahalanobis_distance, VariationalAutoencoder, test_image_pipeline, preprocess_tile

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
monitoring_effort = "Kristineberg_250915"

test_images_folder = os.path.join(ROOT_DIR_C, monitoring_effort, "test", "OG_images")
tile_grid_size = 32 # how many tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = 4512 // tile_grid_size # size of the tiles that the original image is split into
latent_dim = 6 # how many tiles along one side 
model_name = f"models/{monitoring_effort}_vae_model{tile_grid_size}_l{latent_dim}.pth"

# Autoencoder parameters
batch_size = 64
image_size = 128
hidden_channels = 32

val_params_name = model_name.replace(".pth", "_params.pkl")
val_params = pkl.load(open(val_params_name, "rb"))

svm_model_name = model_name.replace("vae_model", "svm_model").replace(".pth", ".pkl")
svm_data = pkl.load(open(svm_model_name, "rb"))
svm_model = svm_data[0]
scaler = svm_data[1]
weight_recon_error = svm_data[2]
svm_score_threshold = svm_data[3] if len(svm_data) > 3 else None

mu_g = val_params["mu_g"]
cov_g = val_params["cov_g"]
threshold = val_params["threshold"]

# tile_size = 4512 // tile_grid_size

test_images_path = list(Path(test_images_folder).glob("*.png"))
print(f"Found {len(test_images_path)} test images in: {test_images_folder}")
print(f"loading VAE model from: {model_name}")
print(f"loading SVM model from: {svm_model_name}")
if svm_score_threshold is None:
    print("SVM score threshold not found in model file -> using raw OneClassSVM decision boundary (predict).")
else:
    print(f"Using saved SVM score threshold: {svm_score_threshold:.4f}")
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
    # tile_gray = cv2.cvtColor(tiles_numpy, cv2.COLOR_BGR2GRAY)  # Convert tile to grayscale for visualization
    tile_list.append(tiles_numpy)  # Store the original tile image for later visualization
    
    # predict obs vs no_obs based on Mahalanobis distance in latent space
    d2 = mahalanobis_distance(mu, mu_g, np.linalg.inv(cov_g))  # Calculate Mahalanobis distance to the "empty" distribution
    is_obs = d2 > threshold   # if threshold was set on d2

    d2_list.append(d2)
    label_list_d2.append(is_obs)
    
    # predict using SVM model
    
    svm_features = np.hstack([mu, recon_err.cpu().detach().numpy().reshape(-1, 1)])  # Combine latent features and reconstruction error for SVM
    svm_features = scaler.transform(svm_features)  # Scale the features using the fitted scaler
    svm_features[:, -1] *= weight_recon_error  # Apply weight to reconstruction error
    svm_scores = -svm_model.decision_function(svm_features)
    label_svm_is_obs = (svm_scores >= svm_score_threshold)

    svm_scores_list.append(svm_scores)
    label_list_svm.append(label_svm_is_obs)
    
    batch_counter += 1
    print(f"processed batch {batch_counter}/{len(tile_loader)} | obs={is_obs.sum()} empty = {(~is_obs).sum()}", end="\r")

mu_array = np.concatenate(mu_list, axis=0)
d2_array = np.concatenate(d2_list, axis=0)

labels_array_d2 = np.concatenate(label_list_d2, axis=0)
labels_array_svm = np.concatenate(label_list_svm, axis=0)
image_names_array = np.concatenate(image_names_list, axis=0)
tile_rows_array = np.concatenate(tile_rows, axis=0)
tile_cols_array = np.concatenate(tile_cols, axis=0)
recon_array = torch.cat(recon_list, dim=0).cpu().numpy()
svm_scores_array = np.concatenate(svm_scores_list, axis=0)
# tile_array = np.concatenate(tile_list, axis=0)

# Create dataframe with image names, tile numbers, and latent representations
latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
df_latent = pd.DataFrame(mu_array, columns=latent_cols)
df_latent.insert(0, "image_name", image_names_array)  # Add image names as the first column
df_latent.insert(1, "tile_row", tile_rows_array)
df_latent.insert(2, "tile_col", tile_cols_array)
df_latent.insert(3, "recon_error", recon_array)  # Add reconstruction error as a column
df_latent.insert(4, "label_predicted_d2", labels_array_d2)  # Add manual labels (True for obs, False for no_obs)
df_latent.insert(5, "label_predicted_svm", labels_array_svm)  # Add SVM predicted labels
df_latent.insert(6, "d2", d2_array)  # Add Mahalanobis distance values
df_latent.insert(7, "svm_scores", svm_scores_array)  # Add SVM scores

# print class distribution for manual labels and predicted labels both methods
print(f"Class distribution for d2 threshold method: {df_latent['label_predicted_d2'].value_counts()}")
print(f"Class distribution for SVM method: {df_latent['label_predicted_svm'].value_counts()}")


#%% chose method for predicted labels (d2 threshold vs SVM)
# labels_predicted = df_latent["label_predicted_d2"].values
labels_predicted = df_latent["label_predicted_svm"].values.astype(bool)

#%% PCA for visualization of latent space
pca = PCA(n_components=2)
mu_pca = pca.fit_transform(mu_array)

plt.figure(figsize=(12, 10))
plt.scatter(mu_pca[labels_predicted, 0], mu_pca[labels_predicted, 1], s=50, c="blue", label="obs", alpha=0.6, edgecolors='black', linewidth=0.5)
plt.scatter(mu_pca[~labels_predicted, 0], mu_pca[~labels_predicted, 1], s=50, c="orange", label="empty", alpha=0.6, edgecolors='black', linewidth=0.5)

plt.xlabel("PCA 1", fontsize=24)
plt.ylabel("PCA 2", fontsize=24)
plt.legend(loc="best", fontsize=12)
plt.title("Outlier detection in 2D latent space", fontsize=20, fontweight="bold")
plt.grid()
plt.show()

#%%

bins = np.linspace(0, df_latent["d2"].max(), int(df_latent["d2"].max()/threshold*4 ))

plt.figure(figsize=(6,4))
plt.hist(df_latent.loc[df_latent.label_predicted_d2==0,'d2'], bins=bins, alpha=0.6, label='no_obs', color='orange')
plt.hist(df_latent.loc[df_latent.label_predicted_d2==1,'d2'], bins=bins, alpha=0.6, label='obs', color='blue')
plt.vlines(threshold, ymin=0, ymax=12000, colors='red', linestyles='dashed', label=f'Threshold (d2={threshold:.1f})')
plt.xlabel("Mahalanobis distance (d2)", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.ylim(0, 12000)
plt.tight_layout()
plt.grid()
plt.legend(); 
plt.title("Mahalanobis distance distributions predicted labels"); plt.show()


#%% Show the original test image with tiles colored by predicted label (obs vs no_obs)

# Extract unique image names from the dataframe
image_names_unique = df_latent["image_name"].unique()

# Loop through each unique image and overlay predicted labels as a grid on original image
for image_name in image_names_unique:
    df_latent_image = df_latent[df_latent["image_name"] == image_name]

    image_path = Path(test_images_folder) / f"{image_name}.png"
    image = cv2.imread(str(image_path))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    # overlay predicted obs tiles
    for df_tile in df_latent_image.itertuples():
        x0 = int(df_tile.tile_col * tile_size)
        y0 = int(df_tile.tile_row * tile_size)
        rect = Rectangle((x0, y0), tile_size, tile_size, linewidth=2,  facecolor="lime", alpha=0.5, edgecolor="none")  # green rectangle for predicted obs tiles
        if df_tile.label_predicted_svm:  # If predicted as obs, add green rectangle
            ax.add_patch(rect)

    ax.set_title(f"Predicted obs tiles in green for image: {image_name}", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.show()

#%% Use heatmap to visualize the predicted obs tiles across the original image
# Global color scale across all images
observation_metric = "svm_scores"  # Choose which metric to visualize in the heatmap (e.g. "d2" or "svm_scores")
treshold_heatmap = svm_score_threshold if observation_metric == "svm_scores" else threshold

obs_likelihood_all = chi2.cdf(df_latent[observation_metric].values, df=2)  # higher d2 -> higher obs-likelihood
likelihood_treshold = 0.9  # Convert threshold to obs likelihood threshold for color scaling

cmap = mpl.colors.LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"])
norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=likelihood_treshold, vmax=1.0) # for midpoint at threshold

print(f"Observation likelihood threshold: {likelihood_treshold:.3f}")

### Loop through each unique image and plot the tiles with a heatmap 
for i, image_name in enumerate(image_names_unique):  # Show heatmap for the first image only
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
        
        # Convert d2 to "obs likelihood" in [0,1] and map to a color
        obs_likelihood = chi2.cdf(df_tile.d2, df=2)
        color = cmap(norm(obs_likelihood))
        rect = Rectangle((x0, y0), tile_size, tile_size, linewidth=0, edgecolor="none", facecolor=color, alpha=0.45)
        ax.add_patch(rect)
    # Colorbar
    
    # sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # ticks = np.linspace(0, 1, 6)   # evenly spaced: 0.0, 0.2, ..., 1.0
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, aspect=30, location="right")
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=20)
    # cbar.set_label("Observation likelihood", rotation=90, fontsize=20, fontweight="bold")
    # plt.imsave(f"heatmap_{image_name}.png", fig, dpi=300)  # Save heatmap as PNG    
    
    plt.show()

#%% Test on single image and visualize the tiles with lowest and highest Mahalanobis distance (most likely obs vs most likely empty)
empty_images_path = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data\Kristineberg_250915\Binary_classifier_test_data\checked\Hard_class\Without_jellyfish\Background"
empty_images_list = list(Path(empty_images_path).glob("*.png"))
random_empty_image = np.random.choice(empty_images_list)

image_path = random_empty_image  # Test on the first image in the folder
grid_size = tile_grid_size
model = model

grid_size = grid_size
threshold = 50# threshold
mu_g = mu_g
cov_g = cov_g
device=device
batch_size=64
image_size = image_size
plot=True

classify_image(image_path, model, grid_size, threshold, mu_g, cov_g, device, batch_size, image_size, plot)

#%%
empty_images_path = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data\Kristineberg_250915\Binary_classifier_test_data\checked\Hard_class\Without_jellyfish\Background"
empty_images_list = list(Path(empty_images_path).glob("*.png"))
random_empty_image = np.random.choice(empty_images_list)

image_path = Path(random_empty_image)
cov_inv = np.linalg.inv(np.asarray(cov_g))
tile_size = 4512 // grid_size

# 1. load & tile 
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

raw_tiles = split_image_into_tiles(image, grid_size=grid_size)  # list of (tile, row, col)
dataset = [(tile, row, col) for tile, row, col in raw_tiles]

# 3. encode & score 
mu_list, d2_list, rows_list, cols_list = [], [], [], []
model.eval()
model.to(device)
with torch.no_grad():
    for raw_tile in raw_tiles:
        tile, row, col = raw_tile
        tile = preprocess_tile(tile)  # Apply same preprocessing as during training
        tile = tile.to(device)  # Add batch dimension and move to device
        mu, _ = model.encode(tile)
        
        mu_np = mu.cpu().numpy()
        d2 = mahalanobis_distance(mu_np, mu_g, cov_inv)
        mu_list.append(mu_np)
        d2_list.append(d2)
        rows_list.append(row)
        cols_list.append(col)

mu_array = np.concatenate(mu_list, axis=0)
d2_array = np.concatenate(d2_list, axis=0)
labels_array = d2_array > threshold

# 4. build df 
latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
df_latent = pd.DataFrame(mu_array, columns=latent_cols)
df_latent.insert(0, "image_name",     image_path.stem)
df_latent.insert(1, "tile_row",       rows_list)
df_latent.insert(2, "tile_col",       cols_list)
df_latent.insert(3, "label_predicted", labels_array)
df_latent.insert(4, "d2",             d2_array)

# 5. heatmap plot 
if plot:
    likelihood_threshold = chi2.cdf(threshold, df=2)
    cmap_ryg = mpl.colors.LinearSegmentedColormap.from_list(
        "red_yellow_green", ["red", "yellow", "green"])
    norm = mpl.colors.TwoSlopeNorm(
        vmin=0.0, vcenter=likelihood_threshold, vmax=1.0)

    image = cv2.imread(str(image_path))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"Observation likelihood: {image_path.stem}",
                 fontsize=16, fontweight="bold")

    for _, row in df_latent.iterrows():
        obs_likelihood = chi2.cdf(row["d2"], df=2)
        color = cmap_ryg(norm(obs_likelihood))
        x0 = int(row["tile_col"] * tile_size)
        y0 = int(row["tile_row"] * tile_size)
        rect = Rectangle(
            (x0, y0), tile_size, tile_size,
            linewidth=0, edgecolor="none",
            facecolor=color, alpha=0.45)
        ax.add_patch(rect)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_ryg)
    sm.set_array([])
    ticks = np.linspace(0, 1, 6)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01,
                            aspect=30, location="right")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=16)
    cbar.set_label("Observation likelihood",
                    rotation=90, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()