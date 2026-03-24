#%% Import packages

### PACKAGES ###
import os
import torch
from torch.utils.data import DataLoader

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

import pickle as pkl
from pathlib import Path
import cv2
import seaborn as sns

from matplotlib.patches import Rectangle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA

from functions import Autoencoder, VariationalAutoencoder, mahalanobis_distance, eval_image_pipeline

print("Packages imported successfully yiho.")

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
# ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
SAMPLE_IMAGE_IDX_TEST = 0  # Index of the sample image to plot process (0-based)
SAMPLE_IMAGE_IDX_TRAIN = 0  # Index of the sample image to plot process (0-based)

DEBUG = False
n_debug = 0.2

grid_size = 32  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)
latent_dim = 6
batch_size = 64

model_name = f"models/vae_model{grid_size}_l{latent_dim}.pth"
if DEBUG:
    model_name = model_name.replace(".pth", "_debug.pth")
print(f"Model to load: {model_name}")

test_tiles_path = os.path.join(ROOT_DIR_C, "test", f"tiles{grid_size}")
test_og_images_path = os.path.join(ROOT_DIR_C, "test", "OG_images")

# images_train = list(Path(train_og_images_path).glob("*.png"))
images_test = list(Path(test_og_images_path).glob("*.png"))
sample_image_path = Path(test_og_images_path) / images_test[SAMPLE_IMAGE_IDX_TEST].name
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

if DEBUG:
    n_test = int(len(images_test) * n_debug)
    # randomly select n_test images from the test set
    images_test = np.random.choice(images_test, size=n_test, replace=False)
    print(f"DEBUG MODE: Using only {n_test} test images ({n_test*grid_size**2} tiles) for evaluation, sample image to plot process: {sample_image_path.name}")

#%% Load model
# Recreate model architecture 
# batch_size = 64
# image_size = 128
# hidden_channels = 32
# latent_dims = 6

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

#%%  Define test dataset and dataloader
test_tiles_obs_path = os.path.join(test_tiles_path, "obs")
test_tiles_no_obs_path = os.path.join(test_tiles_path, "no_obs")

test_tiles_obs = sorted(Path(test_tiles_obs_path).rglob("*.png"))
test_tiles_no_obs = sorted(Path(test_tiles_no_obs_path).rglob("*.png"))

if DEBUG:
    n_test_obs = int(len(test_tiles_obs) * n_debug)
    n_test_no_obs = int(len(test_tiles_no_obs) * n_debug)
    test_tiles_obs = np.random.choice(test_tiles_obs, size=n_test_obs, replace=False)
    test_tiles_no_obs = np.random.choice(test_tiles_no_obs, size=n_test_no_obs, replace=False)
    print(f"DEBUG MODE: Using only {n_test_obs} obs tiles and {n_test_no_obs} no_obs tiles for evaluation")

# Load all test images 

test_loader_obs = DataLoader(
    test_tiles_obs,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_image_pipeline,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

test_loader_no_obs = DataLoader(
    test_tiles_no_obs,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_image_pipeline,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

print("\n" + "="*120)
print(f"Loaded {len(test_tiles_obs)} observation test tiles from: {test_tiles_obs_path}")
print(f"Loaded {len(test_tiles_no_obs)} empty test tiles from: {test_tiles_no_obs_path}")
print("="*120)

#%% Plot sample test tiles from both classes
batch_obs = next(iter(test_loader_obs))
batch_no_obs = next(iter(test_loader_no_obs))

test_tiles_obs = batch_obs[0]
test_tiles_no_obs = batch_no_obs[0]

# Plot observation tiles
N_hor = 8
N_ver = 3
fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= len(test_tiles_obs):
        continue
    tile = test_tiles_obs[idx].cpu().numpy()
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)
    
# Plot non observation tiles
N_hor = 8
N_ver = 3
fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= len(test_tiles_no_obs):
        continue
    tile = test_tiles_no_obs[idx].cpu().numpy()
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

#%% Load all test images and get latent space representations
model.eval()  # Set to evaluation mode

n_batches = len(test_loader_no_obs) + len(test_loader_obs)
mu_list = []
tile_labels = []
tile_rows = []
tile_cols = []
tiles = []

with torch.no_grad():
    for test_loader in [test_loader_obs, test_loader_no_obs]:
        class_name = "obs" if test_loader == test_loader_obs else "no_obs"
        print(f"\n\nProcessing {class_name} test tiles with {len(test_loader.dataset)} tiles and {len(test_loader)} batches...")

        for batch_idx, (images, labels, rows, cols) in enumerate(test_loader):                      
            # encode image to get latent representation
            images = images.to(device)
            mu, _ = model.encode(images, row=rows, col=cols)  # forward pass with row/col info (currently not used in the model, but can be implemented later)
            mu_list.append(mu.cpu().detach())

            # extract image name, tile row, and tile col from the batch paths
            tile_labels.append(labels.cpu().detach())
            tile_rows.append(rows.cpu().detach())
            tile_cols.append(cols.cpu().detach())
            
            # save tiles for later visualization of false positives/negatives
            tiles.append((images.cpu().detach()))

            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}", end="\r")            
            
        print(f"\nProcessing {class_name} test tiles with {len(test_loader.dataset)} tiles and {len(test_loader)} batches... done!", end="\n")
            
print("\nFinished processing all test tiles and extracting latent representations, class distribution in test set:")

# Combine all latent representations and labels into a single array/dataframe
mu_array = torch.cat(mu_list, dim=0).numpy()
tile_labels = torch.cat(tile_labels, dim=0).numpy().astype(bool)
tile_rows = torch.cat(tile_rows, dim=0).numpy()
tile_cols = torch.cat(tile_cols, dim=0).numpy()
tiles = np.concatenate(tiles, axis=0)

# Create dataframe with image names, tile numbers, and latent representations
latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
df_latent = pd.DataFrame(mu_array, columns=latent_cols)
df_latent.insert(0, "label_manual", tile_labels)  # Add manual labels (True for obs, False for no_obs)
df_latent.insert(1, "tile_row", tile_rows)
df_latent.insert(2, "tile_col", tile_cols)
df_latent['tile'] = list(tiles)  # Add tile images for later visualization

# Sanity check
if len(df_latent[df_latent['label_manual']==True]) != len(test_tiles_obs):
    print(f"⚠ Warning: Number of obs tiles in dataframe ({len(df_latent[df_latent['label_manual']==True])}) does not match number of obs tiles loaded ({len(test_tiles_obs)})")

elif len(df_latent[df_latent['label_manual']==False]) != len(test_tiles_no_obs):
    print(f"⚠ Warning: Number of no_obs tiles in dataframe ({len(df_latent[df_latent['label_manual']==False])}) does not match number of no_obs tiles loaded ({len(test_tiles_no_obs)})")

else:
    print("\n" + "="*60)
    print("Sanity check passed: Number of obs and no_obs tiles in dataframe matches number of tiles loaded")
    print("="*60)
    
# Print summary
print(f"Class distribution in test set: obs={len(df_latent[df_latent['label_manual']==True])}, no_obs={len(df_latent[df_latent['label_manual']==False])}")

#%% Calculate observation likelihood based on Mahalanobis distance in full latent space
mu_array= df_latent[latent_cols].values
manual_labels = df_latent["label_manual"].values.astype(bool)  # Convert to boolean: True for obs, False for no_obs

# Seperate into observations on no observations
mask_no_obs = (manual_labels == False)
mask_obs = (manual_labels == True)
mu_array_no_obs = mu_array[mask_no_obs]
mu_array_obs = mu_array[mask_obs]

### Fit Gaussian to no_obs class in 2D latent space
mu_g = mu_array_no_obs.mean(axis=0)
cov_g = np.cov(mu_array_no_obs, rowvar=False) 
cov_inv = np.linalg.inv(cov_g)    

d2 = np.array([mahalanobis_distance(point, mu_g, cov_inv) for point in mu_array])

df_latent["mahalanobis_distance"] = d2

#%% Plot ROC curve for binary classifier
y_true = df_latent["label_manual"].astype(int).values
y_score = df_latent["mahalanobis_distance"].values

fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
target_tpr = 0.8 # e.g., 98% obs found
valid = np.where(tpr >= target_tpr)[0]

### Different treshold options:
# trsh_idx = valid[np.argmin(fpr[valid])]   # lowest FPR at TPR >= target_tpr
# trsh_idx = np.argmin(np.abs(tpr - target_tpr))  # closest to target TPR
trsh_idx = np.argmax(tpr - fpr)  # Youden's J statistic (maximizes TPR-FPR)
# lowest number of wrong obs predictions (false negatives) while still having TPR >= target_tpr

threshold = roc_thresholds[trsh_idx]
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="blue", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
plt.scatter(
    fpr[trsh_idx], tpr[trsh_idx],
    color="red", s=90, zorder=5,
    label=f"threshold={roc_thresholds[trsh_idx]:.3f}\nTPR={tpr[trsh_idx]:.3f}, FPR={fpr[trsh_idx]:.3f}"
)

plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve: Outlier Detection vs Manual Labels", fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.show()

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Selected threshold for Mahalanobis distance: {threshold:.4f}")
print(f"At this threshold, TPR={tpr[trsh_idx]:.3f} and FPR={fpr[trsh_idx]:.3f}")

### Save parameters
model_params = {
    "mu_g": mu_g,
    "cov_g": cov_g,
    "threshold": threshold,
}

val_params_name = model_name.replace(".pth", "_params.pkl")

with open(val_params_name, "wb") as f:
    pkl.dump(model_params, f)

#plot Mahalanobis distance distributions for obs vs no_obs
bins = np.linspace(0, df_latent["mahalanobis_distance"].max(), int(df_latent["mahalanobis_distance"].max()/threshold*4 ))

plt.figure(figsize=(6,4))
plt.hist(df_latent.loc[df_latent.label_manual==0,'mahalanobis_distance'], bins=bins, alpha=0.6, label='no_obs', color='orange')
plt.hist(df_latent.loc[df_latent.label_manual==1,'mahalanobis_distance'], bins=bins, alpha=0.6, label='obs', color='blue')
plt.vlines(threshold, ymin=0, ymax=12000, colors='red', linestyles='dashed', label=f'Threshold (d2={threshold:.1f})')
plt.xlabel("Mahalanobis distance (d2)", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.yscale("log")
plt.ylim(0, 12000)
plt.tight_layout()
plt.grid()
plt.legend(); 
plt.title("Mahalanobis distance distributions manual labels"); plt.show()

# Predict labels based on selected threshold
predicted_labels = d2 > threshold 
df_latent["label_predicted"] = predicted_labels

# Plot confusion matrix
conf_matrix = confusion_matrix(manual_labels, predicted_labels)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["empty", "obs"])
disp.plot(cmap="Blues", values_format="d")
# plt.title("Confusion Matrix: Manual Labels vs Outlier Detection", fontsize=16, fontweight="bold")
plt.show()

print("\nClassification Report:")
print('# correct predictions for obs (true positives):', conf_matrix[1, 1])
print('# correct predictions for empty (true negatives):', conf_matrix[0, 0])
print('# false positives (empty predicted as obs):', conf_matrix[1, 0])
print('# false negatives (obs predicted as empty):', conf_matrix[0, 1])


#%% Compress everything to 2D with PCA for visualization 
if latent_dims > 2:
    pca = PCA(n_components=2)
    mu_array_2d = pca.fit_transform(mu_array)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
else:
    mu_array_2d = mu_array  # If already 2D, just use the original latent representations
    
    # create grid to plot decision boundary
    x_min, x_max = mu_array_2d[:, 0].min() - 1, mu_array_2d[:, 0].max() + 1
    y_min, y_max = mu_array_2d[:, 1].min() - 1, mu_array_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Calculate Mahalanobis distance for each point in the grid
    cov_inv_2d = np.linalg.inv(cov_g[:2, :2])
    d2_grid = np.array([mahalanobis_distance(point, mu_g[:2], cov_inv_2d) for point in grid_points])
    d2_grid = d2_grid.reshape(xx.shape)

### Plot latent space representations colored by manual labels
plt.figure(figsize=(12, 12))
plt.scatter(mu_array_2d[manual_labels, 0], mu_array_2d[manual_labels, 1], s=50, c="blue", label="obs", alpha=0.5, edgecolors='black', linewidth=0.5)
plt.scatter(mu_array_2d[~manual_labels, 0], mu_array_2d[~manual_labels, 1], s=50, c="orange", label="empty", alpha=0.5, edgecolors='black', linewidth=0.5)

# Plot decision boundary contour if latent space is 2D
if latent_dims == 2:
    plt.contour(xx, yy, d2_grid, levels=[threshold], colors="black", linewidths=2, linestyles="--", label=f"Decision boundary (threshold={threshold:.2f})")
plt.xlabel("PCA 1", fontsize=24)
plt.ylabel("PCA 2", fontsize=24)
plt.grid()
plt.legend(loc="best", fontsize=12)
plt.title("Manual labels in 2D latent space", fontsize=20, fontweight="bold")
plt.show()

### Plot latent space representations colored by predicted labels
plt.figure(figsize=(12, 12))
plt.scatter(mu_array_2d[:,0], mu_array_2d[:,1], s=50, c=predicted_labels, label="all tiles", alpha=0.5, edgecolors='black', linewidth=0.5)
plt.scatter(mu_array_2d[predicted_labels, 0], mu_array_2d[predicted_labels, 1], s=50, c="blue", label="obs", alpha=0.5, edgecolors='black', linewidth=0.5)
plt.scatter(mu_array_2d[~predicted_labels, 0], mu_array_2d[~predicted_labels, 1], s=50, c="orange", label="empty", alpha=0.5, edgecolors='black', linewidth=0.5)
plt.xlabel("PCA 1", fontsize=24)
plt.ylabel("PCA 2", fontsize=24)
plt.grid()
plt.legend(loc="best", fontsize=12)
plt.title("Predicted labels in 2D latent space", fontsize=20, fontweight="bold")
plt.show()

df_false_negatives = df_latent[(df_latent["label_manual"] == True) & (df_latent["label_predicted"] == False)]
df_false_positives = df_latent[(df_latent["label_manual"] == False) & (df_latent["label_predicted"] == True)]
df_true_positives = df_latent[(df_latent["label_manual"] == True) & (df_latent["label_predicted"] == True)]
df_true_negatives = df_latent[(df_latent["label_manual"] == False) & (df_latent["label_predicted"] == False)]

#%% Plot false negatives (manualy labeled as obs but classified as no_obs)
print("\nFalse Negatives (obs tiles classified as no_obs):")
# print(df_false_negatives[["image_name", "tile_row", "tile_col", "tile_idx"]])

N_fn = len(df_false_negatives)
N_hor = 8
if N_fn > N_hor * 8:  # If more than 64 false negatives, limit to 64 for visualization
    N_ver = 8 #int(np.ceil(N_fn / N_hor))
    df_false_negatives_plot = df_false_negatives.sample(n=N_hor*N_ver, random_state=42).reset_index(drop=True)
    title_str = f"False Negatives: obs tiles classified as empty (showing random subset of {N_hor*N_ver} out of {N_fn})"
else:
    N_ver = int(np.ceil(N_fn / N_hor))
    df_false_negatives_plot = df_false_negatives
    title_str = "False Negatives: obs tiles classified as empty"

fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()

for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= N_fn:
        continue
    
    tile = df_false_negatives.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

plt.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% Plot false positives (manualy labeled as obs but classified as no_obs)

N_fp = len(df_false_positives)
N_hor = 8
if N_fp > N_hor * 8:  # If more than 64 false positives, limit to 64 for visualization
    N_ver = 8 #int(np.ceil(N_fp / N_hor))
    df_false_positives_plot = df_false_positives.sample(n=N_hor*N_ver, random_state=42).reset_index(drop=True)
    title_str = f"False Positives: empty tiles classified as obs (showing random subset of {N_hor*N_ver} out of {N_fp})"
else:
    N_ver = int(np.ceil(N_fp / N_hor))
    df_false_positives_plot = df_false_positives
    title_str = "False Positives: empty tiles classified as obs"

fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= N_fp:
        continue
    
    tile = df_false_positives.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

plt.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% Plot True positives ranked by Mahalanobis distance (most obs-like to least obs-like)
df_true_positives = df_true_positives.sort_values(by="mahalanobis_distance", ascending=False)

N_tp = len(df_true_positives)
N_hor = 8
N_ver = 1

fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()

for idx, ax in enumerate(axs):
    ax.axis("off")
    tile = df_true_positives.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)
    # ax.set_title(f"d2={df_true_positives.iloc[idx]['mahalanobis_distance']:.2f}", fontsize=8)

# plt.suptitle("True Positives: Highest mahalanobis distances", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# Plot least obs-like true positives (closest to decision boundary)
fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()

for idx, ax in enumerate(axs):
    ax.axis("off")
    tile = df_true_positives.iloc[-(idx+1)]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"d2={df_true_positives.iloc[-(idx+1)]['mahalanobis_distance']:.2f}", fontsize=8)

plt.suptitle("True Positives: lowest mahalanobis distance", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

#%% Plot true negatives ranked by Mahalanobis distance (most empty-like to least empty-like)
df_true_negatives = df_true_negatives.sort_values(by="mahalanobis_distance")
N_tn = len(df_true_negatives)
N_hor = 8
N_ver = 1
fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    tile = df_true_negatives.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)
    # ax.set_title(f"d2={df_true_negatives.iloc[idx]['mahalanobis_distance']:.2f}", fontsize=8)
# plt.suptitle("True Negatives: lowest mahalanobis distance", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
