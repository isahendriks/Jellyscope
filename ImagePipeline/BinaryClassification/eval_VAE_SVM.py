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
from sklearn.model_selection import train_test_split

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

monitoring_effort = "Kristineberg_250915"  # Used in folder names to load data and save results, e.g. "Kristineberg_251128"
grid_size = 32  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)
latent_dim = 6
batch_size = 64

model_name = f"models/{monitoring_effort}_vae_model{grid_size}_l{latent_dim}.pth"
if DEBUG:
    model_name = model_name.replace(".pth", "_debug.pth")
print(f"Model to load: {model_name}")

test_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "test", f"tiles{grid_size}")
test_og_images_path = os.path.join(ROOT_DIR_C, monitoring_effort, "test", "OG_images")

# images_train = list(Path(train_og_images_path).glob("*.png"))
images_test = list(Path(test_og_images_path).glob("*.png"))
print(len(images_test), "test images found in:", test_og_images_path)
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
recon_errors = []

with torch.no_grad():
    for test_loader in [test_loader_obs, test_loader_no_obs]:
        class_name = "obs" if test_loader == test_loader_obs else "no_obs"
        print(f"\n\nProcessing {class_name} test tiles with {len(test_loader.dataset)} tiles and {len(test_loader)} batches...")

        for batch_idx, (images, labels, rows, cols) in enumerate(test_loader):                      
            # encode image to get latent representation
            images = images.to(device)
            mu, _ = model.encode(images, row=rows, col=cols)  # forward pass with row/col info (currently not used in the model, but can be implemented later)
            mu_list.append(mu.cpu().detach())

            # reconstruction error per tile (mean squared error over pixels)
            x_hat = model(images, row=rows, col=cols)
            recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)
            recon_errors.append(recon_err.cpu().detach())

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
recon_errors = torch.cat(recon_errors, dim=0).numpy()

# Create dataframe with image names, tile numbers, and latent representations
latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
# scale latent space for svm training
  # Use all latent dimensions as features for SVM

df_latent = pd.DataFrame(mu_array, columns=latent_cols)
df_latent.insert(0, "label_manual", tile_labels)  # Add manual labels (True for obs, False for no_obs)
df_latent.insert(1, "tile_row", tile_rows)
df_latent.insert(2, "tile_col", tile_cols)
df_latent.insert(3, "recon_error", recon_errors)
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

#%% Split dataset into one for training the SVM and one for evaluating the SVM performance (e.g. plotting confusion matrix, ROC curve, and visualizing false positives/negatives)
# Hyperparameter search space (small grid)
weight_grid =[0.1, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15]  # Weights to apply to the reconstruction error feature before training the SVM
nu_grid = [0.1, 0.2, 0.3, 0.4, 0.5]  # Nu values to try for the One-Class SVM (controls the fraction of outliers allowed in the training data)

# Fallback defaults (used only if search is skipped/empty)
svm_feature_cols_names = [f"latent_{i}" for i in range(mu_array.shape[1])] + ["recon_error"]  # Column names for SVM features
df_svm_train, df_svm_eval = train_test_split(df_latent, test_size=0.5, stratify=df_latent['label_manual'], random_state=42)

# Scale on train split only (avoid data leakage), then apply reconstruction-error weight
scaler = StandardScaler()
X_train_base = scaler.fit_transform(df_svm_train[svm_feature_cols_names].values)
X_eval_base = scaler.transform(df_svm_eval[svm_feature_cols_names].values)
recon_col_idx = len(svm_feature_cols_names) - 1
y_true = df_svm_eval['label_manual'].values
train_no_obs_mask = (df_svm_train['label_manual'].values == False)

# Small grid search for best (weight_recon_error, nu)
search_results = []
for w in weight_grid:
    for nu_val in nu_grid:
        print(f"\nTesting weight_recon_error={w}, nu={nu_val}...", end="")
        X_train_try = X_train_base.copy()
        X_eval_try = X_eval_base.copy()
        X_train_try[:, recon_col_idx] *= w
        X_eval_try[:, recon_col_idx] *= w

        model_try = OneClassSVM(kernel='rbf', gamma='scale', nu=nu_val)
        model_try.fit(X_train_try[train_no_obs_mask])

        y_scores_try = -model_try.decision_function(X_eval_try)
        fpr_try, tpr_try, _ = roc_curve(y_true, y_scores_try)
        auc_try = auc(fpr_try, tpr_try)
        
        optimal_idx = np.argmax(tpr_try - fpr_try)  # Youden's J statistic to find optimal threshold
        fnr_try = 1 - tpr_try
        fnr = fnr_try[optimal_idx]
        fpr = fpr_try[optimal_idx]
        search_results.append({"weight": w, "nu": nu_val, "auc": auc_try, "fnr": fnr, "fpr": fpr})
        print(f"\rTesting weight_recon_error={w}, nu={nu_val}... done! AUC={auc_try:.4f}, FNR={fnr:.4f}, FPR={fpr:.4f}", end="\r")
        
       
search_df = pd.DataFrame(search_results).sort_values("auc", ascending=False).reset_index(drop=True)
best_row = search_df.iloc[0]
weight_recon_error = float(best_row["weight"])
nu = float(best_row["nu"])

print("\nTop hyperparameter combos (by ROC-AUC):")
print(search_df.head(10))
print(f"\nSelected -> weight_recon_error={weight_recon_error}, nu={nu}, AUC={best_row['auc']:.4f}")

# Build final weighted features with best params
X_train = X_train_base.copy()
X_eval = X_eval_base.copy()
X_train[:, recon_col_idx] *= weight_recon_error
X_eval[:, recon_col_idx] *= weight_recon_error

# Train SVM on the latent space representations
labels_train = df_svm_train['label_manual'].values
latent_cols_train = df_svm_train[latent_cols].values

### train SVM on empty class for anomaly detection
latent_cols_train_no_obs = X_train[train_no_obs_mask]
n_empty = latent_cols_train_no_obs.shape[0]

# Define one-class SVM with RBF kernel and gamma='scale' (default) which uses 1/n_features as gamma value, and nu=0.05 to allow for some outliers in the training data
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu) 
# Train only on no_obs class (inliers)
svm.fit(latent_cols_train_no_obs)

# Add predicted labels to the evaluation dataframe
svm_pred = svm.predict(X_eval)  # +1=inlier(no_obs), -1=outlier(obs)
df_svm_eval['label_predicted'] = (svm_pred == -1)

### plot ROC curve
# OneClassSVM has no predict_proba; use anomaly score for obs likelihood
y_scores = -svm.decision_function(X_eval)

fpr, tpr, thresholds = roc_curve(y_true, y_scores) # Compute ROC curve
roc_auc = auc(fpr, tpr) # Compute AUC

# set treshold based on ROC curve (e.g. Youden's J statistic)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold based on Youden's J statistic: {optimal_threshold:.3f}")

# predict test set labels based on optimal threshold
df_svm_eval['label_predicted_SVM'] = (y_scores >= optimal_threshold)

#%% Save the SVM model, scaler, recon-error weight and score threshold for later use
svm_model_name = model_name.replace("vae_model", "svm_model").replace(".pth", ".pkl")
with open(svm_model_name, "wb") as f:
    pkl.dump((svm, scaler, weight_recon_error, optimal_threshold), f)
print(f"\nSVM model trained and saved to: {svm_model_name}")

# plot ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random classifier line
plt.plot(fpr[optimal_idx], tpr[optimal_idx], marker='o', markersize=8, label=f"Optimal threshold = {optimal_threshold:.3f}", color='red')
plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")
plt.title("ROC Curve for SVM on VAE Latent Space")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# plot normalized confusion matrix with the optimal threshold
y_pred_optimal = (y_scores >= optimal_threshold).astype(bool)
cm = confusion_matrix(df_svm_eval['label_manual'], y_pred_optimal, labels=[False, True], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no_obs", "obs"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for One-Class SVM on VAE Latent Space")


#%% Plot false positives/negatives
# False positives: manual no_obs, predicted obs
df_false_positives = df_svm_eval[(df_svm_eval['label_manual'] == False) & (df_svm_eval['label_predicted_SVM'] == True)]
# False negatives: manual obs, predicted no_obs
df_false_negatives = df_svm_eval[(df_svm_eval['label_manual'] == True) & (df_svm_eval['label_predicted_SVM'] == False)]

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
    
    tile = df_false_positives_plot.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

plt.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% Plot false negatives (manualy labeled as obs but classified as no_obs)
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
    
    tile = df_false_negatives_plot.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

plt.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
# %%
