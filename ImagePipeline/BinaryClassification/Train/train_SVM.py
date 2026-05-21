#%% Train SVM Scorer (with both empty and observation data)
"""
Trains both One-Class and Binary SVM scorers on VAE latent features.
Uses the same data loading and augmentation as train_DNN for consistency.

Usage: python train_SVM.py
Output: models/SVM/*/[oc/binary]_svm_model.pkl
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms import v2
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc, f1_score
)
import pickle as pkl
from pathlib import Path
from typing import cast
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import VariationalAutoencoder, image_pipeline_df, load_tiles_from_paths, preprocess_tile

print("Packages imported successfully.")

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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

print("="*40)
print(f"Using device:      {device}")

#%% Configuration and Hyperparameters

ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"
monitoring_effort = "Kristineberg_251128"
grid_size = 16
tile_size = int(4512/grid_size)
image_size_vae = 128
latent_dim = 32
model_name = f"../models/VAE/{monitoring_effort}_vae_model{grid_size}_l{latent_dim}.pth"

batch_size = 512
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Model: {model_name}")

#%% Load VAE model
checkpoint = torch.load(model_name, map_location=device, weights_only=False)

latent_dims = checkpoint.get("latent_dim")
image_size = checkpoint.get("image_size")
hidden_channels = checkpoint.get("hidden_channels")
grid_size = checkpoint.get("grid_size")

vae_model = VariationalAutoencoder(latent_dims=latent_dims, image_size=image_size,
                                   hidden_channels=hidden_channels, grid_size=grid_size)
vae_model.load_state_dict(checkpoint["model_state_dict"])
vae_model.to(device)
vae_model.eval()

print(f"Loaded VAE model with latent_dim={latent_dims}, grid_size={grid_size}")

#%% Load training data
tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train_DNN", f"tiles{grid_size}")

tiles_path_obs = os.path.join(tiles_path, "obs")
tiles_paths_obs = sorted(Path(tiles_path_obs).rglob("*.png"))

tiles_path_no_obs = os.path.join(tiles_path, "no_obs")
tiles_paths_no_obs = sorted(Path(tiles_path_no_obs).rglob("*.png"))

tiles_paths = tiles_paths_obs + tiles_paths_no_obs

df_train_tiles = load_tiles_from_paths(tiles_paths, include_labels=True)
df_train_tiles["group_id"] = df_train_tiles["path"].astype(str)


#%% Augment observation tiles by including flipped and rotated versions
df_train_tiles_obs = df_train_tiles[df_train_tiles['label'] == 1].copy()

for i in range(len(df_train_tiles_obs)):
    df = df_train_tiles_obs.iloc[[i]].copy()

    df_flipped_hor = df.copy()
    df_flipped_hor['image'] = v2.RandomVerticalFlip
    df_flipped_hor['path'] = df_flipped_hor['path'].apply(lambda p: str(p).replace('.png', '_flipped.png'))
    df_flipped_hor['label'] = 1

    df_flipped_vert = df.copy()
    df_flipped_vert['image'] = df_flipped_vert['image'].apply(lambda img: np.flipud(img))
    df_flipped_vert['path'] = df_flipped_vert['path'].apply(lambda p: str(p).replace('.png', '_flippedvert.png'))
    df_flipped_vert['label'] = 1

    df_rotated_90 = df.copy()
    df_rotated_90['path'] = df_rotated_90['path'].apply(lambda p: str(p).replace('.png', '_rotated.png'))
    df_rotated_90['image'] = df_rotated_90['image'].apply(lambda img: np.rot90(img))
    df_rotated_90['label'] = 1

    df_rotated_180 = df.copy()
    df_rotated_180['path'] = df_rotated_180['path'].apply(lambda p: str(p).replace('.png', '_rotated180.png'))
    df_rotated_180['image'] = df_rotated_180['image'].apply(lambda img: np.rot90(img, k=2))
    df_rotated_180['label'] = 1

    df_rotated_270 = df.copy()
    df_rotated_270['path'] = df_rotated_270['path'].apply(lambda p: str(p).replace('.png', '_rotated270.png'))
    df_rotated_270['image'] = df_rotated_270['image'].apply(lambda img: np.rot90(img, k=3))
    df_rotated_270['label'] = 1

    df_contrast_high = df.copy()
    df_contrast_high['image'] = df_contrast_high['image'].apply(lambda img: np.clip((img - img.mean()) * 1.25 + img.mean(), 0, 1))
    df_contrast_high['path'] = df_contrast_high['path'].apply(lambda p: str(p).replace('.png', '_contrasthigh.png'))
    df_contrast_high['label'] = 1

    df_contrast_low = df.copy()
    df_contrast_low['image'] = df_contrast_low['image'].apply(lambda img: np.clip((img - img.mean()) * 0.75 + img.mean(), 0, 1))
    df_contrast_low['path'] = df_contrast_low['path'].apply(lambda p: str(p).replace('.png', '_contrastlow.png'))
    df_contrast_low['label'] = 1

    df_gamma_high = df.copy()
    df_gamma_high['image'] = df_gamma_high['image'].apply(lambda img: np.clip(img ** 1.25, 0, 1))
    df_gamma_high['path'] = df_gamma_high['path'].apply(lambda p: str(p).replace('.png', '_gammahigh.png'))
    df_gamma_high['label'] = 1

    df_gamma_low = df.copy()
    df_gamma_low['image'] = df_gamma_low['image'].apply(lambda img: np.clip(img ** 0.75, 0, 1))
    df_gamma_low['path'] = df_gamma_low['path'].apply(lambda p: str(p).replace('.png', '_gammalow.png'))
    df_gamma_low['label'] = 1

    df_noise = df.copy()
    df_noise['image'] = df_noise['image'].apply(lambda img: np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1))
    df_noise['path'] = df_noise['path'].apply(lambda p: str(p).replace('.png', '_noise.png'))
    df_noise['label'] = 1

    df_train_tiles = pd.concat([df_train_tiles, df_flipped_hor, df_flipped_vert, df_rotated_90, df_rotated_180, df_rotated_270, df_contrast_high, df_contrast_low, df_gamma_high, df_gamma_low, df_noise], ignore_index=True)

print(f"After augmentation, total tiles: {len(df_train_tiles)} (obs={df_train_tiles['label'].sum():.0f}, empty={len(df_train_tiles) - df_train_tiles['label'].sum():.0f})")

group_ids = df_train_tiles["group_id"].values

images_tensor, rows_tensor, cols_tensor, labels_tensor = cast(
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    image_pipeline_df(df_train_tiles, input_image_size_vae=image_size, include_labels=True),
)
test_dataset = TensorDataset(images_tensor, rows_tensor, cols_tensor, labels_tensor)

dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#%% Extract latent features from VAE
latent_list = []
recon_err_list = []
labels = []
rows_list = []
cols_list = []

print("\nExtracting features from tiles...")
with torch.no_grad():
    for images, batch_rows, batch_cols, batch_labels in dataloader:
        images = images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_rows = batch_rows.to(device, non_blocking=True)
        batch_cols = batch_cols.to(device, non_blocking=True)

        mu, _ = vae_model.encode(images, row=batch_rows, col=batch_cols)
        x_hat = vae_model.decode(mu)
        recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)

        latent_list.append(mu.cpu())
        recon_err_list.append(recon_err.cpu())
        labels.extend(batch_labels.cpu().numpy())
        rows_list.extend(batch_rows.cpu().numpy())
        cols_list.extend(batch_cols.cpu().numpy())

latent_features = torch.cat(latent_list, dim=0).numpy()
recon_err_features = torch.cat(recon_err_list, dim=0).numpy()
labels = np.array(labels, dtype=np.float32)
rows = np.array(rows_list, dtype=np.int64)
cols = np.array(cols_list, dtype=np.int64)
group_ids = np.array(group_ids)

print(f"Total samples: {len(labels)}")
print(f"Observation samples: {labels.sum():.0f}")
print(f"Empty samples: {len(labels) - labels.sum():.0f}")

#%% Split into train/test by original tile group
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.3,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(latent_features, labels, groups=group_ids)
)

X_latent_train = latent_features[train_idx]
X_latent_test = latent_features[test_idx]

X_recon_train = recon_err_features[train_idx]
X_recon_test = recon_err_features[test_idx]

y_train = labels[train_idx]
y_test = labels[test_idx]

print(f"\nTrain set: {len(y_train)} samples (obs={y_train.sum():.0f}, empty={(1-y_train).sum():.0f})")
print(f"Test set: {len(y_test)} samples (obs={y_test.sum():.0f}, empty={(1-y_test).sum():.0f})")

train_groups = set(group_ids[train_idx])
test_groups = set(group_ids[test_idx])
leaked_groups = train_groups.intersection(test_groups)
if len(leaked_groups) > 0:
    print(f"WARNING: Data leakage detected! {len(leaked_groups)} groups are in both train and test sets.")
else:
    print("No data leakage detected. Train and test groups are disjoint.")

#%% Prepare SVM features (latent + reconstruction error)
svm_feature_cols = np.concatenate([X_latent_train, X_recon_train.reshape(-1, 1)], axis=1)
svm_feature_cols_test = np.concatenate([X_latent_test, X_recon_test.reshape(-1, 1)], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(svm_feature_cols)
X_test_scaled = scaler.transform(svm_feature_cols_test)

print(f"\nPrepared SVM features: {X_train_scaled.shape[1]} dimensions")


#%% METHOD 1: One-Class SVM

print("\n" + "="*70)
print("Training One-Class SVM")
print("="*70)

oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
train_mask_no_obs = (y_train == 0)
oc_svm.fit(X_train_scaled[train_mask_no_obs])

y_scores_oc = -oc_svm.decision_function(X_test_scaled)
fpr_oc, tpr_oc, thresholds_oc = roc_curve(y_test, y_scores_oc)
roc_auc_oc = auc(fpr_oc, tpr_oc)

fnr_oc = 1 - tpr_oc
cost_oc = 0.4 * fpr_oc + 0.6 * fnr_oc
threshold_idx_oc = np.argmin(cost_oc)
threshold_oc = thresholds_oc[threshold_idx_oc]

print(f"One-Class SVM AUC: {roc_auc_oc:.4f}")
print(f"Optimal threshold: {threshold_oc:.8f}")

# Precision-Recall curve
precision_oc, recall_oc, _ = precision_recall_curve(y_test, y_scores_oc)
aps_oc = average_precision_score(y_test, y_scores_oc)

# ROC curve plot
plt.figure(figsize=(6, 6))
plt.plot(fpr_oc, tpr_oc, label=f'ROC curve (AUC = {roc_auc_oc:.4f})')
plt.plot(fpr_oc[threshold_idx_oc], tpr_oc[threshold_idx_oc], 'ro', label=f'Threshold = {threshold_oc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - One-Class SVM')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Precision-Recall curve plot
plt.figure(figsize=(6, 6))
plt.plot(recall_oc, precision_oc, label=f'Precision-Recall curve (AP = {aps_oc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - One-Class SVM')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Confusion matrix
y_pred_oc = (y_scores_oc >= threshold_oc).astype(int)
cm_oc = confusion_matrix(y_test, y_pred_oc)
disp_oc = ConfusionMatrixDisplay(confusion_matrix=cm_oc, display_labels=['Empty', 'Observation'])
disp_oc.plot(cmap='Blues')
plt.title('Confusion Matrix - One-Class SVM')
plt.show()

# Save One-Class SVM model
oc_svm_model_name = model_name.replace("VAE/", "SVM/OneClass/").replace("vae_model", "oc_svm_model").replace(".pth", ".pkl")
Path(oc_svm_model_name).parent.mkdir(parents=True, exist_ok=True)
with open(oc_svm_model_name, "wb") as f:
    pkl.dump((oc_svm, scaler, threshold_oc), f)
print(f"\n✓ Saved One-Class SVM model to: {oc_svm_model_name}")
print(f"  AUC: {roc_auc_oc:.4f}")
print(f"  Average Precision: {aps_oc:.4f}")


#%% METHOD 2: Binary SVM

print("\n" + "="*70)
print("Training Binary SVM")
print("="*70)

class_weights = {
    0: (y_train == 1).sum() / (y_train == 0).sum(),
    1: 1.0
}

bin_svm = SVC(
    kernel='rbf',
    class_weight=class_weights,
    probability=True,
    gamma='scale'
)

print(f"Training Binary SVM with class weights: {class_weights}...")
bin_svm.fit(X_train_scaled, y_train)
print(f"Training complete!")

# Get prediction scores
y_scores_bin = bin_svm.decision_function(X_test_scaled)
fpr_bin, tpr_bin, thresholds_bin = roc_curve(y_test, y_scores_bin)
roc_auc_bin = auc(fpr_bin, tpr_bin)

# Precision-Recall curve
precision_bin, recall_bin, thresholds_pr_bin = precision_recall_curve(y_test, y_scores_bin)
aps_bin = average_precision_score(y_test, y_scores_bin)

# Find optimal threshold based on F-scores
f0_25_scores = 1.0625 * (precision_bin[:-1] * recall_bin[:-1]) / (0.0625 * precision_bin[:-1] + recall_bin[:-1] + 1e-8)
f0_5_scores = 1.25 * (precision_bin[:-1] * recall_bin[:-1]) / (0.25 * precision_bin[:-1] + recall_bin[:-1] + 1e-8)
f1_scores = 2 * (precision_bin[:-1] * recall_bin[:-1]) / (precision_bin[:-1] + recall_bin[:-1] + 1e-8)
f2_scores = 5 * (precision_bin[:-1] * recall_bin[:-1]) / (4 * precision_bin[:-1] + recall_bin[:-1] + 1e-8)
f3_scores = 10 * (precision_bin[:-1] * recall_bin[:-1]) / (9 * precision_bin[:-1] + recall_bin[:-1] + 1e-8)

best_idx_f3 = np.argmax(f3_scores)
threshold_bin = thresholds_pr_bin[best_idx_f3]

print(f"\nBinary SVM Results:")
print(f"  ROC AUC: {roc_auc_bin:.4f}")
print(f"  Average Precision: {aps_bin:.4f}")
print(f"  Optimal threshold (F3): {threshold_bin:.4f}")
print(f"  F1 score at threshold: {f1_scores[best_idx_f3]:.4f}")

# ROC curve plot
plt.figure(figsize=(6, 6))
plt.plot(fpr_bin, tpr_bin, label=f'ROC curve (AUC = {roc_auc_bin:.4f})')
x_threshold = fpr_bin[np.argmin(np.abs(thresholds_bin - threshold_bin))]
y_threshold = tpr_bin[np.argmin(np.abs(thresholds_bin - threshold_bin))]
plt.plot(x_threshold, y_threshold, 'ro', label=f'Threshold = {threshold_bin:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Binary SVM')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Precision-Recall curve plot
plt.figure(figsize=(6, 6))
plt.plot(recall_bin, precision_bin, label=f'Precision-Recall curve (AP = {aps_bin:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Binary SVM')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Confusion matrix
y_pred_bin = (y_scores_bin >= threshold_bin).astype(int)
cm_bin = confusion_matrix(y_test, y_pred_bin)
disp_bin = ConfusionMatrixDisplay(confusion_matrix=cm_bin, display_labels=['Empty', 'Observation'])
disp_bin.plot(cmap='Blues')
plt.title('Confusion Matrix - Binary SVM')
plt.show()

# Save Binary SVM model
bin_svm_model_name = model_name.replace("VAE/", "SVM/TwoClass/").replace("vae_model", "binary_svm_model").replace(".pth", ".pkl")
Path(bin_svm_model_name).parent.mkdir(parents=True, exist_ok=True)
checkpoint = {
    'model': bin_svm,
    'scaler': scaler,
    'threshold': threshold_bin,
    'roc_auc': roc_auc_bin,
    'average_precision': aps_bin
}
with open(bin_svm_model_name, "wb") as f:
    pkl.dump(checkpoint, f)
print(f"\n✓ Saved Binary SVM model to: {bin_svm_model_name}")
print(f"  ROC AUC: {roc_auc_bin:.4f}")
print(f"  Average Precision: {aps_bin:.4f}")
print(f"  Threshold: {threshold_bin:.4f}")

print("\n" + "="*70)
print("✓ SVM Training Complete!")
print("="*70)
