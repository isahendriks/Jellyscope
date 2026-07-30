#%% Train Neural Network Scorer (with both empty and observation data)
"""
Trains the ObservationScorer in binary mode when BOTH empty and observation data are available.
This is the second stage of progressive learning, after you have collected observation samples.

Usage: python train_scorer_network.py
Output: models/Kristineberg_260424_scorer_binary_model.pth

Before running this:
  1. Train one-class model: python train_scorer_oneclass.py (if only empty data available)
  2. Collect observation samples in Training data/Binary_classifier/Kristineberg_260424/test/tiles16/obs/
  3. Then run this script to upgrade from one-class to binary discrimination
"""

import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import torchvision.transforms as v2
import copy

from typing import cast
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import sys

# import PCA
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import Autoencoder, VariationalAutoencoder, OneClassScorer, TwoClassScorer, image_pipeline_df, load_tiles_from_paths_fast, preprocess_tile, MahalanobisScorer

print("="*70)
print("Training Neural Network Scorer - Binary Mode (Both Classes)")
print("="*70)

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

#%% Configuration and Hyperparameters

# Configuration
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data new\Binary_classifier"   
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data new\Binary_classifier"

monitoring_effort = "Kristineberg_260729"
grid_size = 16
tile_size = int(4512/grid_size)
image_size_vae = 128
latent_dim = 64
offsets_normalized = [0, 0.2, 0.4, 0.6, 0.8]
encoder = "AE"  # Options: "vae" or "ae"
model_name = f"../models/{encoder}/{monitoring_effort}_{encoder}_model{grid_size}_l{latent_dim}_img{image_size_vae}.pth"

# Training parameters
batch_size = 512
learning_rate = 0.001
epochs = 100
train_split = 0.7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Model: {model_name}")

#%% Load encoder model
checkpoint = torch.load(model_name, map_location=device, weights_only=False)

latent_dims = checkpoint.get("latent_dim")
image_size = checkpoint.get("image_size")
hidden_channels = checkpoint.get("hidden_channels")
grid_size = checkpoint.get("grid_size")

if encoder == "VAE":
    encoder_model = VariationalAutoencoder(latent_dims=latent_dims, image_size=image_size, 
                                    hidden_channels=hidden_channels, grid_size=grid_size)
    encoder_model.load_state_dict(checkpoint["model_state_dict"])
    encoder_model.to(device)
    encoder_model.eval()
elif encoder == "AE":
    encoder_model = Autoencoder(latent_dims=latent_dims, image_size=image_size, hidden_channels=hidden_channels, grid_size=grid_size)
    encoder_model.load_state_dict(checkpoint["model_state_dict"])
    encoder_model.to(device)
    encoder_model.eval()
else:
    raise ValueError(f"Invalid encoder type: {encoder}. Must be 'vae' or 'ae'.")

print(f"Loaded {encoder} model with latent_dim={latent_dims}, grid_size={grid_size}")

#%% Load training data
tiles_path = os.path.join(ROOT_DIR_R, monitoring_effort, "train_scorer", f"tiles{grid_size}_offsets{len(offsets_normalized)}")
tiles_path_obs = os.path.join(tiles_path, "obs")
tiles_paths_obs = sorted(Path(tiles_path_obs).rglob("*.png"))

tiles_path_no_obs = os.path.join(tiles_path, "no_obs")
tiles_paths_no_obs = sorted(Path(tiles_path_no_obs).rglob("*.png"))

# Check that we have both observation and empty tiles
# Find unique orginial images in the dataset by extracting the common prefix from tile filenames (remove _row_col.png suffix)
tiles_path_obs_stems = [str(path.stem)[:24] for path in tiles_paths_obs]  # Remove _row_col.png suffix
tiles_path_no_obs_stems = [str(path.stem)[:24] for path in tiles_paths_no_obs]  # Remove _row_col.png suffix
tiles_paths_stems = tiles_path_obs_stems + tiles_path_no_obs_stems
unique_og_imgs = set(tiles_paths_stems)

print(f"Loaded {len(unique_og_imgs)} original images")
print(f"Found {len(tiles_paths_obs)} observation tiles and {len(tiles_paths_no_obs)} empty tiles for training.")

tiles_paths = tiles_paths_obs + tiles_paths_no_obs
df_train_tiles = load_tiles_from_paths_fast(tiles_paths, include_labels=True, to_device = device)  # type: ignore # Load tile paths and labels into a DataFrame

# save the original image as identifier
df_train_tiles["og_img"] = df_train_tiles["path"].apply(lambda x: str(Path(x).stem)[:23])  # Group by original tile (remove _row_col.png suffix)
og_imgs = df_train_tiles["og_img"].values

# Apply image pipeline to load and preprocess tile images, and convert to tensors for training
images_tensor, rows_tensor, cols_tensor, labels_tensor = cast(
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    image_pipeline_df(df_train_tiles, input_image_size_vae=image_size, include_labels=True),
)

#%% Create TensorDataset
dataset = TensorDataset(images_tensor, rows_tensor, cols_tensor, labels_tensor)  # Convert to TensorDataset for DataLoader

# Split twice using GroupShuffleSplit to ensure no data leakage between train/val/test sets based on original image groups (og_img)
X = np.arange(len(dataset))
og_imgs = df_train_tiles["og_img"].values  # Group labels for group-aware splitting

gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(X, groups=og_imgs))

temp_groups = og_imgs[temp_idx]
gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_rel_idx, test_rel_idx = next(gss2.split(X[temp_idx], groups=temp_groups))                                 
                                 
val_idx = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

train_dataset_imgs = torch.utils.data.Subset(dataset, train_idx)
val_dataset_imgs = torch.utils.data.Subset(dataset, val_idx)
test_dataset_imgs = torch.utils.data.Subset(dataset, test_idx)

# Create DataLoaders
dataloader_imgs_val = DataLoader(val_dataset_imgs, batch_size=batch_size, shuffle=False, num_workers=0)
dataloader_imgs_test = DataLoader(test_dataset_imgs, batch_size=batch_size, shuffle=False, num_workers=0)
dataloader_imgs_train = DataLoader(train_dataset_imgs, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"\nDataset splits before augmentation:")
print(f"  Train: {len(train_dataset_imgs)} samples (obs={labels_tensor[train_idx].sum().item():.0f}, empty={len(train_dataset_imgs) - labels_tensor[train_idx].sum().item():.0f})")
print(f"  Val:   {len(val_dataset_imgs)} samples (obs={labels_tensor[val_idx].sum().item():.0f}, empty={len(val_dataset_imgs) - labels_tensor[val_idx].sum().item():.0f})")
print(f"  Test:  {len(test_dataset_imgs)} samples (obs={labels_tensor[test_idx].sum().item():.0f}, empty={len(test_dataset_imgs) - labels_tensor[test_idx].sum().item():.0f})")

AUGMENTATION = True

if AUGMENTATION:
    print("\nData augmentation is enabled. Augmenting only observation samples in the training set...")

    # Augment the trainingdata 
    augmented_images_all = []
    augmented_rows_all = []
    augmented_cols_all = []
    augmented_labels_all = []

    for idx, batch in enumerate(dataloader_imgs_train):
        print(f"Processing batch {idx+1}/{len(dataloader_imgs_train)} for augmentation...", end="\r")
        images, batch_rows, batch_cols, batch_labels = batch
        
        # select only observations for augmentation
        obs_mask = batch_labels == 1
        images = images[obs_mask]
        batch_rows = batch_rows[obs_mask]
        batch_cols = batch_cols[obs_mask]
        batch_labels = batch_labels[obs_mask]
        
        if images.shape[0] == 0:
            print(f"No observation samples in batch {idx+1}, skipping augmentation.")
            continue
        # else:
            # print(f"Augmenting {images.shape[0]} observation samples in batch {idx+1}...", end="\n")
        
        ### Augment by deformations 
        images_eldeform = v2.ElasticTransform(alpha=1.0, sigma=0.25)(images)

        batch_rows_eldeform = batch_rows
        batch_cols_eldeform = batch_cols  # Update column indices for deformed images
        batch_labels_eldeform = 1 * torch.ones_like(batch_labels)  # Observations remain labeled as 1 after deformation
        
        # Add deformed samples to the training dataset
        augmented_images = torch.cat([images, images_eldeform], dim=0)
        augmented_rows = torch.cat([batch_rows, batch_rows_eldeform], dim=0)
        augmented_cols = torch.cat([batch_cols, batch_cols_eldeform], dim=0)
        augmented_labels = torch.cat([batch_labels, batch_labels_eldeform], dim=0)
        
        ### Augment by adding Gaussian noise
        images_guassian = v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))(images)
        batch_rows_gaussian = batch_rows
        batch_cols_gaussian = batch_cols  # Update column indices for Gaussian noise images
        batch_labels_gaussian = 1 * torch.ones_like(batch_labels)  # Observations remain labeled as 1 after adding noise
        
        augmented_images = torch.cat([augmented_images, images_guassian], dim=0)
        augmented_rows = torch.cat([augmented_rows, batch_rows_gaussian], dim=0)
        augmented_cols = torch.cat([augmented_cols, batch_cols_gaussian], dim=0)
        augmented_labels = torch.cat([augmented_labels, batch_labels_gaussian], dim=0)
        
        ### Augment by rotations
        images_rot90 = torch.rot90(images, k=1, dims=[2,3])  # Rotate 90 degrees
        batch_rows_rot90 = batch_rows
        batch_cols_rot90 = batch_cols  # Update column indices for rotated images
        batch_labels_rot90 = 1 * torch.ones_like(batch_labels)  # Observations remain labeled as 1 after rotation
        augmented_images = torch.cat([augmented_images, images_rot90], dim=0)
        augmented_rows = torch.cat([augmented_rows, batch_rows_rot90], dim=0)
        augmented_cols = torch.cat([augmented_cols, batch_cols_rot90], dim=0)
        augmented_labels = torch.cat([augmented_labels, batch_labels_rot90], dim=0)
        
        images_rot180 = torch.rot90(images, k=2, dims=[2,3])  # Rotate 180 degrees
        batch_rows_rot180 = batch_rows
        batch_cols_rot180 = batch_cols  # Update column indices for rotated images
        batch_labels_rot180 = 1 * torch.ones_like(batch_labels)  # Observations remain labeled as 1 after rotation
        augmented_images = torch.cat([augmented_images, images_rot180], dim=0)
        augmented_rows = torch.cat([augmented_rows, batch_rows_rot180], dim=0)
        augmented_cols = torch.cat([augmented_cols, batch_cols_rot180], dim=0)
        augmented_labels = torch.cat([augmented_labels, batch_labels_rot180], dim=0)
        
        images_rot270 = torch.rot90(images, k=3, dims=[2,3])  # Rotate 270 degrees
        batch_rows_rot270 = batch_rows
        batch_cols_rot270 = batch_cols  # Update column indices for rotated images
        batch_labels_rot270 = 1 * torch.ones_like(batch_labels)  # Observations remain labeled as 1 after rotation
        augmented_images = torch.cat([augmented_images, images_rot270], dim=0)
        augmented_rows = torch.cat([augmented_rows, batch_rows_rot270], dim=0)
        augmented_cols = torch.cat([augmented_cols, batch_cols_rot270], dim=0)
        augmented_labels = torch.cat([augmented_labels, batch_labels_rot270], dim=0)

        # Create a new TensorDataset with the augmented data
        train_dataset_augmented = TensorDataset(augmented_images, augmented_rows, augmented_cols, augmented_labels)

        augmented_images_all.append(augmented_images)
        augmented_rows_all.append(augmented_rows)
        augmented_cols_all.append(augmented_cols)
        augmented_labels_all.append(augmented_labels)

    if augmented_images_all:
        train_dataset_augmented = TensorDataset(
            torch.cat(augmented_images_all, dim=0),
            torch.cat(augmented_rows_all, dim=0),
            torch.cat(augmented_cols_all, dim=0),
            torch.cat(augmented_labels_all, dim=0),
        )
    else:
        train_dataset_augmented = TensorDataset(
            torch.empty((0, images_tensor.shape[1], images_tensor.shape[2], images_tensor.shape[3])),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )
        
    # Add augmented data to the original training dataset
    train_dataset_imgs = torch.utils.data.ConcatDataset([train_dataset_imgs, train_dataset_augmented])
    dataloader_imgs_train = DataLoader(train_dataset_imgs, batch_size=batch_size, shuffle=True, num_workers=0)

    train_obs_count = int(labels_tensor[train_idx].sum().item()) + int(torch.cat(augmented_labels_all).sum().item()) if augmented_labels_all else int(labels_tensor[train_idx].sum().item())
    train_empty_count = len(train_dataset_imgs) - train_obs_count

    # Print dataset statistics
    print(f"\nDataset splits after augmentation:")
    print(f"  Train: {len(train_dataset_imgs)} samples (obs={train_obs_count:.0f}, empty={train_empty_count:.0f})")
    print(f"  Val:   {len(val_dataset_imgs)} samples (obs={labels_tensor[val_idx].sum().item():.0f}, empty={len(val_dataset_imgs) - labels_tensor[val_idx].sum().item():.0f})")
    print(f"  Test:  {len(test_dataset_imgs)} samples (obs={labels_tensor[test_idx].sum().item():.0f}, empty={len(test_dataset_imgs) - labels_tensor[test_idx].sum().item():.0f})")


#%% Encode image to obtain latentt features and embed position information
dataloaders_imgs = {
    "train": dataloader_imgs_train,
    "val": dataloader_imgs_val,
    "test": dataloader_imgs_test
}

datasets_latent = {}

print("\nExtracting features from tiles...")
with torch.no_grad():
    for split_name, dataloader_img in dataloaders_imgs.items():
        latent_list = []
        recon_err_list = []
        labels = []
        rows_list = []
        cols_list = []
        
        print(f"  Processing {split_name} set with {len(dataloader_img.dataset)} samples...")
        
        for images, batch_rows, batch_cols, batch_labels in dataloader_img:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_rows = batch_rows.to(device, non_blocking=True)
            batch_cols = batch_cols.to(device, non_blocking=True)
            
            if encoder == "VAE":
                mu, _ = encoder_model.encode(images, row=batch_rows, col=batch_cols)

            elif encoder == "AE":
                mu = encoder_model.encode(images, row=batch_rows, col=batch_cols)
                
            else:
                raise ValueError(f"Invalid encoder type: {encoder}. Must be 'vae' or 'ae'.")
            
            x_hat = encoder_model.decode(mu)
            recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)
                
            latent_list.append(mu.cpu())
            recon_err_list.append(recon_err.cpu())
            labels.extend(batch_labels.cpu().numpy().tolist())
            rows_list.extend(batch_rows.cpu().numpy().tolist())
            cols_list.extend(batch_cols.cpu().numpy().tolist())

        # Concatenate all features
        latent_features = torch.cat(latent_list, dim=0).numpy()
        recon_err_features = torch.cat(recon_err_list, dim=0).numpy()
        labels_array = np.array(labels, dtype=np.float32)
        rows_array = np.array(rows_list, dtype=np.float32)
        cols_array = np.array(cols_list, dtype=np.float32)
        
        dataset = TensorDataset(
            torch.from_numpy(latent_features).float(),
            torch.from_numpy(recon_err_features).float(),
            torch.from_numpy(rows_array).float(),
            torch.from_numpy(cols_array).float(),
            torch.from_numpy(labels_array).float()
        )
        
        datasets_latent[split_name] = dataset

train_dataset_lat = datasets_latent["train"]
val_dataset_lat = datasets_latent["val"]
test_dataset_lat = datasets_latent["test"]


### Define oneclass sets for training the one-class model (only empty samples)
# Only empty samples (label=0) for one-class training
oneclass_mask_train = train_dataset_lat.tensors[4] == 0
train_dataset_oneclass = TensorDataset(
    train_dataset_lat.tensors[0][oneclass_mask_train],
    train_dataset_lat.tensors[1][oneclass_mask_train],
    train_dataset_lat.tensors[2][oneclass_mask_train],
    train_dataset_lat.tensors[3][oneclass_mask_train],
    torch.ones(int(oneclass_mask_train.sum().item()), dtype=torch.float32),
)

# Also filter test set to only empty samples for one-class evaluation
oneclass_mask_test = test_dataset_lat.tensors[4] == 0
val_dataset_oneclass = TensorDataset(
    test_dataset_lat.tensors[0][oneclass_mask_test],
    test_dataset_lat.tensors[1][oneclass_mask_test],
    test_dataset_lat.tensors[2][oneclass_mask_test],
    test_dataset_lat.tensors[3][oneclass_mask_test],
    torch.ones(int(oneclass_mask_test.sum().item()), dtype=torch.float32),
)

#%% Check for data leakage by comparing group IDs in train/val/test splits

train_groups = set(og_imgs[train_idx])
val_groups = set(og_imgs[val_idx])
test_groups = set(og_imgs[test_idx])
leaked_groups = train_groups.intersection(test_groups)

if len(leaked_groups) > 0:
    print(f"WARNING: Data leakage detected! {len(leaked_groups)} groups are in both train and test sets.")
else:
    print("No data leakage detected. Train and test groups are disjoint.")


#%% Train in one class mode
scorer_oc = OneClassScorer(latent_dim=latent_dims, hidden_dim=64, emb_dim= 4, grid_size=grid_size)
scorer_oc.to(device)

train_loader_oneclass = DataLoader(train_dataset_oneclass, batch_size=2048, shuffle=True)
val_loader_oneclass = DataLoader(val_dataset_oneclass, batch_size=2048, shuffle=False)

print(f"\nOne-Class Training dataset: {len(train_loader_oneclass.dataset)} samples (all empty)")
print(f"One-Class Validation dataset: {len(val_loader_oneclass.dataset)} samples (all empty)")

# Loss and optimizer
optimizer = optim.Adam(scorer_oc.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Compute center of empty samples in latent space (for Deep SVDD)
scorer_oc.eval()
zs = []

with torch.no_grad():
    for X_latent, X_recon, rows, cols, y in train_loader_oneclass:
        X_latent = X_latent.to(device)
        X_recon = X_recon.to(device)
        rows = rows.to(device)
        cols = cols.to(device)

        z = scorer_oc(X_latent, X_recon, rows, cols)
        zs.append(z)

center = torch.cat(zs, dim=0).mean(dim=0)

train_losses = []
test_losses = []

print(f"Training for {epochs} epochs...\n")
for epoch in range(epochs):
    # Training
       
    if epoch == 10:
        scorer_oc.eval()
        zs = []

        with torch.no_grad():
            for X_latent, X_recon, rows, cols, y in train_loader_oneclass:
                X_latent = X_latent.to(device)
                X_recon = X_recon.to(device)
                rows = rows.to(device)
                cols = cols.to(device)

                z = scorer_oc(X_latent, X_recon, rows, cols)
                zs.append(z)

        center = torch.cat(zs, dim=0).mean(dim=0)

    scorer_oc.train()
    train_loss = 0.0
    for X_latent, X_recon, rows, cols, y in train_loader_oneclass:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # All 1.0 (normal)
        
        optimizer.zero_grad()
        
        z = scorer_oc(X_latent, X_recon, rows, cols)
        loss = ((z - center) ** 2).sum(dim=1).mean() # MSE loss to push empty samples towards 1.0
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader_oneclass)
    train_losses.append(train_loss)
    
    # Testing (on empty samples only for one-class mode)
    scorer_oc.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_latent, X_recon, rows, cols, y in val_loader_oneclass:
            X_latent = X_latent.to(device, non_blocking=True)
            X_recon = X_recon.to(device, non_blocking=True)
            rows = rows.to(device, non_blocking=True)
            cols = cols.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            z = scorer_oc(X_latent, X_recon, rows, cols)
            loss = ((z - center) ** 2).sum(dim=1).mean()  # MSE loss to push empty samples towards 1.0
            test_loss += loss.item()
    
    test_loss /= len(val_loader_oneclass)
    test_losses.append(test_loss)
    
    scheduler.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.6f}, Test Loss={test_loss:.4f}")

print(f"\n✓ Training complete!")

# Calculate mean probability for empty tiles 

scorer_oc.eval()
all_scores = []
with torch.no_grad():
    for X_latent, X_recon, rows, cols, y in val_loader_oneclass:
        X_latent = X_latent.to(device)
        X_recon = X_recon.to(device)
        rows = rows.to(device)
        cols = cols.to(device)

        z = scorer_oc(X_latent, X_recon, rows, cols)
        scores = ((z - center) ** 2).sum(dim=1)

        all_scores.append(scores.cpu().numpy())

all_scores = np.concatenate(all_scores)

mean_empty_score = all_scores.mean()
std_empty_score = all_scores.std()

threshold_oneclass = np.quantile(all_scores, 0.99)

print(f"\nOne-Class Model Statistics on empty test set:")
print(f"  Mean anomaly score = {mean_empty_score:.4f}")
print(f"  Std anomaly score  = {std_empty_score:.4f}")
print(f"  Min anomaly score  = {all_scores.min():.4f}")
print(f"  Max anomaly score  = {all_scores.max():.4f}")
print(f"  99% threshold      = {threshold_oneclass:.4f}")
print("  Scores above threshold = likely observation/anomaly")

# Recommended threshold for anomaly detection
print(f"\nRecommended threshold for anomaly detection: {threshold_oneclass:.4f}")
print(f"  (Observations < threshold = likely observation, > threshold = likely empty)")

# Plot training curve
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scorer_training.png', dpi=150)
print("Saved training curves to scorer_training.png")
plt.show()

#%% DNN Scorer - Binary Mode (with both empty and observation data)
scorer_bin = TwoClassScorer(latent_dim=latent_dims, hidden_dim=256, dropout=0.3, grid_size=grid_size)
scorer_bin.to(device)


train_losses = []
val_losses = []
val_accuracies = []
val_aps = []

### Initialize checkpoints
best_val_ap = -np.inf
best_model_state = None
best_epoch = None
best_threshold = None

# Balance training dataset by subsampling the empty class to match the number of observation samples
num_obs = train_dataset_lat.tensors[4].sum().item()

obs_mask = train_dataset_lat.tensors[4] == 1
empty_mask = train_dataset_lat.tensors[4] == 0
empty_indices = torch.where(empty_mask)[0]

BALANCE = False

if num_obs > 0 and len(empty_indices) > num_obs and BALANCE:
    empty_indices_subsampled = np.random.choice(empty_indices.cpu().numpy(), size=int(num_obs), replace=False)
    combined_indices = torch.cat([torch.where(obs_mask)[0], torch.from_numpy(empty_indices_subsampled)])
    combined_indices_list = combined_indices.cpu().tolist()
    train_dataset_lat_subset = torch.utils.data.Subset(train_dataset_lat, combined_indices_list)
    
    balanced_labels = train_dataset_lat.tensors[4][combined_indices]
    n_obs = balanced_labels.sum().item()
    n_empty = len(combined_indices) - n_obs

    print(f"\nBalanced training dataset: {len(train_dataset_lat_subset)} samples (obs={n_obs:.0f}, empty={n_empty:.0f})")
    train_loader = DataLoader(train_dataset_lat_subset, batch_size=batch_size, shuffle=True, num_workers=0)

else:
    n_obs = train_dataset_lat.tensors[4].sum().item()
    n_empty = len(train_dataset_lat) - n_obs
    print(f"\nUnbalanced training dataset: {len(train_dataset_lat)} samples (obs={n_obs:.0f}, empty={n_empty:.0f})")

    train_loader = DataLoader(train_dataset_lat, batch_size=batch_size, shuffle=True, num_workers=0)

# Loss and optimizer
pos_weight = torch.tensor(n_empty/n_obs if num_obs > 0 else 1.0)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(scorer_bin.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)

val_loader = DataLoader(val_dataset_lat, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"\nTraining binary model for {epochs} epochs...")
for epoch in range(epochs):
    # Training
    scorer_bin.train()
    train_loss = 0.0
    for X_latent, X_recon, rows, cols, y in train_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Combine features
        recon_err_expanded = X_recon.unsqueeze(1) if X_recon.dim() == 1 else X_recon.unsqueeze(1)
        
        # Get predictions
        logits = scorer_bin(X_latent, X_recon, rows, cols)
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Testing
    scorer_bin.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    val_scores = []
    val_labels = []

    with torch.no_grad():
        for X_latent, X_recon, rows, cols, y in val_loader:
            X_latent = X_latent.to(device, non_blocking=True)
            X_recon = X_recon.to(device, non_blocking=True)
            rows = rows.to(device, non_blocking=True)
            cols = cols.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = scorer_bin(X_latent, X_recon, rows, cols)
            loss = criterion(logits, y)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            val_scores.append(probs.cpu().numpy())
            val_labels.append(y.cpu().numpy())

            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    accuracy = correct / total
    val_accuracies.append(accuracy)

    val_scores = np.concatenate(val_scores)
    val_labels = np.concatenate(val_labels)

    val_ap = average_precision_score(val_labels, val_scores)
    val_aps.append(val_ap)
    
    global_epoch = len(val_aps) - 1

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_epoch = global_epoch
        best_model_state = copy.deepcopy(scorer_bin.state_dict())
    
    scheduler.step(val_ap)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Acc={accuracy:.4f}, Val AP={val_ap:.4f}, best model:  val AP={best_val_ap:.4f}, val acc={accuracy:.4f}")

print(f"\nInitial training complete! Starting Hard negative mining...")
### HARD NEGATIVE MINING
# Number of positives in training set
labels_train = train_dataset_lat.tensors[4]
positive_indices = torch.where(labels_train == 1)[0]
empty_indices = torch.where(labels_train == 0)[0]

n_pos = len(positive_indices)

# Choose how many hard/easy negatives to include
n_hard_neg = min(n_pos, len(empty_indices))
n_easy_neg = min(n_pos, len(empty_indices))

scorer_bin.eval()

empty_probs = []
empty_dataset_indices = []

mine_loader = DataLoader(train_dataset_lat, batch_size=batch_size, shuffle=False)

start_idx = 0

with torch.no_grad():
    for X_latent, X_recon, rows, cols, y in mine_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)

        logits = scorer_bin(X_latent, X_recon, rows, cols)
        probs = torch.sigmoid(logits).cpu()

        batch_size_now = y.size(0)
        batch_indices = torch.arange(start_idx, start_idx + batch_size_now)
        start_idx += batch_size_now

        empty_mask = y == 0

        empty_probs.append(probs[empty_mask])
        empty_dataset_indices.append(batch_indices[empty_mask])

empty_probs = torch.cat(empty_probs)
empty_dataset_indices = torch.cat(empty_dataset_indices)

# Select empty tiles with highest predicted observation probability
topk = torch.topk(empty_probs, k=n_hard_neg)
hard_negative_indices = empty_dataset_indices[topk.indices]

# Also include random easy negatives so the model does not forget normal empty tiles
remaining_empty_indices = empty_indices[~torch.isin(empty_indices, hard_negative_indices)]
easy_negative_indices = remaining_empty_indices[torch.randperm(len(remaining_empty_indices))[:n_easy_neg]]

# Final mined dataset: positives + hard negatives + random easy negatives
mined_indices = torch.cat([positive_indices,
                            hard_negative_indices,
                            easy_negative_indices])

mined_dataset = torch.utils.data.Subset(train_dataset_lat, mined_indices.cpu().tolist())

mined_loader = DataLoader(mined_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0)

print(f"Hard-negative mining dataset:")
print(f"  Positives:       {len(positive_indices)}")
print(f"  Hard negatives:  {len(hard_negative_indices)}")
print(f"  Easy negatives:  {len(easy_negative_indices)}")
print(f"  Total:           {len(mined_dataset)}")
print(f"  Mean hard negative prob: {empty_probs[topk.indices].mean():.4f}")
print(f"  Max hard negative prob:  {empty_probs[topk.indices].max():.4f}")

### Retrain model on mined dataset
print(f" Retraining binary model on mined dataset...")
scorer_bin.train()

mined_labels = labels_train[mined_indices]
n_obs = mined_labels.sum().item()
n_empty = len(mined_labels) - n_obs
pos_weight = torch.tensor([n_empty / n_obs], dtype=torch.float32, device=device)

print(f"  Mined dataset class balance: obs={n_obs:.0f}, empty={n_empty:.0f}, pos_weight={pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(scorer_bin.parameters(), lr=learning_rate * 0.1)  # Lower learning rate for fine-tuning
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)

for epoch in range(epochs):  # Fewer epochs for fine-tuning
    scorer_bin.train()
    train_loss = 0.0
    for X_latent, X_recon, rows, cols, y in mined_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        logits = scorer_bin(X_latent, X_recon, rows, cols)
        
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    
    train_loss /= len(mined_loader)
    train_losses.append(train_loss)
    
    ### Testing on original validation set to check for overfitting
    # Testing
    scorer_bin.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    val_scores = []
    val_labels = []

    with torch.no_grad():
        for X_latent, X_recon, rows, cols, y in val_loader:
            X_latent = X_latent.to(device, non_blocking=True)
            X_recon = X_recon.to(device, non_blocking=True)
            rows = rows.to(device, non_blocking=True)
            cols = cols.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = scorer_bin(X_latent, X_recon, rows, cols)
            loss = criterion(logits, y)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            val_scores.append(probs.cpu().numpy())
            val_labels.append(y.cpu().numpy())

            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    accuracy = correct / total
    val_accuracies.append(accuracy)

    val_scores = np.concatenate(val_scores)
    val_labels = np.concatenate(val_labels)

    val_ap = average_precision_score(val_labels, val_scores)
    val_aps.append(val_ap)

    global_epoch = len(val_aps) - 1

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_epoch = global_epoch
        best_model_state = copy.deepcopy(scorer_bin.state_dict())

        print(f"New best model: epoch {best_epoch+1}, val AP={best_val_ap:.4f}, val acc={accuracy:.4f}")
    
    scheduler.step(val_ap)
        
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={accuracy:.4f}, val AP={val_ap:.4f}")
    
print(f"\n Binary Model Statistics (on empty test set):")

scorer_bin.load_state_dict(best_model_state)

print(f"Restored best model from epoch {best_epoch+1}")
print(f"Best validation AP: {best_val_ap:.4f}")

# Plot training curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(val_accuracies, label='Val Accuracy')
ax2.plot(val_aps, label='Val AP')
# Plot where the best model was selected
ax2.scatter(best_epoch, val_accuracies[best_epoch], color='red', label='Best Model', zorder=5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scorer_training.png', dpi=150)
print("Saved training curves to scorer_training.png")
plt.show()

#%% Mahalanobis distance scorer with torch implementation for fast distance calculation in latent space

# calculate covariance matrice of empty samples in latent space for Mahalanobis distance
# scaler = StandardScaler()

train_maha_latent = train_dataset_oneclass.tensors[0]  # Get latent features of empty samples
X_latent = train_maha_latent

cov = np.cov(X_latent, rowvar=False)
cov_inv = np.linalg.inv(cov)

print(f"\nMahalanobis covariance fit on {X_latent.shape[0]} empty samples, latent_dim={X_latent.shape[1]}")
print(f"  Covariance condition number: {np.linalg.cond(cov):.2e} (>1e6 indicates an unstable/overfit inverse)")

cov = torch.tensor(cov, dtype=torch.float32).to(device)
cov_inv = torch.tensor(cov_inv, dtype=torch.float32).to(device)
mean_vec = torch.mean(X_latent, dim=0).to(device)

# Calibrate the empirical-CDF scoring against a held-out empty split (val_dataset_oneclass,
# drawn from different original images than the ones used to fit mean_vec/cov_inv above) so
# the reference distribution isn't biased by points always looking "close" to their own fit.
calib_latent = val_dataset_oneclass.tensors[0].to(device)
calib_recon = val_dataset_oneclass.tensors[1].to(device)

diff = calib_latent - mean_vec
dist_calib = torch.sqrt(torch.clamp(torch.sum((diff @ cov_inv) * diff, dim=1), min=0))

print(f"Calibration set: {calib_latent.shape[0]} held-out empty samples")

scorer_maha = MahalanobisScorer(
    cov_inv=cov_inv,
    mean_vec=mean_vec,
    dist_calib=dist_calib,
    recon_calib=calib_recon,
)

#%% EVALUATION: Find optimal threshold and plot ROC curve and confusion matrix
test_loader = DataLoader(test_dataset_lat, batch_size=batch_size, shuffle=False, num_workers=0)

n_obs = test_dataset_lat.tensors[4].sum().item()
n_empty = len(test_loader.dataset) - n_obs

print(f"\nEvaluating on test set with {len(test_loader.dataset)} samples, n_obs={n_obs:.0f}, n_empty={n_empty:.0f}")

mode = "mahalanobis"  # Change to "one_class" if evaluating one-class model

if mode == "one_class":
    scorer = scorer_oc
    scorer.eval()
elif mode == "binary":
    scorer = scorer_bin
    scorer.eval()
elif mode == "mahalanobis":
    scorer = scorer_maha
else:
    raise ValueError(f"Invalid mode: {mode}. Must be 'one_class', 'binary', or 'mahalanobis'.")

scorer.to(device)
# Run all test data through the loader
with torch.no_grad():
    all_scores = []
    all_labels = []
    all_recon = []
    for X_latent, X_recon, rows, cols, y in test_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        # y = y.to(device, non_blocking=True)

        if mode == "one_class":
            z = scorer(X_latent, X_recon, rows, cols)
            scores = ((z - center) ** 2).sum(dim=1)
        elif mode == "binary":
            probs = scorer.predict_batch(X_latent, X_recon, rows, cols)
            scores = probs
        elif mode == "mahalanobis":
            probs = scorer.predict_batch(X_latent, X_recon, rows, cols)
            scores = probs
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'one_class' or 'binary'")

        all_scores.append(scores.cpu().numpy())
        all_labels.append(y.numpy())
        all_recon.append(X_recon.cpu().numpy())

# all_scores = np.concatenate(all_scores)
all_labels = np.concatenate(all_labels)
all_recon = np.concatenate(all_recon)
all_scores = np.concatenate(all_scores)  # Combine model score and reconstruction error for evaluation

if n_obs == 0:
    # No observation samples in the test set: classification metrics (ROC/PR/confusion
    # matrix) are undefined without a positive class, so fall back to a one-class threshold
    # chosen from the empty-sample score distribution instead (e.g. "flag the top 1% most
    # anomalous empty tiles").
    print(
        "\nNo observation samples in test set (n_obs=0): skipping ROC/PR/confusion-matrix "
        "evaluation since those metrics require a positive class.\n"
        "Falling back to one-class thresholding based on the empty-tile score distribution."
    )

    avg_precision = float('nan')

    percentile = 99.0
    best_threshold = float(np.percentile(all_scores, percentile))

    print(f"\nEmpty-tile score distribution:")
    print(f"  mean = {all_scores.mean():.4f}, std = {all_scores.std():.4f}")
    print(f"  min = {all_scores.min():.4f}, max = {all_scores.max():.4f}")
    print(f"\nRecommended threshold ({percentile:.0f}th percentile of empty scores): {best_threshold:.4f}")
    print("  (Tiles scoring above this are flagged as anomalies/observations)")

    plt.figure(figsize=(6, 6))
    plt.hist(all_scores, bins=50, alpha=0.7, label='Empty tiles')
    plt.axvline(best_threshold, color='r', linestyle='--', label=f'Threshold = {best_threshold:.4f}')
    plt.xlabel('Anomaly score')
    plt.ylabel('Count')
    plt.title('Score Distribution on Empty Tiles (one-class evaluation)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
else:
    # Plot precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    avg_precision = average_precision_score(all_labels, all_scores)

    # calculate threshold based on optimized F-scores
    # Note: precision and recall have length n_thresholds+1, thresholds has length n_thresholds
    # So we only use the first n_thresholds elements of precision/recall
    f0_25_scores = 1.0625 * (precision[:-1] * recall[:-1]) / (0.0625 * precision[:-1] + recall[:-1] + 1e-8)
    f0_5_scores = 1.25 * (precision[:-1] * recall[:-1]) / (0.25 * precision[:-1] + recall[:-1] + 1e-8)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    f2_scores = 5 * (precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-8)
    f3_scores = 10 * (precision[:-1] * recall[:-1]) / (9 * precision[:-1] + recall[:-1] + 1e-8)

    # Calculate best threshold for each F-score
    best_idx_f0_25 = np.argmax(f0_25_scores)
    best_idx_f0_5 = np.argmax(f0_5_scores)
    best_idx_f1 = np.argmax(f1_scores)
    best_idx_f2 = np.argmax(f2_scores)
    best_idx_f3 = np.argmax(f3_scores)

    best_threshold_f0_25 = thresholds[best_idx_f0_25]
    best_threshold_f0_5 = thresholds[best_idx_f0_5]
    best_threshold_f1 = thresholds[best_idx_f1]
    best_threshold_f2 = thresholds[best_idx_f2]
    best_threshold_f3 = thresholds[best_idx_f3]
    youdens_j = recall - (1 - precision)
    best_idx_youden = np.argmax(youdens_j)
    best_threshold_youden = thresholds[best_idx_youden]

    print(f"Optimal thresholds:")
    print(f"  0.25 score: {best_threshold_f0_25:.4f} (0.25={f0_25_scores[best_idx_f0_25]:.4f})")
    print(f"  0.5 score: {best_threshold_f0_5:.4f} (0.5={f0_5_scores[best_idx_f0_5]:.4f})")
    print(f"  F1 score: {best_threshold_f1:.4f} (F1={f1_scores[best_idx_f1]:.4f})")
    print(f"  F2 score: {best_threshold_f2:.4f} (F2={f2_scores[best_idx_f2]:.4f})")
    print(f"  F3 score: {best_threshold_f3:.4f} (F3={f3_scores[best_idx_f3]:.4f})")

    # Use F3 since you prefer recall (catching obs) over precision
    best_threshold = best_threshold_f3
    print(f"\nRecommended threshold based on Youden's J statistic: {best_threshold:.4f}")

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Plot ROC curve and confusion matrix
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    x_threshold = fpr[np.argmin(np.abs(_ - best_threshold))]
    y_threshold = tpr[np.argmin(np.abs(_ - best_threshold))]

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot(x_threshold, y_threshold, 'ro', label=f'Threshold = {best_threshold:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    preds = (all_scores >= best_threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty', 'Observation'])
    disp.plot(cmap='Blues')

    plt.title('Confusion Matrix')
    plt.show()

# Plot score distribution for empty vs observation samples on PCA of latent space, labels in different shapes, and color by score
# Make colorbar change colors at the threshold for better visualization
color_norm = colors.TwoSlopeNorm(vmin=all_scores.min(), vcenter=best_threshold, vmax=all_scores.max())
#%%
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(test_dataset_lat.tensors[0].cpu().numpy())
latent_2d_obs = latent_2d[all_labels == 1]
latent_2d_empty = latent_2d[all_labels == 0]

max_pca_x = np.max(abs(latent_2d[:, 0]))
max_pca_y = np.max(abs(latent_2d[:, 1]))

max_dim = max(max_pca_x, max_pca_y)

plt.figure(figsize=(3.5, 3.5))
# scatter_obs = plt.scatter(latent_2d_obs[:, 0], latent_2d_obs[:, 1], c=all_scores[all_labels == 1], norm = color_norm, marker='o', cmap='coolwarm', alpha=0.7, s=10, label='Observation')
# scatter_empty = plt.scatter(latent_2d_empty[:, 0], latent_2d_empty[:, 1], c=all_scores[all_labels == 0], norm = color_norm, marker='s', cmap='coolwarm', alpha=0.7, s=10, label='Empty')

scatter_empty = plt.scatter(latent_2d_empty[:, 0], latent_2d_empty[:, 1], c='blue', marker='o',  alpha=0.3, s=2, label='Empty')
scatter_obs = plt.scatter(latent_2d_obs[:, 0], latent_2d_obs[:, 1], c='red', marker='o', alpha=0.3, s=2, label='Observation')
# plt.legend(loc=' right', fontsize=10)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),  # centered above axes
    ncol=2,
    fontsize=10,
)
plt.xlim(-max_dim, max_dim)
plt.ylim(-max_dim, max_dim)
plt.xticks([-10, -5, 0, 5, 10])
plt.yticks([-10, -5, 0, 5, 10])
plt.grid()
# plt.colorbar(scatter_obs, label='Anomaly Score', ticks=[0, best_threshold, 1], format='%.3f')

#%% Save scorer model for future reference

if mode == "one_class" and scorer_oc is not None:
    scorer_model_name = model_name.replace(f"{encoder}/", "NN/OneClass/").replace(f"{encoder}_model", "scorer_oneclass_model")
    checkpoint = {
                'model_state_dict': scorer_oc.state_dict(),
                'threshold': best_threshold,
                'latent_dim': latent_dims,
                'grid_size': grid_size,
                'average_precision': avg_precision
                }
    
elif mode == "binary" and scorer_bin is not None:

    scorer_model_name = model_name.replace(f"{encoder}/", "NN/TwoClass/").replace(f"{encoder}_model", "scorer_binary_model")
    checkpoint = {
                'model_state_dict': scorer_bin.state_dict(),
                'threshold': best_threshold,
                'latent_dim': latent_dims,
                'grid_size': grid_size,
                'average_precision': avg_precision
                }
        
elif mode == "mahalanobis":
    scorer_model_name = model_name.replace(f"{encoder}/", "NN/Mahalanobis/").replace(f"{encoder}_model", "scorer_mahalanobis_model")
    
    checkpoint = {
                'mean_vec': mean_vec.cpu(),
                'cov_inv': cov_inv.cpu(),
                'dist_calib': dist_calib.cpu(),
                'recon_calib': calib_recon.cpu(),
                'threshold': best_threshold,
                'latent_dim': latent_dims,
                'grid_size': grid_size,
                'average_precision': avg_precision
                }
else:
    raise ValueError(f"Invalid mode: {mode}. Cannot save model.")

Path(scorer_model_name).parent.mkdir(parents=True, exist_ok=True)

torch.save(checkpoint, scorer_model_name)
print(f"\n✓ Saved {mode} scorer model to: {scorer_model_name}")
print(f"  Mode: {mode}")
print(f"  Threshold: {best_threshold:.4f}")
print(f"  Average Precision: {avg_precision:.4f}")
# %% Load scorer model checkpoint for inference
# Example of loading the saved model for inference later

checkpoint = torch.load(scorer_model_name, map_location=device)

if mode == "one_class":
    scorer_loaded = OneClassScorer(latent_dim=checkpoint['latent_dim'], hidden_dim=64, emb_dim=4, grid_size=checkpoint['grid_size'])
    scorer_loaded.load_state_dict(checkpoint['model_state_dict'])
    scorer_loaded.to(device)
    scorer_loaded.eval()
    print(f"✓ Loaded one-class scorer model from {scorer_model_name}")
    print(f"  Threshold: {checkpoint['threshold']:.4f}")
    print(f"  Average Precision: {checkpoint['average_precision']:.4f}")
elif mode == "binary":
    scorer_loaded = TwoClassScorer(latent_dim=checkpoint['latent_dim'], hidden_dim=256, dropout=0.3, grid_size=checkpoint['grid_size'])
    scorer_loaded.load_state_dict(checkpoint['model_state_dict'], weight_only=F)
    scorer_loaded.to(device)
    scorer_loaded.eval()
    print(f"✓ Loaded binary scorer model from {scorer_model_name}")
    print(f"  Threshold: {checkpoint['threshold']:.4f}")
    print(f"  Average Precision: {checkpoint['average_precision']:.4f}")
elif mode == "mahalanobis":
    scorer_loaded = MahalanobisScorer(
        cov_inv=checkpoint['cov_inv'].to(device),
        mean_vec=checkpoint['mean_vec'].to(device),
        dist_calib=checkpoint['dist_calib'].to(device),
        recon_calib=checkpoint['recon_calib'].to(device),
    )
    print(f"✓ Loaded Mahalanobis scorer model from {scorer_model_name}")
    print(f"  Threshold: {checkpoint['threshold']:.4f}")
    print(f"  Average Precision: {checkpoint['average_precision']:.4f}")
else:
    raise ValueError(f"Invalid mode: {mode}. Cannot load model.")

#%% Deploy scorer checkpoint to the Jetson
JETSON_MODELS_DIR = "jellyfish@jellyscope:/home/jellyfish/Github/Jellyscope/Jetson_monitoring/models"
# BatchMode=yes so a first-time host-key prompt or a password/passphrase request fails fast
# instead of scp hanging forever waiting for input nothing will ever supply (matches
# Jetson_monitoring/Monitor/transfer.py's own SSH options).
SCP_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]

try:
    scp_result = subprocess.run(
        ["scp", *SCP_SSH_OPTS, scorer_model_name, JETSON_MODELS_DIR],
        capture_output=True, text=True, timeout=120,
    )
    if scp_result.returncode == 0:
        print(f"✓ Copied {Path(scorer_model_name).name} to {JETSON_MODELS_DIR}")
    else:
        print(f"scp failed (exit {scp_result.returncode}): {scp_result.stderr.strip()}")
except subprocess.TimeoutExpired:
    print("scp timed out after 120s -- check that SSH key auth to jellyfish@jellyscope works "
          "non-interactively (test with: ssh -o BatchMode=yes jellyfish@jellyscope whoami)")