#%% Train Neural Network Scorer in One-Class Mode (from empty tiles only)
"""
Trains the ObservationScorer in one-class mode when only empty training data is available.
The network learns what "normal" (empty) tiles look like and detects observations as anomalies.
This is the first stage of progressive learning.

Usage: python train_scorer_oneclass.py
Output: models/Kristineberg_260424_scorer_oneclass_model.pth
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import VariationalAutoencoder, ObservationScorer, image_pipeline, ImagePathDataset


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


#%% Configuration
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
monitoring_effort = "Kristineberg_260424"
grid_size = 16
tile_size = int(4512/grid_size)
image_size_vae = 128
latent_dim = 32
model_name = f"../models/VAE/{monitoring_effort}_vae_model{grid_size}_l{latent_dim}.pth"

# Training parameters
batch_size = 512
learning_rate = 0.001
epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"VAE Model: {model_name}")

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

print(f"✓ Loaded VAE model with latent_dim={latent_dims}, grid_size={grid_size}")

#%% Load empty training data only
test_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "test", f"tiles{grid_size}")
test_tiles_no_obs_path = os.path.join(test_tiles_path, "no_obs")

test_tiles_empty = sorted(Path(test_tiles_no_obs_path).rglob("*.png"))

print(f"\nLoaded {len(test_tiles_empty)} empty tiles")

# Create data loader
test_dataset_empty = ImagePathDataset(
    test_tiles_empty, 
    lambda path: image_pipeline(path, image_size=image_size_vae, include_labels=True)
)
test_loader_empty = DataLoader(test_dataset_empty, batch_size=batch_size, shuffle=False,
                               num_workers=0)

#%% Extract latent features from empty tiles only
latent_list = []
recon_err_list = []
rows_list = []
cols_list = []

print("Extracting features from empty tiles...")
with torch.no_grad():
    for images, batch_labels, batch_rows, batch_cols in test_loader_empty:
        print(f"Processing batch of {len(images)} empty tiles...")
        images = images.to(device, non_blocking=True)
        batch_cols = batch_cols.to(device, non_blocking=True)
        batch_rows = batch_rows.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        
        mu, _ = vae_model.encode(images, row=batch_rows, col=batch_cols)
        x_hat = vae_model.decode(mu)
        recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)
        
        latent_list.append(mu.cpu())
        recon_err_list.append(recon_err.cpu())
        rows_list.extend(batch_rows.cpu().numpy())
        cols_list.extend(batch_cols.cpu().numpy())

# Concatenate all features
latent_features = torch.cat(latent_list, dim=0).numpy()
recon_err_features = torch.cat(recon_err_list, dim=0).numpy()
rows = np.array(rows_list, dtype=np.int64)
cols = np.array(cols_list, dtype=np.int64)

# In one-class mode, all samples are "normal" (empty) = label 1
labels = np.ones(len(latent_features), dtype=np.float32)

print(f"Total empty samples: {len(labels)}")

#%% Split into train/test (80/20 from empty data only)

X_latent_train, X_latent_test, \
X_recon_train, X_recon_test, \
rows_train, rows_test, \
cols_train, cols_test, \
y_train, y_test = train_test_split(
    latent_features, recon_err_features, rows, cols, labels,
    test_size=0.2, random_state=42
)

print(f"Train set: {len(y_train)} empty samples")
print(f"Test set: {len(y_test)} empty samples")

# Create PyTorch tensors and dataloaders
train_dataset = TensorDataset(
    torch.from_numpy(X_latent_train).float(),
    torch.from_numpy(X_recon_train).float(),
    torch.from_numpy(rows_train).long(),
    torch.from_numpy(cols_train).long(),
    torch.from_numpy(y_train).float()
)

test_dataset = TensorDataset(
    torch.from_numpy(X_latent_test).float(),
    torch.from_numpy(X_recon_test).float(),
    torch.from_numpy(rows_test).long(),
    torch.from_numpy(cols_test).long(),
    torch.from_numpy(y_test).float()
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

#%% Initialize scorer in ONE-CLASS mode
scorer = ObservationScorer(latent_dim=latent_dims, hidden_dim=256, dropout=0.3, 
                           grid_size=grid_size, mode='one_class')
scorer.to(device)

print(f"✓ Initialized ObservationScorer in ONE-CLASS mode")
print(f"  Mode: {scorer.mode}")
print(f"  Training objective: Learn to recognize EMPTY tiles as normal")
print(f"  Inference: Observations will appear as anomalies (low probability)\n")

# Loss and optimizer
criterion = nn.BCELoss()  # Empty tiles should output HIGH probability
optimizer = optim.Adam(scorer.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_losses = []
test_losses = []

print(f"Training for {epochs} epochs...\n")
for epoch in range(epochs):
    # Training
    scorer.train()
    train_loss = 0.0
    for X_latent, X_recon, rows, cols, y in train_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # All 1.0 (normal)
        
        optimizer.zero_grad()
        
        probs = scorer(X_latent, X_recon, rows, cols)
        loss = criterion(probs, y)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Testing
    scorer.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_latent, X_recon, rows, cols, y in test_loader:
            X_latent = X_latent.to(device, non_blocking=True)
            X_recon = X_recon.to(device, non_blocking=True)
            rows = rows.to(device, non_blocking=True)
            cols = cols.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            probs = scorer(X_latent, X_recon, rows, cols)
            loss = criterion(probs, y)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    scheduler.step(test_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

print(f"\n✓ Training complete!")

# Calculate mean probability for empty tiles (should be HIGH)
scorer.eval()
with torch.no_grad():
    all_probs = []
    for X_latent, X_recon, rows, cols, y in test_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        
        probs = scorer(X_latent, X_recon, rows, cols)
        all_probs.append(probs.cpu().numpy())

all_probs = np.concatenate(all_probs)
mean_empty_prob = all_probs.mean()
std_empty_prob = all_probs.std()

print(f"\nOne-Class Model Statistics (on empty test set):")
print(f"  Mean prob(empty) = {mean_empty_prob:.4f} (should be HIGH ~0.7-0.9)")
print(f"  Std prob(empty)  = {std_empty_prob:.4f}")
print(f"  Min prob(empty)  = {all_probs.min():.4f}")
print(f"  Max prob(empty)  = {all_probs.max():.4f}")

# Recommended threshold for anomaly detection
threshold_oneclass = mean_empty_prob - std_empty_prob  # Conservative threshold
print(f"\nRecommended threshold for anomaly detection: {threshold_oneclass:.4f}")
print(f"  (Observations < threshold = likely observation, > threshold = likely empty)")

#%% Plot training curves
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(train_losses, label='Train Loss', linewidth=2)
ax.plot(test_losses, label='Test Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (BCE)', fontsize=12)
ax.set_title('One-Class Mode Training: Learning "Empty" Tiles', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%% Save one-class scorer model
scorer_model_name = model_name.replace("VAE/", "NN/oneclass/").replace("vae_model", "scorer_oneclass_model")
Path(scorer_model_name).parent.mkdir(parents=True, exist_ok=True)

checkpoint = {
    'model_state_dict': scorer.state_dict(),
    'threshold': threshold_oneclass,
    'mode': 'one_class',
    'latent_dim': latent_dims,
    'grid_size': grid_size,
    'mean_empty_prob': float(mean_empty_prob),
    'std_empty_prob': float(std_empty_prob),
}

torch.save(checkpoint, scorer_model_name)
print(f"\n✓ Saved one-class scorer to: {scorer_model_name}")
print(f"  Threshold: {threshold_oneclass:.4f}")
print(f"  Mode: one_class")

#%% Final evaluation on test set
scorer.eval()

with torch.no_grad():
    all_probs = []
    for X_latent, X_recon, rows, cols, y in test_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        
        probs = scorer(X_latent, X_recon, rows, cols)
        all_probs.append(probs.cpu().numpy())
        