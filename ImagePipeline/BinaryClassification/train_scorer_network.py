#%% Train Neural Network Scorer in Binary Mode (with both empty and observation data)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from functions import VariationalAutoencoder, ObservationScorer, eval_image_pipeline

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
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
monitoring_effort = "Kristineberg_260424"
grid_size = 16
latent_dim = 32
model_name = f"models/{monitoring_effort}_vae_model{grid_size}_l{latent_dim}.pth"

# Training parameters
batch_size = 512
learning_rate = 0.001
epochs = 50
train_split = 0.7
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
test_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "test", f"tiles{grid_size}")
test_tiles_obs_path = os.path.join(test_tiles_path, "obs")
test_tiles_no_obs_path = os.path.join(test_tiles_path, "no_obs")

test_tiles_obs = sorted(Path(test_tiles_obs_path).rglob("*.png"))
test_tiles_no_obs = sorted(Path(test_tiles_no_obs_path).rglob("*.png"))

print(f"Found {len(test_tiles_obs)} observation tiles")
print(f"Found {len(test_tiles_no_obs)} empty tiles")

# Create data loaders for feature extraction
test_loader_obs = DataLoader(test_tiles_obs, batch_size=batch_size, shuffle=False,
                             collate_fn=eval_image_pipeline, num_workers=0)
test_loader_no_obs = DataLoader(test_tiles_no_obs, batch_size=batch_size, shuffle=False,
                                collate_fn=eval_image_pipeline, num_workers=0)

#%% Encode image to obtain latentt features and embed position information
latent_list = []
recon_err_list = []
labels = []
rows_list = []
cols_list = []

print("\nExtracting features from observation tiles...")
with torch.no_grad():
    for images, batch_labels, batch_rows, batch_cols in test_loader_obs:
        images = images.to(device, non_blocking=True)
        mu, _ = vae_model.encode(images, row=batch_rows, col=batch_cols)
        x_hat = vae_model.decode(mu)
        recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)
        
        latent_list.append(mu.cpu())
        recon_err_list.append(recon_err.cpu())
        labels.extend(batch_labels.numpy())
        rows_list.extend(batch_rows.numpy())
        cols_list.extend(batch_cols.numpy())

print("Extracting features from empty tiles...")
with torch.no_grad():
    for images, batch_labels, batch_rows, batch_cols in test_loader_no_obs:
        images = images.to(device, non_blocking=True)
        mu, _ = vae_model.encode(images, row=batch_rows, col=batch_cols)
        x_hat = vae_model.decode(mu)
        recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)
        
        latent_list.append(mu.cpu())
        recon_err_list.append(recon_err.cpu())
        labels.extend(batch_labels.numpy())
        rows_list.extend(batch_rows.numpy())
        cols_list.extend(batch_cols.numpy())

# Concatenate all features
latent_features = torch.cat(latent_list, dim=0).numpy()
recon_err_features = torch.cat(recon_err_list, dim=0).numpy()
labels = np.array(labels, dtype=np.float32)
rows = np.array(rows_list, dtype=np.int64)
cols = np.array(cols_list, dtype=np.int64)

print(f"Total samples: {len(labels)}")
print(f"Observation samples: {labels.sum():.0f}")
print(f"Empty samples: {(1 - labels).sum():.0f}")

#%% Split into train/test
X_latent_train, X_latent_test, \
X_recon_train, X_recon_test, \
rows_train, rows_test, \
cols_train, cols_test, \
y_train, y_test = train_test_split(
    latent_features, recon_err_features, rows, cols, labels,
    test_size=0.3, random_state=42, stratify=labels
)

print(f"\nTrain set: {len(y_train)} samples (obs={y_train.sum():.0f}, empty={(1-y_train).sum():.0f})")
print(f"Test set: {len(y_test)} samples (obs={y_test.sum():.0f}, empty={(1-y_test).sum():.0f})")

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

#%% Initialize and train scorer network in BINARY mode
scorer = ObservationScorer(latent_dim=latent_dims, hidden_dim=256, dropout=0.3, 
                           grid_size=grid_size, mode='binary')
scorer.to(device)

print(f"\n✓ Initialized ObservationScorer in BINARY mode")
print(f"  Mode: {scorer.mode}")
print(f"  Training objective: Discriminate between empty and observation samples\n")

# Loss and optimizer
criterion = nn.BCELoss()  # Binary cross entropy for probability targets
optimizer = optim.Adam(scorer.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_losses = []
test_losses = []
test_accuracies = []

print(f"\nTraining for {epochs} epochs...")
for epoch in range(epochs):
    # Training
    scorer.train()
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
    correct = 0
    total = 0
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
            
            # Compute accuracy at threshold=0.5
            preds = (probs >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = correct / total
    test_accuracies.append(accuracy)
    
    scheduler.step(test_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Acc={accuracy:.4f}")

print(f"\nTraining complete!")
print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")

#%% Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(test_accuracies, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scorer_training.png', dpi=150)
print("Saved training curves to scorer_training.png")
plt.show()

#%% Find optimal threshold
scorer.eval()
with torch.no_grad():
    all_probs = []
    all_labels = []
    for X_latent, X_recon, rows, cols, y in test_loader:
        X_latent = X_latent.to(device, non_blocking=True)
        X_recon = X_recon.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)
        
        probs = scorer(X_latent, X_recon, rows, cols)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)

# Find threshold that maximizes F1
best_f1 = 0
best_threshold = 0.5
for threshold in np.linspace(0.1, 0.9, 50):
    preds = (all_probs >= threshold).astype(int)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")

#%% Plot ROC curve and confusion matrix
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

preds = (all_probs >= best_threshold).astype(int)
cm = confusion_matrix(all_labels, preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Empty', 'Observation'])
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.show()

#%% Save scorer model and threshold
scorer_model_name = model_name.replace("vae_model", "scorer_binary_model").replace(".pth", ".pth")
Path(scorer_model_name).parent.mkdir(parents=True, exist_ok=True)

checkpoint = {
    'model_state_dict': scorer.state_dict(),
    'threshold': best_threshold,
    'mode': 'binary',
    'latent_dim': latent_dims,
    'grid_size': grid_size,
    'test_accuracy': test_accuracies[-1],
    'best_f1': best_f1
}

torch.save(checkpoint, scorer_model_name)
print(f"\n✓ Saved binary scorer model to: {scorer_model_name}")
print(f"  Mode: binary")
print(f"  Threshold: {best_threshold:.4f}")
print(f"  Test Accuracy: {test_accuracies[-1]:.4f}")
print(f"  Best F1: {best_f1:.4f}")