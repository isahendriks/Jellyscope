#%%
import os
import sys
from pathlib import Path

import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import TwoClassScorer, VariationalAutoencoder, preprocess_tile

print("=" * 70)
print("Full Crop ROC Evaluation")
print("=" * 70)

#%% Configuration
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"
monitoring_effort = "Kristineberg_251128"
grid_size = 16
tile_size = int(4512 / grid_size)
latent_dim = 32
batch_size = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae_model_path = f"../models/VAE/{monitoring_effort}_vae_model{grid_size}_l{latent_dim}.pth"
scorer_model_path = f"../models/NN/TwoClass/{monitoring_effort}_scorer_binary_model{grid_size}_l{latent_dim}.pth"

tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train_DNN", f"tiles{grid_size}_offsets5")
metadata_path = Path(tiles_path) / "crops_metadata.csv"

print(f"Device: {device}")
print(f"VAE model: {vae_model_path}")
print(f"Scorer model: {scorer_model_path}")
print(f"Metadata: {metadata_path}")

if not metadata_path.exists():
    raise FileNotFoundError(f"Missing crop metadata: {metadata_path}")

#%% Load models
vae_checkpoint = torch.load(vae_model_path, map_location=device, weights_only=False)
latent_dims = vae_checkpoint.get("latent_dim")
image_size = vae_checkpoint.get("image_size")
hidden_channels = vae_checkpoint.get("hidden_channels")
grid_size = vae_checkpoint.get("grid_size")

vae_model = VariationalAutoencoder(
    latent_dims=latent_dims,
    image_size=image_size,
    hidden_channels=hidden_channels,
    grid_size=grid_size,
)
vae_model.load_state_dict(vae_checkpoint["model_state_dict"])
vae_model.to(device)
vae_model.eval()

scorer_checkpoint = torch.load(scorer_model_path, map_location=device, weights_only=False)
scorer_model = TwoClassScorer(latent_dim=latent_dims, hidden_dim=256, dropout=0.3, grid_size=grid_size)
scorer_model.load_state_dict(scorer_checkpoint["model_state_dict"])
scorer_model.to(device)
scorer_model.eval()

print(f"Loaded VAE latent_dim={latent_dims}, grid_size={grid_size}")
print(f"Loaded scorer model with latent_dim={latent_dims}, grid_size={grid_size}")

#%% Load crop metadata and labels
df = pd.read_csv(metadata_path)

if "label" not in df.columns:
    raise ValueError("crops_metadata.csv must contain a 'label' column")

df["label"] = df["label"].map({"no_obs": 0, "obs": 1, 0: 0, 1: 1}).astype(int)
image_root = Path(tiles_path).parent / "OG_images"
image_paths = [image_root / f"{image_name}.png" for image_name in sorted(df["image"].astype(str).unique())]
image_paths = [path for path in image_paths if path.exists()]

print(f"Found {len(image_paths)} full crops for evaluation")

#%% Helper to build the full-crop score from offset tiles
def score_full_crop(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    image_h, image_w = image.shape[:2]
    crop_scores = []

    for offset in [0.0, 0.2, 0.4, 0.6, 0.8]:
        offset_px = int(round(tile_size * offset))
        for row in range(grid_size):
            for col in range(grid_size):
                y0 = row * tile_size + offset_px
                x0 = col * tile_size + offset_px
                y1 = y0 + tile_size
                x1 = x0 + tile_size

                if y1 > image_h or x1 > image_w:
                    continue

                tile = image[y0:y1, x0:x1]
                tile_tensor = preprocess_tile(tile, image_size=image_size)
                if tile_tensor is None:
                    continue

                tile_tensor = tile_tensor.unsqueeze(0).to(device)
                row_tensor = torch.tensor([row], dtype=torch.long, device=device)
                col_tensor = torch.tensor([col], dtype=torch.long, device=device)

                mu, _ = vae_model.encode(tile_tensor, row=row_tensor, col=col_tensor)
                x_hat = vae_model.decode(mu)
                recon_err = ((tile_tensor - x_hat) ** 2).flatten(1).mean(dim=1)
                prob = scorer_model(mu, recon_err, row_tensor, col_tensor)
                crop_scores.append(float(prob.item()))

    if not crop_scores:
        raise ValueError(f"No tiles were extracted for {image_path}")

    return float(np.max(crop_scores)), np.asarray(crop_scores, dtype=np.float32)

#%% Compute crop-level scores from all tiles in each crop
all_scores = []
all_labels = []
crop_names = []

with torch.no_grad():
    for image_path in image_paths:
        print(f"Scoring {image_path.name}...", end = "\r") 
        image_name = image_path.stem
        crop_score, crop_scores = score_full_crop(image_path)

        crop_label_rows = df[df["image"].astype(str) == image_name]
        crop_label = int(crop_label_rows["label"].max())

        all_scores.append(crop_score)
        all_labels.append(crop_label)
        crop_names.append(image_name)

all_scores = np.asarray(all_scores, dtype=np.float32)
all_labels = np.asarray(all_labels, dtype=np.int32)

print(f"Evaluated {len(all_scores)} full crops")
print(f"  Observations: {int(all_labels.sum())}")
print(f"  Empty:        {len(all_labels) - int(all_labels.sum())}")

#%% ROC-style threshold search, matching train_DNN.py
precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
avg_precision = average_precision_score(all_labels, all_scores)

f0_25_scores = 1.0625 * (precision[:-1] * recall[:-1]) / (0.0625 * precision[:-1] + recall[:-1] + 1e-8)
f0_5_scores = 1.25 * (precision[:-1] * recall[:-1]) / (0.25 * precision[:-1] + recall[:-1] + 1e-8)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
f2_scores = 5 * (precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-8)
f3_scores = 10 * (precision[:-1] * recall[:-1]) / (9 * precision[:-1] + recall[:-1] + 1e-8)

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

print("Optimal thresholds:")
print(f"  0.25 score: {best_threshold_f0_25:.4f} (0.25={f0_25_scores[best_idx_f0_25]:.4f})")
print(f"  0.5 score: {best_threshold_f0_5:.4f} (0.5={f0_5_scores[best_idx_f0_5]:.4f})")
print(f"  F1 score: {best_threshold_f1:.4f} (F1={f1_scores[best_idx_f1]:.4f})")
print(f"  F2 score: {best_threshold_f2:.4f} (F2={f2_scores[best_idx_f2]:.4f})")
print(f"  F3 score: {best_threshold_f3:.4f} (F3={f3_scores[best_idx_f3]:.4f})")

best_threshold = best_threshold_f3

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"Precision-Recall curve (AP = {avg_precision:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

fpr, tpr, roc_thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)
roc_idx = np.argmin(np.abs(roc_thresholds - best_threshold))

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot(fpr[roc_idx], tpr[roc_idx], "ro", label=f"Threshold = {best_threshold:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

preds = (all_scores >= best_threshold).astype(int)
cm = confusion_matrix(all_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Empty", "Observation"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

print("\nFull crop evaluation complete")
print(f"Average precision: {avg_precision:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Best threshold: {best_threshold:.4f}")
