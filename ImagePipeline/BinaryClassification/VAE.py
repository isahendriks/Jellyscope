#%% import packages
### PACKAGES ###
import json
import os
import torch

import re
import gc 

import deeplay as dl
import deeptrack as dt

from torchvision.datasets import ImageFolder
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Subset

from lightning.pytorch.loggers import CSVLogger

import torch.nn as nn
import lightning as L

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

import pickle as pkl
import random 
from pathlib import Path
import cv2

from matplotlib.patches import Rectangle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
import time
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

#%% Set parameters and paths
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
SAMPLE_IMAGE_IDX_TEST = 0  # Index of the sample image to plot process (0-based)
SAMPLE_IMAGE_IDX_TRAIN = 0  # Index of the sample image to plot process (0-based)

model_name = "vae_model6_aug.pth"

train_tiles_path = os.path.join(ROOT_DIR_R, "train", "tiles6_aug")
train_og_images_path = os.path.join(ROOT_DIR_R, "train", "OG_images")
images_train = list(Path(train_og_images_path).glob("*.png"))

# Load one sample image and its tiles from train set for plotting later
sample_image_path_train = images_train[SAMPLE_IMAGE_IDX_TRAIN] 
sample_image_name_train = sample_image_path_train.stem
sample_image_tiles_path_train = sorted([
    *Path(train_tiles_path).glob(f"{sample_image_name_train}_r?_c?_i?.png"),
    *Path(train_tiles_path).glob(f"{sample_image_name_train}_r?_c?_i??.png"),
])
print(f"Loaded sample train image: {sample_image_name_train}, #tiles: {len(sample_image_tiles_path_train)}")

# training hyperparameters
batch_size = 64
image_size = 128
latent_dim = 6
model_channels = [32, 64, 128, 256]
learning_rate = 1e-3
beta_start = 0.1
beta_end = 2
epochs = 30
recon_loss = torch.nn.MSELoss(reduction="mean")  # was sum

#%% Load train data
train_tiles = sorted(Path(train_tiles_path).rglob("*.png"))
train_source = dt.sources.Source(path=train_tiles)

train_image_pip = (
    dt.LoadImage(train_source.path, to_grayscale=True, ndim=2) >>  # Load as HxW
    dt.Resize((image_size,image_size)) >>  # Resize to 128x128
    dt.Lambda(lambda: (lambda x: x[..., :1])) >>  # Take only first channel -> HxWx1
    dt.Lambda(lambda: (lambda x: x / 255.0)) >>  # Normalize to [0, 1]
    dt.pytorch.ToTensor(dtype=torch.float, permute_mode="always")  # HxWx1 -> 1xHxW
)

train_dataset = dt.pytorch.Dataset(train_image_pip & train_image_pip, inputs=train_source)
train_loader = dl.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("\n" + "="*120)
print(f"Loaded {len(train_dataset)} training tiles from: {train_tiles_path}")
print("="*120)

#%% Plot sample images from train dataset
sample_image_train = cv2.imread(str(sample_image_path_train))
 
num_tiles = len(sample_image_tiles_path_train)
grid_size = int(np.sqrt(num_tiles))
print(f"Original image shape (H, W, C): {sample_image_train.shape}")
print(f"Original tile shape (H, W, C): {4512//grid_size, 4512//grid_size, 3}")

# Get tiles from dataset for the sample image
indices = []
for i, item in enumerate(train_source):
    if item["path"].stem.startswith(sample_image_name_train):
        indices.append(i)
        
train_dataset_subset = Subset(train_dataset, indices)

# plot loaded image
plt.figure(figsize=(6, 6))
plt.imshow(sample_image_train, cmap="gray")
plt.title(f"Original image: {sample_image_name_train}")
plt.axis("off")

# plot original tiles
fig,axis = plt.subplots(grid_size, grid_size, figsize=(grid_size*2.4, grid_size*2.4))
for i in range(num_tiles):
    tile_path = sample_image_tiles_path_train[i]
    tile_image = cv2.imread(str(tile_path))
    row, col = divmod(i, grid_size)
    axis[row, col].imshow(tile_image, cmap="gray", vmin=0, vmax=1)
    # axis[row, col].set_title(f"Tile {i}")
    axis[row, col].axis("off")

fig.suptitle(f"Original {grid_size}x{grid_size} tiles from {sample_image_name_train}", fontsize=20, fontweight="bold")
plt.tight_layout()

# plot tiles from dataloader
fig,axis = plt.subplots(grid_size, grid_size, figsize=(grid_size*2.4, grid_size*2.4))
for i, image in enumerate(sample_image_train): # CxHxW -> HxWxC
    row, col = divmod(i, grid_size)
    axis[row, col].imshow(image[0][0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    # axis[row, col].set_title(f"Tile {i}")
    axis[row, col].axis("off")
plt.suptitle(f"Processed {grid_size}x{grid_size} tiles from dataloader", fontsize=20, fontweight="bold")
plt.tight_layout()

#%% Plot example train images from dataloader
# Get one batch
batch = next(iter(train_loader))
x, y = batch
sample_images = x[0]  # Take the first 64 images from the batch
_, im_height, im_width = sample_images.shape
print(f"Image shape:", im_width, im_height)

plot_grid = (3,7)

fig, axs = plt.subplots(plot_grid[0], plot_grid[1], figsize=(plot_grid[1]*1.2, plot_grid[0]*1.2))
for idx, ax in enumerate(axs.flatten()):
    rowcol = re.search(r'_r(\d+)_c(\d+)', train_source[idx]['path'].name)
    # ax.set_title(f"r{rowcol[1]} c{rowcol[2]}", fontsize=8, fontweight="bold")
    ax.imshow(x[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.suptitle(f"Example Training Images from {'/'.join(train_og_images_path.split(os.sep)[-3:])}", fontsize=16, fontweight="bold")

#%%###########################################################################
#### VARIATIONAL AUTOENCODER (VAE) ARCHITECTURE, TRAINING, AND EVALUATION ####
##############################################################################
vae = dl.VariationalAutoEncoder(
    latent_dim=latent_dim,
    channels=model_channels,
    input_size=(im_width, im_height),
    reconstruction_loss=recon_loss,
    beta=beta_end,
).create()

vae.to(device)

print(vae)

vae_trainer = dl.Trainer(max_epochs=epochs, accelerator="gpu", devices=1)
vae_trainer.fit(vae, train_loader)

vae_history = vae_trainer.history
vae_history.plot()

# save model weights
model_output_path = Path(model_name)  
torch.save(vae.state_dict(), model_output_path)
print(f"Saved model weights to: {model_output_path}")

#%% Generate images in latent space grid
model = vae.to(device)

img_num, img_size = 21, 128

z0_grid = z1_grid = Normal(0, 1).icdf(torch.linspace(0.001, 0.999, img_num))
# z0_grid = z1_grid = torch.linspace(-latent_range, latent_range, img_num)

image = np.zeros((img_num * img_size, img_num * img_size))
vae.eval()

latent_dim = vae.fc_dec.in_features

for i0, z0 in enumerate(z0_grid):
    for i1, z1 in enumerate(z1_grid):
        z = torch.zeros((1, latent_dim), dtype=torch.float32, device=vae.device)
        z[0, 0] = z0
        z[0, 1] = z1
        generated_image = model.decode(z).clone().detach()
        image[i1 * img_size : (i1 + 1) * img_size,
            i0 * img_size : (i0 + 1) * img_size] = \
            generated_image.cpu().numpy().squeeze()

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.xlabel("z0", fontsize=24)
plt.xticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size), np.round(z0_grid.numpy(), 1))
plt.ylabel("z1", fontsize=24)
plt.yticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size), np.round(z1_grid.numpy(), 1))
plt.show()