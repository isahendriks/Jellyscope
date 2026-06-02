#%% import packages
### PACKAGES ###
import os
from pathlib import Path
import random
import sys
from time import time

import cv2
import numpy as np

import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import Autoencoder, VariationalAutoencoder, image_pipeline_df, load_tiles_from_paths_fast, preprocess_tile

torch.manual_seed(0)
plt.rcParams["figure.dpi"] = 200
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

SAMPLE_IMAGE_IDX_TRAIN = 0 # Index of the sample image to plot process (0-based)

# Debugging parameters (set to False for full training)
DEBUG = False
N_DEBUG = 5000
EPOCHS_DEBUG = 10

monitoring_effort = "Kristineberg_251128"  # for titles and saved model names, e.g. "kristineberg_251128" 
grid_size = 16  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)

# model hyperparameters
image_size = 256
latent_dims = 128
hidden_channels = 32

train_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train_VAE", f"tiles{grid_size}")
train_og_images_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train_VAE", "OG_images")

print(f"train_tiles_path: {train_tiles_path}")
print(f"train_og_images_path: {train_og_images_path}")

# check if train_og_images_path has tiles in train_tiles_path, if not remove this image for train_og_images_path and print warning
images_train = sorted(Path(train_og_images_path).rglob("*.png"))

print(f"Total training images with tiles found: {len(images_train)}")
print(f"total training tiles found: {len(list(Path(train_tiles_path).rglob('*.png')))}")

model_name = f"../models/VAE/{monitoring_effort}_vae_model{grid_size}_l{latent_dims}_img{image_size}.pth"
if DEBUG:
    model_name = model_name.replace(".pth", "_debug.pth")
    epochs = EPOCHS_DEBUG
    images_train = images_train[:N_DEBUG // (grid_size**2)]  
    
model_output_path = Path(model_name)
model_output_path.parent.mkdir(parents=True, exist_ok=True)

# training hyperparameters
batch_size = 128
learning_rate = 1e-4 * (batch_size / 64)  # scale learning rate with batch size
epochs = 100
warmup_epochs = 40  # Number of epochs to linearly increase learning rate (helps stabilize early training)
if DEBUG:
    print(f"DEBUG MODE: Using only {N_DEBUG} tiles and training for {EPOCHS_DEBUG} epochs \n if saving enabled model name will be: {model_output_path.name}")
    
    
#%% Plot background image 

# Load all OG background images for training (for plotting later)
og_img_array = np.zeros((len(images_train), 4512, 4512), dtype=np.uint8)
for ind, img_path in enumerate(images_train):
    print(f"Loading image {ind+1}/{len(images_train)}: {img_path.name}", end="\r")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    og_img_array[ind] = img

print(f"Calculating mean background image across {len(images_train)} training images...")
img_bkg_mean = None
img_bkg_mean = np.mean(og_img_array, axis=0).astype(np.uint8)

# plot mean background image across all training images
if img_bkg_mean is not None:
    plt.figure(figsize=(6, 6))
    plt.imshow(img_bkg_mean, cmap="gray")
    plt.title(f"Mean background image across {len(images_train)} training images", fontsize=14)
    plt.axis("off")    

if img_bkg_mean is not None:
    img_bkg = img_bkg_mean
else:
    raise ValueError("Could not calculate background image (both median and mean failed)")

#%% Load train data

train_tiles = sorted(Path(train_tiles_path).rglob("*.png"))

if DEBUG:
    train_tiles = train_tiles[:N_DEBUG]
    epochs = EPOCHS_DEBUG

print(f"Loading training tiles from: {train_tiles_path}, total tiles found: {len(train_tiles)}")    
df_train_tiles = load_tiles_from_paths_fast(train_tiles, include_labels=False, to_device=device)  # Load tile paths and metadata into a DataFrame

print(f"Applying image pipeline to training tiles and creating dataset...")
images_tensor, rows_tensor, cols_tensor = image_pipeline_df(df_train_tiles, input_image_size_vae=image_size)  # type: ignore # Create dataset with preprocessing pipeline applied to each tile

print(f"Dataset created: images tensor shape: {images_tensor.shape}, rows tensor shape: {rows_tensor.shape}, cols tensor shape: {cols_tensor.shape}")
train_dataset = TensorDataset(images_tensor, rows_tensor, cols_tensor)  # Convert to TensorDataset for DataLoader

train_loader = DataLoader(
    train_dataset,                     # dataset object
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,                   # set >0 later if you want
    pin_memory=torch.cuda.is_available()
)

# Display configuration
print("\n" + "="*120)
print(f"Loaded {len(train_loader)*batch_size} training tiles from: {train_tiles_path}")
print("="*120)

# Sanity-check one batch to verify row/col loading
tile_chk, rows_chk, cols_chk = next(iter(train_loader))

print(f"Sanity batch -> x: {tuple(tile_chk.shape)}, rows: {tuple(rows_chk.shape)} ({rows_chk.dtype}), cols: {tuple(cols_chk.shape)} ({cols_chk.dtype})")
print(f"row range: [{rows_chk.min().item()}, {rows_chk.max().item()}], col range: [{cols_chk.min().item()}, {cols_chk.max().item()}]")

assert rows_chk.ndim == 1 and cols_chk.ndim == 1, "rows/cols must be 1D batch tensors"
assert rows_chk.shape[0] == tile_chk.shape[0] and cols_chk.shape[0] == tile_chk.shape[0], "rows/cols must match batch size"
assert rows_chk.min().item() >= 0 and rows_chk.max().item() < grid_size, "row indices out of range"
assert cols_chk.min().item() >= 0 and cols_chk.max().item() < grid_size, "col indices out of range"


#%% Plot sample images from train dataloader
# Load one sample image and its tiles from train set for plotting later
SAMPLE_IMAGE_IDX_TRAIN = random.randint(0, len(images_train) - 1)  # Index of the sample image to plot process (0-based)
sample_image_path_train = images_train[SAMPLE_IMAGE_IDX_TRAIN] 
sample_image_name_train = sample_image_path_train.stem


# Find all tiles corresponding to the sample image
_r_patterns = ("?", "??", "???", "????")
_c_patterns = ("?", "??", "???", "????")

sample_image_tiles_path_train = sorted({
    path
    for r_pat in _r_patterns
    for c_pat in _c_patterns
    for path in Path(train_tiles_path).glob(
        f"{sample_image_name_train}_r{r_pat}_c{c_pat}_o0.png"
    )
})



print(f"Loaded sample train image: {sample_image_name_train}, #tiles: {len(sample_image_tiles_path_train)}")
 
### plot original image
sample_image_train = cv2.imread(str(sample_image_path_train)) # Load image

if sample_image_train is None:
    raise ValueError(f"Could not read sample image: {sample_image_path_train}")

num_tiles = len(sample_image_tiles_path_train)

print(f"Original image shape (H, W, C): {sample_image_train.shape}")
print(f"Original tile shape (H, W, C): {4512//grid_size, 4512//grid_size, 3}")

plt.figure(figsize=(grid_size*2.4, grid_size*2.4))
plt.imshow(sample_image_train, cmap="gray")
plt.title(f"Original image: {sample_image_name_train}")
plt.axis("off")

#%% plot original tiles

sample_image_df = load_tiles_from_paths_fast(sample_image_tiles_path_train, include_labels=False, to_device=device)  # type: ignore # Load tile paths and metadata into a DataFrame

fig,axis = plt.subplots(grid_size, grid_size, figsize=(grid_size*2.4, grid_size*2.4))

for tile_df in sample_image_df.itertuples():
    tile_img =tile_df.image.astype(np.uint8)  # Convert to uint8 for correct display
    row = int(tile_df.row)
    col = int(tile_df.col)
    axis[row, col].imshow(tile_img, cmap="gray", vmin=0, vmax=255)
    axis[row, col].axis("off")
    
fig.suptitle(f"Original {grid_size}x{grid_size} tiles from {sample_image_name_train}", fontsize=20, fontweight="bold")
plt.tight_layout()

#%% plot tiles after image pipeline
sample_image_processed_tensor, rows_tensor, cols_tensor = image_pipeline_df(sample_image_df, input_image_size_vae=image_size)  # type: ignore # Apply image pipeline to sample image tiles

fig,axis = plt.subplots(grid_size, grid_size, figsize=(grid_size*2.4, grid_size*2.4))
for tile_tensor, row, col in zip(sample_image_processed_tensor, rows_tensor, cols_tensor):
    row = int(row.item())
    col = int(col.item())
    tile_img = (tile_tensor.numpy()[0] * 255).astype(np.uint8)  # Convert back to uint8 for display
    axis[row, col].imshow(tile_img, cmap="gray", vmin=0, vmax=255)
    axis[row, col].axis("off")
    
fig.suptitle(f"Processed in pipeline {grid_size}x{grid_size} tiles from {sample_image_name_train}", fontsize=20, fontweight="bold")
plt.tight_layout()

#%% Plot example train images from dataloader
# Get one batch
batch = next(iter(train_loader))
images = batch[0]  # Take the first 64 images from the batch
sample_images = images[:batch_size]  # Ensure we only take batch_size images
_, im_height, im_width = sample_images[0].shape

print(f"Image shape: {im_width} x {im_height}")

N_hor = 8 #int(np.sqrt(batch_size))
N_ver = 3 #int(N_hor/2)

plot_grid = (N_ver, N_hor)

fig, axs = plt.subplots(plot_grid[0], plot_grid[1], figsize=(plot_grid[1]*1.2, plot_grid[0]*1.2))
for idx, ax in enumerate(axs.flatten()):
    ax.imshow(sample_images[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
# plt.tight_layout()
# plt.suptitle(f"Example Training Images from {'/'.join(train_og_images_path.split(os.sep)[-3:])}", fontsize=16, fontweight="bold")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to make room for title
plt.show()

#%%###########################################################################
#### VARIATIONAL AUTOENCODER (VAE) ARCHITECTURE, TRAINING, AND EVALUATION ####
##############################################################################

### Initialize model
model = VariationalAutoencoder(
        latent_dims = latent_dims,
        image_size = image_size,
        hidden_channels = hidden_channels,
        grid_size = grid_size
    ).to(device)

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters.")
print("model device:", next(model.parameters()).device)
print("Model architecture:")
print(model)

#%% Train model
epoch_losses, recon_losses, kl_losses = [], [], []
step_losses, step_recon_losses, step_kl_losses = [], [], []
kl_per_dim_history = []

model.train()

n_batches = len(train_loader)
start_time = time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Learning rate warmup
    if epoch < warmup_epochs:
        lr = learning_rate * (epoch + 1) / warmup_epochs
    else:
        lr = learning_rate

    for g in opt.param_groups:
        g["lr"] = lr

    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for x, rows, cols in train_loader:
        x = x.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)

        # reset gradients
        opt.zero_grad()

        # calculate reconstruction error
        x_hat = model(x, row=rows, col=cols)
        recon = ((x - x_hat) ** 2).sum()

        # calculate KL divergence and total loss with warmup
        kl = model.encoder.kl
        beta = min(1.0, (epoch + 1) / warmup_epochs)
        
        # calculate total loss
        loss = recon + beta * kl

        # backpropagate and optimize
        loss.backward()
        opt.step()
        
        running_loss += loss.item()
        running_recon += recon.item()
        running_kl += kl.item()

        # Log per-step losses (for plotting later)
        step_losses.append(loss.item())
        step_recon_losses.append(recon.item())
        step_kl_losses.append(kl.item())

    # Per-epoch averages
    epoch_losses.append(running_loss / n_batches)
    recon_losses.append(running_recon / n_batches)
    kl_losses.append(running_kl / n_batches)
    kl_per_dim_history.append(model.encoder.kl_per_dim.detach().cpu().numpy())
    
    avg_epoch_time = (time() - start_time) / (epoch + 1)
    expected_time_remaining = avg_epoch_time * (epochs - epoch - 1)
    expected_finish_time = time() + expected_time_remaining
    
    print(f"  loss={epoch_losses[-1]:.3f} (recon={recon_losses[-1]:.3f}, kl={kl_losses[-1]:.3f}, beta={beta:.2f}), average epoch time: {avg_epoch_time:.1f}s, expected finish time: {time():.1f}s + {expected_time_remaining/60:.1f}m = {expected_finish_time:.1f}s")
    
total_elapsed = time() - start_time

print(f"Training finished in {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")


#%% Plot training curves
# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(epoch_losses, label="Total loss")
plt.plot(recon_losses, label="Reconstruction loss")
plt.plot(kl_losses, label="KL loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss curves")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Plot per-step loss curves
eps = 1e-12
plt.figure(figsize=(10, 5))
plt.plot(step_losses, label="Total loss", alpha=0.9)
plt.plot(step_recon_losses, label="Reconstruction loss", alpha=0.8)
plt.plot(step_kl_losses, label="KL loss", alpha=0.8)
plt.yscale("log")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training loss curves per step")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Plot KL per latent dimension over epochs (only for VAE)
if kl_per_dim_history and isinstance(model, VariationalAutoencoder):
    kl_per_dim_history = np.array(kl_per_dim_history)  # shape: (epochs, latent_dims)

    plt.figure(figsize=(8,5))
    for i in range(kl_per_dim_history.shape[1]):
        plt.plot(kl_per_dim_history[:, i], label=f"dim {i}")
    plt.xlabel("Epoch")
    plt.ylabel("KL per dim")
    plt.title("KL divergence per latent dimension")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

#%% save model checkpoint

checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "latent_dim": latent_dims,
    "image_size": image_size,
    "grid_size": grid_size,
    "hidden_channels": hidden_channels,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "epoch_losses": epoch_losses,
    "recon_losses": recon_losses,
    "kl_losses": kl_losses,
    "step_losses": step_losses,
    "step_recon_losses": step_recon_losses,
    "step_kl_losses": step_kl_losses,
    "kl_per_dim_history": kl_per_dim_history
}

torch.save(checkpoint, model_output_path)
print(f"Saved checkpoint to: {model_output_path}")

#%% Load model checkpoint (if restarted kernel)

if model_output_path.exists():
    print(f"Loading checkpoint from: {model_output_path}")
    checkpoint = torch.load(model_output_path, map_location=device, weights_only=False)
    model = VariationalAutoencoder(
        latent_dims=checkpoint["latent_dim"],
        image_size=checkpoint["image_size"],
        hidden_channels=checkpoint["hidden_channels"],
        grid_size=checkpoint.get("grid_size", 16)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded: grid_size={checkpoint.get('grid_size', 16)}, latent_dims={checkpoint['latent_dim']}")
else:
    print(f"Checkpoint not found at {model_output_path}. Please train the model first.")


#%% Generate images in latent space grid
model.to(device)

img_num, img_size = 21, 128

z0_grid = z1_grid = Normal(0, 1).icdf(torch.linspace(0.001, 0.999, img_num))
# z0_grid = z1_grid = torch.linspace(-latent_range, latent_range, img_num)

image = np.zeros((img_num * img_size, img_num * img_size))
model.eval()

for i0, z0 in enumerate(z0_grid):
    for i1, z1 in enumerate(z1_grid):
        z = torch.zeros((1, latent_dims), dtype=torch.float32, device=device)
        z[0, 0] = z0
        z[0, 1] = z1
        generated_image = model.decode(z)
        image[i1 * img_size : (i1 + 1) * img_size,
            i0 * img_size : (i0 + 1) * img_size] = \
            generated_image.cpu().detach().numpy().squeeze()

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.xlabel("z0", fontsize=24)
plt.xticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size), np.round(z0_grid.numpy(), 1).tolist())
plt.ylabel("z1", fontsize=24)
plt.yticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size), np.round(z1_grid.numpy(), 1).tolist())
plt.title("Generated images across latent space grid", fontsize=16)
plt.show()


#%% Feed in same image and see how latent representation changes with row/col position to see if model learned to use positional info

# Use background image as input
img_bkg_tensor = torch.from_numpy(img_bkg).unsqueeze(0).unsqueeze(0).float() / 255.0  # shape: (1, 1, H, W)

empty_tile_result = preprocess_tile(img_bkg, image_size=image_size)
empty_tile = empty_tile_result.unsqueeze(0).to(device)  # type: ignore # shape: (1, 1, H, W)

test_grid_size = 100

# Store reconstructions as floats in [0,1] (don't cast to uint8 — that truncates to 0)
reconstructed_grid = np.zeros((test_grid_size, test_grid_size, image_size, image_size), dtype=np.float32)

with torch.no_grad():
    # Batch processing: create all (r, c) pairs at once
    rows_all = torch.arange(test_grid_size, device=device).repeat_interleave(test_grid_size)  # shape: (test_grid_size^2,)
    cols_all = torch.arange(test_grid_size, device=device).repeat(test_grid_size)  # shape: (test_grid_size^2,)
    
    # Replicate the same tile for all positions
    empty_tile_batch = empty_tile.repeat(test_grid_size * test_grid_size, 1, 1, 1)  # shape: (test_grid_size^2, 1, H, W)
    
    # Single forward pass through encoder
    mu, logvar = model.encode(empty_tile_batch, row=rows_all, col=cols_all)
    
    # Single forward pass through decoder
    tile_reconstructed_batch = model.decode(mu)  # shape: (test_grid_size^2, 1, H, W)
    
    # Reshape back to grid
    reconstructed_grid = tile_reconstructed_batch.squeeze(1).cpu().numpy().reshape(test_grid_size, test_grid_size, image_size, image_size)

vmin = reconstructed_grid.min()
vmax = reconstructed_grid.max()

print(f"Reconstructed grid value range: [{vmin:.3f}, {vmax:.3f}]")

#%% Plot reconstructed grid with mean value of each tile (to see overall trends without visualizing each tile)
mean_reconstructed_grid = reconstructed_grid.mean(axis=(2, 3))  # shape: (test_grid_size, test_grid_size)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(mean_reconstructed_grid, cmap='viridis', vmin=mean_reconstructed_grid.min(), vmax=mean_reconstructed_grid.max(), origin='upper')
ax[0].set_xlabel("Column index (c)", fontsize=12)
ax[0].set_ylabel("Row index (r)", fontsize=12)
ax[0].set_title("Mean reconstructed value across row/col positions", fontsize=16)

ax[1].imshow(img_bkg, cmap='gray', origin='upper')
ax[1].axis('off')
ax[1].set_title("Background Image", fontsize=16)

plt.show()

#%% Plot reconstructed grid for a few selected row/col positions to see how the actual tile reconstructions change across positions
# Plot nexto t

fig, axes = plt.subplots(test_grid_size, test_grid_size, figsize=(12, 12))
for r in range(test_grid_size):
    for c in range(test_grid_size):
        ax = axes[r, c]
        ax.imshow(reconstructed_grid[r, c], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if r == 0:
            ax.text(0.5, 1.05, f"{c}", transform=ax.transAxes, ha='center', va='bottom', fontsize=12)
        if c == 0:
            ax.text(-0.05, 0.5, f"{r}", transform=ax.transAxes, ha='right', va='center', fontsize=12)
plt.suptitle("Reconstructed tiles from empty input across row/col positions", fontsize=16)
plt.tight_layout()
plt.show()

#%% Reconstruct the background image while split up in tiles


#%% Model summary

print(f"Bby you are done stop running cells and relax a lil")
# %%
