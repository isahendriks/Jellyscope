#%% import packages
### PACKAGES ###
import os
from pathlib import Path
import random
import subprocess
import sys
from time import time

import cv2
from copy import deepcopy
import numpy as np

import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import Autoencoder, BackgroundTiles, deploy_checkpoint_to_jetson, image_pipeline_df, load_tiles_from_paths_fast, preprocess_tile

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

ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data new\Binary_classifier"
ROOT_DIR_C = r"C:\Users\IsaH\Documents\Jellyscope\Training data new\Binary_classifier"   

SAMPLE_IMAGE_IDX_TRAIN = 0 # Index of the sample image to plot process (0-based)

# Debugging parameters (set to False for full training)
DEBUG = False
N_DEBUG = 5000
EPOCHS_DEBUG = 10

monitoring_effort = "Faro_260926"  # for titles and saved model names, e.g. "kristineberg_251128" 
grid_size = 16  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)
offsets_normalized = [0.0, 0.2, 0.4, 0.6, 0.8]  # List of normalized offsets [0.0, 0.2, 0.4, 0.6, 0.8] creates crops at 0%, 20%, 40%, 60%, 80% offset; set to [] or [0.0] to disable offset cropping (single crop per tile)

# model hyperparameters
image_size = 128
latent_dims = 64
hidden_channels = 32

train_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train_encoder", f"tiles{grid_size}_offsets{len(offsets_normalized)}", "no_obs")
train_og_images_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train_encoder", "OG_images")

print(f"train_tiles_path: {train_tiles_path}")
print(f"train_og_images_path: {train_og_images_path}")

# check if train_og_images_path has tiles in train_tiles_path, if not remove this image for train_og_images_path and print warning
images_train = sorted(Path(train_og_images_path).rglob("*.png"))

print(f"Total training images with tiles found: {len(images_train)}")
print(f"total training tiles found: {len(list(Path(train_tiles_path).rglob('*.png')))}")

model_name = f"../models/AE/{monitoring_effort}_AE_model{grid_size}_l{latent_dims}_img{image_size}.pth"
if DEBUG:
    model_name = model_name.replace(".pth", "_debug.pth")
    epochs = EPOCHS_DEBUG
    images_train = images_train[:N_DEBUG // (grid_size**2)]  
    
model_output_path = Path(model_name)
model_output_path.parent.mkdir(parents=True, exist_ok=True)

# training hyperparameters
batch_size = 128
learning_rate = 1e-4 * (batch_size / 64)  # scale learning rate with batch size
epochs = 50
warmup_epochs = 2  # Number of epochs to linearly increase learning rate (helps stabilize early training); early stopping only counts after this
patience = 5  # Number of epochs to wait for improvement before early stopping

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

# Residual AE: subtract the mean background at each tile's position, so the AE only has to
# model what differs from the background ("predict the background" is then residual 0).
# Brightness is kept, relative to the background. In place, to avoid a second copy of the tiles.
print(f"Subtracting mean background from {len(images_tensor)} tiles...")
bkg_tiles = BackgroundTiles(img_bkg, grid_size=grid_size, image_size=image_size)
bkg_tiles.subtract_(images_tensor, rows_tensor, cols_tensor)
print(f"Residual range: [{images_tensor.min().item():.3f}, {images_tensor.max().item():.3f}], mean {images_tensor.mean().item():.2e}")

print(f"Dataset created: images tensor shape: {images_tensor.shape}, rows tensor shape: {rows_tensor.shape}, cols tensor shape: {cols_tensor.shape}")
# Split into train and val sets 
val_split = 0.1
n_val = int(len(images_tensor) * val_split)

dataset = TensorDataset(images_tensor, rows_tensor, cols_tensor)  # Convert to TensorDataset for DataLoader
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

train_loader = DataLoader(
    train_dataset,                     # dataset object
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,                   # set >0 later if you want
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,                     # dataset object
    batch_size=batch_size,
    shuffle=False,
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

# Tiles are residuals (tile - background) now: symmetric scale, grey = background, white = brighter
v_res = sample_images[:N_hor * N_ver].abs().max().item()

fig, axs = plt.subplots(plot_grid[0], plot_grid[1], figsize=(plot_grid[1]*1.2, plot_grid[0]*1.2))
for idx, ax in enumerate(axs.flatten()):
    ax.imshow(sample_images[idx].squeeze(), cmap='gray', vmin=-v_res, vmax=v_res)
    ax.axis('off')
# plt.tight_layout()
# plt.suptitle(f"Example Training Images from {'/'.join(train_og_images_path.split(os.sep)[-3:])}", fontsize=16, fontweight="bold")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to make room for title
plt.show()

#%%###########################################################################
#### AUTOENCODER (AE) ARCHITECTURE, TRAINING, AND EVALUATION ####
##############################################################################

### Initialize model
model = Autoencoder(
        latent_dims = latent_dims,
        image_size = image_size,
        hidden_channels = hidden_channels,
        grid_size = grid_size,
        output_activation = "linear",  # residuals (tile - background) go negative, so no sigmoid
    ).to(device)

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters.")
print("model device:", next(model.parameters()).device)
print("Model architecture:")
print(model)

#%% Train model
epoch_losses, step_losses = [], []
epoch_val_losses, step_val_losses = [], []
best_val_loss = float("inf")
patience_counter = 0

n_batches = len(train_loader)
start_time = time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()  # every epoch: the val pass below switches to eval mode

    # Learning rate warmup
    if epoch < warmup_epochs:
        lr = learning_rate * (epoch + 1) / warmup_epochs
    else:
        lr = learning_rate

    for g in opt.param_groups:
        g["lr"] = lr

    running_loss = 0.0

    for x, rows, cols in train_loader:
        x = x.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)

        # reset gradients
        opt.zero_grad()

        # calculate reconstruction error
        x_hat = model(x, row=rows, col=cols)
        loss = ((x - x_hat) ** 2).mean()

        # backpropagate and optimize
        loss.backward()
        opt.step()
        
        running_loss += loss.item()

        # Log per-step losses (for plotting later)
        step_losses.append(loss.item())

    # Per-epoch averages
    epoch_losses.append(running_loss / n_batches)
    
    # Test set evaluation (using training set as proxy for now since we don't have a separate val set)
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for x, rows, cols in val_loader:
            x = x.to(device, non_blocking=True)
            rows = rows.to(device, non_blocking=True)
            cols = cols.to(device, non_blocking=True)

            x_hat = model(x, row=rows, col=cols)
            val_loss += ((x - x_hat) ** 2).mean().item()
    
    epoch_val_losses.append(val_loss / len(val_loader))
    avg_epoch_time = (time() - start_time) / (epoch + 1)
    expected_time_remaining = avg_epoch_time * (epochs - epoch - 1)
    
    print(f"  loss={epoch_losses[-1]:.3e}, val_loss={epoch_val_losses[-1]:.3e}, average epoch time: {avg_epoch_time:.1f}s, expected time remaining: {int(expected_time_remaining // 60)}m {int(expected_time_remaining % 60)}s ")
    
    # Add patience early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = deepcopy(model.state_dict())
        best_epoch = epoch + 1
        patience_counter = 0
    elif epoch >= warmup_epochs:
        patience_counter += 1
        print(f"  No improvement in validation loss for {patience_counter} epoch(s)")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
total_elapsed = time() - start_time

# Keep the best epoch's weights, not the last epoch's
model.load_state_dict(best_model_state)
print(f"Restored best model from epoch {best_epoch} (val_loss={epoch_val_losses[best_epoch - 1]:.3e})")

print(f"Training finished in {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")


#%% Plot training curves
# Plot loss curves

steps_per_epoch = len(train_loader)
epoch_indices = np.arange(1, len(epoch_losses) + 1)
step_indices = np.linspace(0, len(epoch_losses), steps_per_epoch*len(epoch_losses))  # Thinned out step indices for visibility

plt.figure(figsize=(8, 5))

# plot training loss 
plt.plot(step_indices, step_losses, label="Per-step loss", alpha=0.5, linestyle='None', marker='o', markersize=3)
plt.plot(epoch_indices, epoch_losses, label="Total loss")

# plot validation loss 
plt.plot(epoch_indices, epoch_val_losses, label="Validation loss", linestyle='--')

plt.xlabel("Epoch")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Training loss curves")
plt.grid(alpha=0.3)
plt.legend()
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
    "step_losses": step_losses,
    "epoch_val_losses": epoch_val_losses,
    "best_epoch": best_epoch,
    # Residual AE: the model sees tile - background crop (BackgroundTiles), not the raw tile.
    # Anything using this checkpoint must subtract a background the same way before encoding.
    "output_activation": "linear",
    "residual": True,
    "bkg_image": torch.from_numpy(img_bkg),  # 4512x4512 uint8 mean background used in training
}

torch.save(checkpoint, model_output_path)
print(f"Saved checkpoint to: {model_output_path}")

#%% Load model checkpoint (if restarted kernel)
if model_output_path.exists():
    print(f"Loading checkpoint from: {model_output_path}")
    checkpoint = torch.load(model_output_path, map_location=device, weights_only=False)
    model = Autoencoder(
        latent_dims=checkpoint["latent_dim"],
        image_size=checkpoint["image_size"],
        hidden_channels=checkpoint["hidden_channels"],
        grid_size=checkpoint.get("grid_size", 16),
        output_activation=checkpoint.get("output_activation", "sigmoid"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded: grid_size={checkpoint.get('grid_size', 16)}, latent_dims={checkpoint['latent_dim']}")
else:
    print(f"Checkpoint not found at {model_output_path}. Please train the model first.")


#%% Baseline check: AE reconstruction vs. mean background tile (img_bkg) at the same position
# The AE works on residuals r = tile - background, so "predict the mean background" is r_hat = 0
# and its error is simply mean(r^2). The AE should beat that clearly, otherwise it adds nothing
# over plain background subtraction. Uses the val set so tiles are not ones the AE trained on.
# Note: img_bkg is the mean over all training images (incl. the val tiles' source images),
# so this baseline is slightly optimistic -- if the AE still wins, that's a meaningful result.
model.eval()

mse_ae, mse_bkg, mse_bkg_shift = [], [], []
examples = []  # (residual, reconstructed residual, bkg) for plotting

with torch.no_grad():
    for r, rows, cols in tqdm(val_loader, desc="Baseline comparison"):
        r_hat = model(r.to(device, non_blocking=True), row=rows.to(device), col=cols.to(device)).cpu()

        mse_ae.append(((r - r_hat) ** 2).mean(dim=(1, 2, 3)))
        mse_bkg.append((r ** 2).mean(dim=(1, 2, 3)))
        # Brightness-matched baseline: background shifted to the tile's mean brightness
        # (cheap stand-in for per-image illumination changes) = residual minus its own mean
        mse_bkg_shift.append(((r - r.mean(dim=(1, 2, 3), keepdim=True)) ** 2).mean(dim=(1, 2, 3)))

        if len(examples) < 8:
            n = 8 - len(examples)
            examples.extend(zip(r[:n], r_hat[:n], bkg_tiles(rows[:n], cols[:n])))

mse_ae = torch.cat(mse_ae).numpy()
mse_bkg = torch.cat(mse_bkg).numpy()
mse_bkg_shift = torch.cat(mse_bkg_shift).numpy()

print(f"Val tiles compared: {len(mse_ae)}")
print(f"{'':28s}{'mean MSE':>12s}{'median MSE':>14s}")
for name, m in [("AE reconstruction", mse_ae), ("Mean background", mse_bkg), ("Brightness-matched bkg", mse_bkg_shift)]:
    print(f"{name:28s}{m.mean():12.6f}{np.median(m):14.6f}")
print(f"AE beats mean background on       {100 * (mse_ae < mse_bkg).mean():.1f}% of tiles "
      f"(median ratio bkg/AE = {np.median(mse_bkg / mse_ae):.2f}x)")
print(f"AE beats brightness-matched bkg on {100 * (mse_ae < mse_bkg_shift).mean():.1f}% of tiles "
      f"(median ratio bkg/AE = {np.median(mse_bkg_shift / mse_ae):.2f}x)")

# Histogram of per-tile MSE (log x-axis, since values span orders of magnitude)
all_mse = np.concatenate([mse_ae, mse_bkg, mse_bkg_shift])
bins = np.logspace(np.log10(all_mse.min() + 1e-12), np.log10(all_mse.max()), 80)
plt.figure(figsize=(8, 4))
plt.hist(mse_ae, bins=bins, alpha=0.6, label="AE reconstruction")
plt.hist(mse_bkg, bins=bins, alpha=0.6, label="Mean background")
plt.hist(mse_bkg_shift, bins=bins, alpha=0.6, label="Brightness-matched bkg")
plt.xscale("log")
plt.xlabel("Per-tile MSE")
plt.ylabel("Count")
plt.title("Val tiles: AE vs. background baseline")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Example tiles, back in image space: input / AE reconstruction / mean background / error map.
# Rows 1-3 share one grey scale stretched to the brightest input (tiles are too dark for 0..1);
# the error map (r - r_hat)^2 is what segmentation scores on.
v_img = max((r + bk).max().item() for r, _, bk in examples)
v_err = max(((r - rh) ** 2).max().item() for r, rh, _ in examples)
fig, axs = plt.subplots(4, len(examples), figsize=(len(examples) * 1.6, 4 * 1.7))
for i, (r, rh, bk) in enumerate(examples):
    panels = [(r + bk, "Input", v_img), (rh + bk, "AE", v_img), (bk, "Mean bkg", v_img), ((r - rh) ** 2, "Error", v_err)]
    for j, (img, lbl, vmax) in enumerate(panels):
        axs[j, i].imshow(img.squeeze(), cmap="gray" if lbl != "Error" else "magma", vmin=0, vmax=vmax)
        axs[j, i].axis("off")
        if i == 0:
            axs[j, i].set_title(lbl, fontsize=9, loc="left")
plt.tight_layout()
plt.show()

#%% Plot latent space for random subset oftraining tiles to see if it looks structured (e.g. if row/col info is encoded in specific dimensions, or if there are clusters)
model.eval()

N_subset = 10000

train_dataset_random_subset = torch.utils.data.Subset(train_dataset, random.sample(range(len(train_dataset)), min(N_subset, len(train_dataset))))

train_loader_random_subset = DataLoader(
    train_dataset_random_subset,       # dataset object
    batch_size=512,
    shuffle=True,
    num_workers=0,                   # set >0 later if you want
    pin_memory=torch.cuda.is_available()
)

all_mu = []

with torch.no_grad():
    for x, rows, cols in train_loader_random_subset:
        mu = model.encode(
            x.to(device),
            row=rows.to(device),
            col=cols.to(device)
        )
        all_mu.append(mu.cpu())

all_mu = torch.cat(all_mu).numpy()

plt.figure(figsize=(8, 6))
plt.scatter(all_mu[:, 0], all_mu[:, 1], alpha=0.5, s=10)
plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.title("Latent space distribution of training tiles")
plt.grid(alpha=0.3)
plt.show()

### PCA on latent space to see if there are dominant axes of variation
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
z_pca = pca.fit_transform(all_mu)

plt.figure(figsize=(2, 2))
plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.5, s=5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
# plt.title("PCA of latent space distribution")
plt.xlim(-5, 5)
# plt.axis('equal')
plt.ylim(-5, 5)
plt.xticks([-4, -2, 0, 2, 4])
plt.yticks([-4, -2, 0, 2, 4])

plt.grid(alpha=0.7)

print(f'all_mu.shape: {all_mu.shape}')

# Print mean and std of each latent dimension
stds= np.std(all_mu, axis=0)
means = np.mean(all_mu, axis=0)
for i in range(all_mu.shape[1]):
    print(f'Latent dim {i}: mean={means[i]:.3f}, std={stds[i]:.3f}')

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

# Use an empty tile as input: in residual space, "exactly the background" is all zeros
empty_tile = torch.zeros(1, 1, image_size, image_size, device=device)  # shape: (1, 1, H, W)

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
    mu = model.encode(empty_tile_batch, row=rows_all, col=cols_all)
    
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

ax[1].imshow(img_bkg, cmap='gray', origin='upper', vmin=0, vmax=255)
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


#%% Deploy checkpoint to the Jetson
# Copied as <model name>_<DEPLOY_TAG>.pth, so it never replaces a model the Jetson is running.
# Use the same tag in train_DNN.py so the AE and its scorer are easy to pair up, then point
# Jetson_monitoring/Monitor/config.py at the printed names.
DEPLOY_TAG = "residual_v1"

deploy_checkpoint_to_jetson(model_output_path, tag=DEPLOY_TAG)
# %%
