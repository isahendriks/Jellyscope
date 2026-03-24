#%% import packages
### PACKAGES ###
import os
from pathlib import Path
import random
from time import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from functions import Encoder, VariationalEncoder, Decoder, Autoencoder, VariationalAutoencoder, train_image_pipeline, preprocess_tile

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

monitoring_effort = "Kristineberg_250915"  # for titles and saved model names, e.g. "kristineberg_251128" or "skagerrak_220915"
grid_size = 32  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)

# model hyperparameters
image_size = 128
latent_dims = 6
hidden_channels = 32

train_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train", f"tiles{grid_size}")
train_og_images_path = os.path.join(ROOT_DIR_C, monitoring_effort, "train", "OG_images")

print(f"train_tiles_path: {train_tiles_path}")
print(f"train_og_images_path: {train_og_images_path}")

# check if train_og_images_path has tiles in train_tiles_path, if not remove this image for train_og_images_path and print warning
images_train = sorted(Path(train_og_images_path).rglob("*.png"))

print(f"Total training images with tiles found: {len(images_train)}")
print(f"total training tiles found: {len(list(Path(train_tiles_path).rglob('*.png')))}")

model_name = "models/vae_model_" + str(monitoring_effort) + "_" + str(grid_size) + "_l" + str(latent_dims) + ".pth"
if DEBUG:
    model_name = model_name.replace(".pth", "_debug.pth")
model_output_path = Path(model_name)
model_output_path.parent.mkdir(parents=True, exist_ok=True)

# training hyperparameters
batch_size = 64
learning_rate = 1e-4 * (batch_size / 64)  # scale learning rate with batch size
epochs = 100
warmup_epochs = 30  # Number of epochs to linearly increase learning rate (helps stabilize early training)
if DEBUG:
    print(f"DEBUG MODE: Using only {N_DEBUG} tiles and training for {EPOCHS_DEBUG} epochs \n if saving enabled model name will be: {model_output_path.name}")
#%% Load train data

train_tiles = sorted(Path(train_tiles_path).rglob("*.png"))

if DEBUG:
    train_tiles = train_tiles[:N_DEBUG]
    epochs = EPOCHS_DEBUG

train_loader = DataLoader(
    train_tiles,                     # list of paths = dataset
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_image_pipeline,
    num_workers=0,                   # set >0 later if you want
    pin_memory=torch.cuda.is_available()
)

# Display configuration
print("\n" + "="*120)
print(f"Loaded {len(train_loader)*batch_size} training tiles from: {train_tiles_path}")
print("="*120)

# Sanity-check one batch to verify row/col loading
x_chk, y_chk, rows_chk, cols_chk = next(iter(train_loader))
print(
    f"Sanity batch -> x: {tuple(x_chk.shape)}, rows: {tuple(rows_chk.shape)} ({rows_chk.dtype}), "
    f"cols: {tuple(cols_chk.shape)} ({cols_chk.dtype})"
)
print(
    f"row range: [{rows_chk.min().item()}, {rows_chk.max().item()}], "
    f"col range: [{cols_chk.min().item()}, {cols_chk.max().item()}]"
)
assert rows_chk.ndim == 1 and cols_chk.ndim == 1, "rows/cols must be 1D batch tensors"
assert rows_chk.shape[0] == x_chk.shape[0] and cols_chk.shape[0] == x_chk.shape[0], "rows/cols must match batch size"
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
# _i_patterns = ("?", "??", "???", "????", "?????", "??????", "???????", "????????")

sample_image_tiles_path_train = sorted({
    path
    for r_pat in _r_patterns
    for c_pat in _c_patterns
    for path in Path(train_tiles_path).glob(
        f"{sample_image_name_train}_r{r_pat}_c{c_pat}_*.png"
    )
})

# Sort images by index (the last number in the filename) to ensure correct order in grid
sample_image_tiles_path_train = sorted(
    sample_image_tiles_path_train,
    key=lambda p: int(p.stem.split("_")[-1][1:])  # Extract the index after 'i' and convert to int
)

print(f"number of tiles selected for plotting: {len(sample_image_tiles_path_train)}")

#%%
print(f"Loaded sample train image: {sample_image_name_train}, #tiles: {len(sample_image_tiles_path_train)}")
 
### plot original image
sample_image_train = cv2.imread(str(sample_image_path_train)) # Load image
num_tiles = len(sample_image_tiles_path_train)

print(f"Original image shape (H, W, C): {sample_image_train.shape}")
print(f"Original tile shape (H, W, C): {4512//grid_size, 4512//grid_size, 3}")

plt.figure(figsize=(grid_size*2.4, grid_size*2.4))
plt.imshow(sample_image_train, cmap="gray")
plt.title(f"Original image: {sample_image_name_train}")
plt.axis("off")

### plot original tiles
fig,axis = plt.subplots(grid_size, grid_size, figsize=(grid_size*2.4, grid_size*2.4))
for i in range(num_tiles):
    tile_path = sample_image_tiles_path_train[i]
    tile_image = cv2.imread(str(tile_path))
    row, col = divmod(i, grid_size)
    axis[row, col].imshow(tile_image, cmap="gray", vmin=0, vmax=1)
    axis[row, col].axis("off")

fig.suptitle(f"Original {grid_size}x{grid_size} tiles from {sample_image_name_train}", fontsize=20, fontweight="bold")
plt.tight_layout()

### plot tiles after image pipeline
fig,axis = plt.subplots(grid_size, grid_size, figsize=(grid_size*2.4, grid_size*2.4))
for i in range(num_tiles):
    tile_path = sample_image_tiles_path_train[i]
    tile_image_raw = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)
    tile_tensor = preprocess_tile(tile_image_raw, image_size=image_size)
    if tile_tensor is None:
        continue
    tile_image_pip = tile_tensor.squeeze().numpy()
    row, col = divmod(i, grid_size)
    axis[row, col].imshow(tile_image_pip, cmap="gray", vmin=0, vmax=1)
    axis[row, col].axis("off")

fig.suptitle(f"Processed {grid_size}x{grid_size} tiles from {sample_image_name_train}", fontsize=20, fontweight="bold")
plt.tight_layout()

#%% Plot example train images from dataloader
# Get one batch
batch = next(iter(train_loader))
images = batch[0]  # Take the first 64 images from the batch
sample_images = images[:batch_size]  # Ensure we only take batch_size images
_, im_height, im_width = sample_images[0].shape

print(f"Image shape:", im_width, im_height)

N_hor = 8 #int(np.sqrt(batch_size))
N_ver = 3 #int(N_hor/2)

plot_grid = (N_ver, N_hor)

fig, axs = plt.subplots(plot_grid[0], plot_grid[1], figsize=(plot_grid[1]*1.2, plot_grid[0]*1.2))
for idx, ax in enumerate(axs.flatten()):
    ax.imshow(sample_images[idx].squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
# plt.tight_layout()
# plt.suptitle(f"Example Training Images from {'/'.join(train_og_images_path.split(os.sep)[-3:])}", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
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

print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters.")
print("model device:", next(model.parameters()).device)
print("Model architecture:")
print(model)

#%% Train model
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch_losses, recon_losses, kl_losses = [], [], []
step_losses, step_recon_losses, step_kl_losses = [], [], []
kl_per_dim_history = []  # For tracking KL per latent dim over time
model.train()


# before training
n_batches = len(train_loader)
total_steps = epochs * n_batches        # total batches across all epochs
global_step = 0
start_time = time()

# OUTER: epoch-level progress (one bar)
outer = tqdm(
    total=epochs,                    # outer counts epochs now
    desc="Epochs",
    position=0,
    leave=True,
    dynamic_ncols=True,
    smoothing=0.1,
    bar_format="{l_bar}{bar:28} [{elapsed}<{remaining}, {rate_fmt}]",
)

for epoch in range(epochs):
    if epoch < warmup_epochs:
        lr = learning_rate * (epoch + 1) / warmup_epochs
    else:
        lr = learning_rate

    for g in opt.param_groups:
        g['lr'] = lr
        
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    # INNER: per-epoch (batches) progress
    inner = tqdm(
        train_loader,
        desc=f" Epoch {epoch + 1:02d}/{epochs:02d}",
        total=n_batches,
        position=1,
        leave=False,                # inner disappears after epoch
        dynamic_ncols=True,
        smoothing=0.1,
        bar_format="{l_bar}{bar:28} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for batch_idx, (x, y, rows, cols) in enumerate(inner, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        rows = rows.to(device, non_blocking=True)
        cols = cols.to(device, non_blocking=True)

        opt.zero_grad()
        x_hat = model(x, row=rows, col=cols)

        recon = ((x - x_hat) ** 2).sum()
        
        if isinstance(model, Autoencoder):  
            loss = recon
            kl = torch.tensor(0.0)
        elif isinstance(model, VariationalAutoencoder):
            kl = model.encoder.kl
            beta = min(1.0, (epoch + 1) / warmup_epochs)  # Linear warmup of KL weight
            loss = recon + beta * kl
            
        else:
            raise ValueError(f"Unknown model type: {type(model).__name__}")

        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_recon += recon.item()
        running_kl += kl.item()

        step_losses.append(loss.item())
        step_recon_losses.append(recon.item())
        step_kl_losses.append(kl.item())

        global_step += 1
        # NOTE: removed outer.update(1) here (outer is epoch-level)

        # compute ETA (seconds left) for entire training — shown on inner
        elapsed = time() - start_time
        if global_step > 0:
            secs_per_step = elapsed / global_step
            steps_left = max(total_steps - global_step, 0)
            eta_seconds = int(secs_per_step * steps_left)
            hrs, rem = divmod(eta_seconds, 3600)
            mins, secs = divmod(rem, 60)
            eta_str = f"{hrs:d}:{mins:02d}:{secs:02d}"
        else:
            eta_str = "?:?:?"

        step = batch_idx
        postfix = {
            "loss": f"{loss.item():.2f}",
            "recon": f"{recon.item():.2f}",
            "kl": f"{kl.item():.2f}",
            "avg": f"{running_loss / step:.2f}",
            "lr": f"{opt.param_groups[0]['lr']:.1e}",
            "ETA": eta_str,            # ETA for whole training shown here
        }
        if torch.cuda.is_available():
            postfix["gpuGB"] = f"{torch.cuda.memory_allocated(device) / 1e9:.2f}"

        inner.set_postfix(postfix)
    
    inner.close()

    # per-epoch averages
    epoch_losses.append(running_loss / n_batches)
    recon_losses.append(running_recon / n_batches)
    kl_losses.append(running_kl / n_batches)
    if isinstance(model, VariationalAutoencoder):
        kl_per_dim_history.append(model.encoder.kl_per_dim.cpu().numpy())

    # one epoch done -> advance epoch-level outer bar
    outer.update(1)

    # recompute ETA for whole training and show on outer summary
    total_elapsed = time() - start_time
    if global_step > 0:
        secs_per_step = total_elapsed / global_step
        steps_left = max(total_steps - global_step, 0)
        eta_seconds = int(secs_per_step * steps_left)
        hrs, rem = divmod(eta_seconds, 3600)
        mins, secs = divmod(rem, 60)
        eta_str = f"{hrs:d}:{mins:02d}:{secs:02d}"
    else:
        eta_str = "?:?:?"

    outer.set_postfix({
        "epoch": f"{epoch+1}/{epochs}",
        "ep_loss": f"{epoch_losses[-1]:.3f}",
        "ETA": eta_str
    })

    print(
        f"Epoch {epoch+1}/{epochs} | loss={epoch_losses[-1]:.3f} "
        f"| recon={recon_losses[-1]:.3f} | kl={kl_losses[-1]:.3f}"
    )

outer.close()
total_elapsed = time() - start_time
print(f"Training finished in {int(total_elapsed//60)}m {int(total_elapsed%60)}s")


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
plt.xticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size), np.round(z0_grid.numpy(), 1))
plt.ylabel("z1", fontsize=24)
plt.yticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size), np.round(z1_grid.numpy(), 1))
plt.show()


#%% Model summary

print(f"Bby you are done stop running cells and relax a lil")

# %%
