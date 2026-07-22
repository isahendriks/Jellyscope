
#%% import packages
### PACKAGES ###
import os
import torch
import re

import deeplay as dl
import deeptrack as dt
from torch.distributions.normal import Normal

import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import random 
from pathlib import Path
print("Packages imported successfully yiho.")

#%% Wrap to force a 2D flat latent vector
class AE(nn.Module):
    def __init__(self, base_model, input_size=128, latent_dim=2):
        super().__init__()
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            z = self.encoder(dummy)
            self.enc_shape = z.shape[1:]   # (C, H, W)
            flat_dim = z.numel()

        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

    def encode(self, x):
        z = self.encoder(x).view(x.size(0), -1)
        return self.to_latent(z)

    def decode(self, z):
        z = self.from_latent(z).view(-1, *self.enc_shape)
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

#%% Set parameters and paths
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
train_files_path = os.path.join(ROOT_DIR_R, "train", "tiles6_aug")
test_files_path = os.path.join(ROOT_DIR_R, "test", "tiles6")

# Set parameters
train_files = list(Path(train_files_path).rglob("*.png"))
n_train = len(train_files)  # Use all available training images

# training hyperparameters
batch_size = 32
image_size = 128
latent_dim = 6
learning_rate = 1e-3
epochs = 30

#%% Load train data
train_source = dt.sources.Source(path=train_files)

image_pip = (
    dt.LoadImage(train_source.path, to_grayscale=True, ndim=2) >>  # Load as HxW
    dt.Resize((image_size,image_size)) >>  # Resize to 128x128
    dt.Lambda(lambda: (lambda x: x[..., :1])) >>  # Take only first channel -> HxWx1
    # dt.NormalizeMinMax() >> 
    dt.pytorch.ToTensor(dtype=torch.float, permute_mode="always")  # HxWx1 -> 1xHxW
)

train_dataset = dt.pytorch.Dataset(image_pip & image_pip, inputs=train_source)
train_loader = dl.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# load test data (labeled as obs/no_obs folders)
test_files = dt.sources.ImageFolder(root=test_files_path)

# Define pipelines for test data
# label_pip = dt.Value(test_files.label_name[0]) >> int
label_pip = dt.Value(test_files.label)
test_image_pip = (
    dt.LoadImage(test_files.path, to_grayscale=True, ndim=2) >>
    dt.Resize((image_size, image_size)) >>
    dt.Lambda(lambda: (lambda x: x[..., :1])) >>
    # dt.NormalizeMinMax() >>
    dt.pytorch.ToTensor(dtype=torch.float, permute_mode="always")
)
# Create dataset for test data
test_dataset = dt.pytorch.Dataset(test_image_pip & label_pip, inputs=test_files)

# Define dataloader for test dataset
test_loader = dl.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True)

classes = test_files.classes
labels = np.array([item["label"] for item in test_files])
counts = np.bincount(labels)

print("\n" + "="*60)
print(f"Loaded {len(train_dataset)} training images from: {train_files_path}")
print(f"Loaded {len(test_dataset)} test images from: {test_files_path}")
print(f"number of images per class: obs {counts[1]}, no_obs {counts[0]}")
print("="*60)

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
    ax.imshow(x[idx].squeeze(), cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.suptitle(f"Example Training Images from {'/'.join(train_files_path.split(os.sep)[-3:])}", fontsize=16, fontweight="bold")

#%% Plot some example test images with their labels (obs vs no_obs)

# plot 10 test images of obs and 10 test images of no_obs
N = 7
randin_obs = random.sample([i for i, label in enumerate(labels) if label == 1], N)
randin_no_obs = random.sample([i for i, label in enumerate(labels) if label == 0], N)

fig, axs = plt.subplots(2, N, figsize=(N*1.2, 2*1.2))
for idx, rand_i in enumerate(randin_obs):
    image, _ = test_dataset[rand_i]
    axs[0,idx].imshow(image.squeeze(), cmap='gray', vmin=0, vmax=255)
    axs[0,idx].axis('off')
    if idx == 1:
        axs[0,idx].set_ylabel(f"obs", fontsize=8, fontweight='bold')

for idx, rand_i in enumerate(randin_no_obs):
    image, _ = test_dataset[rand_i]
    axs[1,idx].imshow(image.squeeze(), cmap='gray', vmin=0, vmax=255)
    axs[1,idx].axis('off')
    if idx == 1:
        axs[1,idx].set_ylabel(f"no obs", fontsize=8, fontweight='bold')
    
plt.tight_layout()
plt.show()

#%%###########################################################################
#### CONVOLUTIONAL AUTOENCODER (AE) ARCHITECTURE, TRAINING, AND EVALUATION ####
##############################################################################
# Base conv autoencoder
base_ae = dl.ConvolutionalEncoderDecoder2d(
    in_channels=1,
    encoder_channels=[32, 64, 128, 256],  # similar scale to your VAE channels
    out_channels=1,
    out_activation=nn.Identity,
)
base_ae[..., "layer"].configure(nn.Conv2d, kernel_size=3, padding="same")
base_ae[..., "pool"].configure(nn.MaxPool2d, kernel_size=2, stride=2)
base_ae[..., "upsample"].configure(nn.ConvTranspose2d, kernel_size=2, stride=2)

# Build lazy deeplay layers on CPU first
with torch.no_grad():
    _ = base_ae(torch.zeros(1, 1, image_size, image_size))

autoencoder = AE(base_ae, input_size=image_size, latent_dim=latent_dim)

# Define autoencoder regressor
ae = dl.Regressor(
    model=autoencoder,
    loss=nn.L1Loss(),
    optimizer=dl.Adam(lr=learning_rate),
).create()

# Ensure deeplay lazy layers are fully materialized on CPU.
ae = ae.to("cpu")
with torch.no_grad():
    _ = ae(torch.zeros(1, 1, image_size, image_size))

print(ae)

#%% Train the autoencoder

ae_trainer = dl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1)
ae_trainer.fit(ae, train_loader)

ae_history = ae_trainer.history
ae_history.plot()


# save model weights
model_output_path = Path("models/ae_model_aug.pth")
torch.save(ae.state_dict(), model_output_path)
print(f"Saved model weights to: {model_output_path}")

# %% Load model weights (if needed)
model_input_path = Path("models/ae_model.pth")
ae.load_state_dict(torch.load(model_input_path))
print(f"Loaded model weights from: {model_input_path}")

#%% Generate images in latent space grid
model = ae

img_num, img_size = 21, image_size
z0_grid = z1_grid = Normal(0, 1).icdf(torch.linspace(0.001, 0.999, img_num))

image = np.zeros((img_num * img_size, img_num * img_size))
model.eval()

latent_dim_model = model.model.to_latent.out_features
if latent_dim_model < 2:
    raise ValueError(f"Expected latent_dim >= 2 for 2D grid plot, got {latent_dim_model}")

for i0, z0 in enumerate(z0_grid):
    for i1, z1 in enumerate(z1_grid):
        z = torch.zeros((1, latent_dim_model), dtype=torch.float32)
        z[0, 0] = z0
        z[0, 1] = z1
        generated_image = model.model.decode(z).detach()
        image[i1 * img_size : (i1 + 1) * img_size,
              i0 * img_size : (i0 + 1) * img_size] = generated_image.cpu().numpy().squeeze()

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap="gray")
plt.xlabel("z0", fontsize=24)
plt.xticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size),
           np.round(z0_grid.numpy(), 1))
plt.ylabel("z1", fontsize=24)
plt.yticks(np.arange(0.5 * img_size, (0.5 + img_num) * img_size, img_size),
           np.round(z1_grid.numpy(), 1))
plt.show()

#%% Load all test images and get latent space representations
all_z = []
all_labels = []

model.eval()
n_batches = len(test_loader)
z_list, test_labels = [], []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break

        image = batch[0]
        label = batch[1]

        z = model.model.encode(image)
        z_list.append(z.cpu().detach())
        test_labels.append(label)

        print(f"Processed batch {batch_idx + 1}/{n_batches}, "
              f"Class distribution in batch: obs={torch.sum(batch[1]==1)}, "
              f"no_obs={torch.sum(batch[1]==0)}", end="\r")

z_array = torch.cat(z_list, dim=0).numpy()
test_labels = torch.cat(test_labels, dim=0).numpy()

print(f"\nz_array shape: {z_array.shape}")
print(f"test_labels shape: {test_labels.shape}")
print(f"Class distribution in test set: obs={np.sum(test_labels==1)}, no_obs={np.sum(test_labels==0)}")

#%% plot latent space representations colored by obs vs no_obs
z_obs = np.where(test_labels==0, z_array, np.nan)
z_no_obs = np.where(test_labels==1, z_array, np.nan)

z_obs = z_obs[~np.isnan(z_obs).any(axis=1)]
z_no_obs = z_no_obs[~np.isnan(z_no_obs).any(axis=1)]

plt.figure(figsize=(12, 10))
plt.scatter(z_no_obs[:, 0], z_no_obs[:, 1], s=50, c="blue", label="no obs", alpha=0.6, edgecolors='black', linewidth=0.5)
plt.scatter(z_obs[:, 0], z_obs[:, 1], s=50, c="orange", label="obs", alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel("z[:, 0]", fontsize=24)
plt.ylabel("z[:, 1]", fontsize=24)
plt.grid()
plt.gca().invert_yaxis()
plt.axis("equal")
plt.legend(["no obs", "obs"], fontsize=12)
plt.show()
# %%
