
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import chi2
from pathlib import Path
from torch.utils.data import DataLoader

# Define convolutional encoder and decoder
class Encoder(nn.Module):
    def __init__(self, latent_dims=2, image_size=128, hidden_channels=32, grid_size=16):
        super().__init__()
        self.image_size = image_size

        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),  # /8
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, stride=2, padding=1),  # /16
            nn.ReLU(inplace=True),
        )
        
        # Embed the position of each tile in the grid and add to the features before the final FC layer
        embedding_dim = hidden_channels * 2
        self.row_embedding = nn.Embedding(grid_size, embedding_dim)
        self.col_embedding = nn.Embedding(grid_size, embedding_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_size, self.image_size)
            feat = self.conv(dummy)
        self.feat_shape = feat.shape[1:]  # (C, H, W)
        self.feat_dim = feat.numel()      # flattened size

        embedding_total_dim = embedding_dim * 2  # row + col embeddings
        combined_dim = self.feat_dim + embedding_total_dim
        
        self.fc = nn.Linear(combined_dim, latent_dims)

    def forward(self, x, row = None, col = None):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        
        # Get positional embeddings
        if row is not None and col is not None:
            row = row.to(x.device).long()
            col = col.to(x.device).long()
            
            row_emb = self.row_embedding(row)  # shape: (B, embedding_dim)
            col_emb = self.col_embedding(col)  # shape: (B, embedding_dim)
            
            # Concatenate visual features with positional embeddings
            h = torch.cat([h, row_emb, col_emb], dim=1)  # shape: (B, feat_dim + 2*embedding_dim)

        z = self.fc(h)
        return z


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims=2, image_size=128, hidden_channels=32, grid_size=16):
        super().__init__()
        self.image_size = image_size
        self.kl = torch.tensor(0.0)
        self.grid_size = grid_size
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Embed the position of each tile in the grid and add to the features before the final FC layer
        embedding_dim = hidden_channels * 2
        self.row_embedding = nn.Embedding(grid_size, embedding_dim)
        self.col_embedding = nn.Embedding(grid_size, embedding_dim)
        
        # Compute output feature shape 
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_size, self.image_size)
            feat = self.conv(dummy)
        self.feat_shape = feat.shape[1:]
        self.feat_dim = feat.numel()
        
        # Linear layer to combine conv features + embeddings
        embedding_total_dim = embedding_dim * 2  # row + col embeddings
        combined_dim = self.feat_dim + embedding_total_dim

        self.fc_mu = nn.Linear(combined_dim, latent_dims)
        self.fc_logvar = nn.Linear(combined_dim, latent_dims)

    def forward(self, x, row = None, col = None):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)

        if row is not None and col is not None:
            row = row.to(x.device).long()
            col = col.to(x.device).long()

            row_emb = self.row_embedding(row)
            col_emb = self.col_embedding(col)
            h = torch.cat([h, row_emb, col_emb], dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Deterministic reconstruction during evaluation

        kl_per_dim = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)  # shape (B, L)
        kl_per_sample = kl_per_dim.sum(dim=1)                          # shape (B,)
        self.kl = kl_per_sample.mean()                                 # scalar average across batch
        
        # debug info: average KL per latent dim (for logging)
        self.kl_per_dim = kl_per_dim.mean(dim=0).detach().cpu()        # (L,) CPU tensor
        
        return z

    def encode_params(self, x, row = None, col = None):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        
        # Get positional embeddings
        if row is not None and col is not None:
            row = row.to(x.device).long()
            col = col.to(x.device).long()
            
            row_emb = self.row_embedding(row)  # shape: (B, embedding_dim)
            col_emb = self.col_embedding(col)  # shape: (B, embedding_dim)
            
            # Concatenate visual features with positional embeddings
            h = torch.cat([h, row_emb, col_emb], dim=1)  # shape: (B, feat_dim + 2*embedding_dim)
             
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims, feat_shape, image_size=128, hidden_channels=32):
        super().__init__()
        self.image_size = image_size
        self.feat_shape = feat_shape  # (C, H, W)
        feat_dim = feat_shape[0] * feat_shape[1] * feat_shape[2]

        self.fc = nn.Linear(latent_dims, feat_dim)

        c = feat_shape[0]
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, hidden_channels * 4, kernel_size=4, stride=2, padding=1),   # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),  # x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),      # x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, 1, kernel_size=4, stride=2, padding=1),             # x16
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.shape[0], *self.feat_shape)
        x_hat = self.deconv(h)

        if x_hat.shape[-1] != self.image_size or x_hat.shape[-2] != self.image_size:
            x_hat = F.interpolate(x_hat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        return x_hat


class Autoencoder(nn.Module):
    def __init__(self, latent_dims, image_size=128, in_channels=1, hidden_channels=32):
        super().__init__()
        self.encoder = Encoder(latent_dims, image_size, hidden_channels)
        self.decoder = Decoder(
            latent_dims,
            feat_shape=self.encoder.feat_shape,
            image_size=image_size,
            hidden_channels=hidden_channels,
        )

    def encode(self, x, row=None, col=None):
        return self.encoder(x, row=row, col=col)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, row=None, col=None):
        z = self.encoder(x, row=row, col=col)
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, image_size=128, hidden_channels=32, grid_size=16):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims, image_size, hidden_channels, grid_size)
        self.decoder = Decoder(
            latent_dims,
            feat_shape=self.encoder.feat_shape,
            image_size=image_size,
            hidden_channels=hidden_channels,
        )

    def encode(self, x, row=None, col=None):
        return self.encoder.encode_params(x, row=row, col=col)  # returns mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, row=None, col=None):
        z = self.encoder(x, row=row, col=col)
        return self.decoder(z)


def train_image_pipeline(batch_paths, image_size=128):
    images = []
    rows = []
    cols = []

    for path in batch_paths:
        row = int(re.search(r'_r(\d+)_c\d+', path.name).group(1))
        col = int(re.search(r'_r\d+_c(\d+)', path.name).group(1))
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = preprocess_tile(img, image_size=image_size)

        if img is None:
            continue

        images.append(img)
        rows.append(row)
        cols.append(col)

    x = torch.stack(images, dim=0)
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    return x, x, rows, cols


def eval_image_pipeline(batch_paths, image_size=128):
    images = []
    labels = []
    cols = []
    rows = []
    
    for path in batch_paths:
        # if path is in obs folder, label=TrUE, else FALSE (empty)
        label = False if "no_obs" in str(path).lower() else True
        
        # get tile col and row
        tile_match = re.search(r'_r(\d+)_c(\d+)', path.name)

        row = int(tile_match.group(1))
        col = int(tile_match.group(2))

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = preprocess_tile(img, image_size=image_size)
        if img is None:
            continue

        labels.append(label)
        rows.append(row)
        cols.append(col)
        images.append(img)

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.bool)
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    
    return images, labels, rows, cols

def test_image_pipeline(batch, image_size=128):
    """Collate fn for test tiles already split as (tile, row, col, image_name)."""

    all_tiles = []
    cols = []
    rows = []
    image_names = []
    
    for item in batch:
        tile, row, col, image_name = item
        if tile is None:
            continue

        tile_tensor = preprocess_tile(tile, image_size=image_size)
        if tile_tensor is None:
            continue

        rows.append(int(row))
        cols.append(int(col))
        image_names.append(image_name)
        all_tiles.append(tile_tensor)

    tiles_tensor = torch.stack(all_tiles, dim=0)
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)

    return tiles_tensor, rows, cols, image_names


def preprocess_tile(tile, image_size=128):
    """Resize tile and scale dynamic range to [0, 1], returning tensor of shape (1, H, W)."""
    if tile is None:
        print("⚠ Received None tile for preprocessing, skipping")
        
        return None

    if tile.ndim == 3:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

    tile_resized = cv2.resize(tile, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tile_tensor = torch.from_numpy(tile_resized).float().unsqueeze(0) / 255.0
    return tile_tensor


def split_image_into_tiles(image, grid_size):
    """
    Split a 4512x4512 image into square tiles of size tile_size x tile_size.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (4512x4512)
    tile_size : int
        Size of each square tile (width and height)
    
    Returns:
    --------
    list of tuples
        List of (tile_image, row_idx, col_idx) tuples
    """
    tiles = []
    
    im_shape = image.shape
    if im_shape[0] != 4512 or im_shape[1] != 4512:
        raise ValueError(f"Expected image size of 4512x4512, but got {im_shape[0]}x{im_shape[1]}")
    
    # Calculate how many tiles fit (known: 4512x4512 image)  
    n_rows = grid_size
    n_cols = grid_size
    tile_size = 4512 // grid_size
    
    # Extract tiles
    for row in range(n_rows):
        for col in range(n_cols):
            y_start = row * tile_size
            y_end = y_start + tile_size
            x_start = col * tile_size
            x_end = x_start + tile_size
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append((tile, row, col))
    
    return tiles