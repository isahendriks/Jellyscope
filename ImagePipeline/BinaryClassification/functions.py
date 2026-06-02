
import re
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from pathlib import Path
import pandas as pd

# Encoder 
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

### Variational Encoder for segmentation, with multi-level positional embeddings 
class VariationalEncoder(nn.Module):
    """Variational Encoder with continuous positional features"""
    def __init__(self, latent_dims=2, image_size=128, hidden_channels=32, grid_size=16):
        super().__init__()
        self.image_size = image_size
        self.kl = torch.tensor(0.0)
        self.grid_size = grid_size
        
        # Convolutional encoder
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
        
        # Shared positional embedding module
        embedding_total_dim = hidden_channels * 8
        self.pos_embedding = PositionalEmbedding(input_dim=10, output_dim=embedding_total_dim)
        
        # Compute output feature shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_size, self.image_size)
            feat = self.conv(dummy)
        self.feat_shape = feat.shape[1:]
        self.feat_dim = feat.numel()
        
        # Multi-level position fusion layers
        self.pos_fc1 = nn.Linear(embedding_total_dim, hidden_channels * 8)
        self.pos_bn1 = nn.BatchNorm1d(hidden_channels * 8)
        
        combined_dim = self.feat_dim + hidden_channels * 8
        
        self.fusion_fc1 = nn.Linear(combined_dim, hidden_channels * 8)
        self.fusion_bn1 = nn.BatchNorm1d(hidden_channels * 8)
        self.fusion_dropout1 = nn.Dropout(0.2)
        
        self.fusion_fc2 = nn.Linear(hidden_channels * 8 + embedding_total_dim, hidden_channels * 8)
        self.fusion_bn2 = nn.BatchNorm1d(hidden_channels * 8)
        self.fusion_dropout2 = nn.Dropout(0.2)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_channels * 8, latent_dims)
        self.fc_logvar = nn.Linear(hidden_channels * 8, latent_dims)

    def forward(self, x, row=None, col=None):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)

        if row is not None and col is not None:
            pos_features = compute_position_features(row, col, self.grid_size, x.device)
            pos_emb = self.pos_embedding(pos_features)
            
            # LEVEL 1: Process position embeddings through FC
            pos_processed = self.pos_fc1(pos_emb)
            pos_processed = self.pos_bn1(pos_processed)
            pos_processed = F.relu(pos_processed)
            
            # LEVEL 2: First fusion of visual features + processed position
            combined = torch.cat([h, pos_processed], dim=1)
            h = self.fusion_fc1(combined)
            h = self.fusion_bn1(h)
            h = F.relu(h)
            h = self.fusion_dropout1(h)
            
            # LEVEL 3: Second fusion - inject position again at later stage
            h = torch.cat([h, pos_emb], dim=1)
            h = self.fusion_fc2(h)
            h = self.fusion_bn2(h)
            h = F.relu(h)
            h = self.fusion_dropout2(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        
        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Deterministic reconstruction during evaluation

        kl_per_dim = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
        kl_per_sample = kl_per_dim.sum(dim=1)
        self.kl = kl_per_sample.mean()
        self.kl_per_dim = kl_per_dim.mean(dim=0).detach().cpu()
        
        return z

    def encode_params(self, x, row=None, col=None):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)
        
        if row is not None and col is not None:
            pos_features = compute_position_features(row, col, self.grid_size, x.device)
            pos_emb = self.pos_embedding(pos_features)
            
            # LEVEL 1: Process position embeddings through FC
            pos_processed = self.pos_fc1(pos_emb)
            pos_processed = self.pos_bn1(pos_processed)
            pos_processed = F.relu(pos_processed)
            
            # LEVEL 2: First fusion
            combined = torch.cat([h, pos_processed], dim=1)
            h = self.fusion_fc1(combined)
            h = self.fusion_bn1(h)
            h = F.relu(h)
            h = self.fusion_dropout1(h)
            
            # LEVEL 3: Second fusion
            h = torch.cat([h, pos_emb], dim=1)
            h = self.fusion_fc2(h)
            h = self.fusion_bn2(h)
            h = F.relu(h)
            h = self.fusion_dropout2(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def _position_features(self, row, col, device):
        """Deprecated: use compute_position_features() instead."""
        return compute_position_features(row, col, self.grid_size, device)

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

### Autoencoder that combines the Encoder and Decoder
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

### Variational Autoencoder that combines the VariationalEncoder and Decoder
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


class OneClassScorer(nn.Module):
    """
    One-class neural network for anomaly detection in VAE latent space.

    The model learns a compact embedding representation of normal (empty) tiles.
    During training, embeddings of empty tiles are encouraged to cluster around
    a learned center in embedding space.

    At inference:
        - tiles close to the center are considered normal/background
        - tiles far from the center are considered anomalous/observations

    Inputs:
        mu          : VAE latent representation (latent_dim)
        recon_err   : VAE reconstruction error (1)
        rows, cols  : tile positions for positional encoding

    Architecture:
        [latent + recon_error + position_embedding]
            → MLP
            → low-dimensional embedding (emb_dim)

    Anomaly score:
        squared Euclidean distance to the learned center:
            score = ||z - c||²

    Higher scores indicate higher likelihood of containing observations.
    """
    def __init__(self, latent_dim=32, hidden_dim=128, emb_dim=16, grid_size=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.grid_size = grid_size

        pos_emb_dim = max(16, 10 * 2)
        self.pos_embedding = PositionalEmbedding(input_dim=10, output_dim=pos_emb_dim)

        input_dim = latent_dim + 1 + pos_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),

            nn.Linear(hidden_dim // 2, emb_dim),
        )

    def forward(self, mu, recon_err, rows, cols):
        pos_features = compute_position_features(rows, cols, self.grid_size, mu.device)
        pos_emb = self.pos_embedding(pos_features)

        recon_err = recon_err.unsqueeze(1) if recon_err.dim() == 1 else recon_err

        x = torch.cat([mu, recon_err, pos_emb], dim=1)

        z = self.net(x)
        return z

    def anomaly_score(self, mu, recon_err, rows, cols, center):
        z = self.forward(mu, recon_err, rows, cols)
        score = ((z - center) ** 2).sum(dim=1)
        return score

    def predict_batch(self, mu, recon_err, rows, cols, center, threshold):
        with torch.no_grad():
            scores = self.anomaly_score(mu, recon_err, rows, cols, center)
            scores = torch.sigmoid(scores)  # Convert to [0,1] range for interpretability
            preds = (scores >= threshold).long()

        return preds, scores

class TwoClassScorer(nn.Module):
    """
    Dense neural network for classifying tiles as observation or empty.
    Supports two modes:
    - 'binary': Binary classification (observation vs empty)
    - 'one_class': One-class learning (detect anomalies when only empty data available)
    
    Uses shared positional encoding to ensure consistency with VAE.
    Designed for GPU-accelerated inference and training.
    
    Input: latent_features (latent_dim) + recon_error (1) + processed position embedding
    Output: probability (1) of being observation (binary) or anomaly score (one_class)
    """
    def __init__(self, latent_dim=32, hidden_dim=256, dropout=0.3, grid_size=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        # Shared positional embedding module — matches VAE's pos_embedding
        pos_emb_dim = max(16, 10 * 2)  # output dim of PositionalEmbedding with input_dim=10
        self.pos_embedding = PositionalEmbedding(input_dim=10, output_dim=pos_emb_dim)
        
        # Input: latent_dim + recon_error (1) + pos_emb_dim
        input_dim = latent_dim + 1 + pos_emb_dim
        
        # Dense layers with batch norm and dropout for regularization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output probability [0, 1]
        self.fc_out = nn.Linear(hidden_dim // 4, 1)
        
        

        
    def forward(self, mu, recon_err, rows, cols):
        """
        Forward pass.
        
        Args:
            mu: Latent representations (batch_size, latent_dim)
            recon_err: Reconstruction error (batch_size,)
            rows: Tile row indices (batch_size,)
            cols: Tile column indices (batch_size,)
        
        Returns:
            Probability of observation (batch_size,) - values in [0, 1]
        """
        # Use shared positional feature computation and embedding
        pos_features = compute_position_features(rows, cols, self.grid_size, mu.device)
        pos_emb = self.pos_embedding(pos_features)
        
        # Stack features: latent + recon_error + processed position embedding
        recon_err_expanded = recon_err.unsqueeze(1) if recon_err.dim() == 1 else recon_err
        features = torch.cat([mu, recon_err_expanded, pos_emb], dim=1)
        
        # Forward through dense network
        x = self.fc1(features)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # Final linear layer produces one logit per tile
        logits = self.fc_out(x).squeeze(-1)
        return logits

    def predict_batch(self, mu, recon_err, rows, cols, threshold=0.5):
        """
        Batch prediction with thresholding.
        """
        
        with torch.no_grad():
            probs = self.forward(mu, recon_err, rows, cols)            
            preds = (probs >= threshold).long()
            
        return preds, probs

### Shared Positional Encoding Components ###
def compute_position_features(rows, cols, grid_size, device):
    """
    Compute continuous 10-D position features from tile row/col coordinates.
    This function is shared across VAE and ObservationScorer to ensure consistency.
    
    Features: [x, y, x^2, y^2, x*y, r, sin(x), cos(x), sin(y), cos(y)]
    where x,y are normalized to [0,1], and r is radial distance from center.
    
    Args:
        rows: Tensor of row indices (B,)
        cols: Tensor of column indices (B,)
        grid_size: int, number of tiles along one dimension
        device: torch device
    
    Returns:
        pos_features: Tensor of shape (B, 10) with continuous positional features
    """
    rows_t = rows.to(device).float() if isinstance(rows, torch.Tensor) else torch.tensor(rows, dtype=torch.float32, device=device)
    cols_t = cols.to(device).float() if isinstance(cols, torch.Tensor) else torch.tensor(cols, dtype=torch.float32, device=device)

    denom = float(grid_size - 1)
    y = rows_t / denom
    x = cols_t / denom

    # Radial distance from center (normalized) to deal with vignetting/illumination gradients
    r = torch.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    
    # Sin and cos positional features to capture periodicity and suppress edge/corner artifacts
    sin_x = torch.sin(x * 2 * torch.pi)
    cos_x = torch.cos(x * 2 * torch.pi)
    sin_y = torch.sin(y * 2 * torch.pi)
    cos_y = torch.cos(y * 2 * torch.pi)

    return torch.stack([
        x, y, x ** 2, y ** 2, x * y,
        r, sin_x, cos_x, sin_y, cos_y
    ], dim=1)


class PositionalEmbedding(nn.Module):
    """
    Learnable MLP that processes raw positional features into an embedding.
    Shared across VAE and ObservationScorer.
    
    Args:
        input_dim: int, dimension of raw position features (default 10)
        output_dim: int, dimension of output embedding
    """
    def __init__(self, input_dim=10, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, max(16, input_dim * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, input_dim * 2), output_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, pos_features):
        """
        Args:
            pos_features: Tensor of shape (B, 10) from compute_position_features
        
        Returns:
            pos_emb: Tensor of shape (B, output_dim)
        """
        return self.mlp(pos_features)

### Utility functions for image processing, tile extraction, local maxima detection, and region growing
def flood_fill_region(prob_map, start_y, start_x, secondary_threshold):
    """
    Flood-fill from a peak to find the connected region above secondary_threshold.
    Returns a list of (y, x) coordinates in the region.
    
    input:
    - prob_map: 2D array of probabilities
    - start_y, start_x: coordinates of the starting point (peak)
    - secondary_threshold: minimum probability to include in the region
    """
    
    region = set()
    stack = [(start_y, start_x)]
    visited = set()
    
    while stack:
        y, x = stack.pop()
        
        if (y, x) in visited:
            continue
        if y < 0 or y >= prob_map.shape[0] or x < 0 or x >= prob_map.shape[1]:
            continue
        if prob_map[y, x] < secondary_threshold:
            continue
        
        visited.add((y, x))
        region.add((y, x))
        
        # Add 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy != 0 or dx != 0:
                    stack.append((y + dy, x + dx))
    
    return region

def robust_flood_fill(
    prob_map,
    peak_y,
    peak_x,
    peak_threshold=0.98,
    secondary_threshold=0.75,
    min_region_size=9,
    min_region_mean=0.85,
    min_core_pixels=4,
    core_threshold=0.9,
):
    if prob_map[peak_y, peak_x] < peak_threshold:
        return set()

    visited = set()
    region = set()
    stack = [(peak_y, peak_x)]

    h, w = prob_map.shape

    while stack:
        y, x = stack.pop()
        if (y, x) in visited:
            continue
        visited.add((y, x))

        if prob_map[y, x] < secondary_threshold:
            continue

        region.add((y, x))

        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                stack.append((yy, xx))

    if len(region) < min_region_size:
        return set()

    values = np.array([prob_map[y, x] for y, x in region])

    if values.mean() < min_region_mean:
        return set()

    if np.sum(values >= core_threshold) < min_core_pixels:
        return set()

    return region

def find_local_maxima(prob_map, peak_threshold):
    """
    Find local maxima in the probability map (vectorized - much faster).
    
    Args:
        prob_map: 2D probability map
        peak_threshold: absolute threshold value
    
    Returns:
        List of (y, x, value) tuples for local maxima
    """
    # Find local maxima using maximum_filter (8-connected neighborhood)
    local_max = maximum_filter(prob_map, footprint=np.ones((3, 3))) == prob_map
    
    # Apply threshold
    above_threshold = prob_map >= peak_threshold
    maxima_mask = local_max & above_threshold
    
    # Extract coordinates
    y_coords, x_coords = np.where(maxima_mask)
    values = prob_map[y_coords, x_coords]
    
    return list(zip(y_coords, x_coords, values))


def extract_tiles_with_offset(image, tile_size, grid_size, offset_pxl_x=0, offset_pxl_y=0, offset_norm_x = 0, offset_norm_y = 0):
    """
    Extract tiles from image with specified offset.
    
    Args:
        image: input image
        tile_size: size of each tile
        grid_size: number of tiles along one side
        offset_pxl_x: horizontal offset (in pixels)
        offset_pxl_y: vertical offset (in pixels)
        offset_norm_x: horizontal offset (normalized as fraction of row)
        offset_norm_y: vertical offset (normalize as fraction of row)
    
    Returns:
        list of (tile, row, col) tuples
    """
    
    tiles = []
    image_h, image_w = image.shape[:2]
        
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = offset_pxl_y + row * tile_size
            x0 = offset_pxl_x + col * tile_size
            y1 = min(y0 + tile_size, image_h)
            x1 = min(x0 + tile_size, image_w)
            
            # Handle edge cases by padding if necessary
            tile = image[y0:y1, x0:x1]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = np.pad(tile, ((0, tile_size - tile.shape[0]), (0, tile_size - tile.shape[1])), mode='edge')
            
            # Return tile coordinates as 1-based grid position + fractional offset.
            # This encodes the tile center position in grid units for map aggregation.
            model_row = row + offset_norm_y      # 0.0 to 15.8, or clip to 15
            model_col = col + offset_norm_x
            
            tiles.append((tile, model_row, model_col))
    
    return tiles

def load_tiles_from_paths(batch_paths, include_labels=False):
    
    """
    Unified image pipeline for loading tiles from disk.
    
    Args:
        batch_paths: Single Path or list of Path objects to load from disk
        image_size: Target size for preprocessing
        include_labels: Whether to extract labels from path names (True for eval, False for train/test)
    
    Returns:
        Dataframe with columns: 'image', 'path', 'row', 'col', 'label' (if include_labels=True)
    """

    # Handle single path input
    single_input = isinstance(batch_paths, (str, Path))
    if single_input:
        batch_paths = [batch_paths]
    
    images, rows, cols, labels, paths = [], [], [], [], []
    for path in batch_paths:
        path = Path(path) if isinstance(path, str) else path
        
        if not path.exists():
            print(f"⚠ Warning: File not found {path}, skipping")
            continue
        
        # Extract row/col from filename
        tile_match = re.search(r'_r(\d+)_c(\d+)_o([\d]+)', path.name)
        if not tile_match:
            print(f"⚠ Warning: Could not extract row/col from {path.name}, skipping")
            continue
        
        row, col, offset = int(tile_match.group(1)), int(tile_match.group(2)), float(tile_match.group(3))

        # add offset to row/col to encode tile center position in grid units for map aggregation
        row += (offset/100)
        col += (offset/100)

        if include_labels:
            if path.parent.name.lower() == "no_obs":
                label = False
            elif path.parent.name.lower() == "obs":
                label = True
            else:
                print(f"⚠ Warning: Could not determine label from path {path}, skipping")
                continue

            labels.append(label)
                    
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        
        rows.append(row)
        cols.append(col)
        images.append(img)
        paths.append(path)
    
    # Create dataframe with tile info for debugging/analysis
    df_tile_info = pd.DataFrame({
        'image': images,
        'path': paths,
        'row': rows,
        'col': cols,
        'label': labels if include_labels else [None] * len(rows),
    })
    
    return df_tile_info
        
def image_pipeline_df(df_tile_info, input_image_size_vae=128, include_labels=False):
    """
    Image pipeline that takes a DataFrame with tile information and applies preprocessing to convert images to tensors.
    
    Args:
        df_tile_info: DataFrame containing tile information
        input_image_size_vae: Target size for VAE preprocessing
        include_labels: Whether to extract labels from path names (True for eval, False for train/test)
    
    Returns:
        Tuple of tensors:
        - If include_labels=True: (images, labels, rows, cols)
        - If include_labels=False: (images, rows, cols)
    """
    
    # Check dataframe structure
    required_columns = {'image', 'row', 'col'}
    if not required_columns.issubset(df_tile_info.columns):
        raise ValueError(f"DataFrame must contain columns {required_columns}, but got {df_tile_info.columns}")
    
    # Check for invalid entries in dataframe
    if df_tile_info['row'].isnull().any() or df_tile_info['col'].isnull().any():
        raise ValueError("⚠ Warning: Some entries in 'row' or 'col' columns are null")
    
    # Check if labels are present in the DataFrame when include_labels=True, and warn if not
    if df_tile_info['label'].isnull().all() and include_labels:
        raise ValueError("⚠ Warning: include_labels=True but no labels found in DataFrame, returning None for labels")
    elif include_labels==False and not df_tile_info['label'].isnull().all():
        print("⚠ Warning: include_labels=False but labels found in DataFrame, ignoring labels")
    
    images, rows, cols, labels = [], [], [], []
    for tile_info in df_tile_info.itertuples():
        img = tile_info.image
        row = tile_info.row
        col = tile_info.col
        label = tile_info.label

        # If image is None, load from disk using path
        if img is None:
            if not hasattr(tile_info, 'path') or tile_info.path is None:
                print(f"⚠ Warning: Image is None and no path available, skipping")
                continue
            print(f"⚠ Info: Image is None, loading from disk using path {tile_info.path}")
            img = cv2.imread(str(tile_info.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠ Warning: Could not load image from {tile_info.path}, skipping")
                continue
        img_tensor = preprocess_tile(img, image_size=input_image_size_vae)
        
        if img_tensor is None:
            continue
        
        images.append(img_tensor) 
        rows.append(row)
        cols.append(col)
        labels.append(label)
        
    # Stack tensors
    images_tensor = torch.stack(images, dim=0) if images else torch.tensor([])
    rows_tensor = torch.tensor(rows, dtype=torch.float32)
    cols_tensor = torch.tensor(cols, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.bool) if include_labels else None
    
    if len(df_tile_info) == 1:
        if images_tensor.numel() == 0:
            if include_labels:
                return images_tensor, torch.tensor(False, dtype=torch.bool), torch.tensor(-1, dtype=torch.long), torch.tensor(-1, dtype=torch.long)
            return images_tensor, torch.tensor(-1, dtype=torch.long), torch.tensor(-1, dtype=torch.long)

        if include_labels:
            assert labels_tensor is not None
            return images_tensor[0], labels_tensor[0], rows_tensor[0], cols_tensor[0]
        return images_tensor[0], rows_tensor[0], cols_tensor[0]

    if include_labels:
        return images_tensor, rows_tensor, cols_tensor, labels_tensor
    else:
        return images_tensor, rows_tensor, cols_tensor

### Universal image pipeline for loading tiles from disk, with optional label extraction and position encoding
def image_pipeline_paths(batch_paths, image_size=128, include_labels=False):
    """
    Unified image pipeline for loading tiles from disk.
    
    Args:
        batch_paths: Single Path or list of Path objects to load from disk
        image_size: Target size for preprocessing
        include_labels: Whether to extract labels from path names (True for eval, False for train/test)
    
    Returns:
        Tuple of tensors:
        - If include_labels=True: (images, labels, rows, cols)
        - If include_labels=False: (images, rows, cols)
    """
    
    # Handle single path input
    single_input = isinstance(batch_paths, (str, Path))
    if single_input:
        batch_paths = [batch_paths]
    
    images, rows, cols, labels = [], [], [], []
    for path in batch_paths:
        path = Path(path) if isinstance(path, str) else path
        
        if not path.exists():
            print(f"⚠ Warning: File not found {path}, skipping")
            continue
        
        # Extract row/col from filename
        tile_match = re.search(r'_r(\d+)_c(\d+)', path.name)
        if not tile_match:
            print(f"⚠ Warning: Could not extract row/col from {path.name}, skipping")
            continue
        
        row, col = int(tile_match.group(1)), int(tile_match.group(2))
        
        # Load and preprocess image
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img_tensor = preprocess_tile(img, image_size=image_size)
        
        if img_tensor is None:
            continue
        
        images.append(img_tensor) 
        rows.append(row)
        cols.append(col)
        
        # Extract label if requested
        if include_labels:
            label = False if "no_obs" in str(path).lower() else True
            labels.append(label)
    
    # Stack tensors
    images_tensor = torch.stack(images, dim=0) if images else torch.tensor([])
    rows_tensor = torch.tensor(rows, dtype=torch.float32)
    cols_tensor = torch.tensor(cols, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.bool) if include_labels else None
    
    if single_input:
        if images_tensor.numel() == 0:
            if include_labels:
                return images_tensor, torch.tensor(False, dtype=torch.bool), torch.tensor(-1, dtype=torch.long), torch.tensor(-1, dtype=torch.long)
            return images_tensor, torch.tensor(-1, dtype=torch.long), torch.tensor(-1, dtype=torch.long)

        if include_labels:
            assert labels_tensor is not None
            return images_tensor[0], labels_tensor[0], rows_tensor[0], cols_tensor[0]
        return images_tensor[0], rows_tensor[0], cols_tensor[0]

    if include_labels:
        return images_tensor, labels_tensor, rows_tensor, cols_tensor
    
    else:
        return images_tensor, rows_tensor, cols_tensor


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
    tiles_info = []
    
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
            tiles_info.append((tile, row, col))
    
    return tiles_info


# def train_image_pipeline(batch_paths, image_size=128):
#     images = []
#     rows = []
#     cols = []

#     for path in batch_paths:
#         if not path.exists():
#             print(f"⚠ Warning: File not found {path}, skipping")
#             continue
        
#         row_match = re.search(r'_r(\d+)_c\d+', path.name)
#         col_match = re.search(r'_r\d+_c(\d+)', path.name)
        
#         if not row_match or not col_match:
#             print(f"⚠ Warning: Could not extract row/col from {path.name}, skipping")
#             continue
        
#         row = int(row_match.group(1))
#         col = int(col_match.group(1))
#         img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         img = preprocess_tile(img, image_size=image_size)

#         if img is None:
#             continue

#         images.append(img)
#         rows.append(row)
#         cols.append(col)

#     x = torch.stack(images, dim=0)
#     rows = torch.tensor(rows, dtype=torch.long)
#     cols = torch.tensor(cols, dtype=torch.long)
#     return x, x, rows, cols


# def eval_image_pipeline(batch_paths, image_size=128):
#     images = []
#     labels = []
#     cols = []
#     rows = []
    
#     for path in batch_paths:
#         if not path.exists():
#             print(f"⚠ Warning: File not found {path}, skipping")
#             continue

#         # if path is in obs folder, label=TrUE, else FALSE (empty)
#         label = False if "no_obs" in str(path).lower() else True
        
#         # get tile col and row
#         tile_match = re.search(r'_r(\d+)_c(\d+)', path.name)
        
#         if not tile_match:
#             print(f"⚠ Warning: Could not extract row/col from {path.name}, skipping")
#             continue
        
#         row = int(tile_match.group(1))
#         col = int(tile_match.group(2))

#         img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#         img = preprocess_tile(img, image_size=image_size)
#         if img is None:
#             continue

#         labels.append(label)
#         rows.append(row)
#         cols.append(col)
#         images.append(img)

#     images = torch.stack(images, dim=0)
#     labels = torch.tensor(labels, dtype=torch.bool)
#     rows = torch.tensor(rows, dtype=torch.long)
#     cols = torch.tensor(cols, dtype=torch.long)
    
#     return images, labels, rows, cols

# def test_image_pipeline(batch, image_size=128):
#     """Collate fn for test tiles already split as (tile, row, col, image_name)."""

#     all_tiles = []
#     cols = []
#     rows = []
#     image_names = []
    
#     for item in batch:
#         tile, row, col, image_name = item
#         if tile is None:
#             continue

#         tile_tensor = preprocess_tile(tile, image_size=image_size)
#         if tile_tensor is None:
#             continue

#         rows.append(int(row))
#         cols.append(int(col))
#         image_names.append(image_name)
#         all_tiles.append(tile_tensor)

#     tiles_tensor = torch.stack(all_tiles, dim=0)
#     rows = torch.tensor(rows, dtype=torch.long)
#     cols = torch.tensor(cols, dtype=torch.long)

#     return tiles_tensor, rows, cols, image_names

### Dataset class for loading images from file paths and applying preprocessing
# class ImagePathDataset(Dataset):
#     """Dataset that loads images from file paths and applies a preprocessing function."""
#     def __init__(self, image_paths, preprocess_fn):
#         self.image_paths = image_paths
#         self.preprocess_fn = preprocess_fn
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         return self.preprocess_fn(image_path)

def load_tiles_from_paths_fast(batch_paths, image_size=128, include_labels=False, batch_size=256, num_workers=16, to_device=None, return_generator=False):
    """
    Args:
        batch_paths: list of Path or single Path to tile images
        image_size: target size for `preprocess_tile`
        include_labels: whether to extract labels from path parent folder
        batch_size: number of images to transfer to `to_device` at once (if provided)
        num_workers: number of threads for parallel reading and preprocessing
        to_device: if provided (e.g., 'cuda' or torch.device('cuda')), tensors will be moved to this device in batches

    Returns:
        If to_device is None: (images_tensor_cpu, rows_tensor, cols_tensor, labels_tensor?)
        If to_device provided: generator that yields batches (images_on_device, rows, cols, labels) for downstream processing
    """

    single_input = isinstance(batch_paths, (str, Path))
    if single_input:
        batch_paths = [batch_paths]

    def _process_path(path):
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            return None

        tile_match = re.search(r'_r(\d+)_c(\d+)_o?([\d]*)', path.name)
        if not tile_match:
            return None

        # parse row/col and optional offset
        row = int(tile_match.group(1))
        col = int(tile_match.group(2))
        offset = tile_match.group(3)
        try:
            offset = float(offset) / 100.0 if offset != '' else 0.0
        except Exception:
            offset = 0.0

        row_f = row + offset
        col_f = col + offset

        if include_labels:
            label = False if "no_obs" in str(path).lower() else True
        else:
            label = None

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        img_tensor = preprocess_tile(img, image_size=image_size)
        if img_tensor is None:
            return None

        # Return preprocessed tensor for batching, and raw image for generator
        return (img_tensor, row_f, col_f, label, path, img)

    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_process_path, p): p for p in batch_paths}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is None:
                continue
            results.append(res)

    if not results:
        # return empty tensors consistent with other APIs
        empty = torch.tensor([])
        rows_t = torch.tensor([], dtype=torch.long)
        cols_t = torch.tensor([], dtype=torch.long)
        labels_t = torch.tensor([], dtype=torch.bool) if include_labels else None
        return (empty, labels_t, rows_t, cols_t) if include_labels else (empty, rows_t, cols_t)

    # unzip: tensors, rows, cols, labels, paths, raw images
    tensor_list, rows_list, cols_list, labels_list, paths_list, raw_list = zip(*results)

    # Stack into single CPU tensor (used for optional generator)
    images_tensor = torch.stack(list(tensor_list), dim=0)
    rows_tensor = torch.tensor(list(rows_list), dtype=torch.float32)
    cols_tensor = torch.tensor(list(cols_list), dtype=torch.float32)
    labels_tensor = torch.tensor(list(labels_list), dtype=torch.bool) if include_labels else None

    # Build a DataFrame for compatibility with existing code that expects a DataFrame
    # Note: 'image' column contains loaded raw images; image_pipeline_df will use these or load from disk if None
    n = len(paths_list)
    df_tile_info = pd.DataFrame({
        'image': list(raw_list),
        'path': list(paths_list),
        'row': rows_list,  # keep as-is (floats from row_f = row + offset)
        'col': cols_list,  # keep as-is (floats from col_f = col + offset)
        'label': list(labels_list) if include_labels else [None] * n,
    })

    # Always return DataFrame for backward compatibility; ignore to_device unless return_generator=True
    if not return_generator or to_device is None:
        return df_tile_info

    # If caller requested a generator moved to device, yield batches moved to device to avoid large allocations
    dev = torch.device(to_device) if not isinstance(to_device, torch.device) else to_device

    def _batch_generator():
        total = images_tensor.shape[0]
        for i in range(0, total, batch_size):
            j = min(i + batch_size, total)
            batch_imgs = images_tensor[i:j].to(dev)
            batch_rows = rows_tensor[i:j].to(dev)
            batch_cols = cols_tensor[i:j].to(dev)
            batch_labels = labels_tensor[i:j].to(dev) if (include_labels and labels_tensor is not None) else None
            yield batch_imgs, batch_rows, batch_cols, batch_labels

    return _batch_generator()