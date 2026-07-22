
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
    """Regular encoder with continuous positional features"""
    def __init__(self, latent_dims=2, image_size=128, hidden_channels=32, grid_size=16):
        super().__init__()
        self.image_size = image_size
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

        embedding_total_dim = hidden_channels * 8
        self.pos_embedding = PositionalEmbedding(
            input_dim=10,
            output_dim=embedding_total_dim
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_size, self.image_size)
            feat = self.conv(dummy)

        self.feat_shape = feat.shape[1:]
        self.feat_dim = feat.numel()

        self.pos_fc1 = nn.Linear(embedding_total_dim, hidden_channels * 8)
        self.pos_bn1 = nn.BatchNorm1d(hidden_channels * 8)

        combined_dim = self.feat_dim + hidden_channels * 8

        self.fusion_fc1 = nn.Linear(combined_dim, hidden_channels * 8)
        self.fusion_bn1 = nn.BatchNorm1d(hidden_channels * 8)
        self.fusion_dropout1 = nn.Dropout(0.2)

        self.fusion_fc2 = nn.Linear(hidden_channels * 8 + embedding_total_dim, hidden_channels * 8)
        self.fusion_bn2 = nn.BatchNorm1d(hidden_channels * 8)
        self.fusion_dropout2 = nn.Dropout(0.2)

        self.fc_z = nn.Linear(hidden_channels * 8, latent_dims)

    def forward(self, x, row=None, col=None):
        h = self.conv(x)
        h = torch.flatten(h, start_dim=1)

        if row is not None and col is not None:
            pos_features = compute_position_features(row, col, self.grid_size, x.device)
            pos_emb = self.pos_embedding(pos_features)

            pos_processed = self.pos_fc1(pos_emb)
            pos_processed = self.pos_bn1(pos_processed)
            pos_processed = F.relu(pos_processed)

            h = torch.cat([h, pos_processed], dim=1)
            h = self.fusion_fc1(h)
            h = self.fusion_bn1(h)
            h = F.relu(h)
            h = self.fusion_dropout1(h)

            h = torch.cat([h, pos_emb], dim=1)
            h = self.fusion_fc2(h)
            h = self.fusion_bn2(h)
            h = F.relu(h)
            h = self.fusion_dropout2(h)

        z = self.fc_z(h)
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
    def __init__(self, latent_dims, image_size=128, hidden_channels=32, grid_size=16):
        super().__init__()
        self.encoder = Encoder(
            latent_dims=latent_dims,
            image_size=image_size,
            hidden_channels=hidden_channels,
            grid_size=grid_size
        )

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

    def predict_batch(self, mu, recon_err, rows, cols, center):
        with torch.no_grad():
            scores = self.anomaly_score(mu, recon_err, rows, cols, center)
            probs = torch.sigmoid(scores)  # Convert to [0,1] range for interpretability


        return probs

class TwoClassScorer(nn.Module):
    """
    Dense neural network for classifying tiles as observation or empty.
    
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

    def predict_batch(self, mu, recon_err, rows, cols):
        """
        Batch prediction with thresholding.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(mu, recon_err, rows, cols)      
            probs = torch.sigmoid(logits)  # Convert logits to probabilities in [0,1]      

        return probs

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


class MahalanobisScorer(nn.Module):
    def __init__(self, latent_dim=32, grid_size=16, mean_vec=None, cov_inv=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.grid_size = grid_size

        self.register_buffer("mean_vec", mean_vec)
        self.register_buffer("cov_inv", cov_inv)

        self.register_buffer(
            "chi2_half_df",
            torch.tensor(latent_dim / 2, dtype=mean_vec.dtype, device=mean_vec.device)
        )

        pos_emb_dim = max(16, 10 * 2)
        self.pos_embedding = PositionalEmbedding(input_dim=10, output_dim=pos_emb_dim)

    def mahalanobis_distance(self, x):

        diff = x - self.mean_vec

        dist_sq = torch.sum((diff @ self.cov_inv) * diff, dim=1)
        dist_sq = torch.clamp(dist_sq, min=0)

        return torch.sqrt(dist_sq)

    def predict_batch(self, mu, recon_err, rows=None, cols=None):
        with torch.no_grad():
            dist = self.mahalanobis_distance(mu)

            dist_scores = torch.special.gammainc(
                self.chi2_half_df.to(device=dist.device, dtype=dist.dtype),
                dist ** 2 / 2
            )

            recon_scores = torch.special.gammainc(
                self.chi2_half_df.to(device=recon_err.device, dtype=recon_err.dtype),
                recon_err ** 2 / 2
            )

            alpha = 0.5
            probs = alpha * dist_scores + (1 - alpha) * recon_scores

            return probs
import json
from pathlib import Path
import cv2
import numpy as np


def get_manual_crops_from_labelme_json(
    json_path,
    image_shape,
    image_path,
    image_name,
    PXL_TO_MM2,
    crop_padding_pixels=0,
    make_square=True,
    allowed_labels=None,   # set to None to keep all shapes
):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Rasterize LabelMe shapes into a binary mask
    for shape in data.get("shapes", []):
        label = shape.get("label", None)
        if allowed_labels is not None and label not in allowed_labels:
            continue

        pts = np.array(shape.get("points", []), dtype=np.float32)
        if pts.size == 0:
            continue

        shape_type = shape.get("shape_type", "polygon")

        if shape_type == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            x0, x1 = sorted([int(round(x0)), int(round(x1))])
            y0, y1 = sorted([int(round(y0)), int(round(y1))])
            cv2.rectangle(mask, (x0, y0), (x1, y1), 1, thickness=-1)

        elif shape_type in {"polygon", "freehand", "linestrip"}:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)

        else:
            # Skip unsupported shape types for now
            continue

    # Split into connected components so each separated region becomes one crop
    n_labels, cc_mask, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    manual_crop_metadata = []
    crop_id = 0

    for cc_id in range(1, n_labels):  # 0 is background
        x, y, bw, bh, area = stats[cc_id]
        if area == 0:
            continue

        y0 = max(0, y - crop_padding_pixels)
        x0 = max(0, x - crop_padding_pixels)
        y1 = min(h, y + bh + crop_padding_pixels)
        x1 = min(w, x + bw + crop_padding_pixels)

        if make_square:
            crop_h = y1 - y0
            crop_w = x1 - x0
            crop_size = max(crop_h, crop_w)

            y_center = (y0 + y1) // 2
            x_center = (x0 + x1) // 2

            y0 = max(0, y_center - crop_size // 2)
            x0 = max(0, x_center - crop_size // 2)
            y1 = min(h, y0 + crop_size)
            x1 = min(w, x0 + crop_size)

            # Re-clip in case the crop hit an image boundary
            y0 = max(0, y1 - crop_size)
            x0 = max(0, x1 - crop_size)

        region_area_px = int((cc_mask == cc_id).sum())

        manual_crop_metadata.append({
            "OG_image_path": Path(image_path),
            "image_name": image_name,
            "crop_id": crop_id,
            "crop_y0": int(y0),
            "crop_x0": int(x0),
            "crop_y1": int(y1),
            "crop_x1": int(x1),
            "crop_height": int(y1 - y0),
            "crop_width": int(x1 - x0),
            "region_size_pixels": region_area_px,
            "region_size_mm2": float(region_area_px * PXL_TO_MM2),
            "source": "manual",
            "label_manual": True,
            "label_predicted": None,
            "matched_by_pred": False,
            "mask_path": str(json_path),
        })
        crop_id += 1

    return manual_crop_metadata

def crops_overlap(pred_crop, manual_crop,
                  min_manual_covered_pred_larger,
                  min_manual_covered_pred_smaller):
    py0, px0, py1, px1 = pred_crop["crop_y0"], pred_crop["crop_x0"], pred_crop["crop_y1"], pred_crop["crop_x1"]
    my0, mx0, my1, mx1 = manual_crop["crop_y0"], manual_crop["crop_x0"], manual_crop["crop_y1"], manual_crop["crop_x1"]

    inter_y0 = max(py0, my0)
    inter_x0 = max(px0, mx0)
    inter_y1 = min(py1, my1)
    inter_x1 = min(px1, mx1)

    if inter_y1 <= inter_y0 or inter_x1 <= inter_x0:
        return False

    inter_area = (inter_y1 - inter_y0) * (inter_x1 - inter_x0)
    pred_area = (py1 - py0) * (px1 - px0)
    manual_area = (my1 - my0) * (mx1 - mx0)

    if pred_area <= 0 or manual_area <= 0:
        return False

    manual_covered = inter_area / manual_area
    pred_covered = inter_area / pred_area
    thr = min_manual_covered_pred_larger if pred_area > manual_area else min_manual_covered_pred_smaller

    return (manual_covered >= thr) or (pred_covered >= thr)


# class MahalanobisScorer():
#     """
#     Class for scoring usisng mahalnobis distance in latent space, combined with reconstruction error.
    
#     Architecture same as OneClassScorer, so that it can be used interchangeably with different scoring functions.
    
#     """
#     def __init__(self, latent_dim=32, grid_size=16, mean_vec=None, cov_inv=None):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.grid_size = grid_size
#         self.mean_vec = mean_vec
#         self.cov_inv = cov_inv

#         # Shared positional embedding module — matches VAE's pos_embedding
#         pos_emb_dim = max(16, 10 * 2)  # output dim of PositionalEmbedding with input_dim=10
#         self.pos_embedding = PositionalEmbedding(input_dim=10, output_dim=pos_emb_dim)
      
#     def mahalanobis_distance(self, x):
#         # Mahalanobis distance: sqrt((x - mean)^T * cov_inv * (x - mean))
#         # Make compatible for batch processing and single vector input
        
#         mean = self.mean_vec.to(x.device) 
#         cov_inv = self.cov_inv.to(x.device)
        
#         diff = x - mean
#         dist = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))
#         return dist

#     def predict_batch(self, mu, recon_err, rows, cols):
#         """
#         Batch prediction with thresholding.
#         """
        
#         with torch.no_grad():
#             # Compute Mahalanobis distance
#             dist = self.mahalanobis_distance(mu)
            
#             k = self.latent_dim
            
#             # Combine with reconstruction error (you can also try other combinations)
            
#             # Convert to probability using sigmoid (or use raw scores for anomaly detection
#             dist_scores = torch.special.gammainc(torch.tensor(k / 2, device=dist.device), dist ** 2 / 2)  # Convert to [0,1] range for interpretability
#             recon_scores = torch.special.gammainc(torch.tensor(k / 2, device=dist.device), recon_err ** 2 / 2)  # Convert to [0,1] range for interpretability
            
#             alpha = 0.5  # 0.5 = equal weighting

#             probs = alpha * dist_scores + (1 - alpha) * recon_scores
                        
#             return probs
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

### Batch run inference functions
def batch_run_inference(tiles_tensor, rows_tensor, cols_tensor, model, scorer, encoder, device, batch_size):
    """Run scorer inference on a batch of tiles from multiple images."""

    
    return scores_array, mu_array, recon_array

def split_results_by_image(scores_array, rows_list, cols_list, tile_grid_size, offsets, image_tile_counts):
    """Split batched inference results back to per-image probability maps."""
    results_per_image = []
    start_idx = 0
    
    for img_num, num_tiles in enumerate(image_tile_counts):
        if num_tiles == 0:
            results_per_image.append(None)
            continue
        
        end_idx = start_idx + num_tiles
        image_scores = scores_array[start_idx:end_idx]
        image_rows = rows_list[start_idx:end_idx]
        image_cols = cols_list[start_idx:end_idx]
        
        prob_map_small, coverage_map_small = aggregate_scores_to_map(
            image_rows,
            image_cols,
            image_scores,
            tile_grid_size=tile_grid_size,
            offsets_count=len(offsets),
        )
        
        results_per_image.append({
            "prob_map_small": prob_map_small,
            "coverage_map_small": coverage_map_small,
        })
        start_idx = end_idx
    
    return results_per_image

def aggregate_scores_to_map(rows_list, cols_list, scores, tile_grid_size: int, offsets_count: int):
    grid_size_small = tile_grid_size * offsets_count
    prob_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)
    coverage_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)

    for tile_row, tile_col, pred_score in zip(rows_list, cols_list, scores):
        small_row = int(round(float(tile_row) * offsets_count))
        small_col = int(round(float(tile_col) * offsets_count))
        small_row = max(0, min(small_row, grid_size_small - 1))
        small_col = max(0, min(small_col, grid_size_small - 1))

        prob_map_small[small_row : small_row + offsets_count, small_col : small_col + offsets_count] += float(pred_score)
        coverage_map_small[small_row : small_row + offsets_count, small_col : small_col + offsets_count] += 1.0

    prob_map_small = np.divide(
        prob_map_small,
        coverage_map_small,
        out=prob_map_small,
        where=coverage_map_small > 0,
    )
    return prob_map_small, coverage_map_small


def image_to_probability_map(
    image: np.ndarray,
    model,
    scorer,
    device: torch.device,
    tile_size: int,
    tile_grid_size: int,
    image_size: int,
    batch_size: int,
    offsets,
    offsets_norm,
):
    
    tiles_list = []
    rows_list = []
    cols_list = []

    for offset, offset_norm in zip(offsets, offsets_norm):
        print(f"offset = {offset} pixels, ({offset_norm:.2f} of tile size)")
        tiles = extract_tiles_with_offset(image, tile_size, tile_grid_size, offset_pxl_x=offset, offset_pxl_y=offset, offset_norm_x=offset_norm, offset_norm_y = offset_norm )
        for tile, row, col in tiles:
            tile_tensor = preprocess_tile(tile, image_size=image_size)
            if tile_tensor is None:
                continue
            tiles_list.append(tile_tensor)
            rows_list.append(row)
            cols_list.append(col)

    if not tiles_list:
        return None

    tiles_tensor = torch.stack(tiles_list, dim=0)
    rows_tensor = torch.tensor(rows_list, dtype=torch.float32)
    cols_tensor = torch.tensor(cols_list, dtype=torch.float32)

    scores_list = []
    preds_list = []
    mu_list = []
    recon_list = []

    with torch.inference_mode():
        for start_idx in range(0, len(tiles_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(tiles_tensor))
            tiles_batch = tiles_tensor[start_idx:end_idx].to(device, non_blocking=True)
            rows_batch = rows_tensor[start_idx:end_idx].to(device, non_blocking=True)
            cols_batch = cols_tensor[start_idx:end_idx].to(device, non_blocking=True)

            mu, _ = model.encode(tiles_batch, row=rows_batch, col=cols_batch)
            x_hat = model(tiles_batch, row=rows_batch, col=cols_batch)
            recon_err = ((tiles_batch - x_hat) ** 2).flatten(1).mean(dim=1)
            scores = scorer(mu, recon_err, rows_batch, cols_batch)

            mu_list.append(mu.detach().cpu().numpy())
            recon_list.append(recon_err.detach().cpu().numpy())
            scores_list.append(scores.detach().cpu().numpy())


    scores_array = np.concatenate(scores_list, axis=0)
    mu_array = np.concatenate(mu_list, axis=0)
    recon_array = np.concatenate(recon_list, axis=0)

    prob_map_small, coverage_map_small = aggregate_scores_to_map(
        rows_list,
        cols_list,
        scores_array,
        tile_grid_size=tile_grid_size,
        offsets_count=len(offsets),
    )

    return {
        "prob_map_small": prob_map_small,
        "coverage_map_small": coverage_map_small,
        "rows_list": np.asarray(rows_list, dtype=np.float32),
        "cols_list": np.asarray(cols_list, dtype=np.float32),
        "scores_array": scores_array,
        "mu_array": mu_array,
        "recon_array": recon_array,
    }

def batch_extract_tiles_from_images(images_list, tile_size, tile_grid_size, offsets, offsets_norm, image_size):
    """Extract tiles from multiple images and return concatenated batch with image tracking."""
    all_tiles = []
    all_rows = []
    all_cols = []
    image_tile_counts = []  # Track how many tiles per image
    
    for image_gray in images_list:
        tiles_list = []
        rows_list = []
        cols_list = []

        for offset, offset_norm in zip(offsets, offsets_norm):
            tiles = extract_tiles_with_offset(image_gray, tile_size, tile_grid_size, offset_pxl_x=offset, offset_pxl_y=offset, offset_norm_x=offset_norm, offset_norm_y=offset_norm)
            for tile, row, col in tiles:
                tile_tensor = preprocess_tile(tile, image_size=image_size)
                if tile_tensor is None:
                    continue
                tiles_list.append(tile_tensor)
                # row/col already include the fractional offset and are 0-based.
                rows_list.append(row)
                cols_list.append(col)
        
        if tiles_list:
            all_tiles.extend(tiles_list)
            all_rows.extend(rows_list)
            all_cols.extend(cols_list)
            image_tile_counts.append(len(tiles_list))
        else:
            image_tile_counts.append(0)
    
    if not all_tiles:
        return None, None, None, None, None
    
    tiles_tensor = torch.stack(all_tiles, dim=0)
    rows_tensor = torch.tensor(all_rows, dtype=torch.float32)
    cols_tensor = torch.tensor(all_cols, dtype=torch.float32)
    
    return tiles_tensor, rows_tensor, cols_tensor, image_tile_counts, (all_rows, all_cols)

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