
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
        eps = torch.randn_like(std)
        z = mu + eps * std

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

def mahalanobis_distance(x, mu_g, cov_inv):
    if isinstance(x, torch.Tensor):
        x_t = x
        mu_t = torch.as_tensor(mu_g, dtype=x_t.dtype, device=x_t.device)
        cov_inv_t = torch.as_tensor(cov_inv, dtype=x_t.dtype, device=x_t.device)

        single_input = x_t.ndim == 1
        if single_input:
            x_t = x_t.unsqueeze(0)

        diff = x_t - mu_t
        d2 = torch.sum((diff @ cov_inv_t) * diff, dim=-1)
        d = torch.sqrt(torch.clamp_min(d2, 0.0))
        return d[0] if single_input else d

    x_np = np.asarray(x)
    mu_np = np.asarray(mu_g)
    cov_inv_np = np.asarray(cov_inv)

    diff = x_np - mu_np
    
    if diff.ndim == 1:
        return np.sqrt(diff.T @ cov_inv_np @ diff)

    d2 = np.einsum("bi,ij,bj->b", diff, cov_inv_np, diff)
    return np.sqrt(np.maximum(d2, 0.0))


def classify_image(image_path, model, grid_size, threshold, mu_g, cov_g,
                   device=None, batch_size=64, image_size=128, plot=True):
    """
    Classify all tiles of a single image using a VAE and Mahalanobis distance.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image (.png, .jpg, …).
    model : VariationalAutoencoder
        Trained VAE with an .encode() method that returns (mu, logvar).
    grid_size : int
        Number of tiles along each side (e.g. 16 → 16×16 = 256 tiles).
    threshold : float
        Mahalanobis-distance threshold above which a tile is predicted as 'obs'.
    mu_g : array-like, shape (latent_dims,)
        Mean of the 'empty' Gaussian fitted in latent space.
    cov_g : array-like, shape (latent_dims, latent_dims)
        Covariance of the 'empty' Gaussian.
    device : torch.device, optional
        Device to run inference on. Defaults to CUDA if available, else CPU.
    batch_size : int
        Number of tiles per inference batch.
    image_size : int
        Tile resize target (must match the model's input resolution).
    plot : bool
        If True, display the observation-likelihood heatmap overlay.

    Returns
    -------
    df_latent : pd.DataFrame
        One row per tile with columns: image_name, tile_row, tile_col,
        label_predicted, d2, latent_0 … latent_N.
    """


    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = Path(image_path)
    cov_inv = np.linalg.inv(np.asarray(cov_g))
    tile_size = 4512 // grid_size

    # 1. load & tile 
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    raw_tiles = split_image_into_tiles(image, grid_size=grid_size)  # list of (tile, row, col)
    dataset = [(tile, row, col) for tile, row, col in raw_tiles]

    # 3. encode & score 
    mu_list, d2_list, rows_list, cols_list = [], [], [], []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for raw_tile in raw_tiles:
            tile, row, col = raw_tile
            tile_resized = cv2.resize(tile, (image_size, image_size), interpolation=cv2.INTER_AREA)
            tile_tensor = torch.from_numpy(tile_resized).float().unsqueeze(0).unsqueeze(0) / 255.0   # 1x1xHxW in [0,1]
            tile_tensor = tile_tensor.to(device)
            mu, _ = model.encode(tile_tensor)
            
            mu_np = mu.cpu().numpy()
            d2 = mahalanobis_distance(mu_np, mu_g, cov_inv)
            mu_list.append(mu_np)
            d2_list.append(d2)
            rows_list.append(row)
            cols_list.append(col)

    mu_array = np.concatenate(mu_list, axis=0)
    d2_array = np.concatenate(d2_list, axis=0)
    labels_array = d2_array > threshold

    # 4. build df 
    latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
    df_latent = pd.DataFrame(mu_array, columns=latent_cols)
    df_latent.insert(0, "image_name",     image_path.stem)
    df_latent.insert(1, "tile_row",       rows_list)
    df_latent.insert(2, "tile_col",       cols_list)
    df_latent.insert(3, "label_predicted", labels_array)
    df_latent.insert(4, "d2",             d2_array)

    # 5. heatmap plot 
    if plot:
        likelihood_threshold = chi2.cdf(threshold, df=2)
        cmap_ryg = mpl.colors.LinearSegmentedColormap.from_list(
            "red_yellow_green", ["red", "yellow", "green"])
        norm = mpl.colors.TwoSlopeNorm(
            vmin=0.0, vcenter=likelihood_threshold, vmax=1.0)

        image = cv2.imread(str(image_path))
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Observation likelihood: {image_path.stem}",
                    fontsize=16, fontweight="bold")

        for _, row in df_latent.iterrows():
            obs_likelihood = chi2.cdf(row["d2"], df=2)
            color = cmap_ryg(norm(obs_likelihood))
            x0 = int(row["tile_col"] * tile_size)
            y0 = int(row["tile_row"] * tile_size)
            rect = Rectangle(
                (x0, y0), tile_size, tile_size,
                linewidth=0, edgecolor="none",
                facecolor=color, alpha=0.45)
            ax.add_patch(rect)

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_ryg)
        sm.set_array([])
        ticks = np.linspace(0, 1, 6)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01,
                                aspect=30, location="right")
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=16)
        cbar.set_label("Observation likelihood",
                        rotation=90, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    return df_latent


def classify_folder(folder_path, model, grid_size, threshold, mu_g, cov_g,
                    device=None, batch_size=64, image_size=128, plot=False, csv_output_path=None):
    """
    Classify all tiles from all images in a folder using a VAE and Mahalanobis distance.

    Parameters
    ----------
    folder_path : str or Path
        Folder containing input images.
    model : VariationalAutoencoder
        Trained VAE with an .encode() method that returns (mu, logvar).
    grid_size : int
        Number of tiles along each side (e.g. 16 -> 16x16 tiles per image).
    threshold : float
        Mahalanobis-distance threshold above which a tile is predicted as 'obs'.
    mu_g : array-like, shape (latent_dims,)
        Mean of the 'empty' Gaussian fitted in latent space.
    cov_g : array-like, shape (latent_dims, latent_dims)
        Covariance of the 'empty' Gaussian.
    device : torch.device, optional
        Device to run inference on. Defaults to CUDA if available, else CPU.
    batch_size : int
        Number of tiles per inference batch.
    image_size : int
        Tile resize target (must match the model's input resolution).
    plot : bool
        If True, display observation-likelihood heatmaps for each image.
    csv_output_path : str or Path, optional
        Path for saving observation-only results. Defaults to
        <folder_path>/df_latent_observations.csv.

    Returns
    -------
    df_latent_obs : pd.DataFrame
        One row per predicted observation tile with columns: image_name,
        tile_row, tile_col, label_predicted, d2, latent_0 ... latent_N.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = Path(folder_path)
    image_paths = sorted(folder_path.glob("*.png"))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .png images found in folder: {folder_path}")

    cov_inv = np.linalg.inv(np.asarray(cov_g))
    tile_size = 4512 // grid_size

    tiles_list = []
    for image_path in image_paths:
        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            continue

        raw_tiles = split_image_into_tiles(image_gray, grid_size=grid_size)
        for tile, row, col in raw_tiles:
            tiles_list.append((tile, row, col, image_path.stem))

    if len(tiles_list) == 0:
        raise ValueError(f"No valid tiles could be created from images in: {folder_path}")

    tile_loader = DataLoader(
        tiles_list,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: test_image_pipeline(batch, image_size=image_size),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    mu_list = []
    image_names_list = []
    d2_list = []
    label_list = []
    tile_rows = []
    tile_cols = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for tiles, rows, cols, image_names in tile_loader:
            tiles = tiles.to(device, non_blocking=True)
            mu, _ = model.encode(tiles, row=rows, col=cols)

            mu_np = mu.cpu().numpy()
            d2 = mahalanobis_distance(mu_np, mu_g, cov_inv)
            is_obs = d2 > threshold

            mu_list.append(mu_np)
            d2_list.append(d2)
            label_list.append(is_obs)
            tile_rows.append(rows.cpu().numpy())
            tile_cols.append(cols.cpu().numpy())
            image_names_list.append(np.asarray(image_names))

    mu_array = np.concatenate(mu_list, axis=0)
    d2_array = np.concatenate(d2_list, axis=0)
    labels_array = np.concatenate(label_list, axis=0)
    image_names_array = np.concatenate(image_names_list, axis=0)
    tile_rows_array = np.concatenate(tile_rows, axis=0)
    tile_cols_array = np.concatenate(tile_cols, axis=0)

    latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
    df_latent = pd.DataFrame(mu_array, columns=latent_cols)
    df_latent.insert(0, "image_name", image_names_array)
    df_latent.insert(1, "tile_row", tile_rows_array)
    df_latent.insert(2, "tile_col", tile_cols_array)
    df_latent.insert(3, "label_predicted", labels_array)
    df_latent.insert(4, "d2", d2_array)

    df_latent_obs = df_latent[df_latent["label_predicted"]].reset_index(drop=True)

    if csv_output_path is None:
        csv_output_path = folder_path / "df_latent_observations.csv"
    else:
        csv_output_path = Path(csv_output_path)

    df_latent_obs.to_csv(csv_output_path, index=False)
    print(f"Saved {len(df_latent_obs)} observation tiles to: {csv_output_path}")

    if plot:
        likelihood_threshold = chi2.cdf(threshold, df=2)
        cmap_ryg = mpl.colors.LinearSegmentedColormap.from_list(
            "red_yellow_green", ["red", "yellow", "green"])
        norm = mpl.colors.TwoSlopeNorm(
            vmin=0.0, vcenter=likelihood_threshold, vmax=1.0)

        for image_name in df_latent["image_name"].unique():
            df_latent_image = df_latent[df_latent["image_name"] == image_name]
            image_path = folder_path / f"{image_name}.png"
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image)
            ax.axis("off")

            for df_tile in df_latent_image.itertuples():
                x0 = int(df_tile.tile_col * tile_size)
                y0 = int(df_tile.tile_row * tile_size)
                obs_likelihood = chi2.cdf(df_tile.d2, df=2)
                color = cmap_ryg(norm(obs_likelihood))
                rect = Rectangle((x0, y0), tile_size, tile_size, linewidth=0, edgecolor="none", facecolor=color, alpha=0.45)
                ax.add_patch(rect)

            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_ryg)
            sm.set_array([])
            ticks = np.linspace(0, 1, 6)
            cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, aspect=30, location="right")
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks], fontsize=16)
            cbar.set_label("Observation likelihood", rotation=90, fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.show()

    return df_latent_obs