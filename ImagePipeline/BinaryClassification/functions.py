
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
from scipy.ndimage import maximum_filter

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
    """Variational Encoder with multi-level positional embeddings (Method 2)"""
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
        
        # Position embeddings (multi-level)
        embedding_dim = hidden_channels * 4
        self.row_embedding = nn.Embedding(grid_size, embedding_dim)
        self.col_embedding = nn.Embedding(grid_size, embedding_dim)
        
        # Compute output feature shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.image_size, self.image_size)
            feat = self.conv(dummy)
        self.feat_shape = feat.shape[1:]
        self.feat_dim = feat.numel()
        
        embedding_total_dim = embedding_dim * 2  # row + col
        
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
            row = row.to(x.device).long()
            col = col.to(x.device).long()

            # Get position embeddings
            row_emb = self.row_embedding(row)
            col_emb = self.col_embedding(col)
            pos_emb = torch.cat([row_emb, col_emb], dim=1)
            
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
            row = row.to(x.device).long()
            col = col.to(x.device).long()
            
            # Get position embeddings
            row_emb = self.row_embedding(row)
            col_emb = self.col_embedding(col)
            pos_emb = torch.cat([row_emb, col_emb], dim=1)
            
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


class ObservationScorer(nn.Module):
    """
    Dense neural network for classifying tiles as observation or empty.
    Supports two modes:
    - 'binary': Binary classification (observation vs empty)
    - 'one_class': One-class learning (detect anomalies when only empty data available)
    
    Includes explicit position encoding to suppress corner/edge artifacts.
    Designed for GPU-accelerated inference and training.
    
    Input: latent_features (latent_dim) + recon_error (1) + position_features (3)
    Output: probability (1) of being observation (binary) or anomaly score (one_class)
    """
    def __init__(self, latent_dim=32, hidden_dim=256, dropout=0.3, grid_size=16, mode='binary'):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.mode = mode  # 'binary' or 'one_class'
        
        assert mode in ['binary', 'one_class'], f"Mode must be 'binary' or 'one_class', got {mode}"
        
        # Input: latent_dim + recon_error + 3 position features
        input_dim = latent_dim + 1 + 3
        
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
        batch_size = mu.shape[0]
        
        # Compute position features
        center = (self.grid_size - 1) / 2.0
        rows_f = rows.float() if isinstance(rows, torch.Tensor) else torch.tensor(rows, dtype=torch.float32, device=mu.device)
        cols_f = cols.float() if isinstance(cols, torch.Tensor) else torch.tensor(cols, dtype=torch.float32, device=mu.device)
        
        # Normalize to [-1, 1]
        rows_norm = (rows_f - center) / center
        cols_norm = (cols_f - center) / center
        
        # Distance from center [0, 1]
        dist_from_center = torch.sqrt(rows_norm**2 + cols_norm**2) / np.sqrt(2)
        dist_from_center = torch.clamp(dist_from_center, 0, 1)
        
        # Stack features: latent + recon_error + position features
        recon_err_expanded = recon_err.unsqueeze(1) if recon_err.dim() == 1 else recon_err
        features = torch.cat([
            mu,
            recon_err_expanded,
            rows_norm.unsqueeze(1),
            cols_norm.unsqueeze(1),
            dist_from_center.unsqueeze(1)
        ], dim=1)
        
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
        
        x = self.fc_out(x)
        # Sigmoid to output probability in [0, 1]
        prob = torch.sigmoid(x).squeeze(-1)
        
        return prob
    
    def predict_batch(self, mu, recon_err, rows, cols, threshold=0.5):
        """
        Batch prediction with thresholding.
        
        Args:
            mu: Latent representations
            recon_err: Reconstruction error
            rows: Row indices
            cols: Column indices
            threshold: Classification threshold
        
        Returns:
            Binary predictions and probabilities
            
        Note: In one-class mode, the network outputs probability of "normal" (empty).
              We invert to get observation probability for consistency.
        """
        with torch.no_grad():
            probs = self.forward(mu, recon_err, rows, cols)
            
            # In one-class mode, invert prob (high prob = normal/empty, so 1-prob = observation)
            if self.mode == 'one_class':
                probs = 1.0 - probs
            
            preds = (probs >= threshold).long()
        return preds, probs


class SVM_NN(nn.Module):
    """
    One-Class SVM implemented as a neural network for GPU acceleration.
    Learns a nonlinear transformation to separate normal data from outliers.
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature transformation layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Decision boundary
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.threshold = 0.0
        self.is_fitted = False
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x.squeeze(-1)
    
    def fit(self, X, nu=0.05, epochs=100, batch_size=256, learning_rate=0.001, device='cuda'):
        """
        Train the one-class SVM model using deep SVDD approach.
        
        Args:
            X: Training features (numpy array or tensor)
            nu: Upper bound on fraction of outliers (default 0.05)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
        """
        self.to(device)
        self.train()
        
        # Convert to tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X[perm]
            
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                scores = self(X_batch)
                
                # One-class SVM loss (minimize mean score, penalize negative scores)
                # This pushes the decision boundary to enclose most of the data
                margin_loss = -scores.mean()  # Minimize mean distance to center
                penalty_loss = torch.mean(torch.clamp(scores + 1.0, min=0))  # Penalize large negative scores
                loss = margin_loss + 0.1 * penalty_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (n_samples // batch_size):.4f}")
        
        self.eval()
        
        # Set threshold based on nu parameter
        with torch.no_grad():
            train_scores = self(X).cpu().numpy()
            self.threshold = np.quantile(train_scores, 1 - nu)
        
        self.is_fitted = True
        print(f"Training complete. Threshold set to: {self.threshold:.4f}")
    
    def decision_function(self, X, device='cuda'):
        """
        Get decision scores for samples.
        Positive scores = normal, Negative scores = outliers
        
        Args:
            X: Input features (numpy array or tensor)
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            Decision scores (numpy array)
        """
        self.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(device)
        
        with torch.no_grad():
            scores = self(X).cpu().numpy()
        
        return scores
    
    def predict(self, X, device='cuda'):
        """
        Predict class labels.
        1 = normal (inlier), -1 = outlier
        
        Args:
            X: Input features
            device: Device to run on
            
        Returns:
            Class labels (-1 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        scores = self.decision_function(X, device)
        return np.where(scores >= self.threshold, 1, -1)
    
    def predict_proba(self, X, device='cuda'):
        """
        Get probability-like scores (normalized to [0, 1]).
        
        Args:
            X: Input features
            device: Device to run on
            
        Returns:
            Scores in range [0, 1] where higher = more normal
        """
        scores = self.decision_function(X, device)
        # Normalize scores to [0, 1] using sigmoid
        return 1.0 / (1.0 + np.exp(-scores))
    
    def save(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path, device='cuda'):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        # Rebuild model if dimensions don't match
        if self.fc1.in_features != self.input_dim:
            self.__init__(checkpoint['input_dim'], checkpoint['hidden_dim'])
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.is_fitted = checkpoint['is_fitted']
        self.to(device)
        self.eval()
        print(f"Model loaded from {path}")


class BSVM_NN(nn.Module):
    """
    Binary SVM implemented as a neural network for GPU acceleration.
    Classifies data into two classes (e.g., empty vs observation).
    """
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature transformation layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Binary classification output
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.threshold = 0.5
        self.is_fitted = False
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return torch.sigmoid(x).squeeze(-1)  # Output probability in [0, 1]
    
    def fit(self, X, y, epochs=100, batch_size=256, learning_rate=0.001, device='cuda', validation_split=0.2):
        """
        Train the binary classification model.
        
        Args:
            X: Training features (numpy array or tensor)
            y: Training labels (0 or 1, numpy array or tensor)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
            validation_split: Fraction of data to use for validation
        """
        self.to(device)
        self.train()
        
        # Convert to tensors
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        
        X = X.to(device)
        y = y.to(device)
        
        # Split into train and validation
        n_samples = X.shape[0]
        n_train = int(n_samples * (1 - validation_split))
        
        perm = torch.randperm(n_samples)
        X_train = X[perm[:n_train]]
        y_train = y[perm[:n_train]]
        X_val = X[perm[n_train:]]
        y_val = y[perm[n_train:]]
        
        # Initialize optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            # Shuffle training data
            perm = torch.randperm(n_train)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for i in range(0, n_train, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / (n_train // batch_size):.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.eval()
        self.is_fitted = True
        print(f"Training complete. Validation loss: {best_val_loss:.4f}")
    
    def decision_function(self, X, device='cuda'):
        """
        Get decision scores for samples (before sigmoid).
        This is the raw output from the network.
        
        Args:
            X: Input features (numpy array or tensor)
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            Raw decision scores (numpy array)
        """
        self.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(device)
        
        with torch.no_grad():
            # Get raw output before sigmoid
            x = self.fc1(X)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.dropout2(x)
            
            scores = self.fc3(x).squeeze(-1).cpu().numpy()
        
        return scores
    
    def predict_proba(self, X, device='cuda'):
        """
        Get probability predictions for both classes.
        
        Args:
            X: Input features (numpy array or tensor)
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            Probabilities for class 1 (numpy array)
        """
        self.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(device)
        
        with torch.no_grad():
            probs = self(X).cpu().numpy()
        
        return probs
    
    def predict(self, X, threshold=None, device='cuda'):
        """
        Predict class labels.
        0 = class 0, 1 = class 1
        
        Args:
            X: Input features
            threshold: Classification threshold (default 0.5)
            device: Device to run on
            
        Returns:
            Class labels (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        if threshold is None:
            threshold = self.threshold
        
        probs = self.predict_proba(X, device)
        return (probs >= threshold).astype(int)
    
    def set_threshold(self, X_val, y_val, metric='f1', device='cuda'):
        """
        Find optimal threshold using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: 'f1' (default), 'balanced_accuracy', or 'roc_auc'
            device: Device to run on
        """
        from sklearn.metrics import f1_score, balanced_accuracy_score
        
        probs = self.predict_proba(X_val, device)
        
        best_score = -1.0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (probs >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, preds)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_val, preds)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.4f}, {metric}: {best_score:.4f}")
    
    def save(self, path):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path, device='cuda'):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=device)
        
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        
        # Rebuild model if dimensions don't match
        if self.fc1.in_features != self.input_dim:
            self.__init__(checkpoint['input_dim'], checkpoint['hidden_dim'])
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.is_fitted = checkpoint['is_fitted']
        self.to(device)
        self.eval()
        print(f"Model loaded from {path}")


def flood_fill_region(prob_map, start_y, start_x, secondary_threshold):
    """
    Flood-fill from a peak to find the connected region above secondary_threshold.
    Returns a list of (y, x) coordinates in the region.
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


def extract_tiles_with_offset(image, tile_size, grid_size, offset_x=0, offset_y=0):
    """
    Extract tiles from image with specified offset.
    
    Args:
        image: input image
        tile_size: size of each tile
        grid_size: number of tiles along one side
        offset_x: horizontal offset (in pixels)
        offset_y: vertical offset (in pixels)
    
    Returns:
        list of (tile, row, col) tuples
    """
    tiles = []
    image_h, image_w = image.shape[:2]
    
    for row in range(grid_size):
        for col in range(grid_size):
            y0 = offset_y + row * tile_size
            x0 = offset_x + col * tile_size
            y1 = min(y0 + tile_size, image_h)
            x1 = min(x0 + tile_size, image_w)
            
            # Handle edge cases by padding if necessary
            tile = image[y0:y1, x0:x1]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = np.pad(tile, ((0, tile_size - tile.shape[0]), (0, tile_size - tile.shape[1])), mode='edge')
            
            tiles.append((tile, row, col))
    
    return tiles

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