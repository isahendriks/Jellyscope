#%% Cell 1: Import packages
### PACKAGES ###
import os
import torch
from torch.utils.data import DataLoader

import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import pickle as pkl
from pathlib import Path
import cv2
import seaborn as sns

from matplotlib.patches import Rectangle

from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA

from functions import Autoencoder, VariationalAutoencoder, eval_image_pipeline

print("Packages imported successfully.")

#%% Cell 2: Activate GPU
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

#%% Cell 3: Define paths and parameters
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"
SAMPLE_IMAGE_IDX_TEST = 0

DEBUG = False
n_debug = 0.2

monitoring_effort = "Kristineberg_260424" #"Kristineberg_251128"
# monitoring_effort = "Kristineberg_250915"
grid_size = 16
latent_dim = 32
batch_size = 1024

model_name = f"models/{monitoring_effort}_vae_model{grid_size}_l{latent_dim}.pth"
if DEBUG:
    model_name = model_name.replace(".pth", "_debug.pth")
print(f"Model to load: {model_name}")

test_tiles_path = os.path.join(ROOT_DIR_C, monitoring_effort, "test", f"tiles{grid_size}")
test_og_images_path = os.path.join(ROOT_DIR_C, monitoring_effort, "test", "OG_images")

images_test = list(Path(test_og_images_path).glob("*.png"))
sample_image_path = Path(test_og_images_path) / images_test[SAMPLE_IMAGE_IDX_TEST].name
sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

if DEBUG:
    n_test = int(len(images_test) * n_debug)
    images_test = np.random.choice(images_test, size=n_test, replace=False)
    print(f"DEBUG MODE: Using only {n_test} test images ({n_test*grid_size**2} tiles) for evaluation")

#%% Cell 4: Load model
checkpoint = torch.load(model_name, map_location=device, weights_only=False)

latent_dims = checkpoint.get("latent_dim")
if latent_dims!= latent_dim:
    print(f"⚠ Warning: Loaded model latent_dims={latent_dims} does not match expected latent_dim={latent_dim}. Using loaded value.")
    latent_dim = latent_dims
image_size = checkpoint.get("image_size")
hidden_channels = checkpoint.get("hidden_channels")
grid_size = checkpoint.get("grid_size")
state_dict = checkpoint["model_state_dict"]

model = VariationalAutoencoder(latent_dims=latent_dims, image_size=image_size, hidden_channels=hidden_channels, grid_size=grid_size)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

#%% Cell 5: Define test dataset and dataloader
test_tiles_obs_path = os.path.join(test_tiles_path, "obs")
test_tiles_no_obs_path = os.path.join(test_tiles_path, "no_obs")

test_tiles_obs = sorted(Path(test_tiles_obs_path).rglob("*.png"))
test_tiles_no_obs = sorted(Path(test_tiles_no_obs_path).rglob("*.png"))

if DEBUG:
    n_test_obs = int(len(test_tiles_obs) * n_debug)
    n_test_no_obs = int(len(test_tiles_no_obs) * n_debug)
    test_tiles_obs = np.random.choice(test_tiles_obs, size=n_test_obs, replace=False)
    test_tiles_no_obs = np.random.choice(test_tiles_no_obs, size=n_test_no_obs, replace=False)
    print(f"DEBUG MODE: Using only {n_test_obs} obs tiles and {n_test_no_obs} no_obs tiles for evaluation")

test_loader_obs = DataLoader(
    test_tiles_obs,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_image_pipeline,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

test_loader_no_obs = DataLoader(
    test_tiles_no_obs,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_image_pipeline,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

print("\n" + "="*120)
print(f"Loaded {len(test_tiles_obs)} observation test tiles from: {test_tiles_obs_path}")
print(f"Loaded {len(test_tiles_no_obs)} empty test tiles from: {test_tiles_no_obs_path}")
print("="*120)

#%% Cell 6: Plot sample test tiles from both classes
batch_obs = next(iter(test_loader_obs))
batch_no_obs = next(iter(test_loader_no_obs))

test_tiles_obs_batch = batch_obs[0]
test_tiles_no_obs_batch = batch_no_obs[0]

# Plot observation tiles
N_hor = 8
N_ver = 3
fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= len(test_tiles_obs_batch):
        continue
    tile = test_tiles_obs_batch[idx].cpu().numpy()
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)
plt.suptitle("Sample obs tiles")
plt.tight_layout()
plt.show()
    
# Plot no_obs tiles
fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= len(test_tiles_no_obs_batch):
        continue
    tile = test_tiles_no_obs_batch[idx].cpu().numpy()
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)
plt.suptitle("Sample no_obs tiles")
plt.tight_layout()
plt.show()

#%% Cell 7: Load all test data and get latent space representations
model.eval()

mu_list = []
tile_labels = []
tile_rows = []
tile_cols = []
tiles = []
recon_errors = []

# Enable cuDNN benchmarking and use async GPU transfers
torch.backends.cudnn.benchmark = True

with torch.no_grad():
    for test_loader in [test_loader_obs, test_loader_no_obs]:
        class_name = "obs" if test_loader == test_loader_obs else "no_obs"
        print(f"\n\nProcessing {class_name} test tiles with {len(test_loader.dataset)} tiles and {len(test_loader)} batches...")

        for batch_idx, (images, labels, rows, cols) in enumerate(test_loader):
            # Move to device (non_blocking for async transfer)
            images = images.to(device, non_blocking=True)
            rows_gpu = rows.to(device, non_blocking=True)
            cols_gpu = cols.to(device, non_blocking=True)
            
            # Encode and reconstruct
            mu, _ = model.encode(images, row=rows_gpu, col=cols_gpu)
            x_hat = model.decode(mu)
            
            # Compute reconstruction error
            recon_err = ((images - x_hat) ** 2).flatten(1).mean(dim=1)
            
            # Move results back to CPU (async)
            mu_list.append(mu.to('cpu', non_blocking=True))
            recon_errors.append(recon_err.to('cpu', non_blocking=True))
            tile_labels.append(labels)
            tile_rows.append(rows)
            tile_cols.append(cols)
            tiles.append(images.to('cpu', non_blocking=True))

            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}", end="\r")
        
        # Synchronize GPU before moving to next class
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        print(f"\nProcessing {class_name} test tiles... done!", end="\n")
            
print("\nFinished processing all test tiles.")

# Synchronize GPU before concatenating
if torch.cuda.is_available():
    torch.cuda.synchronize()

mu_array = torch.cat(mu_list, dim=0).numpy()
tile_labels = torch.cat(tile_labels, dim=0).numpy().astype(bool)
tile_rows = torch.cat(tile_rows, dim=0).numpy()
tile_cols = torch.cat(tile_cols, dim=0).numpy()
tiles = torch.cat(tiles, dim=0).numpy()
recon_errors = torch.cat(recon_errors, dim=0).numpy()

latent_cols = [f"latent_{i}" for i in range(mu_array.shape[1])]
df_latent = pd.DataFrame(mu_array, columns=latent_cols)
df_latent.insert(0, "label_manual", tile_labels)
df_latent.insert(1, "tile_row", tile_rows)
df_latent.insert(2, "tile_col", tile_cols)
df_latent.insert(3, "recon_error", recon_errors)
df_latent['tile'] = list(tiles)

# Sanity check
if len(df_latent[df_latent['label_manual']==True]) == len(test_tiles_obs) and len(df_latent[df_latent['label_manual']==False]) == len(test_tiles_no_obs):
    print("\n" + "="*60)
    print("Sanity check passed: Tile counts match")
    print("="*60)
    
print(f"Class distribution: obs={len(df_latent[df_latent['label_manual']==True])}, no_obs={len(df_latent[df_latent['label_manual']==False])}")


#%% Split data into train and eval for SVM 
svm_feature_cols_names = [f"latent_{i}" for i in range(mu_array.shape[1])] + ["recon_error"]
df_svm_train, df_svm_eval = train_test_split(df_latent, test_size=0.5, stratify=df_latent['label_manual'], random_state=42)

#%% METHOD 1: One class SVM on latent features + reconstruction error

# Optional: Stratify dataset for faster grid search (only use a subset of empty tiles for training and eval, but keep all obs tiles since they are already rare)
df_svm_train_oc = df_svm_train[df_svm_train['label_manual'] == False]  # Only use no_obs tiles for training the one-class SVM, once class
df_svm_train_sampled = df_svm_train_oc.sample(frac=1, random_state=42)  # Stratify data for faster learning
df_svm_eval_sampled = df_svm_eval.sample(frac=1, random_state=42)  # Sample eval set for faster evaluation, but keep all obs tiles

scaler = StandardScaler()
X_train_base_sampled = scaler.fit_transform(df_svm_train_sampled[svm_feature_cols_names].values)
X_eval_base_sampled = scaler.transform(df_svm_eval_sampled[svm_feature_cols_names].values)
recon_col_idx_sampled = len(svm_feature_cols_names) - 1
y_true_svm = df_svm_eval_sampled['label_manual'].values

train_no_obs_mask_sampled = (df_svm_train_sampled['label_manual'] == False) # select only no_obs tiles for training
train_no_obs_mask_sampled = train_no_obs_mask_sampled.values

# Grid search for best (weight_recon_error, nu)
# Split dataset for SVM training/eval
weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Adjust weights for reconstruction error as needed (e.g. 1, 2, 5, 10, 20)
nu_grid = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001] #, 0.002, 0.005, 0.０1, ０.０15] #０.０2, ０.０3, ０.０4, ０.０5]  # Adjust nu values as needed, nu = number of datapoints within boundary / total number of datapoints (e.g. ０.１ means １０% of data is expected to be within the boundary, so smaller nu = tighter boundary = fewer obs predictions)

search_results = []
for w in weight_grid:
    for nu_val in nu_grid:
        print(f"Testing weight_recon_error={w}, nu={nu_val}...", end="")
        X_train_try = X_train_base_sampled.copy()
        X_eval_try = X_eval_base_sampled.copy()
        X_train_try[:, recon_col_idx_sampled] *= w
        X_eval_try[:, recon_col_idx_sampled] *= w
        
        model_try = OneClassSVM(kernel='rbf', gamma='scale', nu=nu_val)

        model_try.fit(X_train_try[train_no_obs_mask_sampled])

        y_scores_try = -model_try.decision_function(X_eval_try)
        precision_try, recall_try, thresholds_try = precision_recall_curve(y_true_svm, y_scores_try)
        aps_try = average_precision_score(y_true_svm, y_scores_try)
        
        # Choose threshold by best F1
        f1_try = 2 * precision_try[:-1] * recall_try[:-1] / (precision_try[:-1] + recall_try[:-1] + 1e-12)
        trsh_idx_try = np.argmax(f1_try)
        
        best_precision = precision_try[trsh_idx_try]
        best_recall = recall_try[trsh_idx_try]
        best_f1 = f1_try[trsh_idx_try]
        best_threshold = thresholds_try[trsh_idx_try]
        
        search_results.append({"weight": w, 
                               "nu": nu_val,
                               "aps": aps_try,
                               "recall": best_recall,
                               "f1": best_f1,
                               "threshold": best_threshold})
        
        print(f" done! APS={aps_try:.4f}, Recall={best_recall:.4f}, best f1={best_f1:.4f}")
        
search_df = pd.DataFrame(search_results).sort_values("aps", ascending=False).reset_index(drop=True)
best_row = search_df.iloc[0]

weight_recon_error = float(best_row["weight"])
nu = float(best_row["nu"])

print("\nTop hyperparameter combos (by PR-APS):")
print(search_df.head(10))
print(f"\nSelected -> weight_recon_error={weight_recon_error}, nu={nu}, APS={best_row['aps']:.4f}")

#%% Train final SVM with best hyperparameters, evaluate on eval set, plot ROC curve and confusion matrix, save model
weight_recon_error = float(best_row["weight"])
nu = float(best_row["nu"])

# fit scalar to full distribution of latent features in the eval set
scaler = StandardScaler()
scaler.fit(df_latent[svm_feature_cols_names].values)  # Fit on full distribution

# Now scale train and eval using this representative scaler
X_train_base_full = scaler.transform(df_latent.loc[df_svm_train.index, svm_feature_cols_names].values)
X_eval_base = scaler.transform(df_svm_eval[svm_feature_cols_names].values)

# Get training and eval labels  
y_train_full = df_latent.loc[df_svm_train.index, 'label_manual'].values
y_true_svm = df_svm_eval['label_manual'].values

# OneClassSVM trains only on no_obs, but use indices from df_latent to identify them
train_is_no_obs = (y_train_full == False)
recon_col_idx = len(svm_feature_cols_names) - 1

X_train = X_train_base_full.copy()
X_eval = X_eval_base.copy()
X_train[:, recon_col_idx] *= weight_recon_error
X_eval[:, recon_col_idx] *= weight_recon_error
train_no_obs_mask = train_is_no_obs

svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
svm.fit(X_train[train_no_obs_mask])

# ROC curve for SVM
y_scores_svm = -svm.decision_function(X_eval)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_true_svm, y_scores_svm)

roc_auc_svm = auc(fpr_svm, tpr_svm)

#%% set treshold for outlier detection by setting fpr and fnr
lambda_weight = 0.4  # Adjust this weight to balance FPR and FNR according to your needs (0.5 means equal weight)
fnr_svm = 1 - tpr_svm
cost = lambda_weight * fpr_svm + (1 - lambda_weight) * fnr_svm
trsh_idx_svm = np.argmin(cost)
threshold_svm = thresholds_svm[trsh_idx_svm]

print(f"\nSVM Optimal threshold (Youden's J): {threshold_svm:.8f}")

df_svm_eval['label_predicted_SVM'] = (y_scores_svm >= threshold_svm)

plt.figure(figsize=(6,6))
plt.plot(fpr_svm, tpr_svm, label=f"AUC = {roc_auc_svm:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_svm[trsh_idx_svm], tpr_svm[trsh_idx_svm], marker='o', markersize=8, label=f"Optimal threshold = {threshold_svm:.8f}", color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for SVM on VAE Latent Space")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

### PR curve for SVM
precision_svm, recall_svm, thresholds_pr_svm = precision_recall_curve(y_true_svm, y_scores_svm)
aps_svm = average_precision_score(y_true_svm, y_scores_svm)
plt.figure(figsize=(6,6))
plt.plot(recall_svm, precision_svm, label=f"APS = {aps_svm:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for SVM on VAE Latent Space")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Confusion matrix for SVM
y_pred_svm = (y_scores_svm >= threshold_svm).astype(bool)
cm_svm = confusion_matrix(df_svm_eval['label_manual'], y_pred_svm, labels=[False, True], normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=["no_obs", "obs"])
disp.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Confusion Matrix for SVM")
plt.show()

#%% Save SVM model and parameters
### One class SVM
svm_model_name = model_name.replace("vae_model", "oc_svm_model").replace(".pth", ".pkl")
print(f"\nSaving SVM model to: {svm_model_name} with paramters:", end = "\n")
print(f"weight_recon_error={weight_recon_error}, nu={nu}, threshold_svm={threshold_svm:.8f}")
with open(svm_model_name, "wb") as f:
    pkl.dump((svm, scaler, weight_recon_error, nu, threshold_svm), f)
print(f"\n One class SVM model trained and saved to: {svm_model_name}")


#%% METHOD 2: Binary SVM on latent features + reconstruction error

# fit scalar to full distribution of latent features in the eval set
scaler_bsvm = StandardScaler()
scaler_bsvm.fit(df_latent[svm_feature_cols_names].values)  # Fit on full distribution

# Now scale train and eval using this representative scaler
latent_train = scaler_bsvm.transform(df_latent.loc[df_svm_train.index, svm_feature_cols_names].values)
latent_eval = scaler_bsvm.transform(df_svm_eval[svm_feature_cols_names].values)

# Get training and eval labels  
labels_train= df_svm_train['label_manual'].values
labels_eval = df_svm_eval['label_manual'].values

# Define class weights
class_weights = {
    "obs": 1.0,
    "empty": len(labels_train[labels_train == False]) / len(labels_train[labels_train == True])
}

class_weights = {0: class_weights["empty"], 1: class_weights["obs"]}

model = SVC(
    kernel='rbf',          # usually best starting point
    class_weight=class_weights,
    probability=True,      # needed for ROC / PR curves
    gamma='scale'
)

print(f"\nTraining Binary SVM with class weights: {class_weights}...")
model.fit(latent_train, labels_train)
print(f"\rTraining Binary SVM with class weights: {class_weights}... done!")

# Predictions
labels_eval_pred = model.predict(latent_eval)
pred_score = model.predict_proba(latent_eval)[:, 1]

#%% confusion matrix for binary SVM

# Evaluation
pred_scores = model.decision_function(latent_eval)
fpr_bsvm, tpr_bsvm, thresholds_bsvm = roc_curve(labels_eval, pred_scores)

roc_auc_bsvm = auc(fpr_bsvm, tpr_bsvm)

precision_bsvm, recall_bsvm, thresholds_pr_bsvm = precision_recall_curve(labels_eval, pred_scores)
aps_bsvm = average_precision_score(labels_eval, pred_scores)

#%% METHOD 2: Set threshold for binary classification

# METHOD A: Using weighted FPR and FNR (original method)
# lambda_weight_bsvm = 0.4  # Adjust this weight to balance FPR and FNR according to your needs (0.5 means equal weight)
# fnr_bsvm = 1 - tpr_bsvm
# cost_bsvm = lambda_weight_bsvm * fpr_bsvm + (1 - lambda_weight_bsvm) * fnr_bsvm
# best_idx_roc = np.argmin(cost_bsvm)
# threshold_bsvm = thresholds_bsvm[best_idx_roc]

# METHOD B: Using maximum allowed False Discovery Rate (FDR)
# FDR = False Positives / (True Positives + False Positives)
# This means: out of all predictions marked as positive, what fraction are actually false positives
# If you want 1% FDR, that means 99% precision (99 out of 100 positive predictions are correct)
max_fdr_bsvm = 0.9  # 10% false positives allowed out of total positive predictions (= 90% precision)
min_precision = 1 - max_fdr_bsvm  # minimum required precision

# Find the threshold that achieves at least the minimum precision
valid_indices = np.where(precision_bsvm >= min_precision)[0]
if len(valid_indices) > 0:
    # Among valid thresholds, choose the one with highest recall (most sensitive)
    best_idx_precision = valid_indices[np.argmax(recall_bsvm[valid_indices])]
    threshold_bsvm = thresholds_pr_bsvm[best_idx_precision]
    actual_precision = precision_bsvm[best_idx_precision]
    actual_recall = recall_bsvm[best_idx_precision]
    print(f"\nBinary SVM threshold set by FDR: {threshold_bsvm:.8f}")
    print(f"Achieved precision (1-FDR): {actual_precision:.4f} (target: {min_precision:.4f})")
    print(f"Achieved recall: {actual_recall:.4f}")
else:
    print(f"⚠ Warning: Cannot achieve {max_fdr_bsvm:.2%} FDR with current model.")

df_svm_eval['label_predicted_BinarySVM'] = (pred_scores >= threshold_bsvm).astype(int)

# ROC curve for Binary SVM
plt.figure(figsize=(6,6))
plt.plot(fpr_bsvm, tpr_bsvm, label=f"AUC = {roc_auc_bsvm:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr_bsvm[best_idx_roc], tpr_bsvm[best_idx_roc], marker='o', markersize=8, label=f"Optimal threshold = {threshold_bsvm:.8f}", color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Binary SVM on VAE Latent Space")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

## Confusion matrix using optimal threshold
y_pred_bsvm_optimal = (pred_scores >= threshold_bsvm).astype(bool)
cm_bsvm_optimal = confusion_matrix(df_svm_eval['label_manual'], y_pred_bsvm_optimal, labels=[False, True])
disp_optimal = ConfusionMatrixDisplay(confusion_matrix=cm_bsvm_optimal, display_labels=["empty", "observation"])
disp_optimal.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Confusion Matrix for Binary SVM (Optimal Threshold)")
plt.show()

## Confusion matrix using optimal threshold
y_pred_bsvm_optimal = (pred_scores >= threshold_bsvm).astype(bool)
cm_bsvm_optimal = confusion_matrix(df_svm_eval['label_manual'], y_pred_bsvm_optimal, labels=[False, True], normalize='true')
disp_optimal = ConfusionMatrixDisplay(confusion_matrix=cm_bsvm_optimal, display_labels=["empty", "observation"])
disp_optimal.plot(cmap=plt.cm.Blues, values_format=".2f")
plt.title("Confusion Matrix for Binary SVM (Optimal Threshold) normalized")
plt.show()
plt.savefig(f"confusion_matrix_binary_svm_{model_name[7:].replace('.pth', '')}.png", dpi=300)

# PR curve for Binary SVM

plt.figure(figsize=(6,6))
plt.plot(recall_bsvm, precision_bsvm, label=f"APS = {aps_bsvm:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Binary SVM on VAE Latent Space")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#%% Save Binary SVM
bsvm_model_name = model_name.replace("vae_model", "binary_svm_model").replace(".pth", ".pkl")
print(f"\nSaving Binary SVM model to: {bsvm_model_name} with paramters:", end = "\n")
print(f"threshold_bsvm={threshold_bsvm:.8f}")
with open(bsvm_model_name, "wb") as f:
    pkl.dump((model, scaler_bsvm, threshold_bsvm), f)
print(f"\nBinary SVM model trained and saved to: {bsvm_model_name}")

#%% Cell 10: Select model and plot false positives and false negatives
# Choose which method to evaluate (uncomment one)
USE_SVM = True  # Set to True for SVM, False for Mahalanobis


df_eval = df_svm_eval.copy()
predicted_labels_use = y_pred_svm
method_name = "SVM"

# Define false positives and negatives
df_false_positives = df_eval[(df_eval['label_manual'] == False) & (predicted_labels_use == True)]
df_false_negatives = df_eval[(df_eval['label_manual'] == True) & (predicted_labels_use == False)]

print(f"\n{method_name} Results:")
print(f"False Positives: {len(df_false_positives)}")
print(f"False Negatives: {len(df_false_negatives)}")

# Plot false positives
N_fp = len(df_false_positives)
N_hor = 8
if N_fp > N_hor * 8:
    N_ver = 8
    df_false_positives_plot = df_false_positives.sample(n=N_hor*N_ver, random_state=42).reset_index(drop=True)
    title_str = f"False Positives: empty tiles classified as obs (showing random subset of {N_hor*N_ver} out of {N_fp})"
else:
    N_ver = int(np.ceil(N_fp / N_hor))
    df_false_positives_plot = df_false_positives.reset_index(drop=True)
    title_str = f"False Positives: empty tiles classified as obs ({N_fp} total)"

fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
if N_ver == 1:
    axs = axs.reshape(1, -1)
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= len(df_false_positives_plot):
        continue
    
    tile = df_false_positives_plot.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

plt.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Plot false negatives
N_fn = len(df_false_negatives)
if N_fn > N_hor * 8:
    N_ver = 8
    df_false_negatives_plot = df_false_negatives.sample(n=N_hor*N_ver, random_state=42).reset_index(drop=True)
    title_str = f"False Negatives: obs tiles classified as empty (showing random subset of {N_hor*N_ver} out of {N_fn})"
else:
    N_ver = int(np.ceil(N_fn / N_hor))
    df_false_negatives_plot = df_false_negatives.reset_index(drop=True)
    title_str = f"False Negatives: obs tiles classified as empty ({N_fn} total)"

fig, axs = plt.subplots(N_ver, N_hor, figsize=(N_hor*1.2, N_ver*1.2))
if N_ver == 1:
    axs = axs.reshape(1, -1)
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.axis("off")
    if idx >= len(df_false_negatives_plot):
        continue
    
    tile = df_false_negatives_plot.iloc[idx]['tile']
    ax.imshow(tile[0], cmap="gray", vmin=0, vmax=1)

plt.suptitle(title_str, fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

#%% Use Mahalanobic distance and reconstruction error to plot ROC curve and confusion matrix, save parameters
manual_labels = df_latent["label_manual"].values.astype(bool)

# Plot ROC curve
y_true = df_latent["label_manual"].astype(int).values
y_score = df_latent["recon_error"].values

fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)

# set treshold for outlier detection by setting 
lambda_weight = 0.4  # Adjust this weight to balance FPR and FNR according to your needs (0.5 means equal weight)
fnr = 1 - tpr
cost = lambda_weight * fpr + (1 - lambda_weight) * fnr
trsh_idx = np.argmin(cost)
threshold_recon = roc_thresholds[trsh_idx]

roc_auc_recon = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="blue", linewidth=2, label=f"ROC curve (AUC = {roc_auc_recon:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
plt.scatter(
    fpr[trsh_idx], tpr[trsh_idx],
    color="red", s=90, zorder=5,
    label=f"threshold={roc_thresholds[trsh_idx]:.3f}\nTPR={tpr[trsh_idx]:.3f}, FPR={fpr[trsh_idx]:.3f}"
)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve: Reconstruction Error vs Manual Labels", fontsize=16, fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.show()

print(f"Reconstruction Error ROC AUC: {roc_auc_recon:.4f}")
print(f"Selected threshold: {threshold_recon:.4f}")
print(f"At this threshold, TPR={tpr[trsh_idx]:.3f} and FPR={fpr[trsh_idx]:.3f}")

# Save Reconstruction Error parameters
model_params = {
    "threshold": threshold_recon,
}

val_params_name = model_name.replace(".pth", "_recon_params.pkl")
with open(val_params_name, "wb") as f:
    pkl.dump(model_params, f)
print(f"\nReconstruction Error parameters saved to: {val_params_name}")

# Confusion matrix for Reconstruction Error
predicted_labels_recon = df_latent["recon_error"].values > threshold_recon
conf_matrix_recon = confusion_matrix(manual_labels, predicted_labels_recon, labels=[False, True], normalize='true')
disp = ConfusionMatrixDisplay(conf_matrix_recon, display_labels=["empty", "obs"])
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix: Reconstruction Error")
plt.show()