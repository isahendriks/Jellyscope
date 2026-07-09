#%%### PACKAGES ###

from fileinput import filename
import random

from matplotlib.pylab import gamma
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.ops import sigmoid_focal_loss

import copy
import time
import re
import os 
import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report
import random

#%% GPU setup
print("="*40)
print("PyTorch Environment Report")
print("="*40)

print(f"PyTorch version:        {torch.__version__}")

# Check MPS (Metal) availability
mps_available = torch.backends.mps.is_available()
print(f"GPU available (MPS):    {mps_available}")

if mps_available:
    # Apple does not provide detailed GPU properties in PyTorch yet
    print(f"GPU device:             Apple GPU (Metal)")
    print(f"Current device:         mps")

# Set device
device = torch.device("mps") if mps_available else torch.device("cpu")
print("="*40)
print(f"Using device:           {device}")

#%% Set up network and dataloader
class ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)
        vit_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()
        
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=1),
        )

        self.resize = transforms.Resize((224, 224))
        
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.size_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(vit_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, img, size):
        img = self.stem(img)  # [batch, 3, 128, 128] if input is [batch, 1, 256, 256]
        img = self.resize(img)  # [batch, 3, 224, 224]
        img = (img - self.imagenet_mean) / self.imagenet_std  # Normalize using ImageNet stats
        
        img_features = self.vit(img)          # [batch, vit_features]
        size_features = self.size_encoder(size)  # [batch, 32]

        x = torch.cat([img_features, size_features], dim=1)
        return self.classifier(x)

class JellyfishDataset(Dataset):
    def __init__(self, root, transform=None, classes_to_remove=None):
        self.base_dataset = ImageFolder(root, transform=transform)

        classes_to_remove = set(classes_to_remove or [])
                
        self.samples = [
            (path, label)
            for path, label in self.base_dataset.samples
            if self.base_dataset.classes[label] not in classes_to_remove
        ]

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        self.samples = []
        for path, label in self.base_dataset.samples:
            filename = os.path.basename(path)

            if filename.startswith("._"):
                continue
            if not filename.lower().endswith(valid_exts):
                continue
            if self.base_dataset.classes[label] in classes_to_remove:
                continue

            self.samples.append((path, label))

        remaining_classes = sorted({
            self.base_dataset.classes[label]
            for _, label in self.samples
        })
        
        self.classes = remaining_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(remaining_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        path, old_label = self.samples[idx]
        class_name = self.base_dataset.classes[old_label]
        label = self.class_to_idx[class_name]
        
        img = self.base_dataset.loader(path)
        filename = os.path.basename(path)
        
        size = re.search(r"_area_(\d+)\.png$", filename)
        if size is None:
            width, height = img.size
            size = torch.tensor(float(width * height * 0.02**2), dtype=torch.float32)
        else:
            size = torch.tensor(float(size.group(1)), dtype=torch.float32)
        
        if self.base_dataset.transform is not None:
            img = self.base_dataset.transform(img)
              

        return img, size, label

class AugmentedSubset(Dataset):
    def __init__(self, subset, augment=True):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, size, label = self.subset[idx]

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            angle = random.choice([0, 90, 180, 270])
            img = TF.rotate(img, angle)

        return img, size, label

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img

        diff = abs(w - h)
        if w < h:
            left = diff // 2
            right = diff - left
            padding = (left, 0, right, 0)
        else:
            top = diff // 2
            bottom = diff - top
            padding = (0, top, 0, bottom)

        return TF.pad(img, padding, fill=0)


#%% Load training data and model
vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # convert to grayscale
    PadToSquare(),  # pad to square if needed
    transforms.Resize((256, 256)), # resize to 256x256
    transforms.ToTensor(), # convert to tensor
])

### Load data
# monitoring_effort = 'Kristineberg_251128'
# all_training_data_path = f"C:\\Users\\Admin\\Documents\\Jellyscope\\Training data\\class_classifier\\monitoring_data\\{monitoring_effort}\\OG_images_crops"
# all_training_data_path = f"R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data\\Class_classifier\\Monitoring_training_data\\Original_new"

all_training_data_path = f"C:\\Users\\Admin\\Documents\\Jellyscope\\Training data\\class_classifier\\histogram_segmentation\\Original_new"
folder_names = sorted(os.listdir(all_training_data_path))
# remove any non-folder files from folder_names
class_names = [name for name in folder_names if os.path.isdir(os.path.join(all_training_data_path, name))]
num_classes = len(class_names)

print("All classes in dataset:", class_names)

### Define larger groups
Ctenophora = ['mnemiopsis', 'pleurobrachia', 'beroe', 'small_ctenophore']
Hydrozoa = ['clytia_spp1', 'narcomedusa', 'narcomedusa_group']
Other = ['worm_like', 'chaetognath', 'shrimp', 'oikopleura']
Copepods = ['calanus', 'centropages', 'copepod']
Aggregates = ['filament', 'filament_glo', 'detritus', 'detritus_glo', 'macro_filament', 'hairy_stuff']

# remove classes that have less than 65 samples
min_samples = 40

classes_to_remove = Aggregates.copy()  # Start with the predefined list of classes to remove
for class_name in class_names:
    class_path = os.path.join(all_training_data_path, class_name)
    num_samples = len(os.listdir(class_path))
    if num_samples < min_samples:
        classes_to_remove.append(class_name)
        
# Print the classes that will be removed
print(f"Classes to remove (less than {min_samples} samples or aggregates): {classes_to_remove}")

# remove classes with less than 40 samples from dataset
all_training_data = JellyfishDataset(all_training_data_path, transform=preprocess, classes_to_remove=classes_to_remove)
class_names = all_training_data.classes
num_classes = len(class_names)
idx_to_class = all_training_data.idx_to_class
class_to_idx = all_training_data.class_to_idx

# split into train, val, test
train_size = int(0.7 * len(all_training_data))
val_size = int(0.15 * len(all_training_data))
test_size = len(all_training_data) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(all_training_data, [train_size, val_size, test_size])

### Get class counts for training dataset
train_labels = [train_dataset.dataset.samples[i][1] for i in train_dataset.indices]
class_counts = torch.bincount(torch.tensor(train_labels)).float()
# remove all zeros from class_counts
class_counts = class_counts[class_counts > 0]
class_weights = len(train_labels) / (num_classes * class_counts)
class_weights = class_weights.to(device)

# Check if class weights has the same length as the number of classes
if len(class_weights) != num_classes:
    raise ValueError(f"Length of class_weights ({len(class_weights)}) does not match number of classes ({num_classes}).")

# Apply augmentation to the training dataset and no augmentation to the validation and test datasets
train_dataset = AugmentedSubset(train_dataset, augment=True)
val_dataset = AugmentedSubset(val_dataset, augment=False)
test_dataset = AugmentedSubset(test_dataset, augment=False)    

### Plot one example from each class to verify data loading and preprocessing
# select one image from each class in the training set to plot
N_plot = []
labels_seen = []

for i in range(len(train_dataset)):
    _, _, label = train_dataset[i]
    
    if label not in labels_seen:
        N_plot.append(i)
        labels_seen.append(label)
    if len(N_plot) >= num_classes:
        break
        
# plot the selected images with their class names and sizes
fig, axs = plt.subplots(3, len(N_plot)//3+1, figsize=((len(N_plot)//3+1) * 4, 3 * 4))
axs = axs.flatten()

for ax, i in enumerate(N_plot):
    img, size, label = train_dataset[i]
    size = size.item()  # convert tensor to scalar
    # reshape from (3, 224, 224) to (224, 224, 3) for plotting
    img = img.permute(1, 2, 0).numpy()    
    axs[ax].imshow(img, cmap='gray')
    axs[ax].set_title(f"{idx_to_class[label]}, size={size:.0f} mm2")
    axs[ax].axis('off')   
   
#%% Load model 

# Train the model with frozen backbone
epochs = 80
N_unfreeze = 40  # Number of epochs to train with frozen backbone before unfreezing
batch_size = 32
lr_frozen = 1e-4/(batch_size/32)  # Adjust learning rate based on batch size
lr_unfrozen = 1e-5/(batch_size/32)  # Adjust learning rate based on batch size
n_blocks = 4 # Number of ViT blocks to unfreeze for fine-tuning

step_losses = []
train_losses = []
val_losses = []

train_acc_history = []
val_acc_history = []

train_f1_history = []
val_f1_history = []

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
 

best_f1 = 0.0
best_epoch = None

best_model_state = None

# Define new classifier
classifier = ViT(num_classes=num_classes).to(device)

# Continue training classifier
# model_path = f"models\\vit_classifier_f1{best_f1:.4f}_acc{0:.4f}.pth"
# if os.path.exists(model_path):
#     classifier.load_state_dict(torch.load(model_path, map_location=device))
#     print(f"Loaded model from {model_path} for continued training.")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=lr_frozen)

#%% Train the model

# Freeze ViT backbone weights to only train the size encoder, classifier head and stem
for p in classifier.vit.parameters():
    p.requires_grad = False

time_start = time.time()
epoch_times = []

for epoch in range(epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_preds = []
    train_targets = []
    
    # Unfreeze backbone after N_unfreeze epochs
    if epoch == epochs - N_unfreeze:
        
        print(f"\nUnfreezing the last {n_blocks} blocks of the ViT backbone for fine-tuning...\n")
        
        # select model with best validation f1 score from frozen training as starting point for unfreeze training
        if best_model_state is not None:
            classifier.load_state_dict(best_model_state)
        
        optimizer = optim.Adam(classifier.parameters(), lr=lr_unfrozen)
        
        for block in classifier.vit.encoder.layers[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

        # keep the final encoder norm trainable too
        for p in classifier.vit.encoder.ln.parameters():
            p.requires_grad = True

    time_epoch_start = time.time()
    for batch_idx, (images, sizes, labels) in enumerate(train_loader):
        # print(f'loaded batch {batch_idx+1}/{len(train_loader)}', end='\r')
        images = images.to(device)
        sizes = sizes.to(device).unsqueeze(1).float()  # add feature dimension
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = classifier(images, sizes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        step_losses.append(loss.item())
        
        _, preds = torch.max(outputs.data, 1)
                
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        
        print(f"Training Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    epoch_f1 = f1_score(train_targets, train_preds, average='macro')
    train_losses.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    train_f1_history.append(epoch_f1)

    classifier.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for val_batch_idx, (images, sizes, labels) in enumerate(val_loader):
            images = images.to(device)
            sizes = sizes.to(device).unsqueeze(1).float()
            labels = labels.to(device)

            outputs = classifier(images, sizes)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
            
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

            print(f"Validation Epoch {epoch+1}/{epochs} - Batch {val_batch_idx+1}/{len(val_loader)} - Loss: {loss.item():.4f}", end='\r')


    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / val_total
    val_epoch_f1 = f1_score(val_targets, val_preds, average='macro')
    val_losses.append(val_epoch_loss)
    val_acc_history.append(val_epoch_acc)
    val_f1_history.append(val_epoch_f1)

    time_epoch_end = time.time()
    time_epoch = time_epoch_end - time_epoch_start
    epoch_times.append(time_epoch)
    avg_epoch_time = np.mean(epoch_times)
    estimated_time_remaining = avg_epoch_time * (epochs - epoch - 1)
    
    avg_epoch_time_hhmmss_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time))
    estimated_time_remaining_hhmmss_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time_remaining))
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val F1: {val_epoch_f1:.4f} - Avg Epoch Time: {avg_epoch_time_hhmmss_str}, Est. Time Remaining: {estimated_time_remaining_hhmmss_str}          ", flush=True)

    if val_epoch_f1 > best_f1:
        best_epoch = epoch + 1
        best_f1 = val_epoch_f1
        best_model_state = copy.deepcopy(classifier.state_dict())

### Plot training and validation loss and accuracy curves
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(best_epoch-1, val_losses[best_epoch-1], 'ro', label='Best Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.plot(best_epoch-1, val_acc_history[best_epoch-1], 'ro', label='Best Val Accuracy')
plt.vlines(x=epochs-N_unfreeze, ymin=0, ymax=1, colors='gray', linestyles='dashed', label='Unfreeze Backbone')
plt.xlabel('Epoch')
plt.grid()
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.plot(best_epoch-1, val_f1_history[best_epoch-1], 'ro', label='Best Val F1')
plt.xlabel('Epoch')
plt.grid()
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score')
plt.legend()

#%% Load saved model for testing

model_name = "Models\\vit_classifier_F1_0.8697_acc_0.9293.pth"

# Load the model state dictionary
if os.path.exists(model_name):
    classifier.load_state_dict(torch.load(model_name, map_location=device))
    print(f"Loaded model from {model_name} for testing.")

#%% Test the model on the test set and plot confusion matrix

# Use best model state for testing
if best_model_state is not None:
    classifier.load_state_dict(best_model_state)
    print(f"Loaded best model state from epoch {best_epoch} with validation F1 score: {best_f1:.4f}")
else:
    print("Warning: No best model state found. Using the last epoch's model for testing.")

classifier.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, sizes, labels) in enumerate(test_loader):
        print(f'Testing - Batch {batch_idx+1}/{len(test_loader)}', end='\r')
        
        images = images.to(device)
        sizes = sizes.to(device).unsqueeze(1).float()
        labels = labels.to(device)
        
        outputs = classifier(images, sizes)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#%% Plot confusion matrix with classes ordered by taxonomy and lines separating taxonomic groups

accuracy_total = accuracy_score(all_labels, all_preds)
precision_total = precision_score(all_labels, all_preds, average='macro')
recall_total = recall_score(all_labels, all_preds, average='macro')
f1_total = f1_score(all_labels, all_preds, average='macro')

print(f"Test Accuracy: {accuracy_total:.4f}")
print(f"Test Precision (macro): {precision_total:.4f}")
print(f"Test Recall (macro): {recall_total:.4f}")
print(f"Test F1 Score (macro): {f1_total:.4f}")

groups = [Ctenophora, Hydrozoa, Copepods, Other]
group_names = ['Ctenophora', 'Hydrozoa', 'Copepods', 'Other']

order = Ctenophora + Hydrozoa + Copepods + Other
borders = [0, len(Ctenophora), len(Ctenophora)+len(Hydrozoa), len(Ctenophora)+len(Hydrozoa)+len(Copepods), len(Ctenophora)+len(Hydrozoa)+len(Copepods)+len(Other)]
class_names_ordered = [cls for cls in order if cls in class_names]

cm = confusion_matrix(all_labels, all_preds, normalize='true')
cm = cm[[class_names.index(cls) for cls in class_names_ordered], :][:, [class_names.index(cls) for cls in class_names_ordered]]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_ordered)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', colorbar=False, values_format='.2f')
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)
plt.xlabel('Predicted Class', fontsize=16, fontweight='bold')
plt.ylabel('True Class', fontsize=16, fontweight='bold')
# plot lines to separate the groups of classes in the confusion matrix
for border in borders:
    ax.axhline(border-0.5, color='black', linewidth=2)
    ax.axvline(border-0.5, color='black', linewidth=2)

# Plot names of the groups on top and right side of the confusion matrix
for i, group in enumerate(groups):
    if len(group) == 0:
        continue
    mid_point = (borders[i] + borders[i+1]) / 2 - 0.5
    ax.text(mid_point, -1, group_names[i], ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(len(class_names_ordered), mid_point, group_names[i], ha='left', va='center', fontsize=14, fontweight='bold', rotation=90)

plt.show()
plt.savefig(f"confusion_matrix_grouped_f1_{f1_total:.4f}_acc_{accuracy_total:.4f}.png", dpi=300, bbox_inches='tight')

#%% Group the classification report by taxonomic group
all_labels_grouped = []
for label in all_labels:
    class_name = class_names[label]
    for i, group in enumerate(groups):
        if class_name in group:
            all_labels_grouped.append(i)
            break

all_preds_grouped = []
for pred in all_preds:
    class_name = class_names[pred]
    for i, group in enumerate(groups):
        if class_name in group:
            all_preds_grouped.append(i)
            break

accuracy = accuracy_score(all_labels_grouped, all_preds_grouped)
precision = precision_score(all_labels_grouped, all_preds_grouped, average='macro')
recall = recall_score(all_labels_grouped, all_preds_grouped, average='macro')
f1 = f1_score(all_labels_grouped, all_preds_grouped, average='macro')

print(f"Grouped Test Accuracy: {accuracy:.4f}")
print(f"Grouped Test Precision: {precision:.4f}")
print(f"Grouped Test Recall: {recall:.4f}")
print(f"Grouped Test F1 Score: {f1:.4f}")

cm_grouped = confusion_matrix(all_labels_grouped, all_preds_grouped, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm_grouped, display_labels=group_names)
fig, ax = plt.subplots(figsize=(12, 12))

disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', colorbar=False)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=45)
plt.xlabel('Predicted Class', fontsize=16)
plt.ylabel('True Class', fontsize=16)
plt.show()


#%% Save model
# save model after initial training with frozen backbone
model_save_path = f"models\\vit_classifier_F1_{f1_total:.4f}_acc_{accuracy_total:.4f}.pth"
torch.save(best_model_state, model_save_path)
print(f"Model with frozen backbone saved to {model_save_path} with validation accuracy: {accuracy_total:.4f} and F1 score: {f1_total:.4f}")




