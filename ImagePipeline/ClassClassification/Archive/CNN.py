#%% IMport packages

### PACKAGES ###
import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale, RandomHorizontalFlip, RandomVerticalFlip
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint

import deeplay as dl

import matplotlib.pyplot as plt
import os
import numpy as np
import time 

import json
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

print("Packages imported successfully yiho.")

# %% Set user variables
### INPUTS ###
# ROOT_DIR = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Segmented\squared_split_training_data3"
ROOT_DIR = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\class_classification\Monitoring_training_data\squared_split_training_data"
data_dir = r"C:\Users\Admin\Documents\Jellyscope\Training data\class_classifier\monitoring_data\OG_training_data"

train_dir = os.path.join(ROOT_DIR, "train")
val_dir = os.path.join(ROOT_DIR, "val")
test_dir = os.path.join(ROOT_DIR, "test")

weights_save_name = 'zooplanktonet_monitoring'
output_folder = os.path.join(ROOT_DIR, "output")
os.makedirs(output_folder, exist_ok=True)

#%% GPU status report
### ENVIRONMENT REPORT ###
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("="*40)
print(f"Using device:      {device}")
print("="*40)

# %% Datasets and dataloaders
input_size = (128,128)  # Adjust as needed for your model architecture
input_channels = 1  # Grayscale images
input_shape = input_size[0] * input_size[1] * input_channels

transform = Compose([
    Resize(input_size),   # Resize for uniform display
    Grayscale(num_output_channels=1),  # enforce 1 channel
    ToTensor()
])

train = ImageFolder(data_dir, transform=transform)
val = ImageFolder(val_dir, transform=transform)
test = ImageFolder(test_dir, transform=transform)

# set num_workers to 0 when loading from network drive
train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=1, persistent_workers=True, pin_memory=True)
val_loader = DataLoader(val, batch_size=128, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True)

print(f"structure of datset:")
counts = np.zeros((len(train.classes), 3), dtype=int)
for label in train.classes:
    train_count = len([i for i in train.targets if i == train.class_to_idx[label]])
    val_count = len([i for i in val.targets if i == val.class_to_idx[label]])
    test_count = len([i for i in test.targets if i == test.class_to_idx[label]])
    print(f"  {label}: train={train_count}, val={val_count}, test={test_count}")
    counts[train.class_to_idx[label], 0] = train_count
    counts[train.class_to_idx[label], 1] = val_count
    counts[train.class_to_idx[label], 2] = test_count

#%% Visualize three example images of each class in the validation set
dataset = val

# Handle both ImageFolder and Subset objects
classes = dataset.dataset.classes if hasattr(dataset, 'dataset') else dataset.classes
grid = [len(classes), 3]  # 3 images per class, number of classes in dataset

fig, axs = plt.subplots(grid[1], grid[0], figsize=(grid[0]*2, grid[1]*2))  # Adjust figure size as needed

# Pre-compute class indices once instead of in the loop
all_labels = [dataset[i][1] for i in range(len(dataset))]

for i in range(grid[0]):
    ax = axs[:,i]  # Select the row corresponding to the current class
    class_indices = [j for j, label in enumerate(all_labels) if label == i]
    selected_indices = np.random.choice(class_indices, min(grid[1], len(class_indices)), replace=False)
    for j, idx in enumerate(selected_indices):
        image, label = dataset[idx]
        ax[j].imshow(image.squeeze(), cmap='gray')
        if j == 0:  # Only set title for the first image in the row
            ax[j].set_title(f"{classes[label]}", fontsize=14, fontweight='bold')
        ax[j].axis('off')
plt.tight_layout()
plt.show()

# print label shapes and datatype
# print('shape: ', next(iter(train_loader))[1].shape, 'data type:' , next(iter(train_loader))[1].dtype)
#%% Create neural network architecture
### ZooplanktoNet architecture ###
conv_base = nn.Sequential(
                        nn.Conv2d(1, 96, kernel_size=13, padding=6),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2),

                        nn.Conv2d(96, 256, kernel_size=7, padding=3),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2),

                        nn.Conv2d(256, 384, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(384, 384, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(384, 384, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2),

                        nn.Conv2d(384, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2), 
                        )

# Global pooling to convert feature maps into vector
connector = dl.Layer(nn.AdaptiveAvgPool2d, output_size=1)

# Dense top
dense_top = dl.MultiLayerPerceptron(
    in_features=512,                    # has to match the outchannel
    hidden_features=[4096, 4096],
    out_features=17,                    # number of species
    out_activation=None
)

# Full model
zooplanktonet_model = dl.Sequential(
    conv_base,
    connector,
    dense_top
)

# Single wrapper for the model
zooplanktonet_classifier = dl.CategoricalClassifier(
    model=zooplanktonet_model,
    num_classes=17,
    optimizer=dl.Adam(lr=0.001),
    loss=torch.nn.CrossEntropyLoss(),
)

# Build first, then move to GPU
zooplanktonet_classifier.build()
zooplanktonet_classifier = zooplanktonet_classifier.to(device)
# print out complied cnn
print("Compiled ZooplanktoNet Classifier:")
print(zooplanktonet_classifier)

#%% Train the convolutional neural network using DataLoaders

checkpoint_cb = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    filename=f"{weights_save_name}_best-{{epoch:02d}}-{{val_loss:.4f}}",
    save_last=True
)

# Create trainer with callbacks
# trainer = Trainer(
#     max_epochs=100,
#     callbacks=[checkpoint_cb, TQDMProgressBar(refresh_rate=1)],
#     log_every_n_steps=1,
#     accelerator="gpu",
#     devices=1,
#     enable_progress_bar=True,
#     enable_model_summary=True,
#     precision="16-mixed"
# )

# Train the model
zooplanktonet_classifier.fit(train_loader=train,
    val_loader=val,
    num_epochs=100,
    callbacks=[checkpoint_cb]
    )
zooplanktonet_classifier.eval() # set model to evaluation mode after training

print("\nBest checkpoint:", checkpoint_cb.best_model_path)
print("Last checkpoint:", checkpoint_cb.last_model_path)

# Save the best model and last model state dict
torch.save(zooplanktonet_classifier.model.state_dict(), os.path.join(output_folder, f"{weights_save_name}_best.pt"))
torch.save(zooplanktonet_classifier.model.state_dict(), os.path.join(output_folder, f"{weights_save_name}_last.pt"))

#%% Plot training history (extract from checkpoints)
# Note: Since we're not using deeplay's .fit(), we need to extract history differently
# The metrics are logged by Lightning, so we can access them from the trainer
print("\nTraining complete!")
print(f"Final train loss logged in checkpoint: {checkpoint_cb.best_model_path}")

#%% save training history note
# Note: Lightning stores logs differently. You can access metrics via tensorboard or the checkpoint files
print("To view training history, use TensorBoard:")
print(f"  tensorboard --logdir lightning_logs/")

# Load the best model checkpoint after training
ckpt = torch.load(checkpoint_cb.best_model_path, map_location=device)
zooplanktonet_classifier.model.load_state_dict(ckpt["state_dict"], strict=False)
zooplanktonet_classifier.model = zooplanktonet_classifier.model.to(device)

#%% Test best model on test set
test_results = zooplanktonet_classifier.evaluate(test, batch_size=128)
print("Test results:", test_results)


#%% Generate confusion matrix using predict

# Extract inputs and labels from test set
inputs, labels = zip(*test)

# Get predictions using classifier.predict
predicted_scores = zooplanktonet_classifier.predict(inputs)
predicted_labels = torch.argmax(predicted_scores, dim=1)

# Convert to numpy for sklearn
y_true = torch.tensor(labels).cpu().numpy()
y_pred = predicted_labels.cpu().numpy()

# Print first 10 predictions as sanity check
print("\nSample predictions:")
for i in range(min(10, len(labels))):
    status = "✓" if predicted_labels[i] == labels[i] else "✗ Error!"
    print(f"Predicted: {test.classes[predicted_labels[i]]}, True: {test.classes[labels[i]]} {status}")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test.classes)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"{weights_save_name}_confusion_matrix.png"), dpi=300)
plt.show()

# Print detailed classification report
print("\n" + "="*40)
print("Per-Class Performance")
print("="*40)
print(classification_report(y_true, y_pred, target_names=test.classes, digits=3))