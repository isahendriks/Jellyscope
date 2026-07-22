#%% Cell 1: Import Packages

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
import shutil

print("Packages imported successfully yippie!")

#%% Cell 2: Define Functions

def pad_image_to_square(image, target_size=128):
    """
    Pad image to square and resize to target_size x target_size.
    Padding is done with black pixels (0,0,0) to preserve aspect ratio.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    target_size : int
        Target size (square)
    
    Returns:
    --------
    np.ndarray
        Padded and resized square image
    """
    h, w = image.shape[:2]
    
    # Determine padding needed
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        padded = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        padded = image
    
    # Resize to target size
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

print("Functions defined successfully yippie!")

#%% Cell 3: Define User Parameters

# Folder paths
ROOT_DIR = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data"
original_folder = os.path.join(ROOT_DIR, "Monitoring_training_data", "OG_training_data") # Folder containing class subfolders
padded_folder = os.path.join(ROOT_DIR, "Monitoring_training_data", "OG_segments_padded_augmented")                         # Where to save padded images
output_folder = os.path.join(ROOT_DIR, "Monitoring_training_data", "squared_split_training_data")           # Where to save split dataset

# Split fractions
train_ratio = 0.7   # 70% training
val_ratio = 0.15    # 15% validation
test_ratio = 0.15   # 15% testing

# Image parameters
target_size = 128   # Square image size (128x128)

# Augmentation settings
augmentation = True  # Set to True to enable rotation and flipping augmentation
stratified_split = True  # Set to True to ensure class distribution is preserved in splits

# Verify ratios sum to 1.0
total_ratio = train_ratio + val_ratio + test_ratio
print(f"Configuration:")
print(f"  Original folder: {original_folder}")
print(f"  Padded folder: {padded_folder}")
print(f"  Output folder: {output_folder}")
print(f"  Augmentation: {'Enabled' if augmentation else 'Disabled'}")
print(f"  Split ratios: Train {train_ratio:.1%}, Val {val_ratio:.1%}, Test {test_ratio:.1%}")
print(f"  Target size: {target_size}x{target_size}")
print(f"  Ratios sum to: {total_ratio:.1%} (should be 100%)")

if abs(total_ratio - 1.0) > 0.001:
    raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

print("\n" + "="*60)
print(f"number of images in each class in original folder:")
class_counts = {}
input_path = Path(original_folder)  
class_folders = sorted([d for d in input_path.iterdir() if d.is_dir()])
for class_folder in class_folders:
    class_name = class_folder.name  
    image_files = [f for f in class_folder.iterdir() if f.is_file()]
    class_counts[class_name] = len(image_files)
    print(f"  {class_name}: {len(image_files)} images")
print("="*60)
   
    
print("\n Parameters configured successfully cuckedudledoo!")

#%% Cell 4: Load, Pad, augment and Save Images

# Create padded folder
padded_path = Path(padded_folder)
padded_path.mkdir(parents=True, exist_ok=True)

# Get class folders
input_path = Path(original_folder)
class_folders = sorted([d for d in input_path.iterdir() if d.is_dir()])

print(f"Found {len(class_folders)} classes\n")

# Image extensions to look for
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Process each class
total_images = 0
example_images = []  # Store examples for visualization

for class_idx, class_folder in enumerate(class_folders):
    class_name = class_folder.name
    
    # Create output class folder
    output_class_path = padded_path / class_name
    output_class_path.mkdir(exist_ok=True)
    
    # Get image files
    image_files = [f for f in class_folder.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"Class {class_idx+1}/17 ({class_name}): No images found")
        continue
    
    # Progress bar for this class
    for img_idx, image_file in enumerate(image_files):
        try:
            # Read image
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            
            # Pad to square and resize
            processed_img = pad_image_to_square(img, target_size=target_size)
            
            if augmentation:
                # Maximize rotation + flip combinations (8 variants total)
                aug_variants = [
                    ("rot0", processed_img),
                    ("rot90", cv2.rotate(processed_img, cv2.ROTATE_90_CLOCKWISE)),
                    ("rot180", cv2.rotate(processed_img, cv2.ROTATE_180)),
                    ("rot270", cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ]

                # Horizontal flip of each rotation (dihedral group D4)
                aug_variants += [
                    (f"{name}_flip_h", cv2.flip(img, 1)) for name, img in aug_variants
                ]

            # Save original padded image
            output_path_file = output_class_path / image_file.name
            cv2.imwrite(str(output_path_file), processed_img)
            
            if augmentation:
            # Save all augmented images
                for name, img in aug_variants:
                    output_path_file_aug = output_class_path / f"{image_file.stem}_{name}.png"
                    cv2.imwrite(str(output_path_file_aug), img)

            total_images += 1
            
            # Print progress
            print(f"Class {class_idx+1}/17 ({class_name}): {img_idx+1}/{len(image_files)} images", end="\r")
        
        except Exception as e:
            print(f"\nError processing {image_file.name}: {e}")
            continue
    
    ### Plot three randomly selected example images from this class
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    selected_image_files = np.random.choice(image_files, size=min(3, len(image_files)), replace=False)
    for idx, image_file in enumerate(selected_image_files):
        try:
            img = cv2.imread(str(image_file))
            if img is not None:
                processed_img = pad_image_to_square(img, target_size=target_size)
                axes[idx].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(f"{class_name}\n{processed_img.shape[0]}x{processed_img.shape[1]}")
                axes[idx].axis('off')
        except Exception as e:
            print(f"\nError processing {image_file.name} for visualization: {e}")
            continue
    
    plt.tight_layout()
    plt.show()
    print(f"Class {class_idx+1}/17 ({class_name}): Completed {len(image_files)} images ✓")


#%% Print summary of padding results
counts = []
print(f"\n{'='*60}")
print(f"  Padding complete!")
# print(f"  Total images processed: {total_images}")
print(f"  Output folder: {padded_path}")
print(f"  Number of images per class:")

for class_folder in sorted(padded_path.iterdir()):
    if class_folder.is_dir():
        class_name = class_folder.name
        image_files = [f for f in class_folder.iterdir() if f.is_file()]
        count = len(image_files)
        counts.append(count)
        print(f"    {class_name}: {count} images")
    
min_count = min(counts)
print(f"  Minimum images in any class: {min_count}")
print(f"{'='*60}")

print("\nAll classes processed successfully yippie!")

#%% Cell 5: Split Images into Train/Val/Test Folders

# Create output directory structure
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)

splits = ['train', 'val', 'test']
split_paths = {}
split_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

if stratified_split:    
    stratified_sizes = {}
    for split in splits:
        split_ratio = split_ratios[split]
        stratified_sizes[split] = int(min_count * split_ratio)

for split in splits:
    split_path = output_path / split
    split_path.mkdir(exist_ok=True)
    split_paths[split] = split_path

# Get class folders from padded folder
padded_path = Path(padded_folder)
class_folders = sorted([d for d in padded_path.iterdir() if d.is_dir()])

print(f"expected structure of output folder:")


print(f" splitting {len(class_folders)} classes into train/val/test with ratios: {train_ratio/100*min_count:.1%}/{val_ratio:.1%}/{test_ratio:.1%} ...")
    


#%% Process each class
for class_idx, class_folder in enumerate(class_folders):
    class_name = class_folder.name
    
    # Create class subfolder in each split
    for split in splits:
        (split_paths[split] / class_name).mkdir(exist_ok=True)
    
    # Get all images in this class
    image_files = sorted([f for f in class_folder.iterdir() if f.is_file()])
    
    if not image_files:
        print(f"Class {class_idx+1}/17 ({class_name}): No images found")
        continue
    
    # Shuffle and split
    
    if stratified_split:
        # If stratified, we take a fixed number of images per split based on the minimum class count
        n_images = min_count
        n_train = stratified_sizes['train']
        n_val = stratified_sizes['val']
        n_test = stratified_sizes['test']
            
        # Shuffle and take the first n_train + n_val + n_test images
        random.shuffle(image_files)
        image_files = image_files[:n_train + n_val + n_test]
        
    
    else:
        random.shuffle(image_files)
        n_images = len(image_files)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        n_test = n_images - n_train - n_val  # Ensure all images are accounted for
        
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Copy files to respective split folders
    for split, files in splits_data.items():
        output_class_path = split_paths[split] / class_name
        for image_file in files:
            shutil.copy2(str(image_file), str(output_class_path / image_file.name))
    
    print(f"Class {class_idx+1}/17 ({class_name}): {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print(f"\n{'='*60}")
print(f"✓ Train/Val/Test split complete!")
print(f"{'='*60}")

# Print summary
print(f"\nOutput structure:")
for split in splits:
    split_path = split_paths[split]
    class_count = len(list(split_path.iterdir()))
    total_files = sum(len(list((split_path / c).iterdir())) for c in split_path.iterdir() if c.is_dir())
    print(f"  {split}/ - {class_count} classes, {total_files} images")

print(f"\n  Output folder: {output_path}")
