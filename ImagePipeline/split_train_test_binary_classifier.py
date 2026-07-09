#%% Import packages
import os
import random
from pathlib import Path
import shutil

#%% Define user parameters
monitoring_effort = "Kristineberg_250915"# "Kristineberg_250915" #"Kristineberg_251128"

ROOT_C = r"C:\\Users\\Admin\\Documents\\Jellyscope"
ROOT_R = r"R:\\LU24A1037-Jellyscope\\Jellyscope"

manual_labels_folder = f"{ROOT_C}\\Training data\\class_classifier\\monitoring_data\\{monitoring_effort}\\manual_annotations"
source_folder = f"{ROOT_C}\\Training data\\class_classifier\\monitoring_data\\{monitoring_effort}\\OG_images"  # Folder containing the original images to match against (e.g., OG_images folder for the same monitoring effort)
output_folder = f"{ROOT_C}\\Training data\\Binary_classifier\\{monitoring_effort}"

output_folder_dnn = output_folder + "\\train_dnn"
output_folder_test = output_folder + "\\test"

os.makedirs(output_folder_dnn, exist_ok=True)
os.makedirs(output_folder_test, exist_ok=True)

# find number of images already in the output folder (in both train_dnn and test subfolders) to avoid overwriting existing images when copying new ones
existing_images_dnn = list(Path(output_folder_dnn).rglob('*.png'))
existing_images_test = list(Path(output_folder_test).rglob('*.png'))
existing_images = existing_images_dnn + existing_images_test
existing_image_names = [img.stem for img in existing_images]  # List of image names (without extension) already in the output folder

if len(existing_images) > 0:
    print(f"Warning: Found {len(existing_images)} existing images in output folders, skipping these")

# remove images from the manual_labels list if they already exist in the output folder to avoid overwriting existing images when copying new ones
manual_labels = [label for label in Path(manual_labels_folder).rglob('*.json') if label.stem not in existing_image_names]  
split_ratio = 0.75  # Ratio of images to use for training (e.g., 0.8 means 80% for training and 20% for testing)

indexes_dnn = random.sample(range(len(manual_labels)), int(len(manual_labels)*split_ratio))
indexes_test = [i for i in range(len(manual_labels)) if i not in indexes_dnn]

### make output folders
# images
output_folder_dnn_images = Path(output_folder_dnn) / "OG_images"
output_folder_test_images = Path(output_folder_test) / "OG_images"
os.makedirs(output_folder_dnn_images, exist_ok=True)
os.makedirs(output_folder_test_images, exist_ok=True)

# masks
output_folder_dnn_masks = Path(output_folder_dnn) / "manual_masks"
output_folder_test_masks = Path(output_folder_test) / "manual_masks"
os.makedirs(output_folder_dnn_masks, exist_ok=True)
os.makedirs(output_folder_test_masks, exist_ok=True)

# get all image names in the source folder
subfolders = [f for f in Path(source_folder).iterdir() if f.is_dir()]
source_images = list(Path(source_folder).rglob('*.png'))  # Adjust the pattern if you want to filter specific image formats

print("Configuration:")
print(f"  Monitoring effort: {monitoring_effort}")
print(f"  Manual labels folder: {manual_labels_folder}")
print(f"  Source folder: {source_folder}")  
print(f"  Output folder: {output_folder}")
print(f"  Number of manual labels: {len(manual_labels)}")
print(f"  Number of images to use for training: {len(indexes_dnn)}")
print(f"  Number of images to use for testing: {len(indexes_test)}")

#%% Copy original images to the output folder, appending the species name to the image name to avoid overwriting images with the same name from different subfolders

for idx, label_path in enumerate(manual_labels):
    img_name = label_path.stem  # Get the file name without extension
    
    source_img = None
    for img in source_images:
        if img.stem == img_name:
            source_img = Path(img)
            break
    
    if source_img == None:
        print(f"Warning: Could not find source image for label {label_path.name}, expected image name: {img_name}")
        continue  # Skip to the next label if the source image is not found
        
    source_label = label_path  # The label path is the source for the label
    
    if idx in indexes_dnn:
        destination_folder_masks = output_folder_dnn_masks
        destination_folder_images = output_folder_dnn_images
    elif idx in indexes_test:
        destination_folder_masks = output_folder_test_masks
        destination_folder_images = output_folder_test_images
    else:
        raise ValueError(f"Index {idx} is not in either indexes_dnn or indexes_test")

    # Create a new image name by appending the species name to the original image name
    destination_img = Path(destination_folder_images) / (img_name + ".png")
    destination_label = Path(destination_folder_masks) / (img_name + ".json")

    # Copy the original image to the output folder with the new name
    shutil.copy(source_img, destination_img)
    shutil.copy(source_label, destination_label)
    
    print(f"Copied img {idx+1}/{len(manual_labels)}: {source_img.name} to {destination_img.name}", end="\r")

#%% Tile the images in train_dnn