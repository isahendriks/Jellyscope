#%% Import packages
import re
from pathlib import Path
import shutil

import pandas as pd

print("\n" + "=" * 60)
print("Packages imported successfully yippie!")
print("=" * 60)


#%% Define user parameters
monitoring_effort = "Kristineberg_250915" #"Kristineberg_251128"

ROOT_C = r"C:\Users\Admin\Documents\GitHub\Jellyscope\Training data"
ROOT_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data"

folder_crops = f"{ROOT_R}\Class_classifier\Monitoring_training_data\Original"
source_folder = f"{ROOT_R}\Monitoring data\{monitoring_effort}"
output_folder = f"{ROOT_R}\Binary_classifier\{monitoring_effort}\test\cropped_images"



print("Configuration:")
print(f"  Input folder: {folder_crops}")
print(f"  Source folder: {source_folder}")
print(f"  Output folder: {output_folder}")

print(f"  Date(s):      {date}")

#%% Find all subfolders in the input folder
subfolders = [f for f in Path(folder_crops).iterdir() if f.is_dir()]

df_matches = pd.DataFrame(columns=["image_path", "date", "species"])  # Initialize an empty DataFrame to store matches

print(f"Found {len(subfolders)} subfolders in the input folder.")

for subfolder in subfolders:
    # print(f"\nProcessing subfolder: {subfolder.name}")
    
    # Find all images in the subfolder
    images = list(subfolder.rglob('*.png'))  # Adjust the pattern if you want to filter specific image formats
    # print(f"  Found {len(images)} images in this subfolder.")
    
    # Filter images by date using regex
    date_pattern = "|".join(date.split(","))
    matched_images = [img for img in images if re.search(date_pattern, img.stem)]  # Check if the date pattern matches the image name (excluding the last 14 characters)
    img_name = [img.stem for img in matched_images]
    
    # Create a DataFrame to store matched image paths and their corresponding dates
    entry = pd.DataFrame({
        "image_path": matched_images,
        "image_name": img_name,
        "species": subfolder.name,
    })
    
    # Append the matched images to the main DataFrame
    df_matches = pd.concat([df_matches, entry], ignore_index=True)
    print(f" {subfolder.name}: {len(entry)} images matched the date(s).")

print(f"\n Found a total of {len(df_matches)} images matching the date(s) across all subfolders.")

#%%
species_to_copy = df_matches["species"].unique()
# species_to_drop = ['dentritus', 'filament', 'dentritus_glo', 'filament_glo', 'macro_filament']
# species_to_copy = [species for species in species_to_copy if species not in species_to_drop]
df_to_copy = df_matches[df_matches['species'].isin(species_to_copy)]
print(f"\n Species to copy: {species_to_copy}, total number of iamages to copy: {len(df_to_copy)}")

#%% Copy the images from source to output folder

for _, row in df_to_copy[11].iterrows():  # Limit to first 10 images for testing
    image_path = Path(row["image_path"])
    image_name = image_path.stem
    
    species = row["species"]
    
    # Construct the source path based on the image path
    source_path = image_path
    output_path = Path(output_folder) / Path(str(image_name) + f"_{species}.png")  # Append species name to the output image name

    print(f"Copying image: {source_path} to {output_path}")
    
    if source_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create output folder if it doesn't exist
        shutil.copy(source_path, output_path)  # Copy the file to the output folder
    else:
        print(f"Warning: Source image not found: {source_path}")