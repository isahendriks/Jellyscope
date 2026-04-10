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

ROOT_C = r"C:\\Users\\Admin\\Documents\\Jellyscope"
ROOT_R = r"R:\\LU24A1037-Jellyscope\\Jellyscope"

folder_crops = f"{ROOT_R}\\Training data\\Class_classifier\\Monitoring_training_data\\Original_new"
source_folder = f"{ROOT_R}\\Monitoring data\\{monitoring_effort}\\gain10_nobgsub"  # Folder containing the original images to match against (e.g., OG_images folder for the same monitoring effort)
output_folder = f"{ROOT_R}\\Training data\\Binary_classifier\\{monitoring_effort}\\test\\OG_images"

# subfolders_source = [f for f in Path(source_folder).iterdir() if f.is_dir()]  # List of subfolders in the source folder
# subfolders_source_keep = ["gain10", "gain10_nobgsub"]  # List of subfolder names to keep (e.g., those containing the dates of interest)
# subfolders_source = [f for f in subfolders_source if f.name in subfolders_source_keep]  # Limit to specified subfolders for testing

monitoring_image_names = [img.stem for img in Path(source_folder).rglob('*.png')]  # List of image names (without extension) in the source folder
subfolders = [f for f in Path(folder_crops).iterdir() if f.is_dir()]

print("Configuration:")
print(f"  Input folder: {folder_crops}")
print(f"  Source folder: {source_folder}")
print(f"  Output folder: {output_folder}")
print(f"  {len(subfolders)} subfolders in the input folder.")
print(f"  {len(monitoring_image_names)} monitoring images.")

#%% Find all subfolders in the input folder
df_matches = pd.DataFrame(columns=["image_path", "image_name", "species"])  # Initialize an empty DataFrame to store matches

for subfolder in subfolders:  # Limit to first 2 subfolders for testing
    print(f"\nProcessing subfolder: {subfolder.name}")
    
    # Find all images in the subfolder
    crops = list(subfolder.rglob('*.png'))  # Adjust the pattern if you want to filter specific image formats
    
    print(f"  Found {len(crops)} crop images in subfolder: {subfolder.name}")
    
    for crop in crops:
        OG_image_name = crop.stem[:-11]  # Remove the last 15 characters to get the original image name
        OG_image_path = Path(source_folder).joinpath(Path(OG_image_name + ".png"))
        
        # check if the original image exists in the source folder
        if OG_image_name in monitoring_image_names:
            print(f"  Found matching original image for crop: {crop.name}, OG image name: {OG_image_name}")
            entry = pd.DataFrame({
                "image_path": [OG_image_path],
                "image_name": [OG_image_name],
                "species": [subfolder.name],
            })
            df_matches = pd.concat([df_matches, entry], ignore_index=True)  # Append the entry to the matches DataFrame
        else:
            # print(f"Warning: Original image not found for crop: {crop}, expected OG image name: {OG_image_name}")
            continue  # Skip to the next crop if the original image is not found
    
    print(f"  Total matches found in subfolder {subfolder.name}: {len(df_matches)}")
    
print(f"\n Found a total of {len(df_matches)} images matching the date(s) across all subfolders.")

#%% Copy original images to the output folder, appending the species name to the image name to avoid overwriting images with the same name from different subfolders

for idx, row in df_matches.iterrows():
    source_path = row["image_path"]
    species_name = row["species"]
    image_name = row["image_name"]
    
    # Create a new image name by appending the species name to the original image name
    new_image_name = f"{image_name}_{species_name}.png"
    destination_path = Path(output_folder).joinpath(Path(new_image_name))
    
    # Copy the original image to the output folder with the new name
    shutil.copy(source_path, destination_path)
    print(f"Copied img {idx+1}/{len(df_matches)}: {source_path.name} to {destination_path.name}", end="\r")