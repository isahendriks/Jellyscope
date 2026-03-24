#%% Cell 1: Import Packages

import os
import shutil
from turtle import pd
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd

from functions import split_image_into_tiles

print("\n" + "="*60)
print("Packages imported successfully yippie!")
print("="*60)


def parse_obs_indices(user_input, max_tile_idx):
    text = user_input.strip()
    if not text:
        return set()

    text = text.replace("，", ",").replace(";", ",").replace("–", "-").replace("—", "-")
    parts = [p for p in re.split(r"[\s,]+", text) if p]

    parsed = set()
    for part in parts:
        if "-" in part:
            range_parts = part.split("-", 1)
            if len(range_parts) != 2 or not range_parts[0] or not range_parts[1]:
                raise ValueError(part)

            start = int(range_parts[0])
            end = int(range_parts[1])
            if start <= end:
                parsed.update(range(start, end + 1))
            else:
                parsed.update(range(end, start + 1))
        else:
            parsed.add(int(part))

    invalid = {idx for idx in parsed if idx < 0 or idx > max_tile_idx}
    if invalid:
        print(f"⚠ Invalid indices: {sorted(invalid)}, skipping them")
        parsed -= invalid

    return parsed


def center_crop_to_tile_grid(image, tile_size):
    h, w = image.shape[:2]
    n_rows = h // tile_size
    n_cols = w // tile_size

    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"Image too small for tile size {tile_size}: got {w}x{h}"
        )

    crop_h = n_rows * tile_size
    crop_w = n_cols * tile_size
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2

    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
    return cropped, n_rows, n_cols, x0, y0


#%% Cell 2: Define User Parameters


# Tile parameters
grid_size = 32
tile_size = int(4512/grid_size)  # dimensions of tiles, clean divisions: 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96, 144, 288, 576, 1152, 2304, 4512
monitoring_effort = 'Kristineberg_251128'  # e.g. "Kristineberg_251128" or "Kristineberg_251129"
INPUT_NO_OBS_INDICES = True  # True: entered indices are no_obs, False: entered indices are obs

# Folder paths
input_folder = "R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data\\Binary_classifier\\" + monitoring_effort + "\\test\\cropped_images"  # Folder containing images to split
output_folder = "R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data\\Binary_classifier\\" + monitoring_effort + "\\test\\tiles" + str(grid_size)  # Folder to save the tiles (will be created if it doesn't exist)

all_image_files = list(Path(input_folder).rglob('*.*')) # get all image files in the input folder
species_names = [str(image.stem)[35:] for image in all_image_files]

# create dataframe with image names and species names
df_images = pd.DataFrame({
    "image_name": [image.name for image in all_image_files],
    "species": species_names
})

# Display configuration
print(f"Configuration:")
print(f"  Input folder:  {input_folder}")
print(f"  Output folder: {output_folder}")
print(f"  Tile size:     {tile_size}x{tile_size}")
print(f"  Total images:  {len(df_images)}")
print(f"  Species names: {df_images['species'].unique()}")
print(f"  Grid mode:     centered crop per image")
print(f"  Input mode:    {'no_obs indices' if INPUT_NO_OBS_INDICES else 'obs indices'}")

print("\n" + "="*60)
print("Parameters configured successfully cuckedudledoo!")
print("="*60)

#%% Cell 3: Define input and output paths

# Create output folder
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)

# Get input folder
input_path = Path(input_folder)

if not input_path.exists():
    raise ValueError(f"Input folder does not exist: {input_folder}")

# Image extensions to look for
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
# Image range to process
start_image_idx = 12  # Currently done: 30 tiles
end_image_idx = len(df_images)  # Set to total number of images in the input folder

species = df_images['species'].unique()[15]
image_files = df_images[df_images['species'] == species]['image_name'].tolist()
image_files_species = [Path(input_folder) / Path(image_name) for image_name in image_files]
image_files = image_files_species[start_image_idx:end_image_idx] # select images from start to end index

# select n images from the specified range
# all_image_files = list(input_path.rglob('*.*')) # get all image files in the input folder
# image_files = all_image_files[start_image_idx:end_image_idx] # select images from start to end index

print("Folder to process: ", input_folder)
print("Save location: ", output_folder)

# print some info about the selected images
print("\n" + "="*60)
print(f"Processing images of {species}: {len(image_files)} images")
print("="*60)

#%% Cell 4: Interactive tile selection - display with grid and save to obs/noobs folders
# Process images
if not image_files:
    print("No images found!")
else:
    print("\n" + "="*60)
    print("Interactive Tile Selection - Phase 1: Review Images")
    print("="*60)
    print(f"You will select tile indices as {'NO OBS' if INPUT_NO_OBS_INDICES else 'OBS'}")
    print(f"Total images to review: {len(image_files)}\n")
    
    # Dictionary to store selections: image_file -> set of observation indices  
    image_selections = {}
    
    # Enable interactive mode for non-blocking plots
    plt.ion()
    
    # Phase 1: Display all images and collect indices
    for img_idx, image_file in enumerate(image_files):
        try:
            print(f"\nImage {img_idx+1}/{len(image_files)}")
            
            # Read and convert image
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Could not read {image_file.name}")
                continue

            img_cropped, tiles_per_row_img, tiles_per_col_img, crop_x0, crop_y0 = center_crop_to_tile_grid(img, tile_size)
            img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            tiles_per_image = tiles_per_row_img * tiles_per_col_img
            print(
                f"Original: {img.shape[1]}x{img.shape[0]} | "
                f"Cropped: {img_cropped.shape[1]}x{img_cropped.shape[0]} | "
                f"Tiles: {tiles_per_row_img}x{tiles_per_col_img}={tiles_per_image}"
            )
            
            # Create a new figure for each image
            fig, ax = plt.subplots(1, 1, figsize=(25, 25))
            ax.imshow(img_rgb)
            ax.set_title(f"{image_file.name}\nEnter comma-separated indices for OBSERVATION tiles", fontsize=16)
            
            # Draw grid and add tile indices
            tile_index = 0
            
            for row in range(tiles_per_row_img):
                for col in range(tiles_per_col_img):
                    print(f'preparing tile for display {tile_index} (row {row}, col {col}) ...', end="\r")
                    y_start = row * tile_size
                    x_start = col * tile_size
                    
                    # Draw rectangle
                    rect = patches.Rectangle((x_start, y_start), tile_size, tile_size, 
                                             linewidth=2, edgecolor='blue', facecolor='none', alpha=0.5)
                    ax.add_patch(rect)
                    
                    # Add tile index text
                    ax.text(x_start + 10, y_start + 30, str(tile_index), 
                           fontsize=16, color='white', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
                    
                    tile_index += 1
            
            ax.set_xlim(0, img_rgb.shape[1])
            ax.set_ylim(img_rgb.shape[0], 0)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)  # Give the plot time to render
            
            # Get user input
            print(f"Image: {image_file.name}")
            print(f"Total tiles in this image: {tile_index}")
            input_label = "NON-OBSERVATION" if INPUT_NO_OBS_INDICES else "OBSERVATION"
            user_input = input(
                f"Enter indices of {input_label} tiles (e.g., '0,5,10' or '530-555') "
                "or 'skip' to skip this image entirely, or press Enter for none selected: "
            )
            
            # Check if user wants to skip this image entirely
            if user_input.strip().lower() == 'skip':
                print(f"⏭ Skipping {image_file.name} - no tiles will be saved")
                plt.close(fig)
                continue
            
            selected_indices = set()
            if user_input.strip():
                try:
                    max_tile_idx = tile_index - 1
                    selected_indices = parse_obs_indices(user_input, max_tile_idx)
                except ValueError as e:
                    print(f"⚠ Invalid input token: '{e}'. Skipping this image")
                    plt.close(fig)
                    continue
            else:
                if INPUT_NO_OBS_INDICES:
                    print(f"→ No no_obs selected; all {tile_index} tiles will be OBS")
                else:
                    print(f"→ No obs selected; all {tile_index} tiles will be NO OBS")

            if INPUT_NO_OBS_INDICES:
                noobs_indices = selected_indices
                obs_indices = set(range(tile_index)) - noobs_indices
            else:
                obs_indices = selected_indices
                noobs_indices = set(range(tile_index)) - obs_indices
            
            # Phase 1.5: Immediate confirmation with visual overlay for this image
            confirmed = False
            while not confirmed:
                try:
                    # Read and convert image again for overlay
                    img = cv2.imread(str(image_file))
                    if img is None:
                        print(f"Could not read {image_file.name}")
                        break

                    img_cropped, tiles_per_row_img, tiles_per_col_img, _, _ = center_crop_to_tile_grid(img, tile_size)
                    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                    
                    # Create a copy for overlay
                    img_with_overlay = img_rgb.copy()
                    
                    # Draw green overlay on selected tiles
                    tile_idx = 0
                    for row in range(tiles_per_row_img):
                        for col in range(tiles_per_col_img):
                            y_start = row * tile_size
                            y_end = y_start + tile_size
                            x_start = col * tile_size
                            x_end = x_start + tile_size
                            
                            if INPUT_NO_OBS_INDICES and tile_idx in noobs_indices:
                                img_with_overlay[y_start:y_end, x_start:x_end] = (
                                    0.7 * img_with_overlay[y_start:y_end, x_start:x_end].astype(float) +
                                    0.3 * np.array([255, 0, 0]).astype(float)
                                ).astype(np.uint8)
                            elif (not INPUT_NO_OBS_INDICES) and tile_idx in obs_indices:
                                img_with_overlay[y_start:y_end, x_start:x_end] = (
                                    0.7 * img_with_overlay[y_start:y_end, x_start:x_end].astype(float) +
                                    0.3 * np.array([0, 255, 0]).astype(float)
                                ).astype(np.uint8)
                            
                            tile_idx += 1
                    
                    # Display the image with overlay
                    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(25, 25))
                    ax_overlay.imshow(img_with_overlay)
                    if INPUT_NO_OBS_INDICES:
                        overlay_text = f"{len(noobs_indices)} NO OBS tiles selected (red overlay)"
                    else:
                        overlay_text = f"{len(obs_indices)} OBS tiles selected (green overlay)"
                    ax_overlay.set_title(f"{image_file.name}\n{overlay_text}\nPress ENTER to confirm or type 'redo' to re-enter", fontsize=16)
                    ax_overlay.axis('off')
                    plt.tight_layout()
                    plt.show(block=False)
                    plt.pause(0.5)
                    
                    # Get confirmation
                    user_response = input(f"Confirm selection for {image_file.name}? (press ENTER to confirm, or type 'redo'): ").strip().lower()
                    plt.close(fig_overlay)
                    
                    if user_response == '' or user_response == 'confirm':
                        # Selection confirmed
                        print(f"✓ Confirmed: {len(obs_indices)} obs, {len(noobs_indices)} no_obs for {image_file.name}")
                        confirmed = True
                    elif user_response == 'redo':
                        # Allow user to re-enter indices
                        print(f"Re-entering indices for {image_file.name}")
                        prev_indices = sorted(selected_indices)
                        if prev_indices:
                            preview = prev_indices[:30]
                            preview_text = ",".join(str(i) for i in preview)
                            if len(prev_indices) > 30:
                                preview_text += f", ... (total {len(prev_indices)})"
                            print(f"Previous indices: {preview_text}")
                        else:
                            print("Previous indices: none")
                        input_label = "NON-OBSERVATION" if INPUT_NO_OBS_INDICES else "OBSERVATION"
                        user_input = input(
                            f"Enter indices of {input_label} tiles (e.g., '0,5,10' or '530-555') "
                            "or press Enter for none selected: "
                        )
                        
                        selected_indices.clear()
                        if user_input.strip():
                            try:
                                max_tile_idx = tile_index - 1
                                selected_indices = parse_obs_indices(user_input, max_tile_idx)
                            except ValueError as e:
                                print(f"⚠ Invalid input token: '{e}'")

                        if INPUT_NO_OBS_INDICES:
                            noobs_indices = selected_indices
                            obs_indices = set(range(tile_index)) - noobs_indices
                        else:
                            obs_indices = selected_indices
                            noobs_indices = set(range(tile_index)) - obs_indices
                    else:
                        print("Unrecognized input. Please press ENTER or type 'redo'")
                        
                except Exception as e:
                    print(f"✗ Error confirming {image_file.name}: {e}")
                    plt.close('all')
                    confirmed = True
            
            # Store the final selection
            image_selections[image_file] = {
                "obs_indices": set(obs_indices),
                "noobs_indices": set(noobs_indices),
                "total_tiles": tile_index,
            }
            print(f"✓ Final selection stored for {image_file.name}")
            plt.close(fig)
            
        except Exception as e:
            print(f"✗ Error processing {image_file.name}: {e}")
            plt.close('all')
            continue
    
    # Disable interactive mode after all images are reviewed
    plt.ioff()
    
    # Phase 2: Save tiles based on selections
    print("\n" + "="*60)
    print("Phase 2: Saving Tiles")
    print("="*60)
    
    # Create output folders
    folder_obs = Path(output_folder, "obs")
    folder_noobs = Path(output_folder, "no_obs")
    folder_obs.mkdir(parents=True, exist_ok=True)
    folder_noobs.mkdir(parents=True, exist_ok=True)
    
    # Save indices to text file
    indices_file_path = Path(output_folder) / "tile_indices.txt"
    with open(indices_file_path, 'w') as f:
        f.write("Tile Selection Record\n")
        f.write("="*60 + "\n")
        f.write(f"Tile size: {tile_size}x{tile_size}\n")
        f.write(f"Total images processed: {len(image_selections)}\n")
        f.write("="*60 + "\n\n")
        
        for image_file, selection in image_selections.items():
            obs_indices = selection["obs_indices"]
            noobs_indices = selection["noobs_indices"]
            f.write(f"Image: {image_file.name}\n")
            f.write(f"Observation tile indices: {sorted(obs_indices) if obs_indices else 'None'}\n")
            f.write(f"Non-observation tile indices: {sorted(noobs_indices) if noobs_indices else 'None'}\n")
            f.write(f"Number of observation tiles: {len(obs_indices)}\n")
            f.write(f"Number of non-observation tiles: {len(noobs_indices)}\n")
            f.write("-"*60 + "\n")
    
    print(f"Saved tile indices to: {indices_file_path}\n")
    
    # Process and save tiles
    total_tiles_obs = 0
    total_tiles_noobs = 0
    
    for image_file, selection in tqdm(image_selections.items(), desc="Processing images"):
        try:
            obs_indices = selection["obs_indices"]
            # Read image
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Could not read {image_file.name}")
                continue

            img_cropped, tiles_per_row_img, tiles_per_col_img, crop_x0, crop_y0 = center_crop_to_tile_grid(img, tile_size)
            
            # Extract and save tiles
            tile_idx = 0
            for row in range(tiles_per_row_img):
                for col in range(tiles_per_col_img):
                    # Extract the tile from the centered-cropped image
                    y_start = row * tile_size
                    y_end = y_start + tile_size
                    x_start = col * tile_size
                    x_end = x_start + tile_size
                    tile = img_cropped[y_start:y_end, x_start:x_end]
                    
                    tile_filename = f"{image_file.stem}_r{row}_c{col}_i{tile_idx}.png"
                    
                    # Save to appropriate folder
                    if tile_idx in obs_indices:
                        cv2.imwrite(str(folder_obs / tile_filename), tile)
                        total_tiles_obs += 1
                    else:
                        cv2.imwrite(str(folder_noobs / tile_filename), tile)
                        total_tiles_noobs += 1
                    
                    tile_idx += 1
            
            print(
                f"✓ {image_file.name}: {len(obs_indices)} obs, {tile_idx - len(obs_indices)} no_obs "
                f"(crop offset x={crop_x0}, y={crop_y0})"
            )
            
        except Exception as e:
            print(f"✗ Error saving tiles for {image_file.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print(f"Tile selection complete!")
    print(f"Images processed: {len(image_selections)}")
    print(f"Total observation tiles: {total_tiles_obs}")
    print(f"Total non-observation tiles: {total_tiles_noobs}")
    print(f"Observation tiles saved to: {folder_obs}")
    print(f"Non-observation tiles saved to: {folder_noobs}")
    print(f"Tile indices saved to: {indices_file_path}")
    print("="*60)

# %%

