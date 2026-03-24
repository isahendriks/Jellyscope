#%% Cell 1: Import Packages

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from functions import split_image_into_tiles

print("\n" + "="*60)
print("Packages imported successfully yippie!")
print("="*60)
#%% Cell 3: Define User Parameters


# Tile parameters
grid_size = 32  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = int(4512/grid_size)  # dimensions of tiles, clean divisions: 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96, 144, 288, 576, 1152, 2304, 4512
AUGMENTATION = False

# Folder paths
input_folder = "R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data\\Binary_classifier\\Kristineberg_251128\\OG_images\\no_obs"# Folder containing images to split
output_folder = "R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data\\Binary_classifier\\Kristineberg_251128\\tiles32"+str(grid_size)+"\\no_obs"# Folder to save the tiles (will be created if it doesn't exist)

total_images = len(list(Path(input_folder).rglob('*.*')))

# Calculate expected tiles per image
# load first image in the folder to get size (all images are same size)
first_image_path = next(Path(input_folder).rglob('*.*'))
first_image = cv2.imread(str(first_image_path))
image_size = first_image.shape[0]  # Assuming square images
tiles_per_row = image_size // tile_size
tiles_per_col = image_size // tile_size
tiles_per_image = tiles_per_row * tiles_per_col

# find total number of images in the input folder
expected_total_tiles = total_images * tiles_per_image

# whether to apply data augmentation (e.g. rotations, flips) to the tiles (not implemented yet)
if AUGMENTATION:
    output_folder += "_aug"
    
# Display configuration
print(f"Configuration:")
print(f"  Input folder:  {input_folder}")
print(f"  Output folder: {output_folder}")
print(f"  Augmentation:   {'ON' if AUGMENTATION else 'OFF'}")
print(f"  Image size:    {image_size}x{image_size} (all images)")
print(f"  Tile size:     {tile_size}x{tile_size}")
print(f"  Tiles/image:   {tiles_per_row}x{tiles_per_col} = {tiles_per_image} tiles")
print(f"  Total images:  {total_images}")
print(f"  Expected tiles: {expected_total_tiles}")
if AUGMENTATION:
    print(f"  Total tiles with augmentation: {expected_total_tiles * 4 if AUGMENTATION else expected_total_tiles}")

print("\n" + "="*60)
print("Parameters configured successfully cuckedudledoo!")
print("="*60)

#%% Cell 4: Define input and output paths

# Create output folder
output_path = Path(output_folder)
output_path.mkdir(parents=True, exist_ok=True)

# Get input folder
input_path = Path(input_folder)

if not input_path.exists():
    raise ValueError(f"Input folder does not exist: {input_folder}")

# Image extensions to look for
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# select first n images to process
image_files = list(input_path.rglob('*.*')) # get all image files in the input folder
image_files = image_files[:total_images] # select first n images

# print some info about the selected images
print("\n" + "="*60)
print(f"Succesfully selected first {len(image_files)} images from input folder yihoooo")
print("="*60)

#%% Cell 5: Process images and save tiles
if not image_files:
    print("No images found in input folder!")
else:
    print(f"\nFound {len(image_files)} images to process")
    print(f"Expected total tiles: {len(image_files)} images × {tiles_per_image} tiles = {len(image_files) * tiles_per_image} tiles\n")
    print(f"saving tiles at: {output_path}\n")
    print(f"Augmentation: {'ON' if AUGMENTATION else 'OFF'}")
    
    total_tiles = 0
    
    # Process each image
    for img_idx, image_file in enumerate(image_files):
        try:
            # Read image
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"⚠ Could not read {image_file.name}, skipping")
                continue

            # Split image into tiles (all images are 4512x4512)
            print(f"Splitting image {img_idx + 1}/{total_images} ... ", end="")
            tiles = split_image_into_tiles(img, grid_size=grid_size)  # list of (tile, row, col)
            print(f"\nSplitting image {img_idx + 1}/{total_images} ... done! ", end="\r")
            
            if AUGMENTATION:
                print(f"\nAugmenting tiles ... ", end="")
                for tile in tiles:
                    
                    tile_to_aug, col, row = tile
                    print(f"Augmenting tile r{row}/{grid_size} c{col}/{grid_size} ... ", end="\r")
                    tile90 = cv2.rotate(tile_to_aug, cv2.ROTATE_90_CLOCKWISE)
                    tile180 = cv2.rotate(tile_to_aug, cv2.ROTATE_180)
                    tile270 = cv2.rotate(tile_to_aug, cv2.ROTATE_90_COUNTERCLOCKWISE)        
                print(f"Augmenting tiles ... done!", end="\r")
                        
            # Save each tile
            print(f"\nSaving tiles for image {img_idx + 1}/{total_images} ... ", end="\r")    
            tile_counter = 0
            for tile, row, col in tiles:
                print(f"Saving tile r{row}/{grid_size} c{col}/{grid_size} ... ", end="\r")
                tile_counter += 1
                tile_filename = f"{image_file.stem}_r{row}_c{col}_i{row*col}.png"
                tile_path = output_path / tile_filename
                cv2.imwrite(str(tile_path), tile)
                if AUGMENTATION:
                    # save augmented tiles
                    tile90_path = str(tile_path)[:-4] + "_rot90.png"
                    tile180_path = str(tile_path)[:-4] + "_rot180.png"
                    tile270_path = str(tile_path)[:-4] + "_rot270.png"
                    cv2.imwrite(tile90_path, tile90)
                    cv2.imwrite(tile180_path, tile180)
                    cv2.imwrite(tile270_path, tile270)
                    total_tiles += 3  # count augmented tiles
                total_tiles += 1
            print(f"Saving tiles for image {img_idx + 1}/{total_images} ... done! ", end="\r")    

            print()
            
        except Exception as e:
            print(f"\n Error processing {image_file.name}: {e}")
            continue
    

    print(f"✓ Tiling complete!")
    print(f"  Images processed:   {img_idx+1}/{len(image_files)}")
    print(f"  Total tiles:        {total_tiles} (expected: {len(image_files) * tiles_per_image})")
    print(f"  Tiles per image:    {tiles_per_image}")
    print(f"  Output folder:      {output_path}")   
    
    print("\n" + "="*60)
    print("All images processed successfully yippie!")
    print("="*60)

#%% Cell 6: Augmentation

