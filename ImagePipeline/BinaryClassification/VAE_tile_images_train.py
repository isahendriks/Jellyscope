#%% Cell 1: Import Packages

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from functions import split_image_into_tiles
#%% Cell 2: Define User Parameters
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"
ROOT_DIR_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"   

# Tile parameters
grid_size = 16  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = int(4512/grid_size)  # dimensions of tiles, clean divisions: 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96, 144, 288, 576, 1152, 2304, 4512
AUGMENTATION = False

monitoring_effort = "Kristineberg_260424" #"Kristineberg_251128"

# Folder paths
input_folder = f"{monitoring_effort}\\train\\OG_images"# Folder containing images to split
output_folder = f"{monitoring_effort}\\train\\tiles{grid_size}"# Folder to save the tiles (will be created if it doesn't exist)

input_path = Path(ROOT_DIR_C).joinpath(Path(input_folder))

total_images = len(list(Path(input_path).rglob('*.png')))

# Calculate expected tiles per image
# load first image in the folder to get size (all images are same size)
first_image_path = next(input_path.rglob('*.png'))
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
    
#%% Cell 3: Define input and output paths

# Create output folder
output_paths = [Path(ROOT_DIR_C).joinpath(Path(output_folder)), Path(ROOT_DIR_R).joinpath(Path(output_folder))]
for output_path in output_paths:    
    output_path.mkdir(parents=True, exist_ok=True)


if not input_path.exists():
    raise ValueError(f"Input folder does not exist: {input_folder}")

# select first n images to process
image_files = list(input_path.rglob('*.png')) # get all image files in the input folder
image_files = image_files[:total_images] # select first n images

#%% Cell 4: Define worker function for multiprocessing

def process_single_image(args):
    """Process a single image: split into tiles and save (with optional augmentation)"""
    image_file, output_path_str, grid_size, augmentation = args
    output_path = Path(output_path_str)
    
    try:
        # Read image
        img = cv2.imread(str(image_file))
        if img is None:
            return 0, f"⚠ Could not read {image_file.name}"
        
        # Split image into tiles
        tiles = split_image_into_tiles(img, grid_size=grid_size)
        
        tile_count = 0
        
        # Save each tile
        for tile, row, col in tiles:
            tile_filename = f"{image_file.stem}_r{row}_c{col}.png"
            tile_path = output_path / tile_filename
            cv2.imwrite(str(tile_path), tile)
            tile_count += 1
        
        return tile_count, f"✓ {image_file.name}: {tile_count // (4 if augmentation else 1)} tiles"
        
    except Exception as e:
        return 0, f"✗ Error processing {image_file.name}: {e}"


#%% Cell 5: Process images using multiprocessing

if __name__ == "__main__":
    # Print configuration (only in main process)
    print("\n" + "="*60)
    print("Packages imported successfully yippie!")
    print("="*60)
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
        print(f"  Total tiles with augmentation: {expected_total_tiles * 4}")
    print("\n" + "="*60)
    print("Parameters configured successfully cuckedudledoo!")
    print("="*60)
    print("\n" + "="*60)
    print(f"Succesfully selected first {len(image_files)} images from input folder yihoooo")
    print("="*60)
    print(f"\nFound {len(image_files)} images to process")
    print(f"Expected total tiles: {len(image_files)} images × {tiles_per_image} tiles = {len(image_files) * tiles_per_image} tiles\n")
    print(f"Saving tiles at: {output_paths}\n")
    print(f"Augmentation: {'ON' if AUGMENTATION else 'OFF'}")
        
    # Determine number of worker processes (limited to avoid paging file exhaustion with torch/cuda)
    # Task is I/O-bound, not CPU-bound, so fewer workers is faster and uses less memory
    num_workers = 16
    print(f"Using {num_workers} parallel workers\n")
        
    # Prepare arguments for each image
    process_args = [
        (image_file, str(output_path), grid_size, AUGMENTATION) 
        for image_file in image_files
        for output_path in output_paths
    ]
        
    total_tiles = 0
        
    ### Process images in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, process_args),
            total=len(image_files),
            desc="Processing images"
        ))
        
    # Collect results and print summary
    successful = 0
    for tile_count, message in results:
        total_tiles += tile_count
        if tile_count > 0:
            successful += 1
        print(message)
        
    print(f"\n✓ Tiling complete!")
    print(f"  Images processed:   {successful}/{len(image_files)}")
    print(f"  Total tiles:        {total_tiles} (expected: {len(image_files) * tiles_per_image})")
    print(f"  Tiles per image:    {tiles_per_image}")
    print(f"  Output folder(s):      {output_paths}")   
        
    print("\n" + "="*60)
    print("All images processed successfully yippie!")
    print("="*60)
# %%
