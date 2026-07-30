#%% Cell 1: Import Packages

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import split_image_into_tiles
#%% Cell 2: Define User Parameters
ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data new\Binary_classifier"
ROOT_DIR_C = r"C:\Users\IsaH\Documents\Jellyscope\Training data new\Binary_classifier"   

# Tile parameters
grid_size = 16  # number of tiles along one side (e.g. 6 means 6x6=36 tiles per image)
tile_size = int(4512/grid_size)  # dimensions of tiles, clean divisions: 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96, 144, 288, 576, 1152, 2304, 4512
AUGMENTATION = False

# Offset cropping parameters
# List of normalized offsets [0.0, 0.2, 0.4, 0.6, 0.8] creates crops at 0%, 20%, 40%, 60%, 80% offset
# Set to [] or [0.0] to disable offset cropping (single crop per tile)
OFFSET_NORMALIZED = [0.0, 0.2, 0.4, 0.6, 0.8]

monitoring_effort = "Kristineberg_260729" #"Kristineberg_260424" 

# Folder paths
input_folder = f"{monitoring_effort}\\train_scorer\\OG_images"# Folder containing images to split
output_folder = f"{monitoring_effort}\\train_scorer\\tiles{grid_size}_offsets{len(OFFSET_NORMALIZED)}\\no_obs"# Folder to save the tiles (will be created if it doesn't exist)

input_path = Path(ROOT_DIR_R).joinpath(Path(input_folder))
total_images = len(list(Path(input_path).rglob('*.png')))

# Calculate expected tiles per image
# load first image in the folder to get size (all images are same size)
first_image_path = next(input_path.rglob('*.png'))
first_image = cv2.imread(str(first_image_path), cv2.IMREAD_GRAYSCALE)
if first_image is not None:
    if first_image.shape[0] == first_image.shape[1]:
        image_size = first_image.shape[0]  # Assuming square images
    else:
        raise ValueError(f"Unexpected image shape: {first_image.shape}. Expected square images.")
else:
    raise ValueError(f"Could not read image: {first_image_path}")


# find total number of images in the input folder
expected_total_tiles = total_images * grid_size**2 + (total_images * (grid_size-1)**2 * (len(OFFSET_NORMALIZED) - 1))  # account for offset cropping

# whether to apply data augmentation (e.g. rotations, flips) to the tiles (not implemented yet)
if AUGMENTATION:
    output_folder += "_aug"
    
#%% Cell 3: Define input and output paths

# Create output folder
output_paths = [Path(ROOT_DIR_R).joinpath(Path(output_folder))]
for output_path in output_paths:    
    output_path.mkdir(parents=True, exist_ok=True)


if not input_path.exists():
    raise ValueError(f"Input folder does not exist: {input_folder}")

# select first n images to process
image_files = list(input_path.rglob('*.png')) # get all image files in the input folder
image_files = image_files[:total_images] # select first n images

#%% Cell 4: Define worker function for multiprocessing

def process_single_image(args):
    """Process a single image: split into tiles with optional grid offset and save to multiple paths"""
    image_file, output_paths_list, grid_size, offset_normalized_list, tile_size_px = args
    
    try:
        # Read image
        img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, f"⚠ Could not read {image_file.name}"
        
        tile_count = 0
        
        # Process each offset
        for offset in offset_normalized_list:
            # Calculate offset in pixels
            offset_px = int(tile_size_px * offset)
            
            # Extract tiles with offset grid
            for row in range(grid_size):
                for col in range(grid_size):
                    # Calculate tile boundaries with offset
                    y_start = row * tile_size_px + offset_px
                    x_start = col * tile_size_px + offset_px
                    y_end = y_start + tile_size_px
                    x_end = x_start + tile_size_px
                    
                    # Skip if tile goes beyond image boundary
                    if y_end > image_size or x_end > image_size:
                        continue
                    
                    # Extract tile
                    tile = img[y_start:y_end, x_start:x_end]
                    
                    # Create filename with offset suffix (e.g. _o20 for 0.2 offset)
                    tile_filename = f"{image_file.stem}_r{row}_c{col}_o{int(offset*100)}.png"
                    
                    for output_path_str in output_paths_list:
                        output_path = Path(output_path_str)
                        tile_path = output_path / tile_filename
                        cv2.imwrite(str(tile_path), tile)
                    
                    tile_count += 1
        
        return tile_count, f"✓ {image_file.name}: {tile_count} tiles"
        
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
    print(f"  Offset cropping: {len(OFFSET_NORMALIZED)} offsets: ({OFFSET_NORMALIZED})")
    print(f"  Tiles/image:   {grid_size}x{grid_size} = {grid_size**2} tiles")
    print(f"  Total images:  {total_images}")
    print(f"  Expected tiles: {expected_total_tiles}")
    print("\n" + "="*60)
        
    # Determine number of worker processes (limited to avoid paging file exhaustion with torch/cuda)
    # Task is I/O-bound, not CPU-bound, so fewer workers is faster and uses less memory
    num_workers = 16
    print(f"Using {num_workers} parallel workers\n")
        
    # Prepare arguments for each image (one task per image, save to all output paths)
    output_paths_str = [str(p) for p in output_paths]
    process_args = [
        (image_file, output_paths_str, grid_size, OFFSET_NORMALIZED, tile_size) 
        for image_file in image_files
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
    tiles_per_image_count = (grid_size ** 2 if 0.0 in OFFSET_NORMALIZED else 0) + ((grid_size - 1) ** 2 * (len(OFFSET_NORMALIZED) - 1) if len(OFFSET_NORMALIZED) > 1 else 0)
    expected_total = len(image_files) * tiles_per_image_count if tiles_per_image_count > 0 else "N/A"
    print(f"  Total tiles:        {total_tiles} (expected: {expected_total})")
    print(f"  Output folder(s):      {output_paths}")
        
    print("\n" + "="*60)
    print("All images processed successfully yippie!")
    print("="*60)
# %%
