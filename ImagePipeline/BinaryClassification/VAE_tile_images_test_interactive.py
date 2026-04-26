#%% Cell 1: Import Packages
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

print("\n" + "="*60)
print("Interactive Tile Labeler - Simple Grid-Based")
print("="*60)

#%% Cell 2: Define User Parameters

grid_size = 16
tile_size = int(4512 / grid_size)
monitoring_effort = 'Kristineberg_260424'  #

ROOT_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"
ROOT_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"

input_folder = r"{}\test\OG_images".format(monitoring_effort)
output_folder = r"{}\test\tiles{}".format(monitoring_effort, grid_size)

input_path = Path(ROOT_C).joinpath(Path(input_folder))
output_path = Path(ROOT_C).joinpath(Path(output_folder))

all_image_files = sorted(list(Path(input_path).rglob('*.png')))
species_names = [str(img_name)[126:-4] for img_name in all_image_files]  # Extract species name from filename
species_names = np.unique(species_names)  # Get unique species names

species_names_to_exclude =  ['dentritus', 'dentritus_glo', 'filament', 'macro_filament', 'filament_glo'] # Set this to a subset of species if you want to filter
species_names_to_include = [str(name) for name in species_names if name not in species_names_to_exclude] 

# exclude images that do not match the included species names
all_image_files_filtered = []
for img in all_image_files:
    species_name = str(img)[126:-4]  # Extract species name from filename
    if species_name in species_names_to_include:
        all_image_files_filtered.append(img)

print(f"Excluded species names: {species_names_to_exclude}, corresponding to {len(all_image_files) - len(all_image_files_filtered)} images")
print(f"Images after filtering: {len(all_image_files_filtered)}")

# Check if images are already in output folder, exclude those images.
obs_folder = output_path / Path('obs')
no_obs_folder = output_path / Path('no_obs')

processed_images = set()
for patch in obs_folder.glob('*_r0_c0.png'):
    processed_images.add(patch.stem.rsplit('_r', 1)[0])
for patch in no_obs_folder.glob('*_r0_c0.png'):
    processed_images.add(patch.stem.rsplit('_r', 1)[0])

## Filter out images that are already processed
all_image_files_unprocessed = []
for img in all_image_files_filtered:
    if img.stem not in processed_images:
        all_image_files_unprocessed.append(img)

print(f"Configuration:")
print(f"  Input folder:  {input_path}")
print(f"  Output folder: {output_path}")
print(f"  Tile size:     {tile_size}x{tile_size}")
print(f"  Grid size:     {grid_size}x{grid_size}")
print(f"  Total images:  {len(all_image_files)}")
print(f"  Images after filtering by species: {len(all_image_files_filtered)}")
print(f"  Already processed images: {len(processed_images)}")
print(f"  Remaining images to process: {len(all_image_files_unprocessed)}")
print("\n" + "="*60)
print("Parameters configured successfully!")
print("="*60)

# chose image files to process (all or only unprocessed)
image_files_to_process = all_image_files_unprocessed

#%% Cell 3: Create output folders
output_path.mkdir(parents=True, exist_ok=True)
(output_path / Path("obs")).mkdir(exist_ok=True)
(output_path / Path("no_obs")).mkdir(exist_ok=True)

#%% Cell 4: Simple Grid-Based Tile Labeler Class

class GridTileLabeler:
    def __init__(self, image_path, grid_size=32, tile_size=141, fig=None, ax=None, img_idx=0, total_images=1):
        self.image_path = image_path
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.img_idx = img_idx
        self.total_images = total_images
        
        # Read image
        self.img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        
        self.img_h, self.img_w = self.img.shape[:2]
        self.img_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        
        # Store all tile data for the full grid
        self.all_tiles = {}  # {linear_idx: {'row': row, 'col': col, 'x': x, 'y': y, 'data': tile_data}}
        self._extract_all_tiles()
        
        # Tracking selected obs tiles
        self.obs_indices = set()  # Linear indices of tiles marked as obs
        self.undo_stack = []  # Stack of previous selections for undo
        
        # Reuse figure or create new one
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 14))
        else:
            self.fig, self.ax = fig, ax
        
        # Add margin at top to prevent title overlap
        self.fig.subplots_adjust(top=0.93)
        
        self.tile_rects = {}  # {linear_idx: rect_patch}
        self.image_confirmed = False  # Flag to track when user presses 'c'
        self.user_requested_exit = False  # Flag to track when user presses 'e'
        self.show_grid()
        
        # Connect event handlers
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def _extract_all_tiles(self):
        """Extract all tiles from the full grid."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                tile_x = col * self.tile_size
                tile_y = row * self.tile_size
                tile_x_end = tile_x + self.tile_size
                tile_y_end = tile_y + self.tile_size
                
                linear_idx = row * self.grid_size + col
                tile_data = self.img[tile_y:tile_y_end, tile_x:tile_x_end]
                
                self.all_tiles[linear_idx] = {
                    'row': row,
                    'col': col,
                    'x': tile_x,
                    'y': tile_y,
                    'data': tile_data
                }
        
    def show_grid(self):
        """Display image with grid and labeled tiles."""
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.tile_rects = {}
        
        # Draw all tiles with dark blue borders and green hue for selected tiles
        for linear_idx, tile_info in self.all_tiles.items():
            row, col = tile_info['row'], tile_info['col']
            x, y = tile_info['x'], tile_info['y']
            
            # Dark blue border for all tiles
            edge_color = '#00008B'
            
            # Fill color based on selection - slight green hue for obs tiles
            if linear_idx in self.obs_indices:
                facecolor = '#90EE90'  # Light green
                alpha_fill = 0.1
            else:
                facecolor = 'none'
                alpha_fill = 0
            
            rect = patches.Rectangle((x, y), self.tile_size, self.tile_size,
                                    linewidth=1, edgecolor=edge_color, facecolor=facecolor, 
                                    alpha=0.7, zorder=2)
            self.ax.add_patch(rect)
            self.tile_rects[linear_idx] = rect
        
        # Extract species name from filename (last part after final underscore)
        species_name = self.image_path.stem.split('_')[-1]
        
        # Build title with better formatting
        title = f"\n{species_name}\n"
        title += f"Image {self.img_idx + 1} / {self.total_images}\n"
        title += f"OBS (green): {len(self.obs_indices)} | NO_OBS (red): {self.grid_size * self.grid_size - len(self.obs_indices)}\n\n"
        title += "LEFT-CLICK: mark/unmark | 'r': reset | 'u': undo | 'c': save & next | 'e': exit"
        
        self.ax.set_title(title, fontsize=9, fontweight='bold', pad=20)
        self.fig.canvas.draw_idle()
    
    def get_tile_at_click(self, x, y):
        """Find which tile was clicked based on coordinates."""
        for linear_idx, tile_info in self.all_tiles.items():
            tx, ty = tile_info['x'], tile_info['y']
            if tx <= x < tx + self.tile_size and ty <= y < ty + self.tile_size:
                return linear_idx
        return None
    
    def on_click(self, event):
        """Handle mouse clicks on tiles."""
        if event.xdata is None or event.button != 1:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        linear_idx = self.get_tile_at_click(x, y)
        
        if linear_idx is None:
            return
        
        # Save current state to undo stack before making changes
        self.undo_stack.append(self.obs_indices.copy())
        
        # Toggle tile selection
        if linear_idx in self.obs_indices:
            self.obs_indices.remove(linear_idx)
            print(f"Unmarked tile at row {self.all_tiles[linear_idx]['row']}, col {self.all_tiles[linear_idx]['col']}")
        else:
            self.obs_indices.add(linear_idx)
            print(f"Marked tile at row {self.all_tiles[linear_idx]['row']}, col {self.all_tiles[linear_idx]['col']}")
        
        self.show_grid()
    
    def on_key(self, event):
        """Handle keyboard input."""
        if event.key == 'c':
            # Confirm and save
            print("\n✓ Confirming image...")
            self.save_image()
            self.image_confirmed = True
            # Update title to show saved message with species name
            species_name = self.image_path.stem.split('_')[-1]
            self.ax.set_title(f"✓ Saved - {species_name} - Next image coming...", fontsize=10, fontweight='bold', pad=20, color='green')
            self.fig.canvas.draw_idle()
        
        elif event.key == 'r':
            # Reset all tiles
            self.undo_stack.append(self.obs_indices.copy())
            self.obs_indices = set()
            print("✓ Reset all tiles")
            self.show_grid()
        
        elif event.key == 'u':
            # Undo last change
            if self.undo_stack:
                self.obs_indices = self.undo_stack.pop()
                print("✓ Undone last change")
                self.show_grid()
            else:
                print("⚠ Nothing to undo")
        
        elif event.key == 'e':
            # Exit program and save all work
            print(f"\n✗ User exited at image index: {self.img_idx}")
            
            # Save current image's tiles before exiting
            if self.obs_indices:
                print("Saving current image's tiles...")
                self.save_image()
            
            self.save_progress(self.img_idx)
            
            # Signal to save crops and exit
            self.user_requested_exit = True
            self.image_confirmed = True  # Set this so main loop can continue and detect exit
            plt.close('all')
    
    def save_image(self):
        """Store all tiles (obs and no_obs) to be saved later."""
        obs_count = len(self.obs_indices)
        total_tiles = len(self.all_tiles)
        
        # Store all tiles in memory (to be saved at the end)
        for linear_idx, tile_info in self.all_tiles.items():
            row = tile_info['row']
            col = tile_info['col']
            tile_data = tile_info['data']
            filename = f"{self.image_path.stem}_r{row}_c{col}.png"
            
            # Determine label
            label = 'obs' if linear_idx in self.obs_indices else 'no_obs'
            
            # Add to global list
            crops_to_save.append({
                'filename': filename,
                'row': row,
                'col': col,
                'data': tile_data,
                'label': label
            })
        
        no_obs_count = total_tiles - obs_count
        print(f"✓ Saved {total_tiles} tiles from {self.image_path.name} to memory ({obs_count} obs, {no_obs_count} no_obs)")
        print(f"  - Total crops queued: {len(crops_to_save)}")
        
        # Update progress file
        self.save_progress(self.img_idx + 1)
    
    def save_progress(self, index):
        """Save progress to file."""
        progress_file = output_path / "progress.txt"
        progress_file.write_text(str(index))

#%% Cell 5: Process Images Interactively

print("\n" + "="*60)
print("Interactive Tile Labeler - Simple Grid-Based")
print("="*60)

# Global list to store crops for batch saving at the end
crops_to_save = []

# Load progress from file
progress_file = output_path / "progress.txt"
start_index = 0

if progress_file.exists():
    try:
        start_index = int(progress_file.read_text().strip()) + 1  # Start from next image
        print(f"✓ Last session ended at image index {start_index}")
    except:
        start_index = 0

# Ask user for starting index
print(f"\nYou have {len(image_files_to_process)} images to process.")
user_input = input(f"Enter starting index (default: {start_index} to continue): ").strip()

if user_input:
    try:
        start_index = int(user_input)
        if start_index < 0 or start_index >= len(image_files_to_process):
            print(f"⚠ Invalid index. Using default {start_index}")
    except:
        print(f"⚠ Invalid input. Using default {start_index}")
else:
    if progress_file.exists():
        print(f"→ Continuing from image {start_index}")
    else:
        print(f"→ Starting fresh from image {start_index}")

# Create figure once and reuse for all images
fig, ax = plt.subplots(1, 1, figsize=(14, 14))
current_image_index = start_index

try:
    for img_idx, image_path in enumerate(image_files_to_process[start_index:], start=start_index):
        current_image_index = img_idx
        print(f"\n[{img_idx + 1}/{len(image_files_to_process)}] {image_path.name}")
        
        try:
            labeler = GridTileLabeler(image_path, grid_size=grid_size, tile_size=tile_size, 
                                     fig=fig, ax=ax, img_idx=img_idx, total_images=len(image_files_to_process))
            
            # Wait for user to confirm this image
            while not labeler.image_confirmed:
                plt.pause(0.1)
            
            # Check if user pressed exit
            if labeler.user_requested_exit:
                print("✓ Saving all work before exit...")
                break
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

except KeyboardInterrupt:
    print(f"\n✗ Interrupted by user at image index: {current_image_index}")
    # Save progress on interrupt
    progress_file.write_text(str(current_image_index))

# Clean up
plt.close('all')

print("\n" + "="*60)
print("Saving all crops to disk...")
print("="*60)

# Save all obs crops and create metadata dataframe
if crops_to_save:
    df_data = []
    duplicates = []
    
    for crop_idx, crop_info in enumerate(crops_to_save, 1):
        filename = crop_info['filename']
        row = crop_info['row']
        col = crop_info['col']
        tile_data = crop_info['data']
        label = crop_info['label']
        
        # Print progress
        print(f"  Saving crop {crop_idx}/{len(crops_to_save)}: {filename}", end='\r')
        
        # Save crop to disk in appropriate folder
        filepath = output_path / label / filename
        if filepath.exists():
            duplicates.append(filename)
        
        cv2.imwrite(str(filepath), tile_data)
        
        # Add to dataframe data
        df_data.append({
            'filename': filename,
            'row': row,
            'col': col,
            'label': label,
            'image': filename.rsplit('_r', 1)[0]  # Extract image name
        })
    
    print()  # New line after progress output
    
    # Create and save dataframe
    df = pd.DataFrame(df_data)
    csv_path = output_path / "crops_metadata.csv"
    df.to_csv(csv_path, index=False)
    
    # Count obs and no_obs
    obs_count = sum(1 for item in df_data if item['label'] == 'obs')
    no_obs_count = sum(1 for item in df_data if item['label'] == 'no_obs')
    
    print(f"✓ Saved {len(crops_to_save)} total crops")
    print(f"  - OBS: {obs_count} crops")
    print(f"  - NO_OBS: {no_obs_count} crops")
    print(f"✓ Created metadata CSV: {csv_path}")
    
    if duplicates:
        print(f"\n⚠ WARNING: {len(duplicates)} crops already existed and were overwritten:")
        for dup in duplicates[:5]:
            print(f"    - {dup}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")
else:
    print("⚠ No obs crops were labeled")

print("\n" + "="*60)
print("Interactive Tile Labeler - COMPLETE")
print("="*60)
print(f"All labeled tiles saved to: {output_path}")

# Delete progress file when done
# if progress_file.exists():
#     progress_file.unlink()
#     print("✓ Progress file cleared - restart will begin at image 0")

# %%
