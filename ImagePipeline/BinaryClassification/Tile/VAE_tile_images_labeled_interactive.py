#%% Cell 1: Import Packages
import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import pandas as pd

print("\n" + "="*60)
print("Interactive Tile Labeler - Simple Grid-Based")
print("="*60)


def extract_species_name(image_path):
    """Extract the species name from an image filename."""
    return Path(image_path).stem.split('_')[-1]

#%% Cell 2: Define User Parameters

grid_size = 16
tile_size = int(4512 / grid_size)
monitoring_effort = 'Kristineberg_251128' 

OFFSET_NORMALIZED = [0.0, 0.2, 0.4, 0.6, 0.8]
ANY_OBS_WINS = True

sort_species = False
ROOT_C = r"C:\Users\Admin\Documents\Jellyscope\Training data\Binary_classifier"
ROOT_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier"

input_folder = r"{}\test\OG_images".format(monitoring_effort)
output_folder = r"{}\test\tiles{}_offsets{}".format(monitoring_effort, grid_size, len(OFFSET_NORMALIZED))

input_path = Path(ROOT_C).joinpath(Path(input_folder))
output_path = Path(ROOT_C).joinpath(Path(output_folder))
manual_outline_dir = output_path / Path('manual_outlines')
manual_outline_mask_dir = manual_outline_dir / Path('masks')
manual_outline_contour_dir = manual_outline_dir / Path('contours')

all_image_files = sorted(list(Path(input_path).rglob('*.png')))

if sort_species: 
    species_names = [str(img_name)[132:-4] for img_name in all_image_files]  # Extract species name from filename
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

else:
    all_image_files_filtered = all_image_files
    
# Check if images are already in output folder, exclude those images.
obs_folder = output_path / Path('obs')
no_obs_folder = output_path / Path('no_obs')

processed_images = set()
for patch in obs_folder.glob('*.png'):
    processed_images.add(patch.stem.rsplit('_r', 1)[0])
for patch in no_obs_folder.glob('*.png'):
    processed_images.add(patch.stem.rsplit('_r', 1)[0])

crop_metadata_path = output_path / "crop_metadata.csv"
crop_metadata_df = None
if crop_metadata_path.exists():
    try:
        crop_metadata_df = pd.read_csv(crop_metadata_path)
        if 'image' in crop_metadata_df.columns:
            processed_images.update(crop_metadata_df['image'].dropna().astype(str).tolist())
        elif 'filename' in crop_metadata_df.columns:
            processed_images.update(crop_metadata_df['filename'].dropna().astype(str).tolist())
        print(f"  Loaded {len(crop_metadata_df)} labeled crop records from {crop_metadata_path}")
    except Exception as e:
        print(f"⚠ Could not read {crop_metadata_path}: {e}")

## Filter out images that are already processed
all_image_files_unprocessed = []
for img in all_image_files_filtered:
    if img.stem not in processed_images:
        all_image_files_unprocessed.append(img)

species_to_total = defaultdict(int)
for img in all_image_files_filtered:
    species_to_total[extract_species_name(img)] += 1

species_to_labeled = defaultdict(int)
if crop_metadata_df is not None and 'image' in crop_metadata_df.columns:
    for image_name in crop_metadata_df['image'].dropna().astype(str).tolist():
        species_to_labeled[extract_species_name(Path(image_name))] += 1

species_progress = {}
for species, total_count in species_to_total.items():
    labeled_count = species_to_labeled.get(species, 0)
    species_progress[species] = {
        'labeled': labeled_count,
        'remaining': max(total_count - labeled_count, 0),
        'total': total_count,
    }

random.Random(42).shuffle(all_image_files_unprocessed)
randomized_image_files = all_image_files_unprocessed

print(f"Configuration:")
print(f"  Input folder:  {input_path}")
print(f"  Output folder: {output_path}")
print(f"  Tile size:     {tile_size}x{tile_size}")
print(f"  Grid size:     {grid_size}x{grid_size}")
print(f"  Offsets:       {len(OFFSET_NORMALIZED)} layers: {OFFSET_NORMALIZED}")
print(f"  Total images:  {len(all_image_files)}")
print(f"  Images after filtering by species: {len(all_image_files_filtered)}")
print(f"  Already processed images: {len(processed_images)}")
print(f"  Remaining images to process: {len(all_image_files_unprocessed)}")
print(f"  Randomized images to process: {len(randomized_image_files)}")
print("  Species progress:")
for species in sorted(species_progress):
    stats = species_progress[species]
    print(f"    - {species}: labeled={stats['labeled']}, remaining={stats['remaining']}, total={stats['total']}")
print("\n" + "="*60)
print("Parameters configured successfully!")
print("="*60)

# chose image files to process (all or only unprocessed)
image_files_to_process = randomized_image_files

#%% Cell 3: Create output folders
output_path.mkdir(parents=True, exist_ok=True)
(output_path / Path("obs")).mkdir(exist_ok=True)
(output_path / Path("no_obs")).mkdir(exist_ok=True)
manual_outline_mask_dir.mkdir(parents=True, exist_ok=True)
manual_outline_contour_dir.mkdir(parents=True, exist_ok=True)

#%% Cell 4: Simple Grid-Based Tile Labeler Class

class GridTileLabeler:
    def __init__(self, image_path, grid_size=32, tile_size=141, offsets_normalized=None, fig=None, ax=None, img_idx=0, total_images=1):
        self.image_path = image_path
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.offsets_normalized = list(offsets_normalized or [0.0])
        self.img_idx = img_idx
        self.total_images = total_images
        self.image_tile_extent = self.grid_size + len(self.offsets_normalized) - 1
        self.panel_stride = self.img_w if 'img_w' in self.__dict__ else None
        
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        self.img = img
        
        self.img_h, self.img_w = self.img.shape[:2]
        self.panel_stride = self.img_w
        # High-resolution grid parameters
        self.k = len(self.offsets_normalized)
        self.hr_rows = self.grid_size * self.k
        self.hr_cols = self.grid_size * self.k
        self.hr_cell_h = float(self.img_h) / float(self.hr_rows)
        self.hr_cell_w = float(self.img_w) / float(self.hr_cols)
        color_img = cv2.imread(str(image_path))
        if color_img is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        self.img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        
        # Store all tile data for the full offset-aware grid
        self.all_tiles = {}  # {(offset_idx, row, col): {'offset_idx': idx, 'offset': offset, 'row': row, 'col': col, 'x': x, 'y': y, 'data': tile_data}}
        self._extract_all_tiles()
        
        # Tracking selected high-resolution grid cells (hr_row, hr_col)
        self.hr_selected = set()
        self.undo_stack = []  # Stack of previous selections for undo (stores hr_selected snapshots)
        self.saved_message = None
        
        # Reuse figure or create new one
        if fig is None or ax is None:
            self.fig = plt.figure(figsize=(18, 10))
            self.ax = self.fig.add_axes([0.03, 0.05, 0.74, 0.9])
        else:
            self.fig, self.ax = fig, ax
            self.ax.set_position([0.03, 0.05, 0.74, 0.9])

        # Side panel for species name and instructions
        existing_info_ax = getattr(self.fig, '_jelly_info_ax', None)
        if existing_info_ax is not None:
            try:
                existing_info_ax.remove()
            except Exception:
                pass

        self.info_ax = self.fig.add_axes([0.80, 0.05, 0.18, 0.90])
        self.fig._jelly_info_ax = self.info_ax
        self.info_ax.set_axis_off()
        
        self.tile_rects = {}  # {(offset_idx, row, col): rect_patch}
        self.image_confirmed = False  # Flag to track when user presses 'c'
        self.user_requested_exit = False  # Flag to track when user presses 'e'
        self.show_grid()
        
        # Connect event handlers
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.release_cid = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Drag selection state
        self.dragging = False
        self.drag_start = None
        self.drag_rect = None

    def _build_manual_outline_mask(self):
        """Build a full-resolution binary mask from selected HR cells."""
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        for hr_row, hr_col in self.hr_selected:
            x0 = int(round(hr_col * self.hr_cell_w))
            y0 = int(round(hr_row * self.hr_cell_h))
            x1 = int(round((hr_col + 1) * self.hr_cell_w))
            y1 = int(round((hr_row + 1) * self.hr_cell_h))

            x0 = min(max(x0, 0), self.img_w)
            y0 = min(max(y0, 0), self.img_h)
            x1 = min(max(x1, 0), self.img_w)
            y1 = min(max(y1, 0), self.img_h)

            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 255

        return mask

    def _save_manual_outline_assets(self):
        """Save the manual outline mask and contour metadata for the current crop."""
        mask = self._build_manual_outline_mask()
        base_name = self.image_path.stem

        mask_path = manual_outline_mask_dir / f"{base_name}_mask.png"
        contour_path = manual_outline_contour_dir / f"{base_name}_contours.json"

        cv2.imwrite(str(mask_path), mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_payload = []
        for contour in contours:
            contour_xy = contour.squeeze().astype(int)
            if contour_xy.ndim == 1:
                contour_xy = contour_xy.reshape(-1, 2)
            contour_payload.append(contour_xy.tolist())

        with open(contour_path, 'w', encoding='utf-8') as f:
            json.dump(contour_payload, f)

        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            bbox = {
                'x0': int(xs.min()),
                'y0': int(ys.min()),
                'x1': int(xs.max()) + 1,
                'y1': int(ys.max()) + 1,
            }
        else:
            bbox = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}

        selected_area_px = int((mask > 0).sum())
        return mask_path, contour_path, bbox, selected_area_px, len(contour_payload)
    
    def _extract_all_tiles(self):
        """Extract all sampled tiles for a full kxk subdivision per base tile.

        For display we will create a composite image of the same size as the original
        but subdivided into (grid_size * k) cells per axis, where k = len(offsets).
        Each sampled tile (size tile_size) is resized to a small cell (tile_size // k)
        for rendering and placed at: (row*tile_size + oy_idx*small, col*tile_size + ox_idx*small).
        """
        k = len(self.offsets_normalized)
        small = max(1, self.tile_size // k)

        for oy_idx, oy in enumerate(self.offsets_normalized):
            oy_px = int(round(self.tile_size * oy))
            for ox_idx, ox in enumerate(self.offsets_normalized):
                ox_px = int(round(self.tile_size * ox))

                for row in range(self.grid_size):
                    for col in range(self.grid_size):
                        tile_x = col * self.tile_size + ox_px
                        tile_y = row * self.tile_size + oy_px
                        tile_x_end = tile_x + self.tile_size
                        tile_y_end = tile_y + self.tile_size

                        if tile_x_end > self.img_w or tile_y_end > self.img_h:
                            continue

                        tile_data = self.img[tile_y:tile_y_end, tile_x:tile_x_end]
                        # compute display position (within original image coordinates)
                        disp_x = col * self.tile_size + ox_idx * small
                        disp_y = row * self.tile_size + oy_idx * small

                        tile_key = (oy_idx, ox_idx, row, col)
                        self.all_tiles[tile_key] = {
                            'oy_idx': oy_idx,
                            'ox_idx': ox_idx,
                            'offset_y': oy,
                            'offset_x': ox,
                            'row': row,
                            'col': col,
                            'x': disp_x,
                            'y': disp_y,
                            'small': small,
                            'data': tile_data,
                        }
        
    def show_grid(self):
        """Display image with grid and labeled tiles."""
        # Display the original image and overlay a high-resolution grid of size (grid_size * k)
        self.ax.clear()
        self.ax.imshow(self.img_rgb)
        self.tile_rects = {}

        # Draw HR grid with a neutral grey outline for every offset cell
        k = self.k
        hr_h = self.hr_cell_h
        hr_w = self.hr_cell_w

        # Draw each HR cell border in the same grey style
        for r in range(self.hr_rows):
            for c in range(self.hr_cols):
                x = c * hr_w
                y = r * hr_h
                rect = patches.Rectangle((x, y), hr_w, hr_h, linewidth=0.3, edgecolor='grey', facecolor='none', alpha=0.5, zorder=1)
                self.ax.add_patch(rect)

        # Shade selected HR cells
        for (r, c) in self.hr_selected:
            x = c * hr_w
            y = r * hr_h
            rect = patches.Rectangle((x, y), hr_w, hr_h, linewidth=0, facecolor='#90EE90', alpha=0.5, zorder=3)
            self.ax.add_patch(rect)

        self.ax.set_xlim(0, self.img_w)
        self.ax.set_ylim(self.img_h, 0)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self._update_info_panel()
        self.fig.canvas.draw_idle()
    
    def _update_info_panel(self):
        """Render crop details and instructions in the side panel."""
        self.info_ax.clear()
        self.info_ax.set_axis_off()

        species_name = extract_species_name(self.image_path)
        sampled_tiles = self.hr_rows * self.hr_cols
        status_line = f"{self.saved_message}\n\n" if self.saved_message else ""
        info_text = (
            f"{status_line}"
            f"Species\n{species_name}\n\n"
            f"Image\n{self.img_idx + 1} / {self.total_images}\n\n"
            f"Selected OBS\n{len(self.hr_selected)}\n\n"
            f"Unselected\n{sampled_tiles - len(self.hr_selected)}\n\n"
            f"Grid\n{self.grid_size * self.k} x {self.grid_size * self.k}\n"
            f"({self.grid_size}x{self.grid_size} base tiles, {len(self.offsets_normalized)} offsets)\n\n"
            "Controls\n"
            "Left-click: mark/unmark\n"
            "Drag: select region\n"
            "r: reset\n"
            "u: undo\n"
            "c: save & next\n"
            "e: exit"
        )

        self.info_ax.text(
            0.02,
            0.98,
            info_text,
            va='top',
            ha='left',
            fontsize=12,
            fontweight='bold',
            family='sans-serif',
            linespacing=1.3,
        )
    def get_tile_at_click(self, x, y):
        """Find which tile was clicked based on coordinates."""
        # Map click to high-resolution grid cell
        if x < 0 or y < 0 or x >= self.img_w or y >= self.img_h:
            return None
        hr_col = int(x / self.hr_cell_w)
        hr_row = int(y / self.hr_cell_h)
        hr_col = min(max(hr_col, 0), self.hr_cols - 1)
        hr_row = min(max(hr_row, 0), self.hr_rows - 1)
        return (hr_row, hr_col)
        return None
    
    def on_click(self, event):
        """Handle mouse button press; start drag selection for left-button."""
        if event.xdata is None or event.button != 1:
            return

        # Begin dragging selection rectangle
        self.dragging = True
        self.drag_start = (event.xdata, event.ydata)

        # Create a rectangle patch to show selection area
        if self.drag_rect is None:
            self.drag_rect = patches.Rectangle((event.xdata, event.ydata), 0, 0,
                                               linewidth=1, edgecolor='cyan', facecolor='cyan', alpha=0.25, zorder=4)
            self.ax.add_patch(self.drag_rect)
        else:
            self.drag_rect.set_xy((event.xdata, event.ydata))
            self.drag_rect.set_width(0)
            self.drag_rect.set_height(0)

        self.fig.canvas.draw_idle()

    def on_motion(self, event):
        """Update drag rectangle visual while mouse moves."""
        if not self.dragging or self.drag_start is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0 = self.drag_start
        x1, y1 = event.xdata, event.ydata
        xmin, ymin = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)

        if self.drag_rect is not None:
            self.drag_rect.set_xy((xmin, ymin))
            self.drag_rect.set_width(w)
            self.drag_rect.set_height(h)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse button release; finalize drag or treat as single click."""
        if event.button != 1:
            return
        if not self.dragging:
            return

        self.dragging = False
        if self.drag_start is None:
            return

        # If release outside axes, just clear rectangle
        if event.xdata is None or event.ydata is None:
            if self.drag_rect is not None:
                try:
                    self.drag_rect.remove()
                except Exception:
                    pass
                self.drag_rect = None
                self.fig.canvas.draw_idle()
            self.drag_start = None
            return

        x0, y0 = self.drag_start
        x1, y1 = event.xdata, event.ydata
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        # Small movement -> treat as single click toggle
        if dx < 4 and dy < 4:
            x, y = int(event.xdata), int(event.ydata)
            hr_cell = self.get_tile_at_click(x, y)
            if hr_cell is not None:
                self.undo_stack.append(self.hr_selected.copy())
                if hr_cell in self.hr_selected:
                    self.hr_selected.remove(hr_cell)
                    print(f"Unmarked HR cell {hr_cell}")
                else:
                    self.hr_selected.add(hr_cell)
                    print(f"Marked HR cell {hr_cell}")
                self.show_grid()
        else:
            # Rectangle selection: add all HR cells overlapping the box
            xmin = max(min(x0, x1), 0)
            ymin = max(min(y0, y1), 0)
            xmax = min(max(x0, x1), self.img_w)
            ymax = min(max(y0, y1), self.img_h)

            hr_r0 = int(ymin / self.hr_cell_h)
            hr_r1 = int((ymax - 1) / self.hr_cell_h)
            hr_c0 = int(xmin / self.hr_cell_w)
            hr_c1 = int((xmax - 1) / self.hr_cell_w)

            hr_r0 = min(max(hr_r0, 0), self.hr_rows - 1)
            hr_r1 = min(max(hr_r1, 0), self.hr_rows - 1)
            hr_c0 = min(max(hr_c0, 0), self.hr_cols - 1)
            hr_c1 = min(max(hr_c1, 0), self.hr_cols - 1)

            # Save state for undo
            self.undo_stack.append(self.hr_selected.copy())

            added = 0
            for rr in range(hr_r0, hr_r1 + 1):
                for cc in range(hr_c0, hr_c1 + 1):
                    if (rr, cc) not in self.hr_selected:
                        self.hr_selected.add((rr, cc))
                        added += 1

            print(f"Marked {added} HR cells from rectangle selection")
            self.show_grid()

        # Remove drag rectangle
        if self.drag_rect is not None:
            try:
                self.drag_rect.remove()
            except Exception:
                pass
            self.drag_rect = None

        self.drag_start = None
    
    def on_key(self, event):
        """Handle keyboard input."""
        if event.key == 'c':
            # Confirm and save
            print("\n✓ Confirming image...")
            self.save_image()
            self.image_confirmed = True
            self.saved_message = f"Saved - {extract_species_name(self.image_path)}"
            self._update_info_panel()
            self.fig.canvas.draw_idle()
        
        elif event.key == 'r':
            # Reset all tiles
            self.undo_stack.append(self.hr_selected.copy())
            self.hr_selected = set()
            print("✓ Reset all tiles")
            self.show_grid()
        
        elif event.key == 'u':
            # Undo last change
            if self.undo_stack:
                self.hr_selected = self.undo_stack.pop()
                print("✓ Undone last change")
                self.show_grid()
            else:
                print("⚠ Nothing to undo")
        
        elif event.key == 'e':
            # Exit program and save all work
            print(f"\n✗ User exited at image index: {self.img_idx}")
            
            # Save current image's tiles before exiting
            if self.hr_selected:
                print("Saving current image's tiles...")
                self.save_image()
            
            self.save_progress(self.img_idx)
            
            # Signal to save crops and exit
            self.user_requested_exit = True
            self.image_confirmed = True  # Set this so main loop can continue and detect exit
            plt.close('all')
    
    def save_image(self):
        """Extract sampled tiles for each offset and decide obs/no_obs by HR overlap.

        DECISION POINT: a sampled tile is labeled 'obs' if any high-resolution grid cell
        that overlaps the sampled tile was selected by the user (hr_selected). Change
        the logic here if you prefer majority voting instead.
        """
        queued_before = len(crops_to_save)
        offsets = self.offsets_normalized
        obs_count = 0
        no_obs_count = 0

        for off_idx, off in enumerate(offsets):
            off_px = int(round(self.tile_size * off))
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    y0 = row * self.tile_size + off_px
                    x0 = col * self.tile_size + off_px
                    y1 = y0 + self.tile_size
                    x1 = x0 + self.tile_size

                    if y1 > self.img_h or x1 > self.img_w:
                        continue

                    tile = self.img[y0:y1, x0:x1]

                    # Map sampled tile bounds to HR grid indices
                    hr_r0 = int(y0 / self.hr_cell_h)
                    hr_r1 = int((y1 - 1) / self.hr_cell_h)
                    hr_c0 = int(x0 / self.hr_cell_w)
                    hr_c1 = int((x1 - 1) / self.hr_cell_w)

                    # DECISION POINT: sampled tile is obs if any overlapping HR cell selected
                    is_obs = False
                    for rr in range(hr_r0, hr_r1 + 1):
                        for cc in range(hr_c0, hr_c1 + 1):
                            if (rr, cc) in self.hr_selected:
                                is_obs = True
                                break
                        if is_obs:
                            break

                    label = 'obs' if is_obs else 'no_obs'
                    if label == 'obs':
                        obs_count += 1
                    else:
                        no_obs_count += 1

                    filename = f"{self.image_path.stem}_r{row}_c{col}_o{int(off*100):02d}.png"
                    crops_to_save.append({
                        'filename': filename,
                        'row': row,
                        'col': col,
                        'offset_idx': off_idx,
                        'offset_norm': off,
                        'data': tile,
                        'label': label,
                        'image': self.image_path.stem,
                        'hr_overlap_count': sum(1 for rr in range(hr_r0, hr_r1 + 1) for cc in range(hr_c0, hr_c1 + 1) if (rr, cc) in self.hr_selected),
                    })

        mask_path, contour_path, bbox, selected_area_px, contour_count = self._save_manual_outline_assets()

        crop_metadata_records.append({
            'image': self.image_path.stem,
            'image_path': str(self.image_path),
            'label': 'obs' if selected_area_px > 0 else 'no_obs',
            'selected_hr_cells': len(self.hr_selected),
            'selected_area_px': selected_area_px,
            'selected_area_fraction': selected_area_px / float(self.img_h * self.img_w),
            'mask_path': str(mask_path),
            'contour_path': str(contour_path),
            'contour_count': contour_count,
            'bbox_x0': bbox['x0'],
            'bbox_y0': bbox['y0'],
            'bbox_x1': bbox['x1'],
            'bbox_y1': bbox['y1'],
        })

        queued_after = len(crops_to_save)
        print(f"✓ Extracted tiles for {len(offsets)} offsets: {obs_count} obs, {no_obs_count} no_obs")
        print(f"  - Saved manual outline: {mask_path.name}")
        print(f"  - Total crops queued: {queued_after - queued_before}")

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
crop_metadata_records = []

# Load progress from file
progress_file = output_path / "progress.txt"
start_index = 0

if progress_file.exists():
    try:
        start_index = int(progress_file.read_text().strip()) + 1  # Start from next image
        print(f"✓ Last session ended at image index {start_index}")
    except:
        start_index = 0

print(f"\nYou have {len(image_files_to_process)} images to process.")
if progress_file.exists():
    print(f"→ Continuing from saved progress at image {start_index}")
else:
    print("→ Starting from the first randomized image")

# Create figure once and reuse for all images
fig, ax = plt.subplots(1, 1, figsize=(14, 14))
current_image_index = start_index

try:
    for img_idx, image_path in enumerate(image_files_to_process[start_index:], start=start_index):
        current_image_index = img_idx
        print(f"\n[{img_idx + 1}/{len(image_files_to_process)}] {image_path.name}")
        
        try:
            labeler = GridTileLabeler(image_path, grid_size=grid_size, tile_size=tile_size, 
                                     offsets_normalized=OFFSET_NORMALIZED,
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
        offset_idx = crop_info.get('offset_idx')
        offset_norm = crop_info.get('offset_norm')
        source_obs_count = crop_info.get('source_obs_count')
        sampled_tiles_in_group = crop_info.get('sampled_tiles_in_group')
        
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
            'offset_idx': offset_idx,
            'offset_norm': offset_norm,
            'label': label,
            'image': crop_info.get('image', filename.rsplit('_r', 1)[0]),
            'source_obs_count': source_obs_count,
            'sampled_tiles_in_group': sampled_tiles_in_group,
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

if crop_metadata_records:
    crop_metadata_df = pd.DataFrame(crop_metadata_records)
    crop_metadata_path = output_path / "crop_metadata.csv"
    crop_metadata_df.to_csv(crop_metadata_path, index=False)
    print(f"✓ Created crop-level metadata CSV: {crop_metadata_path}")

print("\n" + "="*60)
print("Interactive Tile Labeler - COMPLETE")
print("="*60)
print(f"All labeled tiles saved to: {output_path}")

# Delete progress file when done
# if progress_file.exists():
#     progress_file.unlink()
#     print("✓ Progress file cleared - restart will begin at image 0")

# %%
