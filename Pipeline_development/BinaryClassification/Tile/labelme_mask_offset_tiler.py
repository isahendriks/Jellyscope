# labelme_mask_offset_tiler.py
# Converts LabelMe annotations/masks into offset-grid tiles labeled obs/no_obs.
#
# Expected inputs:
#   - Original images in input_path
#   - Either LabelMe JSON files OR generated LabelMe mask PNGs
#
# Output:
#   output_path/
#     obs/
#     no_obs/
#     manual_outlines/
#       masks/
#       contours/
#     tiles_metadata.csv
#     crops_metadata.csv

import json
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# =========================
# USER PARAMETERS
# =========================

grid_size = 16
tile_size = int(4512 / grid_size)

monitoring_effort = "Kristineberg_260730"

OFFSET_NORMALIZED = [0.0, 0.2, 0.4, 0.6, 0.8]

# If True: a tile is obs if ANY mask pixel overlaps it.
# If False: obs if mask overlap fraction >= MIN_MASK_FRACTION.
ANY_OBS_WINS = True
MIN_MASK_FRACTION = 0.01

sort_species = False
ROOT_C = f"C:\\Users\\Admin\\Documents\\Jellyscope\\Training data\\Binary_classifier\\{monitoring_effort}"

folder = Path("test\\obs")

input_folder = rf"OG_images"
output_folder = rf"tiles{grid_size}_offsets{len(OFFSET_NORMALIZED)}_labelme"

# Choose one:
# 1. If you have LabelMe .json files, set mask_source = "json"
# 2. If you already ran labelme_json_to_dataset and have label.png files, set mask_source = "label_png"
mask_source = "json"

# Folder containing LabelMe JSON files, one per image: image_name.json
labelme_json_dir = Path(ROOT_C) / folder / "manual_masks"

# LabelMe labels to count as observation.
# Use None to count all non-background shapes/classes as observation.
OBS_LABELS = None
# Example:
# OBS_LABELS = {"jelly", "organism", "object"}

print(f"images folder: {input_folder}")
print(f"output folder: {output_folder}")
print(f"mask source:   {mask_source}")

# =========================
# PATHS
# =========================

input_path = Path(ROOT_C) / folder / input_folder
output_path = Path(ROOT_C) / folder / output_folder

obs_folder = output_path / "obs"
no_obs_folder = output_path / "no_obs"

manual_outline_dir = output_path / "manual_outlines"
manual_outline_mask_dir = manual_outline_dir / "masks"
manual_outline_contour_dir = manual_outline_dir / "contours"

metadata_tiles_path = output_path / "tiles_metadata.csv"
metadata_crops_path = output_path / "crops_metadata.csv"

for p in [obs_folder, no_obs_folder, manual_outline_mask_dir, manual_outline_contour_dir]:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS
# =========================

def extract_species_name(image_path):
    """Extract the species name from an image filename."""
    return Path(image_path).stem.split("_")[-1]


def find_image_files(folder):
    # Look into all subfolders and find image files with common extensions
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(Path(folder).rglob(ext))
    return sorted(files)


def polygon_to_mask_from_labelme_json(json_path, image_shape, obs_labels=None):
    """
    Build a binary mask from a LabelMe JSON file.
    White/255 = observation, black/0 = no observation.
    """
    img_h, img_w = image_shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        label = shape.get("label", "")

        if obs_labels is not None and label not in obs_labels:
            continue

        points = np.array(shape.get("points", []), dtype=np.float32)
        if points.size == 0:
            continue

        shape_type = shape.get("shape_type", "polygon")

        if shape_type in ["polygon", None]:
            pts = np.round(points).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

        elif shape_type == "rectangle":
            # LabelMe rectangle usually has two corner points.
            x0, y0 = points[0]
            x1, y1 = points[1]
            x0, x1 = sorted([int(round(x0)), int(round(x1))])
            y0, y1 = sorted([int(round(y0)), int(round(y1))])
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

        elif shape_type == "circle":
            # LabelMe circle usually has center and perimeter point.
            center = points[0]
            edge = points[1]
            radius = int(round(np.linalg.norm(edge - center)))
            cv2.circle(mask, tuple(np.round(center).astype(int)), radius, 255, thickness=-1)

        elif shape_type == "line" or shape_type == "linestrip":
            pts = np.round(points).astype(np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=3)

        elif shape_type == "point":
            x, y = np.round(points[0]).astype(int)
            cv2.circle(mask, (x, y), 3, 255, thickness=-1)

    return mask


def load_labelme_label_png(image_path, image_shape):
    """
    Load LabelMe-generated label.png.
    Expected location:
      labelme_dataset_dir / f"{image_stem}_json" / "label.png"
    """
    candidate = labelme_dataset_dir / f"{image_path.stem}_json" / "label.png"

    if not candidate.exists():
        raise FileNotFoundError(f"Missing LabelMe label PNG: {candidate}")

    label_img = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
    if label_img is None:
        raise FileNotFoundError(f"Could not read mask: {candidate}")

    if label_img.ndim == 3:
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

    # LabelMe label.png usually has background 0 and labels as 1, 2, ...
    mask = (label_img > 0).astype(np.uint8) * 255

    img_h, img_w = image_shape[:2]
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    return mask


def load_mask_for_image(image_path, image_shape):
    """Load or build the binary observation mask for one image."""
    if mask_source == "json":
        json_path = labelme_json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing LabelMe JSON: {json_path}")
        return polygon_to_mask_from_labelme_json(json_path, image_shape, OBS_LABELS), json_path

    if mask_source == "label_png":
        mask = load_labelme_label_png(image_path, image_shape)
        mask_path = labelme_dataset_dir / f"{image_path.stem}_json" / "label.png"
        return mask, mask_path

    raise ValueError("mask_source must be either 'json' or 'label_png'")


def save_mask_and_contours(image_path, mask):
    """Save copied/generated mask and contour JSON in the same style as the old script."""
    base_name = image_path.stem

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

    with open(contour_path, "w", encoding="utf-8") as f:
        json.dump(contour_payload, f)

    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        bbox = {
            "x0": int(xs.min()),
            "y0": int(ys.min()),
            "x1": int(xs.max()) + 1,
            "y1": int(ys.max()) + 1,
        }
    else:
        bbox = {"x0": 0, "y0": 0, "x1": 0, "y1": 0}

    selected_area_px = int((mask > 0).sum())

    return mask_path, contour_path, bbox, selected_area_px, len(contour_payload)


def already_processed_images():
    """Avoid reprocessing images whose tiles or crop metadata already exist."""
    processed = set()

    for patch in obs_folder.glob("*.png"):
        processed.add(patch.stem.rsplit("_r", 1)[0])

    for patch in no_obs_folder.glob("*.png"):
        processed.add(patch.stem.rsplit("_r", 1)[0])

    if metadata_crops_path.exists():
        try:
            df = pd.read_csv(metadata_crops_path)
            if "image" in df.columns:
                processed.update(df["image"].dropna().astype(str).tolist())
        except Exception as e:
            print(f"Could not read existing crop metadata: {e}")

    return processed


def tile_is_obs(mask_tile):
    """Decide whether a tile is observation based on mask overlap."""
    obs_pixels = int((mask_tile > 0).sum())

    if ANY_OBS_WINS:
        return obs_pixels > 0, obs_pixels, obs_pixels / float(mask_tile.size)

    overlap_fraction = obs_pixels / float(mask_tile.size)
    return overlap_fraction >= MIN_MASK_FRACTION, obs_pixels, overlap_fraction


# =========================
# MAIN PROCESSING
# =========================

def process_images():
    all_image_files = find_image_files(input_path)
    
    print(f"Found {len(all_image_files)} image files in {input_path}")

    if sort_species:
        # Keep the simpler species filtering from your original script if needed.
        species_names_to_exclude = {"dentritus", "dentritus_glo", "filament", "macro_filament", "filament_glo"}
        image_files = [
            img for img in all_image_files
            if extract_species_name(img) not in species_names_to_exclude
        ]
    else:
        image_files = all_image_files

    processed = already_processed_images()
    image_files = [img for img in image_files if img.stem not in processed]

    random.Random(42).shuffle(image_files)

    print("=" * 60)
    print("LabelMe Mask Offset Tiler")
    print("=" * 60)
    print(f"Input folder:  {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Mask source:   {mask_source}")
    print(f"Grid size:     {grid_size}x{grid_size}")
    print(f"Tile size:     {tile_size}x{tile_size}")
    print(f"Offsets:       {OFFSET_NORMALIZED}")
    print(f"Images found:  {len(all_image_files)}")
    print(f"Images left:   {len(image_files)}")
    print("=" * 60)

    tile_records = []
    crop_records = []

    total_obs = 0
    total_no_obs = 0

    for img_idx, image_path in enumerate(image_files, 1):
        print(f"[{img_idx}/{len(image_files)}] {image_path.name}")

        img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"  Skipping: could not read image")
            continue

        img_h, img_w = img_gray.shape[:2]

        try:
            mask, mask_input_path = load_mask_for_image(image_path, img_gray.shape)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        obs_count = 0
        no_obs_count = 0

        for off_idx, off in enumerate(OFFSET_NORMALIZED):
            off_px = int(round(tile_size * off))

            for row in range(grid_size):
                for col in range(grid_size):
                    y0 = row * tile_size + off_px
                    x0 = col * tile_size + off_px
                    y1 = y0 + tile_size
                    x1 = x0 + tile_size

                    if y1 > img_h or x1 > img_w:
                        continue

                    tile = img_gray[y0:y1, x0:x1]
                    mask_tile = mask[y0:y1, x0:x1]

                    is_obs, mask_overlap_px, mask_overlap_fraction = tile_is_obs(mask_tile)
                    label = "obs" if is_obs else "no_obs"

                    if is_obs:
                        obs_count += 1
                        total_obs += 1
                    else:
                        no_obs_count += 1
                        total_no_obs += 1

                    filename = f"{image_path.stem}_r{row}_c{col}_o{int(off * 100):02d}.png"
                    out_file = output_path / label / filename
                    cv2.imwrite(str(out_file), tile)

                    tile_records.append({
                        "filename": filename,
                        "row": row,
                        "col": col,
                        "offset_idx": off_idx,
                        "offset_norm": off,
                        "label": label,
                        "image": image_path.stem,
                        "image_path": str(image_path),
                        "tile_x0": x0,
                        "tile_y0": y0,
                        "tile_x1": x1,
                        "tile_y1": y1,
                        "mask_overlap_px": mask_overlap_px,
                        "mask_overlap_fraction": mask_overlap_fraction,
                    })

        mask_path, contour_path, bbox, selected_area_px, contour_count = save_mask_and_contours(image_path, mask)

        crop_records.append({
            "image": image_path.stem,
            "image_path": str(image_path),
            "mask_input_path": str(mask_input_path),
            "label": "obs" if selected_area_px > 0 else "no_obs",
            "selected_area_px": selected_area_px,
            "selected_area_fraction": selected_area_px / float(img_h * img_w),
            "mask_path": str(mask_path),
            "contour_path": str(contour_path),
            "contour_count": contour_count,
            "bbox_x0": bbox["x0"],
            "bbox_y0": bbox["y0"],
            "bbox_x1": bbox["x1"],
            "bbox_y1": bbox["y1"],
            "tiles_obs": obs_count,
            "tiles_no_obs": no_obs_count,
        })

        print(f"  Saved tiles: {obs_count} obs, {no_obs_count} no_obs")
        print(f"  Saved mask:  {mask_path.name}")

    if tile_records:
        tile_df = pd.DataFrame(tile_records)

        if metadata_tiles_path.exists():
            old = pd.read_csv(metadata_tiles_path)
            tile_df = pd.concat([old, tile_df], ignore_index=True)

        tile_df.to_csv(metadata_tiles_path, index=False)
        print(f"Saved tile metadata: {metadata_tiles_path}")

    if crop_records:
        crop_df = pd.DataFrame(crop_records)

        if metadata_crops_path.exists():
            old = pd.read_csv(metadata_crops_path)
            crop_df = pd.concat([old, crop_df], ignore_index=True)

        crop_df.to_csv(metadata_crops_path, index=False)
        print(f"Saved crop metadata: {metadata_crops_path}")

    print("=" * 60)
    print("COMPLETE")
    print(f"Total OBS tiles:    {total_obs}")
    print(f"Total NO_OBS tiles: {total_no_obs}")
    print(f"Output saved to:    {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    process_images()
