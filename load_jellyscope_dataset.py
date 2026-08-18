"""
Load Jellyscope crops + metadata into FiftyOne for browsing and labeling.

Expected layout:

    CROPS_DIR/
        2026-06-01/
            crop_0001.jpg
            crop_0002.jpg
        2026-06-02/
            ...

    METADATA_DIR/
        2026-06-01/
            crop_0001.json
            crop_0002.json
        2026-06-02/
            ...

Each crop is matched to its metadata file by identical basename
(crop_0001.jpg <-> crop_0001.json) within the same date subfolder.

Install:
    pip install fiftyone

Run:
    python load_jellyscope_dataset.py
"""

import os
import json
from pathlib import Path

import fiftyone as fo

# ---- CONFIG -----------------------------------------------------------
# Update these to point at your network drive. On Windows this can be a
# mapped drive letter ("Z:\\Jellyscope\\crops") or a UNC path
# ("\\\\server\\share\\Jellyscope\\crops"). Use raw strings (r"...") either way.

CROPS_DIR = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data\Kristineberg_260725_ongoing\crops"
METADATA_DIR = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data\Kristineberg_260725_ongoing\crop_metadata"

DATASET_NAME = "jellyscope_crops"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Set True to reload from scratch each run, False to reuse/update an
# existing persistent dataset (faster on repeat runs, keeps any labels
# you've already added in the App).
RESET_DATASET = False

# ------------------------------------------------------------------------


def build_dataset() -> fo.Dataset:
    if fo.dataset_exists(DATASET_NAME):
        if RESET_DATASET:
            fo.delete_dataset(DATASET_NAME)
            dataset = fo.Dataset(DATASET_NAME, persistent=True)
        else:
            dataset = fo.load_dataset(DATASET_NAME)
            print(f"Loaded existing dataset '{DATASET_NAME}' "
                  f"with {len(dataset)} samples. Scanning for new samples...")
    else:
        dataset = fo.Dataset(DATASET_NAME, persistent=True)

    existing_filepaths = set(dataset.values("filepath")) if len(dataset) else set()

    samples = []
    n_missing_meta = 0

    crops_root = Path(CROPS_DIR)
    metadata_root = Path(METADATA_DIR)

    date_folders = sorted(p.name for p in crops_root.iterdir() if p.is_dir())

    for date_folder in date_folders:
        crop_subdir = crops_root / date_folder
        meta_subdir = metadata_root / date_folder

        for img_path in sorted(crop_subdir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            img_path_str = str(img_path)
            if img_path_str in existing_filepaths:
                continue  # already in dataset from a previous run

            sample = fo.Sample(filepath=img_path_str)
            sample["date"] = date_folder

            json_path = meta_subdir / f"{img_path.stem}.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    meta = json.load(f)
                # Dump every field from the JSON straight onto the sample.
                # (FiftyOne handles dynamic fields fine; flatten nested
                # dicts here first if you have any and want them queryable.)
                for key, value in meta.items():
                    sample[key] = value
            else:
                sample["metadata_missing"] = True
                n_missing_meta += 1

            samples.append(sample)

    if samples:
        dataset.add_samples(samples)

    print(f"Added {len(samples)} new samples "
          f"({n_missing_meta} missing metadata).")
    print(f"Dataset '{DATASET_NAME}' now has {len(dataset)} samples total.")

    return dataset


if __name__ == "__main__":
    dataset = build_dataset()

    # Handy default view: newest date first
    dataset = dataset.sort_by("date", reverse=True)

    session = fo.launch_app(dataset)
    session.wait()  # keep the script alive while you browse/label