"""
Load Jellyscope crops + metadata into FiftyOne for browsing and labeling.

Expected layout (per source in SOURCES):

    crops_dir/
        2026-06-01/
            crop_0001.jpg
            crop_0002.jpg
        2026-06-02/
            ...

    metadata_dir/
        2026-06-01/
            crop_0001.json
            crop_0002.json
        2026-06-02/
            ...

Each crop is matched to its metadata file by identical basename
(crop_0001.jpg <-> crop_0001.json) within the same date subfolder.
Images placed directly in crops_dir (no date subfolder) are also picked up;
their metadata is then looked up directly in metadata_dir.

Every sample gets a "source" field naming the folder it came from, so in the
App you can filter on it or use it in the sidebar to switch between folders.

Install:
    pip install fiftyone

Run:
    python load_jellyscope_dataset.py
"""

import argparse
import json
from pathlib import Path
import time
import fiftyone as fo
from fiftyone import ViewField as F

# ---- CONFIG -----------------------------------------------------------
# Update these to point at your network drive. On Windows this can be a
# mapped drive letter ("Z:\\Jellyscope\\crops") or a UNC path
# ("\\\\server\\share\\Jellyscope\\crops"). Use raw strings (r"...") either way.
#
# Add one entry per image folder. "name" is stored on each sample as its
# "source" field. "metadata_dir" may be None if the folder has no JSON files.

DATA_ROOT = r"\\uw.lu.se\research\LU24A1037-Jellyscope\Jellyscope\Monitoring data"

SOURCES = [
    {
        "name": "Kristineberg_260725_260911",
        "crops_dir": DATA_ROOT + r"\Kristineberg_260725_260911\crops",
        "metadata_dir": DATA_ROOT + r"\Kristineberg_260725_260911\crop_metadata",
    },
        {
        "name": "Faro_260926_ongoing",
        "crops_dir": DATA_ROOT + r"\Faro_260926_ongoing\crops",
        "metadata_dir": DATA_ROOT + r"\Faro_260926_ongoing\crop_metadata",
    },
    # {
    #     "name": "other_folder",
    #     "crops_dir": r"...",
    #     "metadata_dir": None,
    # },
]

DATASET_NAME = "jellyscope_crops"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Set True to reload from scratch each run, False to reuse/update an
# existing persistent dataset (faster on repeat runs, keeps any labels
# you've already added in the App).
RESET_DATASET = False

REFRESH_INTERVAL = 3600  # how often to rescan for new crops [seconds]


def make_sample(img_path: Path, meta_subdir, source_name: str, date: str):
    """Build one sample; returns (sample, metadata_found)."""
    sample = fo.Sample(filepath=str(img_path))
    sample["source"] = source_name
    sample["date"] = date

    json_path = meta_subdir / f"{img_path.stem}.json" if meta_subdir else None
    if json_path is not None and json_path.exists():
        with open(json_path, "r") as f:
            meta = json.load(f)
        # Dump every field from the JSON straight onto the sample.
        # (FiftyOne handles dynamic fields fine; flatten nested
        # dicts here first if you have any and want them queryable.)
        for key, value in meta.items():
            sample[key] = value
        return sample, True

    sample["metadata_missing"] = True
    return sample, False


def scan_source(source: dict, existing_filepaths: set):
    """Collect new samples from one source folder."""
    crops_root = Path(source["crops_dir"])
    metadata_root = Path(source["metadata_dir"]) if source.get("metadata_dir") else None

    if not crops_root.is_dir():
        print(f"WARNING: crops folder for '{source['name']}' not found: {crops_root}")
        return [], 0

    samples = []
    n_missing_meta = 0

    # (image folder, metadata folder, date label) for the root and each date subfolder
    folders = [(crops_root, metadata_root, "")]
    for sub in sorted(p for p in crops_root.iterdir() if p.is_dir()):
        folders.append((sub, metadata_root / sub.name if metadata_root else None, sub.name))

    for crop_subdir, meta_subdir, date in folders:
        for img_path in sorted(crop_subdir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if str(img_path) in existing_filepaths:
                continue  # already in dataset from a previous run

            sample, has_meta = make_sample(img_path, meta_subdir, source["name"], date)
            n_missing_meta += not has_meta
            samples.append(sample)

    return samples, n_missing_meta


def build_dataset(sources=None) -> fo.Dataset:
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

    for source in sources or SOURCES:
        samples, n_missing_meta = scan_source(source, existing_filepaths)
        if samples:
            dataset.add_samples(samples)
            existing_filepaths.update(sample.filepath for sample in samples)
        print(f"[{source['name']}] added {len(samples)} new samples "
              f"({n_missing_meta} missing metadata).")

        # Backfill "source" on samples added before this field existed
        untagged = dataset.match(
            (F("source") == None) & F("filepath").starts_with(str(Path(source["crops_dir"])))
        )
        if len(untagged):
            untagged.set_field("source", source["name"]).save()

    print(f"Dataset '{DATASET_NAME}' now has {len(dataset)} samples total.")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Load Jellyscope images into FiftyOne")
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("NAME", "CROPS_DIR"),
        help="add an image folder; repeat for multiple folders",
    )
    parser.add_argument(
        "--metadata-dir",
        nargs=2,
        action="append",
        metavar=("NAME", "METADATA_DIR"),
        help="metadata folder for a named source; repeat as needed",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="import the configured folders and exit without launching the App",
    )
    return parser.parse_args()


def sources_from_args(args):
    sources = list(SOURCES)
    metadata_dirs = dict(args.metadata_dir or [])
    for name, crops_dir in args.source or []:
        sources.append({
            "name": name,
            "crops_dir": crops_dir,
            "metadata_dir": metadata_dirs.get(name),
        })
    return sources


if __name__ == "__main__":
    args = parse_args()
    sources = sources_from_args(args)
    dataset = build_dataset(sources)

    if args.once:
        raise SystemExit(0)

    # Default view: newest date first
    view = dataset.sort_by("date", reverse=True)

    # address="0.0.0.0" so the app is reachable from other devices on the
    # Tailscale network, not just this machine (default only binds localhost).
    session = fo.launch_app(view, remote=True, address="0.0.0.0", port=5151)

    try:
        while True:
            time.sleep(REFRESH_INTERVAL)
            build_dataset(sources)  # rescan the crop folders
            session.refresh()  # push the update to connected browsers
    except KeyboardInterrupt:
        pass
