#!/usr/bin/env python3
"""Copy N images evenly spaced (by sorted filename order) from a source folder into a destination folder."""
import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def select_uniform(files: list[Path], n: int) -> list[Path]:
    if n >= len(files):
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing the source images")
    parser.add_argument("dest", type=Path, help="Folder to copy the selected images into")
    parser.add_argument("-n", "--num-images", type=int, default=200, help="Number of images to select (default: 200)")
    args = parser.parse_args()

    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images found in {args.source}")

    selected = select_uniform(files, args.num_images)

    args.dest.mkdir(parents=True, exist_ok=True)
    for src in selected:
        shutil.copy2(src, args.dest / src.name)

    print(f"Copied {len(selected)} of {len(files)} images from {args.source} to {args.dest}")


if __name__ == "__main__":
    main()
