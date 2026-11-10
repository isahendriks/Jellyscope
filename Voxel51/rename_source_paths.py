"""
Point existing samples at a renamed folder instead of re-importing them.

load_jellyscope_dataset.py skips crops whose filepath is already in the
dataset, so renaming a folder on the share makes every crop look new. This
script deletes any duplicates already imported from the new path, then
rewrites the old samples' filepaths (keeping their labels/tags) and sets
their "source" to the new name.

Stop load_jellyscope_dataset.py first, then run (dry run without --apply):

    python rename_source_paths.py OLD_DIR NEW_DIR NEW_SOURCE_NAME [--apply]
"""

import argparse

import fiftyone as fo
from fiftyone import ViewField as F

DATASET_NAME = "jellyscope_crops"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old_dir", help="folder prefix as stored before the rename")
    parser.add_argument("new_dir", help="folder prefix after the rename")
    parser.add_argument("source_name", help='new value for the "source" field')
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    old_dir = args.old_dir.rstrip("\\/")
    new_dir = args.new_dir.rstrip("\\/")

    dataset = fo.load_dataset(DATASET_NAME)
    old_view = dataset.match(F("filepath").starts_with(old_dir + "\\"))
    new_view = dataset.match(F("filepath").starts_with(new_dir + "\\"))

    print(f"Samples under old path: {len(old_view)}")
    print(f"Samples already re-imported under new path (to delete): {len(new_view)}")
    if not args.apply:
        print("Dry run; re-run with --apply to make these changes.")
        return

    dataset.delete_samples(new_view)

    ids, filepaths = old_view.values(["id", "filepath"])
    moved = dataset.select(ids, ordered=True)
    moved.set_values("filepath", [new_dir + fp[len(old_dir):] for fp in filepaths])
    moved.set_values("source", [args.source_name] * len(ids))

    print(f"Updated {len(ids)} samples; dataset now has {len(dataset)} samples.")


if __name__ == "__main__":
    main()
