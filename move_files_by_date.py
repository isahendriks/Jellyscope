"""
Script to move files created on a specific date to a different folder.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


def move_files_by_creation_date(source_dir, target_dir, creation_date):
    """
    Move files created on a specific date to a target directory.
    
    Args:
        source_dir (str): Path to source directory
        target_dir (str): Path to destination directory
        creation_date (str): Date in format 'YYYY-MM-DD' (e.g., '2026-02-17')
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Parse the target date
    target_datetime = datetime.strptime(creation_date, '%Y-%m-%d')
    target_date = target_datetime.date()
    
    moved_count = 0
    failed_count = 0
    
    # Iterate through all files in source directory
    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            try:
                # Get file creation time
                creation_time = datetime.fromtimestamp(file_path.stat().st_ctime)
                file_date = creation_time.date()
                
                # Check if file was created on the target date
                if file_date == target_date:
                    # Move file to target directory
                    destination = target_path / file_path.name
                    
                    # Handle duplicate filenames
                    if destination.exists():
                        stem = file_path.stem
                        suffix = file_path.suffix
                        counter = 1
                        while destination.exists():
                            destination = target_path / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(destination))
                    print(f"✓ Moved: {file_path.name}")
                    moved_count += 1
                    
            except Exception as e:
                print(f"✗ Failed to move {file_path.name}: {e}")
                failed_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Files moved: {moved_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Target date: {creation_date}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Configuration - modify these values
    SOURCE_DIRECTORY = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier\Kristineberg_250915_train\without_jellyfish\all_tiles"  # Folder containing files to move
    TARGET_DIRECTORY = r"R:\LU24A1037-Jellyscope\Jellyscope\Training data\Binary_classifier\Kristineberg_250915_train\with_jellyfish\all_tiles"  # Folder to move files to
    CREATION_DATE = "2026-02-16"  # Format: YYYY-MM-DD
    
    print(f"Moving files created on {CREATION_DATE}...")
    print(f"From: {SOURCE_DIRECTORY}")
    print(f"To:   {TARGET_DIRECTORY}\n")
    
    move_files_by_creation_date(SOURCE_DIRECTORY, TARGET_DIRECTORY, CREATION_DATE)
