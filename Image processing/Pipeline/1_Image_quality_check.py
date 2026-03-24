
#### PREPARING BATCHES WITH QUALITY CHECKED IMAGES ####

### PACKAGRES ####
import yaml
from pathlib import Path
import os 
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import sys

#============================================================================
#### GET INPUT DATA FROM CONFIG FILE ####
config_path = Path(sys.argv[1]).resolve()
#config_path =  ""

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

input_folder = Path(config["input_folder"]).resolve() 
batch_size = config['batch_size']
output_folder = Path(config["output_folder"]).resolve()

#============================================================================
#### FUNCTIONS ####

def collect_image_files(input_dir, date = True):
    """
    Collect all image files recursively from input directory. 
    
    Arguments
    - input_dir: Path to the input directory containing images.
    - date: when date is included in the file name sort by date and time

    Returns:
    - list of files with extentions: ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"
    """
    
    files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]
    
    ordered_files = []
    if date:
        # Sort files by date in filename (assuming format includes img_YYYYMMDD_HHMMSS_XXX.png)
        ordered_files = sorted(files)
    else:
        ordered_files = files
    
    return ordered_files


def check_files(file_list):
    """
    Quality check on images

    Arguments:
    - file_list: list of image files

    Returns
    - valid files
    """
    valid_files = []
    for filename in tqdm(file_list, desc="Quality checking images"):
        try:
            with Image.open(filename) as img:
                img.verify()
            valid_files.append(filename)
        except Exception:
            continue
    return valid_files


def create_batches(file_list, split_size):
    """ 
    Split a list of valid image files into batches

    Arguments:
    - file_list: list of valid files 
    - split_size: size of the batches

    """
    return [
        file_list[i : i + split_size]
        for i in range(0, len(file_list), split_size)
    ]

#============================================================================
### MAIN CODE ####

# make sure output directories exist
os.makedirs(output_folder, exist_ok=True)

# collect all images in input folder
all_images = collect_image_files(input_folder)

# check image integrity
valid_files = check_files(all_images)

# log invalid files
invalid_files = len(all_images) - len(valid_files)
print(f"Found {invalid_files} invalid files out of {len(all_images)} total files.")

invalid_file_list = set(all_images) - set(valid_files)
# save invalid files log
if invalid_files > 0:
    invalid_log_path = os.path.join(output_folder, 'logs', 'quality_logs')
    os.makedirs(invalid_log_path, exist_ok=True)
    with open(os.path.join(invalid_log_path, "invalid_files.txt"), "w") as f:
        for item in invalid_file_list:
            f.write(f"{item}\n")

# create batches
batches = create_batches(valid_files, split_size=batch_size)

# create logs
summary_rows = []
for batch_idx, batch in enumerate(tqdm(batches, desc="Creating batches")):
    
    # Log batch summary
    summary_rows.append({
        "q_batch": batch_idx,
        "total_files_in_batch": len(batch)
    })

    # Optional: save detailed valid files per batch
    log_rows = []
    for file in batch:
        log_rows.append({
            "q_batch": batch_idx,
            "full_path": file,
            "base_name": os.path.basename(file)
        })

    log_df = pd.DataFrame(log_rows)
    save_to = os.path.join(output_folder, 'logs', 'quality_logs')
    os.makedirs(save_to, exist_ok=True)
    log_df.to_csv(
        os.path.join(save_to, f"q_batch_{batch_idx:03}_log.csv"),
        index=False
    )

# Save overall summary
summary_df = pd.DataFrame(summary_rows)
save_to = os.path.join(output_folder, 'logs', 'summaries')
os.makedirs(save_to, exist_ok=True)
summary_df.to_csv(os.path.join(save_to, "q_log_summary.csv"), index=False)

print("Quality check completed.")


