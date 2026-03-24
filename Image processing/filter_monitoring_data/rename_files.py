#%%
import os
import re
import shutil
from datetime import datetime

#%% Define function to get imgage metadata
def get_image_datetime(path):
    """Extract datetime from EXIF, return as string 'YYYYMMDD-HHMMSS' or None."""
    try:
        img = Image.open(path)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTime"):
                    # Format like 2025:08:26 17:30:15 -> 20250826-173015
                    dt_str = value.replace(":", "").replace(" ", "-")
                    return dt_str
    except Exception as e:
        print(f"Could not read EXIF from {path}: {e}")
        return None

#%%
# Folder containing the images
folder_old = "C:\\Users\\Admin\\Documents\\Jellyscope Data\\Monitoring_Data\\Kristineberg_250820\\set2"
folder_new = "C:\\Users\\Admin\\Documents\\Jellyscope Data\\Monitoring_Data\\Kristineberg_250820\\set2_renamed"
# Number to add
offset = 21475

# Regex to match filenames like img_123.png
pattern = re.compile(r"^img_(\d+)\.png$")

filenames = os.listdir(folder_old)

#%% Add offset
for filename in filenames:
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        new_number = number + offset
        new_name = f"img_{new_number}.png"

        old_path = os.path.join(folder_old, filename)
        new_path = os.path.join(folder_new, new_name)

        shutil.copy2(old_path, new_path)
        print(f"Renamed {filename} -> {new_name}")

#%% Change number of digits to constant length

# Folder containing the images
folder_old = "C:\\Users\\Admin\\Documents\\Jellyscope Data\\Monitoring_Data\\Kristineberg_250820\\set1"
folder_new = "C:\\Users\\Admin\\Documents\\Jellyscope Data\\Monitoring_Data\\Kristineberg_250820\\set1_renamed"

# Make sure the new folder exists
os.makedirs(folder_new, exist_ok=True)

# Regex to match filenames like img_123.png
pattern = re.compile(r"^img_(\d+)\.png$")

# First, figure out how many digits we need
numbers = []
for filename in os.listdir(folder_old):
    match = pattern.match(filename)
    if match:
        numbers.append(int(match.group(1)))

if not numbers:
    raise ValueError("No files matching img_<number>.png found!")

max_digits = len(str(max(numbers)))  # width = number of digits of largest number

# Now rename with zero padding
for filename in os.listdir(folder_old):
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        new_name = f"img_{number:0{max_digits}d}.png"

        old_path = os.path.join(folder_old, filename)
        new_path = os.path.join(folder_new, new_name)

        shutil.copy2(old_path, new_path)
        print(f"Copied {filename} -> {new_name}")

# %% Find out date
folder = "C:\\Users\\Admin\\Documents\\Jellyscope Data\\Monitoring_Data\\Kristineberg_250820\\pictures_with_jellies"
img_ind = "04036"
img_path = os.path.join(folder, f"img_{img_ind}.png")

mtime = os.path.getmtime(img_path)
dt_str = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
print(dt_str)

# %% Rename with date

folder = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_manual"
pattern = re.compile(r"^img_(\d+)\.png$")

for filename in os.listdir(folder):

    img_path = os.path.join(folder, filename)

    mtime = os.path.getmtime(img_path)
    dt_str = datetime.fromtimestamp(mtime).strftime("%Y%m%d-%H%M%S")

    new_name = f"img_{dt_str}.png"
    path_new = os.path.join(folder, new_name)
    path_old = os.path.join(folder, filename)
    os.rename(path_old, path_new)
    print(f"Renamed {filename} -> {new_name}")
# %%
