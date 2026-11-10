#%% Import packages
import pandas as pd
import os

faro_folder = "E:\\jellyscope_preprocessed_frames"
kristineberg_folder = "R:\\LU24A1037-Jellyscope\\Jellyscope\\Monitoring data\\Kristineberg_260725_260911"

#%% Load data

### Faro
faro_subfolders = [f for f in os.listdir(faro_folder) if os.path.isdir(os.path.join(faro_folder, f))]

file_counts = []
for subfolder in faro_subfolders:
    print(f"Processing subfolder: {subfolder}")
    subfolder_path = os.path.join(faro_folder, subfolder)
    file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
    file_counts.append(file_count)
    print(f"Number of files in subfolder {subfolder}: {file_count}")

print(f"Number of files in Faro folder: {sum(file_counts)}")


#%% Kristineberg
kristineberg_subfolders = [f for f in os.listdir(kristineberg_folder) if os.path.isdir(os.path.join(kristineberg_folder, f))]

file_counts = []
for subfolder in kristineberg_subfolders:
    print(f"Processing subfolder: {subfolder}")
    subfolder_path = os.path.join(kristineberg_folder, subfolder)
    file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
    file_counts.append(file_count)
    print(f"Number of files in subfolder {subfolder}: {file_count}")

print(f"Number of files in Kristineberg folder: {sum(file_counts)}")
