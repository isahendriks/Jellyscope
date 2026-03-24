#%%

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage as ski
from scipy import ndimage as ndi
import shutil

#%% Functions


def hist_equalization(img):
    """Apply histogram equalization to enhance contrast."""
    img_eq = ski.exposure.equalize_hist(img)
    img_eq = (img_eq * 255).astype(np.uint8)  # Scale back to [0, 255]
    return img_eq

#%%

load_path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_manual"
image_files = glob.glob(os.path.join(load_path, "*"))

#%% Plot image and histogram

index = 20
selected_path = image_files[index] if 0 <= index < len(image_files) else None

if selected_path:
    img = ski.io.imread(image_files[index])
    print("Loaded:", selected_path)
else:
    print(f"No image found for index {index}")

hist_raw, bin_edges = np.histogram(img, bins=64, range=(0, 256))
entropy_raw = ski.measure.shannon_entropy(img)
img=hist_equalization(img)

# Calculate mean intensity from histogram
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
I_mean = np.average(bin_centers, weights=hist_raw)

# Plot image and its histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Show the image
ax1.imshow(img, cmap='gray')
ax1.set_title('Image')
ax1.axis('off')

# Show the histogram
ax2.bar(bin_edges[:-1], hist_raw, width=bin_edges[1]-bin_edges[0], color='gray', align='edge')
ax2.set_title('Histogram, mean pixel intensity: {:.2f}'.format(I_mean))
ax2.set_xlabel('Pixel Intensity')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()


#%% Calculate mean intensity for set of images with histogram and rolling median
# fig, axes = plt.subplots(4, 5, figsize=(15, 12))
# axes = axes.flatten()  # Flatten to 1D for easy indexing


#make txt file to log observations
output_txt = "C:/Users/Admin/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_hist/observations.txt"
with open(output_txt, "w") as f:
    rolling_window = 20 #number of previous images to consider for rolling median
    index = 270 #starting index
    num_obs = 1000 #number of observations to check

    img_count = 0

    rolling_mean = []
    rolling_median = []

    output_folder = "C:/Users/Admin/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_hist"
    os.makedirs(output_folder, exist_ok=True)

    while img_count < num_obs:
        selected_path = find_image_by_index(image_files, index)
        if selected_path:
            img = ski.io.imread(selected_path)

            hist, bin_edges = np.histogram(img, bins=64, range=(0, 256))
            # Calculate mean intensity from histogram
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            I_mean = np.average(bin_centers, weights=hist)

            rolling_mean.append(float(I_mean))
            if len(rolling_mean) > rolling_window:
                rolling_mean.pop(0)

            median_rolling_mean = np.median(rolling_mean[:-1])
            rolling_median.append(float(median_rolling_mean))

            if I_mean > median_rolling_mean*1.021:
                line = f"Observation {img_count} at index: {index} with mean intensity: {I_mean:.3f}, median rolling mean: {median_rolling_mean:.3f}, ratio: {I_mean/median_rolling_mean:.3f}\n"
                print(line.strip())
                f.write(line)
                
                shutil.copy2(selected_path, output_folder)
                # ax = axes[img_count]
                # ax.imshow(img, cmap='gray')
                # ax.set_title(f'Index {index}\n {I_mean/median_rolling_mean:.3f} x rolling median')
                # ax.axis('off')
                img_count += 1

        else:
            print(f"No image found for index {index}")
            
        # if img_count >= len(axes):
        #     break

        index += 1

# plt.tight_layout()
# plt.show()
# %%
