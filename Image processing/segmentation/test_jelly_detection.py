#%% Import packages
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.measure import label, regionprops
from skimage import feature
from skimage.transform import downscale_local_mean
from skimage.segmentation import chan_vese

#%% Define functions
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

#%% Define path

path = "C:\\Users\\is5046he\\Work Folders\\Documents\\PhD\\JellyScope\\TrainingData\\Unlabeled\\Tjarnö_082024_Lab"
image_files = glob.glob(os.path.join(path, "*.bmp"))

#%% Show image
image = normalize_image(plt.imread(image_files[0]))
downscaled_image = downscale_local_mean(image, (3, 3))
print(image.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(downscaled_image, cmap="gray")
ax[1].set_title("Downscaled Image")
ax[1].axis("off")

# %% Treshold, erode, dilate
thresholded_lim = 0.1
disk_size = 20/3

thresholded_image = downscaled_image > thresholded_lim
eroded_image = binary_erosion(thresholded_image, disk(disk_size))
dilated_image = binary_dilation(eroded_image, disk(disk_size))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(dilated_image, cmap="gray")
ax[1].set_title("Thresholded Image")
ax[1].axis("off")

# %% edge detection with canny 
from scipy import ndimage as ndi
sigma_canny = 1  # Adjust sigma for different edge sensitivity
disk_size = 10
edges = feature.canny(downscaled_image, sigma=sigma_canny)  # Adjust sigma for different edge sensitivity
fill_blobs = ndi.binary_fill_holes(edges)

# erode dilate
# edges_image = edges.astype(np.uint8)
# eroded_image = binary_erosion(thresholded_image, disk(disk_size))
# dilated_image = binary_dilation(eroded_image, disk(disk_size))

# Plot original and edge-detected image
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(edges, cmap="gray")
ax[1].set_title("Canny Edge Detection")
ax[1].axis("off")

ax[2].imshow(fill_blobs, cmap="gray")
ax[2].set_title("Eroded + dilated Image")
ax[2].axis("off")

#%% Chan-vese segmentation
cv = chan_vese(
    downscaled_image,
    mu=0.25,
    lambda1=1,
    lambda2=1,
    tol=1e-4,
    max_num_iter=500,
    dt=0.2,
    init_level_set="checkerboard",
    extended_output=True,
)

# %%Plot original and edge-detected image
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
ax = axes.flatten()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(cv[2])} iterations'
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()
# %% Get ROIs
blobs, num_blobs = label(dilated_image, return_num=True)
props = regionprops(blobs)
positions = np.array([prop.centroid for prop in props])
bboxs = np.array([prop.bbox for prop in props])
crops = [downscaled_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] for bbox in bboxs]

print(num_blobs)

#%% Plot bounding boxes

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(downscaled_image, cmap='gray')
ax.set_title("Bounding Boxes")
ax.axis("off")

for bbox in bboxs:
    min_row, min_col, max_row, max_col = bbox
    width = max_col - min_col
    height = max_row - min_row
    rect = patches.Rectangle((min_col, min_row), width, height, 
                              linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

plt.show()


#%% plot ROIs
ind=0
hor_windows = 6 # number of horizontal windows
ver_windows = np.ceil(len(crops)/hor_windows).astype(int) # number of vertical windows

fig, axs = plt.subplots(ver_windows, hor_windows, figsize=(16, 8))
for ax in axs.ravel():
    ROI = crops[ind]
    ax.imshow(ROI, cmap='gray')
    ax.set_title(f"ROI #{ind}", fontsize=20)
    ind=ind+1
plt.tight_layout()
plt.show()