#%% Import packages
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage as ski
from scipy import ndimage as ndi
from skimage.morphology import dilation, disk
from shapely.geometry import box
from shapely.ops import unary_union


#%% Define Functions
def preprocess_image(image, f_s=1):
    # Preprocessing of image

    image = ski.transform.pyramid_reduce(image, downscale=f_s) # Downsample image
    image = image / np.max(image)
    image = ski.filters.gaussian(image, sigma=1) # Gaussian filter to smooth the image
    image = ski.exposure.equalize_adapthist(image, clip_limit=0.01) # CLAHE to enhance contrast
    image = ski.filters.unsharp_mask(image, radius=1, amount=1) # Sharpen image

    image = (image * 255).astype(np.uint8) # Convert to uint8 for compatibility with watershed algorithm

    return image

def watershed_jellies(image):
    """ 
    Region based segmenation using watershed algorithm

    Parameters
    ----------
    image : array_like
        Input image to be segmented.
    
    Returns
    -------
    labels : array_like
        Binary image with segmenteded regions
    """
    hist, hist_centers = ski.exposure.histogram(image)

    # Calculate cumulative distribution from the histogram
    dist = np.cumsum(hist) / np.sum(hist)

    # Find lower and upper intensity thresholds (e.g., 1% and 99%)
    ind_low = np.searchsorted(dist, 0.01)
    ind_high = np.searchsorted(dist, 0.99)

    I_low = hist_centers[ind_low]
    I_high = hist_centers[ind_high]

    # Find elevation map
    elevation_map = ski.filters.sobel(image)

    # Find markers of the background and jellies based on the extreme parts of the histogram of gray values.
    markers = np.zeros_like(image)
    markers[image < I_low] = 1
    markers[image > I_high] = 2

    segmentation_image = ski.segmentation.watershed(elevation_map, markers)

    return segmentation_image>1

#%% Load image
species  = "Pleurobrachia pileus"

# load_path = "C:\\Users\\is5046he\\Work Folders\\Documents\\PhD\\JellyScope\\TrainingData\\Unlabeled\\"
load_path = "F:\\Kristineberg0425\\" 
image_files = glob.glob(os.path.join(load_path, species, "*"))

# im_rand_ind = random.randint(0, len(image_files)-1)
im_ind = 5
f_s = 2 # Downscale factor
dilation_factor = 30 # Dilation factor for bounding boxes

im_test = plt.imread(image_files[im_ind]) # Read image

im_preprocessed = preprocess_image(im_test, f_s)
im_segmented = watershed_jellies(im_preprocessed)

#%% Plot segmentation results
# Plot segmentation results
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
ax = axes.flatten()

ax[0].imshow(im_test, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(im_segmented, cmap='gray')
ax[1].set_title("segmented Image")
ax[1].axis("off")

ax[2].imshow(im_segmented_rmsmall, cmap='gray')
ax[2].set_title("small obvjects removed")
ax[2].axis("off")

ax[3].imshow(dilated_segmented, cmap="gray")
ax[3].set_title("dilated segmented Image")
ax[3].axis("off")

#%% Plot bounding boxes
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(im_preprocessed, cmap='gray')
ax.set_title("Bounding Boxes")
ax.axis("off")

for bbox in bboxs:
    min_row, min_col, max_row, max_col = bbox
    width = max_col - min_col
    height = max_row - min_row

    # Draw the rectangle
    rect = patches.Rectangle((min_col, min_row), width, height, 
                            linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

#%% Get bounding boxes

# Step 1: Remove small objects
im_segmented_rmsmall = ski.morphology.remove_small_objects(im_segmented, 100)

# Step 2: Dilate the segmented regions
dilated_segmented = dilation(im_segmented_rmsmall, disk(dilation_factor))

# Step 3: Label connected components
blobs, _ = ski.measure.label(dilated_segmented, return_num=True)
props = ski.measure.regionprops(blobs)

# Step 4: Extract bounding boxes
bboxs = []
for prop in props:
    min_row, min_col, max_row, max_col = prop.bbox
    bboxs.append((min_row, min_col, max_row, max_col))

# Step 5: Make bounding boxes square
final_bboxs = []
for bbox in bboxs:
    min_row, min_col, max_row, max_col = bbox
    height = max_row - min_row
    width = max_col - min_col

    # Make the box square by expanding the smaller dimension
    size = max(height, width)
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    half_size = size // 2

    # Calculate new square bounding box
    new_min_row = max(0, center_row - half_size)
    new_max_row = min(im_segmented.shape[0], center_row + half_size)
    new_min_col = max(0, center_col - half_size)
    new_max_col = min(im_segmented.shape[1], center_col + half_size)

    # Append the adjusted square bounding box
    final_bboxs.append((new_min_row, new_min_col, new_max_row, new_max_col))