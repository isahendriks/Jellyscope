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
    image = ski.filters.gaussian(image, sigma=1) # Gaussian filter to smooth the image
    image = ski.exposure.equalize_adapthist(image, clip_limit=0.01) # CLAHE to enhance contrast
    image = ski.filters.unsharp_mask(image, radius=1, amount=1) # Sharpen image

    image = (image * 255).astype(np.uint8) # Convert to uint8 for compatibility with watershed algorithm

    return image

def canny_jellies(image):

    edges = ski.feature.canny(image, sigma=4)  # Adjust sigma for different edge sensitivity
    
    closed_edges = ski.morphology.closing(edges, ski.morphology.disk(50)) # Apply morphological closing to close gaps
    fill_blobs = ndi.binary_fill_holes(closed_edges)

    return fill_blobs

def watershed_jellies(image, dilation_factor=30):
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

    # Remove small objects
    im_segmented_rmsmall = ski.morphology.remove_small_objects(segmentation_image, 200)

    # Dilate the segmented regions
    dilated_segmented = dilation(im_segmented_rmsmall, disk(dilation_factor))

    return dilated_segmented>1

def get_bbox(im_segmented):
    """
    Extract bounding boxes after dilating the segmented regions, merge overlapping boxes,
    and make them square with an added margin.

    Parameters:
    ----------
    im_segmented : array_like
        Binary segmented image.
    margin : float
        Fraction of the bounding box size to add as a margin (default: 10%).
    dilation_factor : int
        Radius of the structuring element used for dilation (default: 5).

    Returns:
    -------
    bboxs : list
        List of adjusted bounding boxes.
    """
    # Step 1: Label connected components
    blobs, _ = ski.measure.label(im_segmented, return_num=True)
    props = ski.measure.regionprops(blobs)

    # Step 2: Extract bounding boxes
    bboxs = []
    for prop in props:
        min_row, min_col, max_row, max_col = prop.bbox
        bboxs.append((min_row, min_col, max_row, max_col))

    # Step 3: Append bounding boxes to a list
    final_bboxs = []
    for bbox in bboxs:
        min_row, min_col, max_row, max_col = bbox

        # Append the adjusted square bounding box
        final_bboxs.append((min_row, min_col, max_row, max_col))

    return final_bboxs

def get_ROIs(image, bboxs, f_s=1):
    """
    Extracts Regions of Interest (ROIs) from the image based on the provided bounding boxes.

    Parameters:
    ----------
    image : array_like
        Input image from which ROIs will be extracted.
    bboxs : list of tuples
        List of bounding boxes, each defined as (min_row, min_col, max_row, max_col).

    Returns:
    -------
    crops : list of array_like
        List of cropped ROIs extracted from the image.
    """

    bboxs_original = [
        (
            int(bbox[0] * f_s),  # Scale min_row
            int(bbox[1] * f_s),  # Scale min_col
            int(bbox[2] * f_s),  # Scale max_row
            int(bbox[3] * f_s)   # Scale max_col
        )
        for bbox in bboxs]
    
    crops = [image[bbox[0]:bbox[2], bbox[1]:bbox[3]] for bbox in bboxs_original] # extract ROIs

    return crops

def save_crops(crops, output_folder, prefix="crop"):
    """
    Save all crops in a list as PNG files in a specified folder.

    Parameters:
    ----------
    crops : list of array_like
        List of cropped images to save.
    output_folder : str
        Path to the folder where the crops will be saved.
    prefix : str
        Prefix for the saved file names (default: "crop").
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save each crop as a PNG file
    for i, crop in enumerate(crops):
        output_path = os.path.join(output_folder, f"{prefix}_{i+1}.png")
        ski.io.imsave(output_path, crop.astype(np.uint8))  # Scale to 0-255 if needed
        print(f"Saved: {output_path}")

def is_focus(image, edge_threshold= 300):
    """
    Quickly check if an image is in focus using the variance of the Laplacian.

    Parameters:
    ----------
    image : array_like
        Input image to check.
    threshold : float
        Minimum variance of the Laplacian required to consider the image in focus.

    Returns:
    -------
    in_focus : bool
        True if the image is in focus, False otherwise.
    edge_count : float
        Number of edges, used as a measure of focus
    """
    # Apply Canny edge detection
    edges = ski.feature.canny(image, sigma=1)

    # Count the number of edges
    edge_count = np.sum(edges)

    # Check if the edge count exceeds the threshold
    in_focus = edge_count > edge_threshold

    return [in_focus, edge_count]

def plot_ROIs(crops):
    # Dynamically calculate grid size based on the number of crops
    num_crops = len(crops)
    hor_windows = int(np.ceil(np.sqrt(num_crops)))  # Number of horizontal windows (columns)
    ver_windows = int(np.ceil(num_crops / hor_windows))  # Number of vertical windows (rows)

    # Create the grid and plot the ROIs
    fig, axs = plt.subplots(ver_windows, hor_windows, figsize=(16, 8))

    # Flatten the axes array for easier iteration
    if num_crops == 1:
        axs = [axs]  # Wrap single Axes object in a list
    else:
        axs = axs.ravel()  # Flatten the array of Axes for easier iteration

    # Plot each ROI
    for ind, ax in enumerate(axs):
        if ind < num_crops:
            ROI = crops[ind]
            ax.imshow(ROI, cmap='gray')
            ax.set_title(f"ROI #{ind+1}", fontsize=10)
        else:
            ax.axis("off")  # Turn off unused subplots

    plt.tight_layout()
#%% Define path to images
<<<<<<< Updated upstream
species = "leucartiaria_octona"
=======
# measurement = "Tjarnö_082024_lab"
>>>>>>> Stashed changes
# measurement = "Kristineberg_022025_Jellies"
species  = "Pleurobrachia pileus"
species = "bolinopis infundibulum"

<<<<<<< Updated upstream
load_path ="C:\\Users\\Admin\\Documents\\Jellyscope Data\\Trainingdata"
=======
# load_path = "C:\\Users\\is5046he\\Work Folders\\Documents\\PhD\\JellyScope\\TrainingData\\Unlabeled\\"
load_path = "F:\\Kristineberg0425\\" 
>>>>>>> Stashed changes
image_files = glob.glob(os.path.join(load_path, species, "*"))

edge_lim = 100

#%% Test image preprocessing and segmentation
<<<<<<< Updated upstream
im_test = plt.imread(image_files[7]) # Read image
f_s = 5 #Downsampling factor
=======

# im_rand_ind = random.randint(0, len(image_files)-1)
im_ind = 5

im_test = plt.imread(image_files[im_ind]) # Read image
f_s = 3 #Downsampling factor
>>>>>>> Stashed changes

# preprocess image and apply both canny and watershed segmentation
im_preprocessed = preprocess_image(im_test, f_s)
im_segmented_watershed = watershed_jellies(im_preprocessed)
# im_segmented_canny = canny_jellies(im_preprocessed)

# Plot segmentation results
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
ax = axes.flatten()

ax[0].imshow(im_test, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(im_preprocessed, cmap='gray')
ax[1].set_title("preprocessed Image")
ax[1].axis("off")

# ax[2].imshow(im_segmented_canny, cmap='gray')
ax[2].set_title("Edge detection")
ax[2].axis("off")

ax[3].imshow(im_segmented_watershed, cmap="gray")
ax[3].set_title("Region detection")
ax[3].axis("off")

# %% Plot bounding boxes

# # get bounding boxes
# bboxs = get_bbox(im_segmented_watershed)

# # Get ROIs
# crops = get_ROIs(im_test, bboxs, f_s)
# focus = [is_focus(crop) for crop in crops]  # Check focus for each crop

f_s = 3 # Downsampling factor

for i in [1083]: #range(len(image_files)):
    print(f"image {i}/{len(image_files)}")
    im_test = plt.imread(image_files[i]) # Read image
    print(f"Pixel range raw image: {im_test.min()} to {im_test.max()}")

    # preprocess image and apply both canny and watershed segmentation
    im_preprocessed = preprocess_image(im_test, f_s)
<<<<<<< Updated upstream
    im_segmented = canny_jellies(im_preprocessed)
=======
    print(f"Pixel range after preprocessing: {im_preprocessed.min()} to {im_preprocessed.max()}")
>>>>>>> Stashed changes

    im_segmented = watershed_jellies(im_preprocessed)
    print(f"Pixel range after segmentation: {im_segmented.min()} to {im_segmented.max()}")
    plt.imshow(im_segmented, cmap='gray')
    # Get ROIs
    bboxs = get_bbox(im_segmented)

    # plot ROIs
    crops = get_ROIs(im_test, bboxs, f_s)
    print(f"Pixel range after cropping: {crops[1].min()} to {crops[1].max()}")

    crops_preprocessed = get_ROIs(im_preprocessed, bboxs, 1)
    focus = [is_focus(crop, edge_lim) for crop in crops_preprocessed] # Check focus for each crop on preprocessed image

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im_preprocessed, cmap='gray')
    ax.set_title("Bounding Boxes")
    ax.axis("off")

    for bbox, (in_focus, focus_measure) in zip(bboxs, focus):
        min_row, min_col, max_row, max_col = bbox
        width = max_col - min_col
        height = max_row - min_row

        # Set color based on focus status
        color = 'green' if in_focus else 'red'

        # Draw the rectangle
        rect = patches.Rectangle((min_col, min_row), width, height, 
                                linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()

#%% Check focus of images

f_s = 3 # Downsampling factor
for i in [1083]:
    im_test = plt.imread(image_files[i]) # Read image

    # preprocess image and apply both canny and watershed segmentation
    im_preprocessed = preprocess_image(im_test, f_s)
    im_segmented_watershed = watershed_jellies(im_preprocessed)

    bboxs = get_bbox(im_segmented_watershed)
    crops = get_ROIs(im_test, bboxs, f_s)
    crops_preprocessed = get_ROIs(im_preprocessed, bboxs, 1)
    focus = [is_focus(crop, edge_lim) for crop in crops_preprocessed] # Check focus for each crop on preprocessed image

    num_crops = len(crops)
    hor_windows = int(np.ceil(np.sqrt(num_crops)))  # Number of horizontal windows (columns)
    ver_windows = int(np.ceil(num_crops / hor_windows))  # Number of vertical windows (rows)

    # Create the grid and plot the ROIs
    fig, axs = plt.subplots(ver_windows, hor_windows, figsize=(16, 8))

    # Flatten the axes array for easier iteration
    if num_crops == 1:
        axs = [axs]  # Wrap single Axes object in a list
    else:
        axs = axs.ravel()  # Flatten the array of Axes for easier iteration

    # Plot each ROI
    for ind, ax in enumerate(axs):
        if ind < num_crops:
            ROI = crops[ind]
            ax.imshow(ROI, cmap='gray')
            ax.set_title(f"ROI #{ind+1}, focus = {focus[ind]}", fontsize=10)
        else:
            ax.axis("off")  # Turn off unused subplots

    plt.tight_layout()
    plt.show()

# %% Loop through all images and save ROIs

f_s = 3 # Downsampling factor
save_path = "F:\\Kristineberg0425\\Segmented\\" 
species = "Pleurobrachia pileus2"
method="pleorobrachiapileus"

output_folder = os.path.join(save_path, species)

im_indxs = np.linspace(0, len(image_files)-1, 100).astype(int) # Randomly select images to process

crops_all = [] # Initialize empty list to store crops

for i in range(len(image_files)):
    im_path = image_files[i]

    im = plt.imread(im_path) # Read image

    # preprocess image and apply both canny and watershed segmentation
    im_preprocessed = preprocess_image(im, f_s)

    # Plot image
    plt.imshow(im_preprocessed, cmap='gray')
    plt.title( f"preprocessed image {i}/{len(image_files)}")
    plt.show()

    im_segmented = watershed_jellies(im_preprocessed)

    bboxs = get_bbox(im_segmented)
    
    # Get ROIs
    crops_preprocessed = get_ROIs(im_preprocessed, bboxs, 1) # Cut crops out of the preprocessed image
    crops = get_ROIs(im, bboxs, f_s) # Cut crops out of the original image (so not the preprocessed one)

    # Check focus for each crop
    focus = [is_focus(crop, edge_lim)[0] for crop in crops_preprocessed] # check focus for each crop on preprocessed image
    print(f"Focus results: {focus}")

    # Filter crops that are in focus
    crops_in_focus = [crop for crop, in_focus in zip(crops, focus) if in_focus]

    save_crops(crops_in_focus, output_folder, prefix=f"{method}_crop_{i+1}")
# %%
