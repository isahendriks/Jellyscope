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
save_path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_hist_eq"
os.makedirs(save_path, exist_ok=True)

for index in range(len(image_files)):
    selected_path = image_files[index] if 0 <= index < len(image_files) else None
    filename = os.path.basename(selected_path)

    if selected_path:
        img = ski.io.imread(image_files[index])
        print("Loaded:", filename)
    else:
        print(f"No image found for index {index}")

    img=hist_equalization(img)

    #plt.imshow(img, cmap='gray')

    plt.imsave(save_path + '/' +filename,img, cmap='gray')
