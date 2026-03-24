#%%

import os
import glob
import numpy as np
import skimage as ski
from skimage import filters
import shutil
import matplotlib.pyplot as plt

#%% Functions

<<<<<<< HEAD
def calc_fft_mean(img, r_in=70, r_out=1000):
    """
    Calculate mean of masked FFT magnitude spectrum for the input image.
=======
>>>>>>> 4836dcc77d3d25889e8605c824b5276175d48c30

    Parameters:
    - img: 2D numpy array (image after CLAHE)
    - r_in: inner radius of circular mask
    - r_out: outer radius of circular mask

    Returns:
    - fft_mean: mean of all non-zero masked FFT magnitude values
    """
    # FFT
    fft_img = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_shift)

    # Mask creation
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = (x - ccol) ** 2 + (y - crow) ** 2
    mask[dist_from_center <= r_in**2] = 0
    mask[dist_from_center >= r_out**2] = 0

    # Apply mask and compute mean of non-zero values
    magnitude_spectrum_masked = magnitude_spectrum * mask
    non_zero_values = magnitude_spectrum_masked[magnitude_spectrum_masked != 0]
    
    if non_zero_values.size > 0:
        fft_mean = np.mean(non_zero_values)
    else:
        fft_mean = 0

    return fft_mean
#%%

load_path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250915/gain10_nobgsub"
image_files = glob.glob(os.path.join(load_path, "*"))

output_path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250915/gain10_bgsub_sorted"
output_txt = os.path.join(output_path, "observationlog.txt")
output_folder_obs = os.path.join(output_path, "obs")
output_folder_noobs = os.path.join(output_path, "no_obs")

#%% Load images

index = 100
selected_path = image_files[index] if 0 <= index < len(image_files) else None

if selected_path:
    img = ski.io.imread(selected_path)
    print("Loaded:", selected_path)
else:
    print(f"No image found for index {index}")

# Plot image and FFT space
fig, axes = plt.subplots(3,3, figsize=(10, 4))
axes = axes.flatten()

for index,ax in enumerate(axes):

    selected_path = image_files[index] if 0 <= index < len(image_files) else None
    if selected_path:
        img = ski.io.imread(selected_path)
        print("Loaded:", selected_path)
    else:
        print(f"No image found for index {index}")

    # FFT analysis
    fft_img = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_shift)
    fft_mean = np.mean(magnitude_spectrum)

    # Edge detection (Sobel)
    edges = filters.sobel(img)
    edge_mean = np.mean(edges)

    img_eq=hist_equalization(img) #histogram equalization
    ax.axis('off') # Turn off unused subplots

    ax.imshow(img_eq, cmap='gray')
    ax.set_title('Mean: {:.2f}'.format(np.mean(img)) + 'Entropy: {:.2f}'.format(ski.measure.shannon_entropy(img)) )


plt.tight_layout()
plt.show()


#%% Calculate mean intensity for set of images with histogram and rolling median and plot
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()  # Flatten to 1D for easy indexing

rolling_window = 20 #number of previous images to consider for rolling median
index = 0 #starting index
# num_obs = 20 #number of observations to check

img_count = 0

rolling_mean = []
rolling_median = []

output_folder = "C:/Users/Admin/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_hist"
os.makedirs(output_folder, exist_ok=True)

while img_count <=len(image_files):
    selected_path = image_files[index] if 0 <= index < len(image_files) else None
    if selected_path:
        img = ski.io.imread(selected_path)

        # FFT analysis
        fft_img = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft_img)
        magnitude_spectrum = np.abs(fft_shift)
        fft_mean = np.mean(magnitude_spectrum)

        # Edge detection (Sobel)
        edges = filters.sobel(img)
        edge_mean = np.mean(edges)

        line = f"Observation {img_count} at index: {index} with fft mean: {fft_mean:.3f}, edge mean: {edge_mean:.3f}"
        print(line.strip())

        ax = axes[img_count]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Index {index}\n fft mean: {fft_mean:.2f}\n edge mean: {edge_mean:.4f}')
        ax.axis('off')

        img_count += 1

    else:
        print(f"No image found for index {index}")

    index += 1

# plt.tight_layout()
# plt.show()
# %%


#%% Detect jellies based on FFT and edge detection with rolling median threshold, copy to new folder
path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/"
input_folder = "raw_data" #"images_with_jellies_manual"

input_path = os.path.join(path, input_folder)

image_files = glob.glob(os.path.join(input_path, "*"))

rolling_window = 20  # number of previous images to consider for rolling median
ind_start = 0  # starting index
ind_end = 10 #len(image_files)-1  # ending index (exclusive)


rolling_fft_mean = []
rolling_entr_mean = []
median_fft_list = []
median_entr_list = []

img_count = 0
im_index = 0

for filename in image_files[ind_start:ind_end]:
    im_index += 1
    file_path = os.path.join(input_folder, filename)
    try:
        img = ski.io.imread(file_path)
    except Exception as e:
        print(f"Could not read {filename}: {e}")
        continue

    img=hist_equalization(img) #histogram equalization

    # FFT analysis
    fft_img = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_shift)
    fft_mean = np.mean(magnitude_spectrum)

    # Edge detection (Sobel)
    entropy = ski.measure.shannon_entropy(img)

    # Update rolling means
    rolling_fft_mean.append(float(fft_mean))
    rolling_entr_mean.append(float(entropy))
    if len(rolling_fft_mean) > rolling_window:
        rolling_fft_mean.pop(0)
        rolling_entr_mean.pop(0)

    # Use rolling median (excluding current value)
    if len(rolling_fft_mean) > 1:
        median_fft = np.median(rolling_fft_mean[:-1])
        median_entr = np.median(rolling_entr_mean[:-1])
    else:
        median_fft = fft_mean
        median_entr = entropy


    plt.imshow(img, cmap='gray')
    plt.title(f'Index {im_index}\n fft mean: {fft_mean/median_fft:.2f}\n edge mean: {entropy/median_entr:.4f}')
    plt.axis('off')
    plt.show()


    # Threshold: copy if both FFT and edge mean are above threshold
    # if (fft_mean > median_fft * 1.014) and (edge_mean > median_entr * 1.001):
    #     f.write(line)
    #     plt.imsave(os.path.join(output_folder, filename), img, cmap='gray')
    #     img_count += 1

#%% Detect jellies based on FFT and edge detection with rolling median threshold, copy to new folder

input_folder = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/raw_data"
output_folder = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_fft2"
output_txt = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_fft2/observationlog.txt"

os.makedirs(output_folder, exist_ok=True)
image_files = glob.glob(os.path.join(input_folder, "*"))

rolling_window = 20  # number of previous images to consider for rolling median
ind_start = 0  # starting index
ind_end = len(image_files)-1  # ending index (exclusive)


rolling_fft_mean = []
rolling_entr_mean = []
median_fft_list = []
median_entr_list = []

img_count = 0

image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.png')])

with open(output_txt, "w") as f:
    for filename in image_files[ind_start:ind_end]:
        file_path = os.path.join(input_folder, filename)
        try:
            img = ski.io.imread(file_path)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue

        img=hist_equalization(img) #histogram equalization

        # FFT analysis
        fft_img = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft_img)
        magnitude_spectrum = np.abs(fft_shift)
        fft_mean = np.mean(magnitude_spectrum)

        # Edge detection (Sobel)
        entropy = ski.measure.shannon_entropy(img)

        # Update rolling means
        rolling_fft_mean.append(float(fft_mean))
        rolling_entr_mean.append(float(entropy))
        if len(rolling_fft_mean) > rolling_window:
            rolling_fft_mean.pop(0)
            rolling_entr_mean.pop(0)

        # Use rolling median (excluding current value)
        if len(rolling_fft_mean) > 1:
            median_fft = np.median(rolling_fft_mean[:-1])
            median_entr = np.median(rolling_entr_mean[:-1])
        else:
            median_fft = fft_mean
            median_entr = entropy

        median_fft_list.append(float(median_fft))
        median_entr_list.append(float(median_entr))

        line = (
            f"Observation {img_count} file: {filename}\n"
            f"  fft mean: {fft_mean:.3f}, rolling median fft: {median_fft:.3f}, ratio: {fft_mean/median_fft:.5f}\n"
            f"  edge mean: {entropy:.5f}, rolling median edge: {median_entr:.5f}, ratio: {edge_mean/median_entr:.5f}\n"
        )
        print(line.strip())

        # Threshold: copy if both FFT and edge mean are above threshold
        if (fft_mean > median_fft * 1.014) and (entropy > median_entr * 1.001):
            f.write(line)
            plt.imsave(os.path.join(output_folder, filename), img, cmap='gray')
            img_count += 1

np.savetxt(
    "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_fft/rolling_medians.csv",
    np.column_stack([median_fft_list, median_entr_list]),
    delimiter=",",
    header="median_fft,median_edge",
    comments=''
)

# %%
