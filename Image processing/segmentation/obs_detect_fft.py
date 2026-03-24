#%% 
import os
import glob
import numpy as np
import skimage as ski
import shutil
import matplotlib.pyplot as plt
from PIL import Image

#%% set path
path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data"
monitoring_date = "Kristineberg_250820"
# load_path_obs = os.path.join(path, monitoring_date, "images_with_jellies_manual")
load_path_raw = os.path.join(path, monitoring_date, "raw_data")

#%% Define functions
def calc_fft_mean(img, mask):
    """
    Calculate mean of masked FFT magnitude spectrum for the input image.

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

    # Apply mask and compute mean of non-zero values
    magnitude_spectrum_masked = magnitude_spectrum * mask
    non_zero_values = magnitude_spectrum_masked[magnitude_spectrum_masked != 0]
    
    if non_zero_values.size > 0:
        fft_mean = np.mean(non_zero_values)
    else:
        fft_mean = 0

    return fft_mean

#%% Define input/output paths and parameters
# input_folder = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/raw_data"
# "\RemoteTransfer\Test_bkg\img_20250915_113032.png"
input_folder = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250915/gain10_nobgsub"
output_path = "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250915/gain10_bgsub_sorted"
output_txt = os.path.join(output_path, "observationlog.txt")
output_folder_obs = os.path.join(output_path, "obs")
output_folder_noobs = os.path.join(output_path, "no_obs")

os.makedirs(output_folder_obs, exist_ok=True)
os.makedirs(output_folder_noobs, exist_ok=True)

image_files = glob.glob(os.path.join(input_folder, "*"))

threshold_fft = 1.014  # threshold for FFT mean ratio
fft_mask_r_in = 50
fft_mask_r_out = 700
rolling_window = 100  # number of previous images to consider for rolling median
clip_val = 0.05  # CLAHE clip limit

ind_start = 0  # starting index
ind_end = 7803 #len(image_files)-1  # ending index (exclusive)

# Create mask for FFT
# Mask creation
test_img = ski.io.imread(image_files[0])
rows, cols = test_img.shape
crow, ccol = rows // 2, cols // 2
mask = np.ones((rows, cols), np.uint8)
y, x = np.ogrid[:rows, :cols]
dist_from_center = (x - ccol) ** 2 + (y - crow) ** 2
mask[dist_from_center <= fft_mask_r_in**2] = 0
mask[dist_from_center >= fft_mask_r_out**2] = 0

#%%

ind_plot = 10

### Plot example image and its FFT
# Load image
example_img = ski.io.imread(image_files[ind_plot])  

# Convert to grayscale if needed
if example_img.ndim == 3:
    if example_img.shape[2] == 4:
        example_img = example_img[:, :, :3]  # Strip alpha channel
    if example_img.shape[2] == 3:
        example_img = ski.color.rgb2gray(example_img)

# Apply contrast enhancement
example_img_CLAHE = ski.exposure.equalize_adapthist(example_img, clip_limit=clip_val)  # CLAHE

# Calculate FFT and fft mean
fft_mean_example = calc_fft_mean(example_img_CLAHE, mask) # mean fft value

fft_img = np.fft.fft2(example_img_CLAHE)
fft_shift = np.fft.fftshift(fft_img)
magnitude_spectrum = np.abs(fft_shift)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)

plt.imshow(example_img, cmap='gray')
plt.title('Example Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(magnitude_spectrum)), cmap='gray')
plt.title('FFT Magnitude Spectrum, fft mean: {:.2f}'.format(fft_mean_example))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.log(1 + np.abs(magnitude_spectrum*mask)), cmap='gray')
plt.title('FFT Mask')
plt.axis('off')


#%% Select images with jellyfish based on FFT mean

rolling_fft_mean = []
median_fft_list = []

img_count = 0
ind = 0

with open(output_txt, "w") as f:
    for filename in image_files[ind_start:ind_end]:
        file_path = os.path.join(input_folder, filename)
        
        filename = os.path.basename(file_path)  

        try:
            img = ski.io.imread(file_path)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue
        
        ind += 1
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = img[:, :, :3]  # Strip alpha channel
            if img.shape[2] == 3:
                img = ski.color.rgb2gray(img)

        img = ski.exposure.equalize_adapthist(img, clip_limit=clip_val)  # CLAHE

        # FFT analysis
        fft_mean = calc_fft_mean(img, mask)

        # Update rolling means
        rolling_fft_mean.append(float(fft_mean))

        if len(rolling_fft_mean) > rolling_window:
            rolling_fft_mean.pop(0)

        # Use rolling median (excluding current value)
        if len(rolling_fft_mean) > 1:
            median_fft = np.median(rolling_fft_mean[:-1])
        else:
            median_fft = fft_mean

        median_fft_list.append(float(median_fft))

        line = (
            f"Image {ind}, current observations {img_count} file: {filename}\n"
            f"fft mean: {fft_mean:.3f}, rolling median fft: {median_fft:.3f}, ratio: {fft_mean/median_fft:.5f}\n"
        )
        print(line.strip())

        # Threshold: copy if both FFT and edge mean are above threshold
        if (fft_mean / median_fft > threshold_fft):
            f.write(line)
            
            img_uint8 = (img * 255).astype('uint8')
            img_pil = Image.fromarray(img_uint8, mode='L')  # 'L' = grayscale
            img_pil.save(os.path.join(output_folder, filename))
            
            img_count += 1

np.savetxt(
    "C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250820/images_with_jellies_fft/rolling_medians.csv",
    np.column_stack([median_fft_list]),
    delimiter=",",
    header="median_fft",
    comments=''
)