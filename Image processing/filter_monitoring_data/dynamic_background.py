# %%
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from scipy.ndimage import median_filter
from scipy.signal import convolve2d

# %% Subtract dynamic background
dirname = "F:/Tjarno082024/20240819 evening long/COE-200-M-USB-080-IR-C (J1230000591)"
imagefiles = [f for f in os.listdir(dirname) if f.endswith('.bmp')]
nfiles = len(imagefiles)    # Number of files found

# Load images
nmin = 1
nmax = 20

if nmax == 0:
    nmax = nfiles

images_raw = []

for ii in range(nmin, nmax + 1):
    filename = imagefiles[ii - 1]
    im = cv2.imread(os.path.join(dirname, filename), cv2.IMREAD_GRAYSCALE)
    images_raw.append(im)
    print(f'Loading image {ii}/{nmax}')

# %% Dynamic background subtraction
dr = 2  # Dynamic range
images = []

imh, imw = images_raw[0].shape

for ii in range(nmin + dr, nmax - dr + 1):
    bkg_images = np.array(images_raw[ii - dr - 1:ii + dr])
    bkg = np.median(bkg_images, axis=0)  # Calculate dynamic background
    Ibs = images_raw[ii - 1] - bkg       # Subtract background
    images.append(Ibs)
    print(f'Subtracting background from image {ii - dr}/{nmax - 2 * dr}')

# Plot and process images
color = [1, 1, 1]
nColors = 256
cmap = np.linspace(0, color, nColors)

savedirname = "C:\Users\is5046he\Work Folders\Documents\PhD\JellyScope\Tjarno0824\imageprocessing\subtractbkg"

for i in range(len(images)):
    im = images[i]
    im = im.astype(np.float64) / np.max(im)  # Normalize

    # Apply highpass filter
    threshold = 0.1 * np.max(im)
    im[im < threshold] = 0

    # Apply gamma correction
    gamma = 0.7
    im = exposure.adjust_gamma(im, gamma)

    # Increase contrast using CLAHE
    im = exposure.equalize_adapthist(im)

    # Reduce grain using averaging filter
    kernel = np.ones((5, 5)) / 25.0
    im = convolve2d(im, kernel, mode='same')

    # Reduce noise using median filter
    im = median_filter(im, size=5)

    # Plot image
    plt.imshow(im, cmap='gray')
    plt.title('original')
    plt.axis('off')

    # Save image
    savename = os.path.join(savedirname, f'image{i+1}.png')
    plt.savefig(savename)
    print(f'Processing image {i + 1}/{nmax - 2 * dr}')

    plt.show()
