
#### SEGMENTS BATCH OF IMAGES ####

### PACKAGRES ###
import yaml
import sys
from pathlib import Path
import os 
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
import time
import glob

#============================================================================
#### GET INPUT DATA FROM CONFIG FILE ####
config_path = Path(sys.argv[1]).resolve()
#config_path = "" # for manual runs

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

output_folder = Path(config["output_folder"]).resolve()
fraction = config['fraction']
min_area = config['min_area']
margin = config['margin']
save_crop = config['save_crop']
num_examples_per_batch = config['num_examples_per_batch']
save_box = config['save_box']
save_steps = config['save_steps']

#============================================================================
#### FUNCTIONS ####

def get_files_from_log(log_file):
    
    """
    Get list of image files from a log file of checked images.
    Arguments:
    - log_files: path to the log file
    Returns:
    - file_list: list of valid image file paths
    """

    df_log = pd.read_csv(log_file)
    
    file_list = []
    for filename in df_log['full_path']:
        file_list.append(filename)  

    return file_list

def fraction_batch_for_bg_removal(file_list, fraction =0.1):
    total_files = len(file_list)
    batch_size = max(1, int(total_files * fraction))  # ensure at least 1 item per batch
    batches = {}

    for i in range(0, total_files, batch_size):
        batch_idx = i // batch_size
        batches[batch_idx] = file_list[i:i+batch_size]

    return batches

def compute_median_background(image_list):
    """
    Takes a list of N images (all same size) and returns
    the pixel-wise median image.
    """

    imgs = []
    for path in image_list:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE) 
        imgs.append(img.astype(np.float32))

    # convert all to float32 and stack them
    stack = np.stack(imgs, axis=0)
    
    # pixel-wise median
    median_background = np.median(stack, axis=0)
    
    # convert back to uint8
    return median_background.astype(np.uint8)

def subtract_background(img, background, method="subtract", eps=1e-6):
    """
    Subtracts or divides the background from one image.
    Supports grayscale or BGR.
    """

    # ensure float
    img_f = img.astype(np.float32)
    back_f = background.astype(np.float32)

    if method == "subtract":
        corrected = img_f - back_f

    elif method == "divide":
        corrected = img_f / (back_f + eps)

    elif method == "hybrid":
        corrected = (img_f - back_f) / (back_f + eps)

    # Normalize to 0–255
    corrected = cv.normalize(corrected, None, 0, 255, cv.NORM_MINMAX)
    return corrected.astype(np.uint8)

def adaptive_binary_threshold(img, imageShow=False):
    
    """ 
    Performs Binary thresholding on histogram equalised image (CLAHE)
    Build in billateral denoising 

    Arguments:
    - img: Grayscale or BGR image
    - imageShow = panel of the process

    Return:
    - Binary image 
    """

    if len(img.shape) == 3:     # BGR image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:                       # already grayscale
        gray = img.copy()

    # Histogram equalisation
    clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(4,4))
    img_clahe = clahe.apply(gray)
    vals = img_clahe.flatten()
    hist, bin_edges = np.histogram(vals, bins=256, range=(0, 256))
        
    mean_intensity = np.mean(vals)
    std_intensity = np.std(vals)

    denoised_1 = cv.bilateralFilter(img_clahe, 9, 75, 75)
            
    binary_threshold = mean_intensity + std_intensity
    _, binary = cv.threshold(denoised_1, binary_threshold, 255, cv.THRESH_BINARY)

    if imageShow: 
        plt.figure(figsize=(10,10))
        plt.suptitle(f"Visualization of binary thresholding")


        plt.subplot(221)
        plt.imshow(img, cmap='gray')
        plt.title('Background Corrected Image')

        plt.subplot(222)
        plt.imshow(img_clahe, cmap='gray')
        plt.title('CLAHE Image')

        plt.subplot(223)
        plt.hist(vals)
        plt.axvline(binary_threshold, color='red')
        plt.title('Histogram with Threshold')

        plt.subplot(224)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Image')

        #plt.show()
        plt.tight_layout()
        save_to = os.path.join(output_folder, "ROIs", f"q_batch_{q_batch_value:03}" , batch_comb, 'example', f"threshold_example{counter}")
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        plt.savefig(save_to, dpi = 300, bbox_inches = "tight")
        plt.close()

    return binary

#============================================================================
#### MAIN CODE ####
## FINDING ROIs ##

# Search for files matching the pattern
quality_logs = glob.glob(os.path.join(output_folder, 'logs', '**', 'q_batch_*_log.csv'), recursive=True)

# filter from q_batch_087_log.csv to q_batch_119_log.csv
#section = range(87, 120)
#quality_logs = [f for f in quality_logs if any(f"q_batch_{i:03}" in f for i in section)]

for i, current_log_file in enumerate(quality_logs):

    print(f"\nProcessing quality batch {i+1}/{len(quality_logs)}: {current_log_file}")

    df_current_log = pd.read_csv(current_log_file)
    q_batch_value = df_current_log['q_batch'].iloc[0]

    image_files = get_files_from_log(current_log_file)
    batches = fraction_batch_for_bg_removal(image_files, fraction=0.1)

    batch_rows = []
    for batch_idx, file_list in tqdm(
        batches.items(),
        desc=f"q_batch {q_batch_value:03}",
        total=len(batches),
        leave=True
    ):
        
        log_rows = []
        
        batch_median = compute_median_background(file_list)
        batch_comb = f'f_batch_{q_batch_value:03}_{batch_idx:02}'

        time_per_image = []

        # RANDOM VISUALISATION EXAMPLES
        n_examples = min(num_examples_per_batch, len(file_list))
        visualize_indices = random.sample(range(len(file_list)), n_examples)
        #visualize_indices = [10, 20] for manual setting
        counter = 0  # to track global image index

        for image in file_list:

            # crop per image 
            crop_counter = 0
            # time tracking
            start_time = time.time()

            # update example counter
            do_visualize = counter in visualize_indices
            counter += 1
            
            ## READ IMAGE 
            img = cv.imread(image)
            base_name = os.path.splitext(os.path.basename(image))[0]
            # make it grey scale
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ## PREPARE SEGMENTATION MASK
            # background correction
            img_corrected = subtract_background(img_gray, batch_median, method="subtract", eps=1e-6)

            # binary thresholding based on histrogram equalisation
            if save_steps & do_visualize:
                img_threshold = adaptive_binary_threshold(img_corrected, imageShow=True)
            else:
                img_threshold = adaptive_binary_threshold(img_corrected, imageShow=False)

            # Find contours and compute blob metrics
            contours, _ = cv.findContours(img_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            blob_metrics = []
            for c in contours:
                area = cv.contourArea(c)
                if area < min_area:  # apply size filter here and not after dilation
                    continue

                # calculate compactness and ESD for adaptive dilation
                x, y, w, h = cv.boundingRect(c)
                compactness = area / (w * h + 1e-5)
                esd_diameter = 2 * np.sqrt((area) / np.pi)
                blob_metrics.append({'contour': c, 'area': area, 'compactness': compactness, 'esd': esd_diameter})

            # binary image with blobs after filtering 
            img_filtered = np.zeros_like(img_threshold)

             # prepare saving box info
            img_boxes = img.copy()
            boxes = [] 

            for b in blob_metrics:
            
                # blob parameters for log
                blob = b['contour']
                area = b['area']
                compactness = b['compactness']
                esd_diameter = b['esd']

                # fill filtered mask
                cv.drawContours(img_filtered, [blob], -1, 255, -1)

                crop_counter = crop_counter + 1
                crop_filename = f"{base_name}_crop_{crop_counter:05}.png"
                save_path = os.path.join(output_folder, "ROIs", f"q_batch_{q_batch_value:03}" ,batch_comb, batch_comb, crop_filename)
                
                # mark ROI on the image
                x, y, w, h = cv.boundingRect(blob)
                boxes.append((x, y, w, h))
                cv.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # apply margin
                x_start = max(x - margin, 0)
                y_start = max(y - margin, 0)
                x_end = min(x + w + margin, img.shape[1])
                y_end = min(y + h + margin, img.shape[0])
                
                # fill in log 
                log_rows.append({
                    "q_batch": q_batch_value,
                    "f_batch": batch_idx,
                    "image_path": save_path,
                    "original_image": os.path.basename(image),
                    "crop_filename": crop_filename,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "compactness": compactness,
                    "ESD": esd_diameter, 
                    "x0": x_start,
                    "y0": y_start,
                    "x1": x_end,
                    "y1": y_end
                })

                # SAVE CROPS
                if save_crop:

                    object_crop = img[y_start:y_end, x_start:x_end]

                    save_to = save_path
                    os.makedirs(os.path.dirname(save_to), exist_ok=True)
                    cv.imwrite(save_to, object_crop)  
                
            # save boxed image   
            if do_visualize & save_box:
                save_to = os.path.join(output_folder, "ROIs", f"q_batch_{q_batch_value:03}" ,batch_comb, 'example', f"{base_name}_boxed.png")
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                cv.imwrite(save_to, img_boxes)
            
            # Visualization of steps
            if do_visualize & save_steps:
                plt.figure(figsize=(15, 10))
                plt.suptitle(f"Visualization for {base_name}")

                plt.subplot(231)
                plt.imshow(img, cmap='gray')
                plt.title("Original Image")
                #plt.axis('off')

                plt.subplot(232)
                plt.imshow(batch_median, cmap='gray')
                plt.title("Median Background")
                #plt.axis('off')
                
                plt.subplot(233)
                plt.imshow(img_corrected, cmap='gray')
                plt.title("Background Corrected")
                #plt.axis('off')

                plt.subplot(234)
                plt.imshow(img_threshold, cmap='gray')
                plt.title("Binary Threshold")
                #plt.axis('off')
                
                plt.subplot(235)
                plt.imshow(img_filtered, cmap='gray')
                plt.title("Size filter")
                #plt.axis('off')

                plt.subplot(236)
                plt.imshow(img_boxes)
                plt.title("Bounding Boxes")
                #plt.axis('off')

                plt.tight_layout()
                save_to = os.path.join(output_folder, "ROIs", f"q_batch_{q_batch_value:03}" ,batch_comb, 'example', f"example{counter}")
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                plt.savefig(save_to, dpi = 300, bbox_inches = "tight")
                plt.close()

            # end of image processing time    
            end_time = time.time() 
            elapsed = end_time - start_time

            batch_rows.append({
                'q_batch': q_batch_value,
                'f_batch': batch_idx,
                'full_path': image,
                'base_name': f"{base_name}.png",
                'num_crops': crop_counter, 
                'time_per_image': elapsed 
            })

        log_seg_df = pd.DataFrame(log_rows)
        
        # create empty logs if no ROIs found
        if len(log_seg_df) != 0:
            log_seg_df["roi_id"] = range(1, len(log_seg_df) + 1)
            
        save_to = os.path.join(output_folder, "ROIs", f"q_batch_{q_batch_value:03}", batch_comb)
        os.makedirs(save_to, exist_ok=True)
        log_seg_df.to_csv(
            os.path.join(save_to, f"f_batch_{q_batch_value:03}_{batch_idx:02}_log.csv"),
            index=False
            )

    batch_df = pd.DataFrame(batch_rows)
    save_to = os.path.join(output_folder, 'logs', 'segmentation_logs')
    os.makedirs(save_to, exist_ok=True)
    batch_df.to_csv(os.path.join(save_to, f"f_batch_{q_batch_value:03}_log.csv"),
                    index=False)

#============================================================================
#### create summary log of all segmentation batches ####
# summarise segmentation logs

# Search for files matching the pattern
f_batch_logs = glob.glob(os.path.join(output_folder, 'logs', 'segmentation_logs', '**', 'f_batch_*_log.csv'), recursive=True)

summary_rows = []
for current_file in tqdm(f_batch_logs, desc="\nSummarising segmentation batches"):
    
    log_df = pd.read_csv(current_file)

    # Group inside logfile
    for (q_batch, f_batch), g in log_df.groupby(['q_batch', 'f_batch']):
        num_images = g.shape[0]
        num_images_with_roi = (g['num_crops'] > 0).sum()
        num_rois = g['num_crops'].sum()
        mean_time = g['time_per_image'].mean()

        summary_rows.append({
            'q_batch': q_batch,
            'f_batch': f_batch,
            'num_images': num_images,
            'num_images_with_roi': num_images_with_roi,
            'num_rois': num_rois,
            'mean_segmentation_time': mean_time
        })

# Save final summary
summary_df = pd.DataFrame(summary_rows)

save_to = os.path.join(output_folder, 'logs', 'summaries')
os.makedirs(save_to, exist_ok=True)
summary_df.to_csv(os.path.join(save_to, 'seg_log_summary.csv'), index=False)

print("Segmentation completed.")
