# Import packages and set up environment
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import time
from matplotlib.patches import Rectangle

sys.path.append(str(Path(__file__).resolve().parents[1]))

from functions import (  # noqa: E402
    ObservationScorer,
    VariationalAutoencoder,
    extract_tiles_with_offset,
    find_local_maxima,
    flood_fill_region,
    preprocess_tile,
)


def report_torch_environment() -> torch.device:
    print("=" * 40)
    print("PyTorch Environment Report")
    print("=" * 40)
    print(f"PyTorch version:        {torch.__version__}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"GPU available:          {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_index = 0
        print(f"GPU name:               {torch.cuda.get_device_name(gpu_index)}")
        print(f"Compute capability:     {torch.cuda.get_device_capability(gpu_index)}")
        print(f"Current device:         {torch.cuda.current_device()}")
        print(f"Total memory (GB):      {torch.cuda.get_device_properties(gpu_index).total_memory / 1e9:.2f}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print("=" * 40)
    print(f"Using device:      {device}")
    return device


def load_vae_model(device: torch.device, model_name: str, expected_latent_dim: int):
    print(f"loading VAE model from: {model_name}")
    checkpoint = torch.load(model_name, map_location=device, weights_only=False)

    latent_dims = checkpoint.get("latent_dim")
    if latent_dims != expected_latent_dim:
        print(
            f"⚠ Warning: Loaded model latent_dims={latent_dims} does not match expected latent_dim={expected_latent_dim}. "
            "Using loaded value."
        )

    image_size = checkpoint.get("image_size")
    hidden_channels = checkpoint.get("hidden_channels")
    grid_size = checkpoint.get("grid_size")
    state_dict = checkpoint["model_state_dict"]

    model = VariationalAutoencoder(
        latent_dims=latent_dims,
        image_size=image_size,
        hidden_channels=hidden_channels,
        grid_size=grid_size,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(
        f"✓ Loaded VAE model with latent_dim={latent_dims}, image_size={image_size}, "
        f"hidden_channels={hidden_channels}, grid_size={grid_size}"
    )
    return model, int(image_size), int(hidden_channels), int(grid_size)


def load_scorer_model(device: torch.device, model_name: str, scorer_mode: str, latent_dim: int, tile_grid_size: int):
    print("\n" + "=" * 70)
    print("Loading Neural Network Scorer")
    print("=" * 70)

    if scorer_mode == "one_class":
        scorer_name = model_name.replace("VAE/", "NN/OneClass/").replace("vae_model", "scorer_oneclass_model")
    elif scorer_mode == "binary":
        scorer_name = model_name.replace("VAE/", "NN/TwoClass/").replace("vae_model", "scorer_binary_model")
    else:
        raise ValueError(f"Invalid scorer_mode: {scorer_mode}. Must be 'binary' or 'one_class'.")

    checkpoint = torch.load(scorer_name, map_location=device, weights_only=False)

    scorer = ObservationScorer(
        latent_dim=latent_dim,
        hidden_dim=256,
        dropout=0.3,
        grid_size=tile_grid_size,
    )
    scorer.load_state_dict(checkpoint["model_state_dict"])
    scorer_threshold = float(checkpoint.get("threshold", 0.5))
    scorer.to(device)
    scorer.eval()

    print(f"✓ Loaded scorer checkpoint: {scorer_name}")
    print(f"✓ Loaded scorer threshold: {scorer_threshold:.4f}")
    return scorer, scorer_threshold, scorer_name


def aggregate_scores_to_map(rows_list, cols_list, scores, tile_grid_size: int, offsets_count: int):
    grid_size_small = tile_grid_size * offsets_count
    prob_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)
    coverage_map_small = np.zeros((grid_size_small, grid_size_small), dtype=np.float32)

    for tile_row, tile_col, pred_score in zip(rows_list, cols_list, scores):
        small_row = int(round((float(tile_row) - 1.0) * offsets_count))
        small_col = int(round((float(tile_col) - 1.0) * offsets_count))
        small_row = max(0, min(small_row, grid_size_small - offsets_count))
        small_col = max(0, min(small_col, grid_size_small - offsets_count))

        prob_map_small[small_row : small_row + offsets_count, small_col : small_col + offsets_count] += float(pred_score)
        coverage_map_small[small_row : small_row + offsets_count, small_col : small_col + offsets_count] += 1.0

    prob_map_small = np.divide(
        prob_map_small,
        coverage_map_small,
        out=prob_map_small,
        where=coverage_map_small > 0,
    )
    return prob_map_small, coverage_map_small

def image_to_probability_map(
    image: np.ndarray,
    model,
    scorer,
    scorer_threshold: float,
    device: torch.device,
    tile_size: int,
    tile_grid_size: int,
    image_size: int,
    batch_size: int,
    offsets,
    offsets_norm,
):
    
    tiles_list = []
    rows_list = []
    cols_list = []

    for offset, offset_norm in zip(offsets, offsets_norm):
        tiles = extract_tiles_with_offset(image, tile_size, tile_grid_size, offset_x=offset, offset_y=offset)
        for tile, row, col in tiles:
            tile_tensor = preprocess_tile(tile, image_size=image_size)
            if tile_tensor is None:
                continue
            tiles_list.append(tile_tensor)
            rows_list.append(row + 1 + offset_norm)
            cols_list.append(col + 1 + offset_norm)

    if not tiles_list:
        return None

    tiles_tensor = torch.stack(tiles_list, dim=0)
    rows_tensor = torch.tensor(rows_list, dtype=torch.float32)
    cols_tensor = torch.tensor(cols_list, dtype=torch.float32)

    scores_list = []
    preds_list = []
    mu_list = []
    recon_list = []

    with torch.inference_mode():
        for start_idx in range(0, len(tiles_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(tiles_tensor))
            tiles_batch = tiles_tensor[start_idx:end_idx].to(device, non_blocking=True)
            rows_batch = rows_tensor[start_idx:end_idx].to(device, non_blocking=True)
            cols_batch = cols_tensor[start_idx:end_idx].to(device, non_blocking=True)

            mu, _ = model.encode(tiles_batch, row=rows_batch, col=cols_batch)
            x_hat = model(tiles_batch, row=rows_batch, col=cols_batch)
            recon_err = ((tiles_batch - x_hat) ** 2).flatten(1).mean(dim=1)
            scores = scorer(mu, recon_err, rows_batch, cols_batch)
            preds = scores >= scorer_threshold

            mu_list.append(mu.detach().cpu().numpy())
            recon_list.append(recon_err.detach().cpu().numpy())
            scores_list.append(scores.detach().cpu().numpy())
            preds_list.append(preds.detach().cpu().numpy())

    scores_array = np.concatenate(scores_list, axis=0)
    preds_array = np.concatenate(preds_list, axis=0)
    mu_array = np.concatenate(mu_list, axis=0)
    recon_array = np.concatenate(recon_list, axis=0)

    prob_map_small, coverage_map_small = aggregate_scores_to_map(
        rows_list,
        cols_list,
        scores_array,
        tile_grid_size=tile_grid_size,
        offsets_count=len(offsets),
    )

    return {
        "prob_map_small": prob_map_small,
        "coverage_map_small": coverage_map_small,
        "rows_list": np.asarray(rows_list, dtype=np.float32),
        "cols_list": np.asarray(cols_list, dtype=np.float32),
        "scores_array": scores_array,
        "preds_array": preds_array,
        "mu_array": mu_array,
        "recon_array": recon_array,
    }


def batch_extract_tiles_from_images(images_list, tile_size, tile_grid_size, offsets, offsets_norm, image_size):
    """Extract tiles from multiple images and return concatenated batch with image tracking."""
    all_tiles = []
    all_rows = []
    all_cols = []
    image_tile_counts = []  # Track how many tiles per image
    
    for image_gray in images_list:
        tiles_list = []
        rows_list = []
        cols_list = []

        for offset, offset_norm in zip(offsets, offsets_norm):
            tiles = extract_tiles_with_offset(image_gray, tile_size, tile_grid_size, offset_x=offset, offset_y=offset)
            for tile, row, col in tiles:
                tile_tensor = preprocess_tile(tile, image_size=image_size)
                if tile_tensor is None:
                    continue
                tiles_list.append(tile_tensor)
                rows_list.append(row + 1 + offset_norm)
                cols_list.append(col + 1 + offset_norm)
        
        if tiles_list:
            all_tiles.extend(tiles_list)
            all_rows.extend(rows_list)
            all_cols.extend(cols_list)
            image_tile_counts.append(len(tiles_list))
        else:
            image_tile_counts.append(0)
    
    if not all_tiles:
        return None, None, None, None, None
    
    tiles_tensor = torch.stack(all_tiles, dim=0)
    rows_tensor = torch.tensor(all_rows, dtype=torch.float32)
    cols_tensor = torch.tensor(all_cols, dtype=torch.float32)
    
    return tiles_tensor, rows_tensor, cols_tensor, image_tile_counts, (all_rows, all_cols)


def batch_run_inference(tiles_tensor, rows_tensor, cols_tensor, model, scorer, scorer_threshold, device, batch_size, image_tile_counts):
    """Run VAE/scorer inference on a batch of tiles from multiple images."""
    scores_list = []
    preds_list = []
    mu_list = []
    recon_list = []

    total_tiles = len(tiles_tensor)
    # print(f"    Running inference on {total_tiles} tiles in batches of {batch_size}...")

    with torch.inference_mode():
        for start_idx in range(0, len(tiles_tensor), batch_size):
            end_idx = min(start_idx + batch_size, len(tiles_tensor))
            # print(f"    → Batch {start_idx//batch_size + 1}: tiles {start_idx}-{end_idx}...", flush=True)
            
            tiles_batch = tiles_tensor[start_idx:end_idx].to(device, non_blocking=True)
            rows_batch = rows_tensor[start_idx:end_idx].to(device, non_blocking=True)
            cols_batch = cols_tensor[start_idx:end_idx].to(device, non_blocking=True)
            
            torch.cuda.synchronize()  # Ensure GPU is ready
            
            # print(f"      Encoding...", flush=True)
            mu, _ = model.encode(tiles_batch, row=rows_batch, col=cols_batch)
            
            # print(f"      Decoding...", flush=True)
            x_hat = model(tiles_batch, row=rows_batch, col=cols_batch)
            
            # print(f"      Computing reconstruction error...", flush=True)
            recon_err = ((tiles_batch - x_hat) ** 2).flatten(1).mean(dim=1)
            
            # print(f"      Scoring...", flush=True)
            scores = scorer(mu, recon_err, rows_batch, cols_batch)
            preds = scores >= scorer_threshold

            torch.cuda.synchronize()  # Ensure all ops complete
            
            # print(f"      Transferring to CPU...", flush=True)
            mu_list.append(mu.detach().cpu().numpy())
            recon_list.append(recon_err.detach().cpu().numpy())
            scores_list.append(scores.detach().cpu().numpy())
            preds_list.append(preds.detach().cpu().numpy())
            
            # free temporary GPU tensors and clear cache (guarded)
            try:
                del tiles_batch, rows_batch, cols_batch, mu, x_hat, recon_err, scores, preds
            except NameError:
                pass
            torch.cuda.empty_cache()
            
            # print(f"      ✓ Batch complete", flush=True)

    # print(f"    Concatenating results...", flush=True)
    scores_array = np.concatenate(scores_list, axis=0)
    preds_array = np.concatenate(preds_list, axis=0)
    mu_array = np.concatenate(mu_list, axis=0)
    recon_array = np.concatenate(recon_list, axis=0)
    
    return scores_array, preds_array, mu_array, recon_array


def split_results_by_image(scores_array, rows_list, cols_list, tile_grid_size, offsets, image_tile_counts):
    """Split batched inference results back to per-image probability maps."""
    results_per_image = []
    start_idx = 0
    
    for img_num, num_tiles in enumerate(image_tile_counts):
        if num_tiles == 0:
            results_per_image.append(None)
            continue
        
        end_idx = start_idx + num_tiles
        image_scores = scores_array[start_idx:end_idx]
        image_rows = rows_list[start_idx:end_idx]
        image_cols = cols_list[start_idx:end_idx]
        
        prob_map_small, coverage_map_small = aggregate_scores_to_map(
            image_rows,
            image_cols,
            image_scores,
            tile_grid_size=tile_grid_size,
            offsets_count=len(offsets),
        )
        
        results_per_image.append({
            "prob_map_small": prob_map_small,
            "coverage_map_small": coverage_map_small,
        })
        start_idx = end_idx
    
    return results_per_image


def save_crop_metadata_rows(csv_path: Path, rows: list[dict]):

    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, index=False, mode="a", header=write_header)


def main():
    device = report_torch_environment()

    # Define paths and parameters
    ROOT_DIR_R = r"R:\LU24A1037-Jellyscope\Jellyscope\Monitoring data"
    monitoring_effort = "Kristineberg_260424"
    images_folder = os.path.join(ROOT_DIR_R, monitoring_effort)
    IMAGE_SIZE_PX = 4512  # All images are square 4512x4512
    tile_grid_size = 16
    
    tile_size = IMAGE_SIZE_PX // tile_grid_size
    latent_dim = 32
    model_name = f"../models/VAE/{monitoring_effort}_vae_model{tile_grid_size}_l{latent_dim}.pth"
    output_folder = Path(f"C:/Users/Admin/Documents/Jellyscope/Segmentation_results_{monitoring_effort}")

    batch_size = 4096*2  # Process 8192 tiles per batch (adjust based on GPU memory)
    batch_size_images = 32  # Process 32 images per batch (32 * 1280 tiles = 40960)
    vae_image_size = 128

    # timing accumulators
    total_images_processed = 0
    total_processing_time = 0.0
    total_image_load_time = 0.0

    offsets_norm = [0, 0.2, 0.4, 0.6, 0.8]
    offsets = [int(norm * tile_size) for norm in offsets_norm]

    grid_size_small = tile_grid_size * len(offsets)
    upscale_factor = IMAGE_SIZE_PX/grid_size_small 
    min_region_size_patches = 1
    crop_padding_pixels = 10
    
    # set preak detection parameters
    peak_threshold = 0.8
    secondary_threshold = 0.3
    
    # Set scorer mode (binary or one_class) and load corresponding model and threshold
    scorer_mode = "binary"
    print(f"Using scorer mode: {scorer_mode}")
    
    images_path = sorted(str(p) for p in Path(images_folder).glob("*.png"))
    print(f"Found {len(images_path)} images in: {images_folder}")

    model, _, hidden_channels, grid_size_loaded = load_vae_model(device, model_name, latent_dim)
    scorer, scorer_threshold, _ = load_scorer_model(device, model_name, scorer_mode, latent_dim, tile_grid_size)

    if grid_size_loaded != tile_grid_size:
        raise ValueError(f"⚠ Warning: VAE model grid_size={grid_size_loaded} does not match expected tile_grid_size={tile_grid_size}. ")

    output_folder_crops = output_folder / "crops_peak_detection"
    output_folder_crops.mkdir(parents=True, exist_ok=True)

    crop_metadata_csv = output_folder_crops / "crop_metadata.csv"
    if crop_metadata_csv.exists():
        crop_metadata_csv.unlink()

    print(f"\nStarting processing of {len(images_path)} tiles with {len(offsets)} offsets...\n")

    crop_counter = 0
    images_batch = []
    image_paths_batch = []
    batch_load_time_accumulator = 0.0


    for image_idx, image_path_str in enumerate(images_path, start=1):
        image_path = Path(image_path_str)
        img_load_start = time.perf_counter()
        image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        batch_load_time = time.perf_counter() - img_load_start
        total_image_load_time += batch_load_time

        if image_gray is None:
            print(f"\n⚠ Warning: failed to read image {image_path}, skipping")
            continue
        
        images_batch.append(image_gray)
        image_paths_batch.append(image_path)
        batch_load_time_accumulator += batch_load_time
        
        # Process batch when full or at end of list
        if len(images_batch) == batch_size_images or image_idx == len(images_path):
            # print(f"\n→ Processing batch of {len(images_batch)} images...")
            batch_start_time = time.perf_counter()
            tiles_tensor, rows_tensor, cols_tensor, image_tile_counts, (rows_list, cols_list) = batch_extract_tiles_from_images(images_batch, tile_size, tile_grid_size, offsets, offsets_norm, vae_image_size) # type: ignore
            
            if tiles_tensor is not None:
                # print(f"  Extracted {len(tiles_tensor)} tiles total, running inference...")
                scores_array, preds_array, mu_array, recon_array = batch_run_inference(
                    tiles_tensor, rows_tensor, cols_tensor, model, scorer, scorer_threshold, device, batch_size, image_tile_counts
                )
                
                # print(f"  Splitting results by image...")
                results_per_image = split_results_by_image(scores_array, rows_list, cols_list, tile_grid_size, offsets, image_tile_counts)
                # print(f"  Processing {len(results_per_image)} results...")
                
                # Collect crops for deferred writing (after post-processing)
                deferred_crops = []
                
                # Process each image's results
                for batch_img_idx, (image_gray, image_path) in enumerate(zip(images_batch, image_paths_batch)):
                    image_name = image_path.stem
                    # print(f"  [{batch_img_idx+1}/{len(images_batch)}] {image_name}: ", end="", flush=True)
                    
                    result = results_per_image[batch_img_idx]
                    if result is None:
                        # print("no tiles")
                        continue
                    
                    prob_map_small = result["prob_map_small"]
                    maxima_small = find_local_maxima(prob_map_small, peak_threshold)
                    if len(maxima_small) == 0:
                        continue

                    # Convert each small-grid flood-filled region into a pixel crop bbox
                    candidates = []
                    for peak_y_small, peak_x_small, peak_val in maxima_small:
                        region_small = flood_fill_region(prob_map_small, peak_y_small, peak_x_small, secondary_threshold)
                        
                        if len(region_small) < min_region_size_patches ** 2:
                            continue

                        region_small_points = np.asarray(list(region_small), dtype=np.int32)

                        min_y_small = int(region_small_points[:, 0].min())
                        max_y_small = int(region_small_points[:, 0].max())
                        min_x_small = int(region_small_points[:, 1].min())
                        max_x_small = int(region_small_points[:, 1].max())

                        # Map small-grid bounds to full-resolution pixel bbox
                        y0 = int(round(min_y_small * upscale_factor))
                        y1 = int(round((max_y_small + 1) * upscale_factor))
                        x0 = int(round(min_x_small * upscale_factor))
                        x1 = int(round((max_x_small + 1) * upscale_factor))

                        crop_y0 = max(0, y0 - crop_padding_pixels)
                        crop_x0 = max(0, x0 - crop_padding_pixels)
                        crop_y1 = min(IMAGE_SIZE_PX, y1 + crop_padding_pixels)
                        crop_x1 = min(IMAGE_SIZE_PX, x1 + crop_padding_pixels)

                        candidates.append({"region_small_points": region_small_points,
                                            "peak_val": float(peak_val),
                                            "crop_coords": (crop_y0, crop_x0, crop_y1, crop_x1)})

                    if len(candidates) == 0:
                        continue

                    # Pick the strongest peaks first
                    candidates = sorted(candidates, key=lambda c: c["peak_val"], reverse=True)

                    accepted = []
                    for cand in candidates:
                        crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
                        crop_h = crop_y1 - crop_y0
                        crop_w = crop_x1 - crop_x0
                        crop_area = crop_h * crop_w

                        is_duplicate = False
                        for prev in accepted:
                            prev_y0, prev_x0, prev_y1, prev_x1 = prev["crop_coords"]
                            prev_area = prev["crop_area"]
                            inter_y0 = max(crop_y0, prev_y0)
                            inter_x0 = max(crop_x0, prev_x0)
                            inter_y1 = min(crop_y1, prev_y1)
                            inter_x1 = min(crop_x1, prev_x1)

                            if inter_y1 > inter_y0 and inter_x1 > inter_x0:
                                inter_area = (inter_y1 - inter_y0) * (inter_x1 - inter_x0)
                                union_area = crop_area + prev_area - inter_area
                                iou = inter_area / union_area if union_area > 0 else 0
                                
                                if iou > 0.5:
                                    is_duplicate = True
                                    break

                        if not is_duplicate:
                            cand["crop_area"] = crop_area
                            accepted.append(cand)

                    # Save crops and metadata
                    crop_rows = []
                    for crop_idx, cand in enumerate(accepted):
                        region_points = cand["region_small_points"]
                        peak_val = cand["peak_val"]
                        crop_y0, crop_x0, crop_y1, crop_x1 = cand["crop_coords"]
                        crop_h = crop_y1 - crop_y0
                        crop_w = crop_x1 - crop_x0

                        region_mean = prob_map_small[region_points[:, 0], region_points[:, 1]].mean() if len(region_points) > 0 else 0.0
                        region_max = prob_map_small[region_points[:, 0], region_points[:, 1]].max() if len(region_points) > 0 else 0.0
                    
                        crop_rows.append({  "image_name": image_name,
                                            "crop_id": crop_idx,
                                            "crop_width": crop_w,
                                            "crop_height": crop_h,
                                            "crop_y0": crop_y0,
                                            "crop_x0": crop_x0,
                                            "crop_y1": crop_y1,
                                            "crop_x1": crop_x1,
                                            "region_size_pixels": int(region_points.shape[0]),
                                            "peak_intensity": peak_val,
                                            "region_mean_intensity": region_mean,
                                            "region_max_intensity": region_max})
                        
                        crop_image = image_gray[crop_y0:crop_y1, crop_x0:crop_x1]
                        crop_path = output_folder_crops / f"{image_name}_crop_{crop_idx:03d}.png"
                        deferred_crops.append((crop_image.copy(), crop_path))  # defer write

                        crop_counter += 1

                    save_crop_metadata_rows(crop_metadata_csv, crop_rows)
                    # print(f"{len(crop_rows)} crops")
                    
                    del prob_map_small, maxima_small, candidates, accepted, crop_rows
                
                # Write all deferred crops (CPU/disk work while GPU can extract next batch)
                for crop_image, crop_path in deferred_crops:
                    cv2.imwrite(str(crop_path), crop_image)
                del deferred_crops
                
                # Batch timing and cleanup
                images_in_batch = len(image_paths_batch)
                batch_end_time = time.perf_counter()
                batch_process_time = batch_end_time - batch_start_time
                total_images_processed += images_in_batch
                total_processing_time += batch_process_time
                total_image_load_time += batch_load_time_accumulator
                avg_process_time = batch_process_time / images_in_batch if images_in_batch > 0 else 0.0
                total_avg_time = total_processing_time / total_images_processed if total_images_processed > 0 else 0.0
                avg_process_time_per_image = (total_processing_time - total_image_load_time) / total_images_processed if total_images_processed > 0 else 0.0
                remaining_images = len(images_path) - total_images_processed
                eta_seconds = remaining_images * avg_process_time_per_image if avg_process_time_per_image > 0 else 0
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                total_time = total_image_load_time + total_processing_time
                load_pct = 100*total_image_load_time/total_time if total_time > 0 else 0.0
                print(
                    f"Batch: {images_in_batch} img | load: {batch_load_time_accumulator:.2f}s | process: {batch_process_time:.2f}s | "
                    f"processed: {total_images_processed}/{len(images_path)} | ETA: {eta_time.strftime('%H:%M:%S')} | crops: {crop_counter}",
                    end="\r",
                    flush=True,
                )

                # Clean up batch
                del images_batch, tiles_tensor, rows_tensor, cols_tensor, scores_array, preds_array, mu_array, recon_array, results_per_image
                images_batch = []
                image_paths_batch = []
                batch_load_time_accumulator = 0.0
        
    if total_images_processed > 0:
        total_time = total_image_load_time + total_processing_time
        load_pct = 100*total_image_load_time/total_time
        process_pct = 100*total_processing_time/total_time
        print(f"\nOverall timing:")
        print(f"  Image loading: {total_image_load_time:.2f}s ({load_pct:.1f}% of total)")
        print(f"  Batch processing: {total_processing_time:.2f}s ({process_pct:.1f}% of total)")
        print(f"  Total elapsed: {total_time:.2f}s")
    
    print(f"\n✓ Complete! Total crops extracted: {crop_counter}")
    if crop_metadata_csv.exists():
        print(f"✓ Crop metadata saved to {crop_metadata_csv}")
    
if __name__ == "__main__":
    main()