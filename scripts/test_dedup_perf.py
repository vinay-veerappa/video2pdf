
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import imagehash
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import image_dedup

def optimized_sequential_dedup(images_data, threshold=10, hist_thresh=0.90):
    """
    Optimized sequential deduplication with:
    1. Reference image caching (avoid re-opening)
    2. Hamming distance pre-check (avoid full compare if hashes differ)
    """
    duplicates = []
    
    if len(images_data) == 0:
        return []
        
    current_ref_idx = 0
    # Cache the reference image object
    ref_img_obj = image_dedup.get_processed_image(images_data[current_ref_idx])
    
    comparisons = 0
    full_comparisons = 0
    
    for i in range(1, len(images_data)):
        comparisons += 1
        candidate_data = images_data[i]
        ref_data = images_data[current_ref_idx]
        
        # 1. Fast Hash Pre-check
        # Calculate minimum hash distance without loading images
        distances = {k: abs(ref_data['hashes'][k] - candidate_data['hashes'][k]) 
                     for k in ref_data['hashes'].keys()}
        min_dist = min(distances.values())
        
        # If hash distance is significantly larger than threshold, skip full check
        # We add a safety margin (e.g. +5) to be sure
        if min_dist > threshold + 5:
            # Different! New reference.
            current_ref_idx = i
            ref_img_obj = image_dedup.get_processed_image(images_data[current_ref_idx])
            continue
            
        # 2. Full Comparison (Only if hashes are close)
        full_comparisons += 1
        
        # We need to manually call compare logic but inject our cached image
        # Since compare_images_smart loads images internally using get_processed_image,
        # we can temporarily inject our cached object into the data dict
        
        # Ensure ref_data has the processed image cached
        ref_data['processed'] = ref_img_obj
        
        # Perform comparison
        comp = image_dedup.compare_images_smart(ref_data, candidate_data, threshold, hist_thresh)
        
        if comp['is_duplicate']:
            duplicates.append({
                'image1': ref_data['name'],
                'image2': candidate_data['name'],
                **comp
            })
        else:
            # Different! New reference.
            current_ref_idx = i
            ref_img_obj = image_dedup.get_processed_image(images_data[current_ref_idx])
            
    return duplicates, comparisons, full_comparisons

def run_test(image_dir):
    print(f"Testing on: {image_dir}")
    
    # 1. Load Images (Common Step)
    print("\nLoading images and computing hashes...")
    start_load = time.time()
    image_paths = image_dedup.get_image_files(image_dir)
    if not image_paths:
        print("No images found!")
        return
        
    # Use existing parallel loader
    crop_dir = Path(image_dir) / 'perf_test_crops'
    crop_dir.mkdir(exist_ok=True)
    
    # Mock args for process_image_for_dedup
    # path, blur, downscale, crop_dir
    pool_args = [(p, True, False, crop_dir) for p in image_paths]
    
    import multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.map(image_dedup.process_image_for_dedup, pool_args)
    
    images_data = [r for r in results if r]
    load_time = time.time() - start_load
    print(f"Loaded {len(images_data)} images in {load_time:.2f}s")
    
    # 2. Baseline Run (Existing Sequential Logic)
    print("\n--- Baseline: Existing Sequential Logic ---")
    start_base = time.time()
    
    # We need to clear any 'processed' keys to simulate fresh run
    for img in images_data:
        if 'processed' in img: del img['processed']
        
    base_dups = []
    current_ref_idx = 0
    for i in range(1, len(images_data)):
        # Existing logic calls compare_images_smart which reloads images
        comp = image_dedup.compare_images_smart(images_data[current_ref_idx], images_data[i], threshold=10, hist_thresh=0.90)
        if comp['is_duplicate']:
            base_dups.append(comp)
        else:
            current_ref_idx = i
            
    base_time = time.time() - start_base
    print(f"Baseline Time: {base_time:.2f}s")
    print(f"Duplicates Found: {len(base_dups)}")
    
    # 3. Optimized Run
    print("\n--- Optimized: Caching + Hash Pre-check ---")
    
    # Clear cache again
    for img in images_data:
        if 'processed' in img: del img['processed']
        
    start_opt = time.time()
    opt_dups, comps, full_comps = optimized_sequential_dedup(images_data, threshold=10, hist_thresh=0.90)
    opt_time = time.time() - start_opt
    
    print(f"Optimized Time: {opt_time:.2f}s")
    print(f"Duplicates Found: {len(opt_dups)}")
    print(f"Total Comparisons: {comps}")
    print(f"Full Comparisons (Image Load): {full_comps} ({full_comps/comps*100:.1f}%)")
    
    # 4. Results
    print("\n=== RESULTS ===")
    print(f"Speedup: {base_time / opt_time:.2f}x")
    print(f"Time Saved: {base_time - opt_time:.2f}s")
    
    if len(base_dups) != len(opt_dups):
        print("WARNING: Duplicate count mismatch!")
        print(f"Baseline: {len(base_dups)}, Optimized: {len(opt_dups)}")
    else:
        print("Verification: Duplicate counts match.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Directory containing images")
    args = parser.parse_args()
    
    run_test(args.image_dir)
