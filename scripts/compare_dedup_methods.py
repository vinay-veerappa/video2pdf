import cv2
import numpy as np
import os
import glob
import argparse
from skimage.metrics import structural_similarity as ssim
from similarity import grid_ssim, calculate_similarity

def compare_methods(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if len(files) < 2:
        print("Need at least 2 images")
        return

    print(f"{'Comparison vs ' + os.path.basename(files[1]):<40} | {'CNN':<10} | {'Grid SSIM':<10}")
    print("-" * 70)
    
    # Use files[1] (vis_000) as the reference, skipping test_border.png (files[0])
    ref_idx = 1
    ref_img = cv2.imread(files[ref_idx])
    ref_name = os.path.basename(files[ref_idx])
    
    for i in range(len(files)):
        if i == ref_idx: continue
        
        img_target = cv2.imread(files[i])
        target_name = os.path.basename(files[i])
        
        # 1. CNN
        cnn_score = calculate_similarity(ref_img, img_target, method='cnn')
        
        # 2. Grid SSIM
        is_dup, grid_score = grid_ssim(ref_img, img_target)
        grid_str = f"{grid_score:.1%} ({'DUP' if is_dup else 'DIFF'})"
        
        print(f"{target_name:<40} | {cnn_score:.3f}      | {grid_str:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    compare_methods(args.input_dir)
