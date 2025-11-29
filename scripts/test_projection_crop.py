import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

def get_projection_crop(img, threshold_ratio=0.1):
    """
    Find crop coordinates using projection profiles (row/col variance).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Calculate variance along rows and columns
    # High variance = content (text, edges)
    # Low variance = flat background/border
    
    # We use Sobel edges to detect "activity"
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Project onto axes
    row_projection = np.sum(magnitude, axis=1)
    col_projection = np.sum(magnitude, axis=0)
    
    # Normalize
    row_projection = row_projection / np.max(row_projection)
    col_projection = col_projection / np.max(col_projection)
    
    # Threshold
    # We look for the first and last index where activity exceeds threshold
    active_rows = np.where(row_projection > threshold_ratio)[0]
    active_cols = np.where(col_projection > threshold_ratio)[0]
    
    if len(active_rows) == 0 or len(active_cols) == 0:
        return 0, 0, img.shape[1], img.shape[0]
        
    y_min, y_max = active_rows[0], active_rows[-1]
    x_min, x_max = active_cols[0], active_cols[-1]
    
    # Refinement: Check if the detected boundaries are just the border edges themselves
    # If the "activity" at the boundary is very high but immediately drops, it's likely a border line
    # We move inwards until we hit consistent activity or non-black pixels
    
    # Simple heuristic: Move inwards by 1% and check again
    h, w = img.shape[:2]
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.01)
    
    x_min += margin_x
    y_min += margin_y
    x_max -= margin_x
    y_max -= margin_y
    
    # Ensure valid coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    return x_min, y_min, x_max - x_min, y_max - y_min

def process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        
        x, y, w, h = get_projection_crop(img)
        cropped = img[y:y+h, x:x+w]
        
        out_path = os.path.join(output_dir, os.path.basename(f))
        cv2.imwrite(out_path, cropped)
        print(f"Processed {os.path.basename(f)}: {img.shape} -> {cropped.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    process(args.input_dir, args.output_dir)
