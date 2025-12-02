import os
import shutil
import glob
import subprocess
import sys
import json
import cv2
import numpy as np

# Import cropping logic
from experiment_cropping import get_content_bbox

def setup_test_data(source_dir, test_root):
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
    os.makedirs(test_root)

    original_dir = os.path.join(test_root, "original")
    cropped_dir = os.path.join(test_root, "cropped")
    os.makedirs(original_dir)
    os.makedirs(cropped_dir)

    # Get all images
    images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    
    print(f"Preparing {len(images)} images...")

    # Calculate Union Crop for consistency (simulating optimize_images.py)
    crop_boxes = []
    valid_images = []
    
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None: continue
        valid_images.append((img_path, img))
        crop_boxes.append(get_content_bbox(img))

    if not crop_boxes:
        print("No valid images found.")
        return None, None

    boxes = np.array(crop_boxes)
    # Union logic (min x, min y, max x2, max y2)
    x1s = boxes[:, 0]
    y1s = boxes[:, 1]
    x2s = boxes[:, 0] + boxes[:, 2]
    y2s = boxes[:, 1] + boxes[:, 3]
    
    # Use 5th/95th percentile to avoid outliers
    union_x1 = int(np.percentile(x1s, 5))
    union_y1 = int(np.percentile(y1s, 5))
    union_x2 = int(np.percentile(x2s, 95))
    union_y2 = int(np.percentile(y2s, 95))
    
    final_crop = (union_x1, union_y1, union_x2 - union_x1, union_y2 - union_y1)
    print(f"Union Crop: {final_crop}")

    # Save images
    for img_path, img in valid_images:
        basename = os.path.basename(img_path)
        
        # Save Original
        shutil.copy2(img_path, os.path.join(original_dir, basename))
        
        # Save Cropped
        x, y, w, h = final_crop
        # Ensure bounds
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        
        cropped_img = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(cropped_dir, basename), cropped_img)

    return original_dir, cropped_dir

def run_dedup(image_dir):
    print(f"Running deduplication on {image_dir}...")
    cmd = [
        sys.executable,
        os.path.join("scripts", "image_dedup.py"),
        image_dir,
        "--mode", "compare-all",
        "--sequential",
        "--skip-blanks" # Skip blank detection to focus on dedup
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dedup on {image_dir}:")
        print(e.stderr.decode())
        raise
    
    # Read results
    json_path = os.path.join(image_dir, "dedup_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            return len(data['duplicates'])
    return 0

if __name__ == "__main__":
    source_dir = r"C:\Users\vinay\video2pdf\output\Bootcamp Classroom - Week 4 Day 1 - Candle Stick Science intro\images"
    test_root = r"C:\Users\vinay\video2pdf\dedup_test"
    
    orig_dir, crop_dir = setup_test_data(source_dir, test_root)
    
    if orig_dir:
        dupes_orig = run_dedup(orig_dir)
        dupes_crop = run_dedup(crop_dir)
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Original Images Duplicates: {dupes_orig}")
        print(f"Cropped Images Duplicates:  {dupes_crop}")
        print("="*50)
        
        if dupes_crop > dupes_orig:
            print("WARNING: Cropping caused MORE duplicates (False Positives?)")
        elif dupes_crop < dupes_orig:
            print("NOTE: Cropping caused FEWER duplicates (False Negatives?)")
        else:
            print("SUCCESS: Cropping had no impact on duplicate count.")
