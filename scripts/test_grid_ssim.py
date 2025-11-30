import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import glob
import argparse

def grid_ssim(img1, img2, grid_size=(8, 8), threshold=0.95, min_matching_cells=0.8, visualize=False, output_path=None):
    """
    Compare images using a grid-based SSIM.
    Returns True if images are duplicates.
    """
    # Resize to same size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h, w = g1.shape
    rows, cols = grid_size
    cell_h = h // rows
    cell_w = w // cols
    
    matching_cells = 0
    total_cells = rows * cols
    
    # Visualization image (copy of img2 with overlay)
    vis_img = img2.copy() if visualize else None
    
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            cell1 = g1[y1:y2, x1:x2]
            cell2 = g2[y1:y2, x1:x2]
            
            score = ssim(cell1, cell2, data_range=255)
            is_match = score > threshold
            
            if is_match:
                matching_cells += 1
            
            if visualize:
                color = (0, 255, 0) if is_match else (0, 0, 255)
                # Draw rectangle
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                # Draw score
                cv2.putText(vis_img, f"{score:.2f}", (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if visualize and output_path:
        cv2.imwrite(output_path, vis_img)
                
    match_ratio = matching_cells / total_cells
    return match_ratio >= min_matching_cells, match_ratio

def test_duplicates(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if len(files) < 2:
        print("Need at least 2 images")
        return

    print(f"Comparing {len(files)} images...")
    
    for i in range(len(files) - 1):
        img1 = cv2.imread(files[i])
        img2 = cv2.imread(files[i+1])
        
        name1 = os.path.basename(files[i])
        name2 = os.path.basename(files[i+1])
        vis_path = os.path.join(output_dir, f"compare_{name1}_vs_{name2}")
        
        is_dup, score = grid_ssim(img1, img2, visualize=True, output_path=vis_path)
        status = "DUPLICATE" if is_dup else "DISTINCT"
        print(f"{name1} vs {name2}: {status} (Match: {score:.1%}) -> Saved to {vis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output", default="scripts/grid_vis")
    args = parser.parse_args()
    test_duplicates(args.input_dir, args.output)
