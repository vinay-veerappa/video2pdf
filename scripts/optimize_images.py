import cv2
import numpy as np
import os
import glob
import argparse
from PIL import Image

def get_content_bbox(img, threshold_ratio=0.1):
    """
    Find crop coordinates using projection profiles (row/col variance).
    Returns (x, y, w, h)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Calculate variance along rows and columns using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Project onto axes
    row_projection = np.sum(magnitude, axis=1)
    col_projection = np.sum(magnitude, axis=0)
    
    # Normalize
    row_max = np.max(row_projection)
    col_max = np.max(col_projection)
    
    if row_max == 0 or col_max == 0:
        return 0, 0, img.shape[1], img.shape[0]
        
    row_projection = row_projection / row_max
    col_projection = col_projection / col_max
    
    # Threshold
    active_rows = np.where(row_projection > threshold_ratio)[0]
    active_cols = np.where(col_projection > threshold_ratio)[0]
    
    if len(active_rows) == 0 or len(active_cols) == 0:
        return 0, 0, img.shape[1], img.shape[0]
        
    y_min, y_max = active_rows[0], active_rows[-1]
    x_min, x_max = active_cols[0], active_cols[-1]
    
    # Refinement: Move inwards to avoid border edges
    h, w = img.shape[:2]
    margin_x = int(w * 0.01)
    margin_y = int(h * 0.01)
    
    x_min = min(x_min + margin_x, w)
    y_min = min(y_min + margin_y, h)
    x_max = max(x_max - margin_x, 0)
    y_max = max(y_max - margin_y, 0)
    
    # Ensure valid coordinates
    if x_max <= x_min or y_max <= y_min:
        return 0, 0, w, h
        
    return x_min, y_min, x_max - x_min, y_max - y_min

def process_images(input_dir, output_dir=None, crop=False, compress=False, format='png'):
    """
    Process images: crop borders and/or compress.
    Uses a two-pass approach for cropping to ensure consistent slide sizes.
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "optimized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Pass 1: Calculate crop coordinates
    crop_boxes = []
    if crop:
        print("Analyzing crop boundaries...")
        for i, img_path in enumerate(image_files):
            img = cv2.imread(img_path)
            if img is None: continue
            
            box = get_content_bbox(img)
            crop_boxes.append(box)
            
        # Calculate UNION crop box (safest)
        # We want the smallest x,y (top-left) and the largest w,h (bottom-right)
        # to ensure we include content that might only appear on some slides.
        if crop_boxes:
            boxes = np.array(crop_boxes)
            
            # Calculate min/max coordinates
            # box = [x, y, w, h]
            # x2 = x + w, y2 = y + h
            x1s = boxes[:, 0]
            y1s = boxes[:, 1]
            x2s = boxes[:, 0] + boxes[:, 2]
            y2s = boxes[:, 1] + boxes[:, 3]
            
            # Use 10th percentile for min and 90th for max to ignore extreme outliers
            # but still capture almost everything
            union_x1 = int(np.percentile(x1s, 5))
            union_y1 = int(np.percentile(y1s, 5))
            union_x2 = int(np.percentile(x2s, 95))
            union_y2 = int(np.percentile(y2s, 95))
            
            final_crop = (union_x1, union_y1, union_x2 - union_x1, union_y2 - union_y1)
            print(f"Calculated consistent crop (Union): {final_crop}")
    
    # Pass 2: Apply processing
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"Processing {filename} ({i+1}/{len(image_files)})...", end='\r')
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Crop if requested
        if crop and crop_boxes:
            x, y, w, h = final_crop
            # Ensure crop is within bounds for this specific image
            h_img, w_img = img.shape[:2]
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            
            img = img[y:y+h, x:x+w]
        
        # Save/Compress
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.' + format)
        
        if compress:
            # Convert BGR (OpenCV) to RGB (PIL)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            if format == 'png':
                # Optimize PNG: Quantize to 256 colors (P mode) reduces size significantly for slides
                pil_img = pil_img.quantize(colors=256, method=2)
                pil_img.save(output_path, "PNG", optimize=True)
            elif format == 'webp':
                pil_img.save(output_path, "WEBP", quality=80, method=6)
            elif format == 'jpg':
                pil_img.save(output_path, "JPEG", quality=85, optimize=True)
        else:
            cv2.imwrite(output_path, img)
            
    print(f"\nProcessing complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize and crop images")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument("--output", help="Output directory (default: input_dir/optimized)")
    parser.add_argument("--crop", action="store_true", help="Auto-crop dark borders")
    parser.add_argument("--compress", action="store_true", help="Compress images")
    parser.add_argument("--format", choices=['png', 'webp', 'jpg'], default='png', help="Output format")
    
    args = parser.parse_args()
    
    process_images(args.input_dir, args.output, args.crop, args.compress, args.format)
