import cv2
import numpy as np
import os
import glob
import argparse
from PIL import Image

def get_content_bbox(img, threshold=30):
    """
    Find the bounding box of the content, ignoring dark borders.
    Returns (x, y, w, h)
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Threshold to find non-black regions
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, img.shape[1], img.shape[0]

    # Find the bounding box of all contours combined
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0

    found_content = False
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out very small noise
        if w * h > 100: 
            found_content = True
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

    if not found_content:
        return 0, 0, img.shape[1], img.shape[0]

    # Add a small padding
    padding = 10
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img.shape[1], x_max + padding)
    y_max = min(img.shape[0], y_max + padding)

    return x_min, y_min, x_max - x_min, y_max - y_min

def process_images(input_dir, output_dir=None, crop=False, compress=False, format='png'):
    """
    Process images: crop borders and/or compress.
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "optimized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"Processing {filename} ({i+1}/{len(image_files)})...", end='\r')
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Crop if requested
        if crop:
            x, y, w, h = get_content_bbox(img)
            # Only crop if we are removing a significant amount (e.g. > 5% of area)
            original_area = img.shape[0] * img.shape[1]
            new_area = w * h
            if new_area < original_area * 0.95:
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
