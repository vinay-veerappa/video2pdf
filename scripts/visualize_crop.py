import cv2
import argparse
import json
import os
import sys

def visualize_crop(image_path, crop_params, output_path):
    """
    Draw a colored rectangle on the image representing the crop box.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = img.shape[:2]
    
    # Check if params are standard Qwen dict or manual config
    if isinstance(crop_params, dict):
        ymin = crop_params.get('ymin', 0.0)
        xmin = crop_params.get('xmin', 0.0)
        ymax = crop_params.get('ymax', 1.0)
        xmax = crop_params.get('xmax', 1.0)
    else:
        # Default fallback
        ymin, xmin, ymax, xmax = 0,0,1,1

    # Convert to pixels
    top = int(ymin * h)
    left = int(xmin * w)
    bottom = int(ymax * h)
    right = int(xmax * w)

    # Draw semi-transparent overlay meant for removal
    overlay = img.copy()
    
    # 1. Red Mask over eliminated areas
    # Top strip
    cv2.rectangle(overlay, (0, 0), (w, top), (0, 0, 255), -1)
    # Bottom strip
    cv2.rectangle(overlay, (0, bottom), (w, h), (0, 0, 255), -1)
    # Left strip
    cv2.rectangle(overlay, (0, top), (left, bottom), (0, 0, 255), -1)
    # Right strip
    cv2.rectangle(overlay, (right, top), (w, bottom), (0, 0, 255), -1)
    
    # Apply overlay
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 2. Green Border around Kept Area
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)

    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--xmin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    parser.add_argument("--xmax", type=float, default=1.0)
    
    args = parser.parse_args()
    
    params = {
        'ymin': args.ymin,
        'xmin': args.xmin,
        'ymax': args.ymax,
        'xmax': args.xmax
    }
    
    visualize_crop(args.image, params, args.output)
