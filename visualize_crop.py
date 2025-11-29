import cv2
import numpy as np
import sys
import os

def visualize_crop(image_path, crop_percent=0.1, output_path=None):
    """Draw a rectangle showing the cropped area used for comparison."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return

    h, w = img.shape[:2]
    h_crop = int(h * crop_percent)
    w_crop = int(w * crop_percent)
    
    # Draw rectangle (Green, 3px thickness)
    # The rectangle represents the area KEPT for comparison
    cv2.rectangle(img, (w_crop, h_crop), (w - w_crop, h - h_crop), (0, 255, 0), 3)
    
    # Add text
    cv2.putText(img, f"Comparison Area ({int((1-2*crop_percent)*100)}%)", 
                (w_crop, h_crop - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")
    else:
        # If no output path, just save to current dir with prefix
        filename = os.path.basename(image_path)
        out = f"vis_{filename}"
        cv2.imwrite(out, img)
        print(f"Saved visualization to {out}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_crop.py <image_path> [crop_percent]")
        return

    img_path = sys.argv[1]
    crop = 0.1
    if len(sys.argv) > 2:
        crop = float(sys.argv[2])
        
    visualize_crop(img_path, crop)

if __name__ == "__main__":
    main()
