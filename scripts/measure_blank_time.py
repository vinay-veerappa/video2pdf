
import time
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.getcwd())
from scripts import image_dedup

if __name__ == '__main__':
    image_dir = r"C:\Users\vinay\video2pdf\output\4 Secret FVGs (Fractal Quarterly Shifts) - Ep. 3\images"
    print(f"Measuring blank detection time for: {image_dir}")

    start = time.time()
    # Run with default settings
    image_dedup.find_blank_images(image_dir, debug=False)
    end = time.time()

    print(f"Time taken: {end - start:.2f}s")
