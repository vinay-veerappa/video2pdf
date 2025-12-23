#!/usr/bin/env python3
"""
Analyze captured images for duplicates and optimization opportunities
"""

import os
import glob
import cv2
import re
from skimage.metrics import structural_similarity as ssim
import numpy as np

def parse_image_timestamp(filename):
    """Parse timestamp from image filename"""
    try:
        match = re.search(r'_(\d+\.\d+)\.png', filename)
        if match:
            return float(match.group(1))  # minutes
    except:
        pass
    return None

def calculate_similarity(img1_path, img2_path):
    """Calculate structural similarity between two images"""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return None
        
        # Resize to same size
        img1_resized = cv2.resize(img1, (300, 200))
        img2_resized = cv2.resize(img2, (300, 200))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        similarity = ssim(gray1, gray2, data_range=255)
        return similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None

def analyze_images(images_folder):
    """Analyze images for duplicates and optimization opportunities"""
    print("="*60)
    print("IMAGE ANALYSIS REPORT")
    print("="*60)
    
    # Get all images
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"\nTotal images: {len(image_files)}\n")
    
    # Analyze timestamps
    print("TIMESTAMP ANALYSIS:")
    print("-" * 60)
    image_data = []
    for img_file in image_files:
        timestamp = parse_image_timestamp(os.path.basename(img_file))
        if timestamp:
            image_data.append((timestamp, img_file))
    
    # Find images that are too close together
    print("\n1. Images captured very close together (< 10 seconds):")
    close_pairs = []
    for i in range(len(image_data) - 1):
        time1, path1 = image_data[i]
        time2, path2 = image_data[i + 1]
        time_diff_seconds = (time2 - time1) * 60
        
        if time_diff_seconds < 10:
            close_pairs.append((i, i+1, time_diff_seconds, path1, path2))
            print(f"   {os.path.basename(path1)} ({time1:.2f} min) -> "
                  f"{os.path.basename(path2)} ({time2:.2f} min) = "
                  f"{time_diff_seconds:.1f} seconds apart")
    
    # Find large gaps
    print("\n2. Large time gaps (> 2 minutes):")
    large_gaps = []
    for i in range(len(image_data) - 1):
        time1, path1 = image_data[i]
        time2, path2 = image_data[i + 1]
        time_diff_minutes = time2 - time1
        
        if time_diff_minutes > 2:
            large_gaps.append((i, i+1, time_diff_minutes, path1, path2))
            print(f"   {os.path.basename(path1)} ({time1:.2f} min) -> "
                  f"{os.path.basename(path2)} ({time2:.2f} min) = "
                  f"{time_diff_minutes:.2f} minutes gap")
    
    # Check for visual duplicates
    print("\n3. Checking for visual duplicates (similarity > 0.95):")
    print("   (This may take a while...)")
    duplicates = []
    checked = 0
    
    for i in range(len(image_data) - 1):
        time1, path1 = image_data[i]
        time2, path2 = image_data[i + 1]
        time_diff_seconds = (time2 - time1) * 60
        
        # Only check if images are close together
        if time_diff_seconds < 30:
            similarity = calculate_similarity(path1, path2)
            checked += 1
            if similarity and similarity > 0.95:
                duplicates.append((i, i+1, similarity, path1, path2))
                print(f"   DUPLICATE: {os.path.basename(path1)} <-> "
                      f"{os.path.basename(path2)} (similarity: {similarity:.3f}, "
                      f"time diff: {time_diff_seconds:.1f}s)")
    
    print(f"\n   Checked {checked} image pairs")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS:")
    print("="*60)
    print(f"\nTotal images: {len(image_data)}")
    print(f"Images too close together (< 10s): {len(close_pairs)} pairs")
    print(f"Large gaps (> 2 min): {len(large_gaps)} gaps")
    print(f"Visual duplicates found: {len(duplicates)} pairs")
    
    print("\nOPTIMIZATION SUGGESTIONS:")
    print("-" * 60)
    
    if len(close_pairs) > 0:
        print(f"\n1. MINIMUM TIME INTERVAL:")
        print(f"   - {len(close_pairs)} pairs are captured within 10 seconds")
        print(f"   - Recommendation: Add minimum time interval (e.g., 15-30 seconds)")
        print(f"   - This would eliminate ~{len(close_pairs)} potential duplicates")
    
    if len(duplicates) > 0:
        print(f"\n2. SIMILARITY THRESHOLD:")
        print(f"   - {len(duplicates)} visual duplicates detected")
        print(f"   - Recommendation: Lower similarity threshold to 0.92-0.93")
        print(f"   - Or implement post-processing duplicate removal")
    
    if len(large_gaps) > 0:
        print(f"\n3. MISSING SLIDES:")
        print(f"   - {len(large_gaps)} large gaps detected (possible missing slides)")
        print(f"   - Recommendation: Check if slides were missed during transitions")
        print(f"   - Consider increasing frame rate or adjusting motion thresholds")
    
    print(f"\n4. POST-PROCESSING:")
    print(f"   - Implement post-processing to compare all images, not just consecutive")
    print(f"   - Compare each image with last N images (e.g., last 5-10)")
    print(f"   - Remove images that are too similar to recent captures")
    
    print(f"\n5. TIME-BASED FILTERING:")
    print(f"   - Enforce minimum time between captures (e.g., 20-30 seconds)")
    print(f"   - This prevents rapid-fire captures of the same slide")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        images_folder = sys.argv[1]
    else:
        images_folder = "output/Nov 21 2025 - Reengineering/images"
    
    analyze_images(images_folder)

