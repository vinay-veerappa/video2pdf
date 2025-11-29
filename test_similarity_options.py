import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import sys
import os

def load_and_preprocess(path, size=(300, 200), crop_percent=0.0):
    """Load image, optionally crop borders, and resize."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    if crop_percent > 0:
        h, w = img.shape[:2]
        h_crop = int(h * crop_percent)
        w_crop = int(w * crop_percent)
        img = img[h_crop:h-h_crop, w_crop:w-w_crop]
    
    return cv2.resize(img, size)

def compare_ssim(img1, img2):
    """Standard SSIM comparison."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2, data_range=255)

def compare_histogram(img1, img2):
    """Compare using Histogram Correlation."""
    # Convert to HSV for better color separation
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare (Correlation)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def compare_orb(img1, img2):
    """Compare using ORB feature matching."""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0.0
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate a score based on number of good matches relative to total keypoints
    # This is a heuristic
    if len(kp1) == 0: return 0
    score = len(matches) / len(kp1)
    return min(score, 1.0) # Cap at 1.0

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_similarity_options.py <image1_path> <image2_path>")
        return

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    
    print(f"Comparing:\n  A: {os.path.basename(img1_path)}\n  B: {os.path.basename(img2_path)}\n")
    
    # 1. Standard SSIM (Baseline)
    img1_std = load_and_preprocess(img1_path)
    img2_std = load_and_preprocess(img2_path)
    score_ssim = compare_ssim(img1_std, img2_std)
    print(f"1. Standard SSIM:           {score_ssim:.4f}")
    
    # 2. Center Crop SSIM (Focus on core content, ignore 10% border)
    img1_crop = load_and_preprocess(img1_path, crop_percent=0.1)
    img2_crop = load_and_preprocess(img2_path, crop_percent=0.1)
    score_crop = compare_ssim(img1_crop, img2_crop)
    print(f"2. Center Crop SSIM (10%):  {score_crop:.4f}")
    
    # 3. Heavy Center Crop SSIM (Focus on core content, ignore 20% border)
    img1_crop_heavy = load_and_preprocess(img1_path, crop_percent=0.2)
    img2_crop_heavy = load_and_preprocess(img2_path, crop_percent=0.2)
    score_crop_heavy = compare_ssim(img1_crop_heavy, img2_crop_heavy)
    print(f"3. Center Crop SSIM (20%):  {score_crop_heavy:.4f}")

    # 4. Histogram Correlation
    score_hist = compare_histogram(img1_std, img2_std)
    print(f"4. Histogram Correlation:   {score_hist:.4f}")
    
    # 5. ORB Feature Matching
    score_orb = compare_orb(img1_std, img2_std)
    print(f"5. ORB Feature Match:       {score_orb:.4f}")

if __name__ == "__main__":
    main()
