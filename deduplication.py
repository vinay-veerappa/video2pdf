
import os
import cv2
import numpy as np
import imagehash
from PIL import Image, ImageFilter, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from pathlib import Path
import json

# Optional OCR
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
except Exception:
    OCR_AVAILABLE = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class ContentAnalyzer:
    """Methods for analyzing image content quality and detecting blanks."""
    
    @staticmethod
    def analyze_image_content(img, debug=False):
        """
        Analyze image to detect if it has minimal content.
        img: PIL Image
        """
        # Downscale for speed
        max_size = 640
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Convert
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        gray = np.array(img.convert('L'))
        
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Analyze center region (excludes 10% margin)
        top_margin = int(height * 0.10)
        bottom_margin = int(height * 0.90)
        left_margin = int(width * 0.10)
        right_margin = int(width * 0.90)
        
        center_gray = gray[top_margin:bottom_margin, left_margin:right_margin]
        center_array = img_array[top_margin:bottom_margin, left_margin:right_margin, :]
        
        # Metrics
        variance_full = np.var(img_array)
        variance_center = np.var(center_array)
        std_dev_full = np.std(gray)
        std_dev_center = np.std(center_gray)
        mean_brightness = np.mean(gray)
        mean_brightness_center = np.mean(center_gray)
        
        # Entropy
        gray_hist = np.histogram(gray, bins=256)[0]
        gray_hist = gray_hist[gray_hist > 0]
        gray_hist = gray_hist / gray_hist.sum()
        entropy = -np.sum(gray_hist * np.log2(gray_hist))
        
        # Color diversity
        hist_r = np.histogram(img_array[:,:,0], bins=32)[0]
        hist_g = np.histogram(img_array[:,:,1], bins=32)[0]
        hist_b = np.histogram(img_array[:,:,2], bins=32)[0]
        color_bins_used = np.sum(hist_r > 0) + np.sum(hist_g > 0) + np.sum(hist_b > 0)
        color_diversity = color_bins_used / 96
        
        # Dominant intensity
        gray_hist_full = np.histogram(gray.flatten(), bins=256)[0]
        max_intensity_percentage = np.max(gray_hist_full) / total_pixels * 100
        
        metrics = {
            'variance': float(variance_full),
            'variance_center': float(variance_center),
            'std_dev': float(std_dev_full),
            'std_dev_center': float(std_dev_center),
            'mean_brightness': float(mean_brightness),
            'mean_brightness_center': float(mean_brightness_center),
            'entropy': float(entropy),
            'color_diversity': float(color_diversity),
            'dominant_intensity_percentage': float(max_intensity_percentage)
        }
        
        # Blank Detection Logic
        blank_score = 0
        blank_reasons = []
        
        if variance_center < 500:
            blank_score += 1
            blank_reasons.append(f"Low variance in center ({variance_center:.1f})")
        if std_dev_center < 15:
            blank_score += 1
            blank_reasons.append(f"Low std deviation ({std_dev_center:.2f})")
        if entropy < 3.0:
            blank_score += 2
            blank_reasons.append(f"Very low entropy ({entropy:.2f})")
        if max_intensity_percentage > 85:
            blank_score += 1
            blank_reasons.append(f"Dominated by single intensity ({max_intensity_percentage:.1f}%)")
        if mean_brightness_center > 235 or mean_brightness_center < 30:
            blank_score += 1
            blank_reasons.append(f"Extreme brightness ({mean_brightness_center:.1f})")
        if color_diversity < 0.30:
            blank_score += 1
            blank_reasons.append(f"Low color diversity ({color_diversity:.2f})")
            
        # Non-slide detection (Low intensity + High diversity = Camera feed?)
        if max_intensity_percentage < 10 and color_diversity > 0.9:
            blank_score += 3
            blank_reasons.append("Likely non-slide (Low intensity, High diversity)")

        has_significant_content = (
            (variance_center > 2000 and std_dev_center > 35) or
            variance_center > 5000
        )
        
        if "Likely non-slide" in str(blank_reasons):
            has_significant_content = False
            
        # Force blank if entropy is very low
        if entropy < 3.0:
            is_blank = True
            blank_reasons.append(f"FORCE BLANK: Entropy {entropy:.2f} < 3.0")
        elif blank_score >= 2 and not has_significant_content:
            is_blank = True
        else:
            is_blank = False
            
        metrics['is_blank'] = is_blank
        metrics['blank_reasons'] = blank_reasons
        metrics['blank_score'] = blank_score
        
        # Content Score (0-100)
        score = 0
        score += min(25, variance_center / 100)
        score += min(25, std_dev_center * 5)
        score += min(25, entropy * 3)
        score += color_diversity * 25
        if is_blank:
            score = max(0, score - 50)
            
        metrics['content_score'] = float(score)
        return metrics

class DuplicateDetector:
    """Methods for detecting duplicate images using hashing and histograms."""
    
    @staticmethod
    def get_crop_box(img, method='auto', margin=0.10):
        """Get the crop box (left, top, right, bottom)."""
        width, height = img.size
        
        if method == 'content_aware':
            left = int(width * margin)
            top = int(height * margin)
            right = int(width * (1 - margin))
            bottom = int(height * (1 - margin))
            return (left, top, right, bottom)
        
        elif method == 'auto':
            # Simple UI detection logic
            img_array = np.array(img.convert('RGB'))
            row_vars = [np.var(img_array[i, :, :]) for i in range(height)]
            row_bright = [np.mean(img_array[i, :, :]) for i in range(height)]
            
            top_crop = 0
            for i in range(min(int(height * 0.15), 100)):
                if row_bright[i] < 50 or row_vars[i] < 100:
                    top_crop = i + 1
                else:
                    break
            
            bottom_crop = height
            for i in range(height - 1, max(height - int(height * 0.15), height - 100), -1):
                if row_bright[i] < 50 or row_vars[i] < 100:
                    bottom_crop = i
                else:
                    break
            return (0, top_crop, width, bottom_crop)
            
        else:
            return (0, 0, width, height)

    @staticmethod
    def compute_hashes(img):
        """Compute multiple hash types for an image."""
        return {
            'phash': imagehash.phash(img),
            # 'avg_hash': imagehash.average_hash(img), # Optional, can add if needed
            'dhash': imagehash.dhash(img),
            # 'whash': imagehash.whash(img)
        }

    @staticmethod
    def compute_histogram_similarity(img1, img2):
        """Compare images using histogram correlation."""
        try:
            h1 = np.array(img1.convert('RGB').histogram()).astype('float')
            h2 = np.array(img2.convert('RGB').histogram()).astype('float')
            h1 /= (h1.sum() + 1e-10)
            h2 /= (h2.sum() + 1e-10)
            return np.corrcoef(h1, h2)[0, 1]
        except:
            return 0.0

    @staticmethod
    def compare_images(data1, data2, threshold=10, hist_thresh=0.90):
        """
        Compare two image data dicts.
        data dicts must contain 'hashes' and preferably 'entropy'.
        """
        hashes1 = data1['hashes']
        hashes2 = data2['hashes']
        
        distances = {k: abs(hashes1[k] - hashes2[k]) for k in hashes1.keys() if k in hashes2}
        if not distances:
            return {'is_duplicate': False, 'score': 100}
            
        min_dist = min(distances.values())
        avg_dist = sum(distances.values()) / len(distances)
        
        # Entropy-based threshold adjustment
        ent1 = data1.get('entropy', 5.0)
        ent2 = data2.get('entropy', 5.0)
        avg_entropy = (ent1 + ent2) / 2
        
        adj_threshold = threshold
        if avg_entropy < 4.0:
            adj_threshold = max(threshold - 3, 5) # Stricter for simple images
        elif avg_entropy > 7.0:
            adj_threshold = threshold + 3 # Lenient for complex images
            
        # We need histogram similarity for confirmation
        # This requires the actual images or precomputed histograms.
        # If passed in data, use it.
        hist_sim = 0.0
        # If caller provided loaded images in data (optional)
        if 'image_obj' in data1 and 'image_obj' in data2:
             hist_sim = DuplicateDetector.compute_histogram_similarity(data1['image_obj'], data2['image_obj'])
        
        is_min_match = min_dist <= adj_threshold
        
        # If weak hash match but strong hist match?
        # Logic from image_dedup.py:
        if is_min_match and avg_dist > adj_threshold:
             if hist_sim > 0 and hist_sim < 0.85:
                 is_min_match = False
                 
        is_hist_match = hist_sim >= hist_thresh
        is_avg_match = avg_dist <= adj_threshold + 3 and hist_sim >= hist_thresh - 0.05
        
        is_dup = is_min_match or is_hist_match or is_avg_match
        
        return {
            'is_duplicate': is_dup,
            'score': min_dist,
            'avg_score': avg_dist,
            'histogram_similarity': hist_sim,
            'distances': distances
        }

def process_image_file(path, crop_method='auto', blur=True, downscale=False):
    """
    Process a single image file for deduplication.
    Returns a dict with hashes, metrics, and optionally the loaded image (if requested).
    """
    try:
        img = Image.open(path).convert('RGB')
        
        # Crop
        crop_box = DuplicateDetector.get_crop_box(img, method=crop_method)
        cropped = img.crop(crop_box)
        
        # Process for Hashing
        proc = cropped.copy()
        if blur:
            proc = proc.filter(ImageFilter.GaussianBlur(radius=2))
        if downscale:
             w, h = proc.size
             proc = proc.resize((w//2, h//2), Image.LANCZOS)
             
        # Compute Content Metrics (on original cropped)
        metrics = ContentAnalyzer.analyze_image_content(cropped)
        
        # Compute Hashes (on processed)
        hashes = DuplicateDetector.compute_hashes(proc)
        
        return {
            'path': str(path),
            'name': os.path.basename(path),
            'hashes': hashes,
            'metrics': metrics,
            'entropy': metrics['entropy'],
            # We don't return the image object to avoid pickling issues in multiprocessing
            # But we can re-open it later if needed for histogram
        }
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None
