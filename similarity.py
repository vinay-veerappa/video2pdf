import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

try:
    from imagededup.methods import PHash, CNN
    from sklearn.metrics.pairwise import cosine_similarity
    IMAGEDEDUP_AVAILABLE = True
except ImportError:
    IMAGEDEDUP_AVAILABLE = False

# Global instances for caching
_phasher_instance = None
_cnn_instance = None

def get_phasher():
    global _phasher_instance
    if _phasher_instance is None and IMAGEDEDUP_AVAILABLE:
        _phasher_instance = PHash()
    return _phasher_instance

def get_cnn():
    global _cnn_instance
    if _cnn_instance is None and IMAGEDEDUP_AVAILABLE:
        _cnn_instance = CNN()
    return _cnn_instance

def calculate_similarity(img1, img2, method='ssim'):
    """
    Calculate similarity between two images using specified method.
    Methods: 'ssim' (default), 'phash', 'cnn'
    """
    if method == 'ssim':
        # Center Crop SSIM (10%)
        h, w = img1.shape[:2]
        crop_h = int(h * 0.1)
        crop_w = int(w * 0.1)
        
        if h > 20 and w > 20:
            img1 = img1[crop_h:h-crop_h, crop_w:w-crop_w]
            img2 = img2[crop_h:h-crop_h, crop_w:w-crop_w]
        
        img1_resized = cv2.resize(img1, (300, 200))
        img2_resized = cv2.resize(img2, (300, 200))
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2, data_range=255)
        
    elif method == 'phash':
        if not IMAGEDEDUP_AVAILABLE:
            print("Warning: imagededup not installed, falling back to SSIM")
            return calculate_similarity(img1, img2, 'ssim')
            
        phasher = get_phasher()
        # PHash expects RGB usually, but works with BGR array
        # encode_image accepts image_array
        try:
            h1 = phasher.encode_image(image_array=img1)
            h2 = phasher.encode_image(image_array=img2)
            dist = phasher.hamming_distance(h1, h2)
            # Normalize distance to 0-1 similarity (0 dist = 1.0 sim)
            # Max ham dist for 64-bit hash is 64
            return 1.0 - (dist / 64.0)
        except Exception as e:
            print(f"PHash error: {e}")
            return 0.0

    elif method == 'cnn':
        if not IMAGEDEDUP_AVAILABLE:
            print("Warning: imagededup not installed, falling back to SSIM")
            return calculate_similarity(img1, img2, 'ssim')
            
        cnn = get_cnn()
        try:
            emb1 = cnn.encode_image(image_array=img1)
            emb2 = cnn.encode_image(image_array=img2)
            
            if isinstance(emb1, np.ndarray):
                emb1 = emb1.reshape(1, -1)
                emb2 = emb2.reshape(1, -1)
            
            return cosine_similarity(emb1, emb2)[0][0]
        except Exception as e:
            print(f"CNN error: {e}")
            return 0.0
            
    elif method == 'grid':
        # Grid SSIM
        # Returns boolean is_dup and score (ratio of matching cells)
        # We return the score as similarity
        is_dup, score = grid_ssim(img1, img2)
        return score

    return 0.0


def dhash(image, hash_size=8):
    """
    Calculate the difference hash (dHash) of an image.
    This is a perceptual hash that is robust to scaling and minor color changes.
    """
    # Resize to (hash_size + 1, hash_size)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Compare adjacent pixels
    diff = gray[:, 1:] > gray[:, :-1]
    # Convert to integer
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def calculate_hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes"""
    return bin(int(hash1) ^ int(hash2)).count('1')

def grid_ssim(img1, img2, grid_size=(8, 8), threshold=0.95, min_matching_cells=0.8):
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
    
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            cell1 = g1[y1:y2, x1:x2]
            cell2 = g2[y1:y2, x1:x2]
            
            score = ssim(cell1, cell2, data_range=255)
            if score > threshold:
                matching_cells += 1
                
    match_ratio = matching_cells / total_cells
    return match_ratio >= min_matching_cells, match_ratio
