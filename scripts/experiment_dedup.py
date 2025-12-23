import os
import sys
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
import cv2
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def compute_histogram_similarity(img1, img2):
    """Compare images using histogram correlation."""
    hist1 = np.array(img1.convert('RGB').histogram())
    hist2 = np.array(img2.convert('RGB').histogram())
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    return np.corrcoef(hist1, hist2)[0, 1]

def process_images_multi_tier(image_dir, output_report_dir):
    image_dir = Path(image_dir)
    image_paths = sorted([p for p in image_dir.glob("*.png")])
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    results = []
    auto_discarded = []
    ambiguous = []
    unique = []

    # Cache hashes and images
    data = []
    print(f"Analyzing {len(image_paths)} images...")
    for p in image_paths:
        img = Image.open(p)
        data.append({
            'name': p.name,
            'path': p,
            'hash': imagehash.phash(img),
            'img': img
        })

    # Sequential comparison
    for i in range(len(data)):
        if i == 0:
            unique.append(data[i])
            continue
        
        is_duplicate = False
        best_match = None
        
        # Compare with previous 5 images (common pattern in slide extraction)
        search_range = range(max(0, i-5), i)
        for j in search_range:
            dist = data[i]['hash'] - data[j]['hash']
            hist_sim = compute_histogram_similarity(data[i]['img'], data[j]['img'])
            
            # Tier 1: Auto-Discard (High Confidence)
            if dist <= 2 and hist_sim >= 0.98:
                auto_discarded.append({
                    'image': data[i]['name'],
                    'matched_with': data[j]['name'],
                    'dist': dist,
                    'hist_sim': hist_sim,
                    'tier': 1
                })
                is_duplicate = True
                break
            
            # Tier 2: Ambiguous (Needs Review)
            if dist <= 8 or hist_sim >= 0.85:
                # We save all possible matches for review
                ambiguous.append({
                    'image': data[i]['name'],
                    'matched_with': data[j]['name'],
                    'dist': int(dist),
                    'hist_sim': float(hist_sim),
                    'tier': 2
                })
                is_duplicate = True
                # In experimental mode, we might want to continue searching, 
                # but for deduplication we stop at the first match.
                break
        
        if not is_duplicate:
            unique.append(data[i])

    # Save results to experimental folder
    report = {
        'total_images': len(image_paths),
        'auto_discarded_count': len(auto_discarded),
        'ambiguous_count': len(ambiguous),
        'unique_count': len(unique),
        'auto_discarded': auto_discarded,
        'ambiguous': ambiguous
    }
    
    report_file = Path(output_report_dir) / "dedup_results.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nDeduplication Results:")
    print(f"  Total: {len(image_paths)}")
    print(f"  Auto-Discarded (Tier 1): {len(auto_discarded)}")
    print(f"  Ambiguous (Tier 2): {len(ambiguous)}")
    print(f"  Unique: {len(unique)}")
    print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python experiment_dedup.py <image_dir> <output_report_dir>")
        sys.exit(1)
    
    process_images_multi_tier(sys.argv[1], sys.argv[2])
