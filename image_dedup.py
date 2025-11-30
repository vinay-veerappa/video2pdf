"""
Smart Video Conference Screenshot Deduplication
Automatically detects and crops out video conference UI artifacts
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw
import imagehash
from sklearn.metrics.pairwise import cosine_similarity


def detect_ui_regions(img, debug=False):
    """
    Automatically detect video conference UI regions to crop.
    Looks for:
    - Dark bars (speaker names, controls)
    - Consistent color regions (UI panels)
    - Text overlays (captions, badges)
    """
    img_array = np.array(img.convert('RGB'))
    height, width = img_array.shape[:2]
    
    # Analyze rows to find dark/consistent regions
    row_variances = []
    row_brightnesses = []
    
    for i in range(height):
        row = img_array[i, :, :]
        variance = np.var(row)
        brightness = np.mean(row)
        row_variances.append(variance)
        row_brightnesses.append(brightness)
    
    # Find top boundary (skip dark/low-variance top regions)
    top_crop = 0
    for i in range(min(int(height * 0.15), 100)):  # Check top 15% or 100px
        if row_brightnesses[i] < 50 or row_variances[i] < 100:
            top_crop = i + 1
        else:
            break
    
    # Find bottom boundary (skip dark/low-variance bottom regions)
    bottom_crop = height
    for i in range(height - 1, max(height - int(height * 0.15), height - 100), -1):
        if row_brightnesses[i] < 50 or row_variances[i] < 100:
            bottom_crop = i
        else:
            break
    
    # Analyze columns for side panels
    col_variances = []
    for j in range(width):
        col = img_array[:, j, :]
        variance = np.var(col)
        col_variances.append(variance)
    
    # Find left boundary
    left_crop = 0
    for j in range(min(int(width * 0.1), 50)):
        if col_variances[j] < 100:
            left_crop = j + 1
        else:
            break
    
    # Find right boundary
    right_crop = width
    for j in range(width - 1, max(width - int(width * 0.1), width - 50), -1):
        if col_variances[j] < 100:
            right_crop = j
        else:
            break
    
    crop_box = (left_crop, top_crop, right_crop, bottom_crop)
    
    if debug:
        print(f"  Detected UI regions:")
        print(f"    Top crop: {top_crop}px ({top_crop/height*100:.1f}%)")
        print(f"    Bottom crop: {height-bottom_crop}px ({(height-bottom_crop)/height*100:.1f}%)")
        print(f"    Left crop: {left_crop}px ({left_crop/width*100:.1f}%)")
        print(f"    Right crop: {width-right_crop}px ({(width-right_crop)/width*100:.1f}%)")
    
    return crop_box


def smart_crop_video_conference(img, method='auto', debug=False):
    """
    Intelligently crop video conference artifacts.
    
    Methods:
    - 'auto': Automatically detect UI regions
    - 'conservative': Remove top 5%, bottom 10%
    - 'aggressive': Remove top 8%, bottom 15%
    - 'sides': Also remove left/right 2%
    """
    width, height = img.size
    
    if method == 'auto':
        crop_box = detect_ui_regions(img, debug=debug)
        cropped = img.crop(crop_box)
    elif method == 'conservative':
        top = int(height * 0.05)
        bottom = int(height * 0.90)
        cropped = img.crop((0, top, width, bottom))
    elif method == 'aggressive':
        top = int(height * 0.08)
        bottom = int(height * 0.85)
        cropped = img.crop((0, top, width, bottom))
    elif method == 'sides':
        top = int(height * 0.05)
        bottom = int(height * 0.90)
        left = int(width * 0.02)
        right = int(width * 0.98)
        cropped = img.crop((left, top, right, bottom))
    else:
        cropped = img
    
    return cropped


def visualize_crop_regions(img_path, output_path=None, method='auto'):
    """
    Create a visualization showing what will be cropped.
    Useful for debugging and tuning.
    """
    img = Image.open(img_path)
    
    # Get crop box
    if method == 'auto':
        crop_box = detect_ui_regions(img, debug=True)
        left, top, right, bottom = crop_box
    else:
        width, height = img.size
        if method == 'conservative':
            top = int(height * 0.05)
            bottom = int(height * 0.90)
            left, right = 0, width
        elif method == 'aggressive':
            top = int(height * 0.08)
            bottom = int(height * 0.85)
            left, right = 0, width
        else:
            top, bottom = 0, height
            left, right = 0, width
    
    # Create visualization
    vis_img = img.copy()
    draw = ImageDraw.Draw(vis_img, 'RGBA')
    
    # Draw red overlay on regions to be cropped
    if top > 0:
        draw.rectangle([0, 0, img.width, top], fill=(255, 0, 0, 100))
    if bottom < img.height:
        draw.rectangle([0, bottom, img.width, img.height], fill=(255, 0, 0, 100))
    if left > 0:
        draw.rectangle([0, 0, left, img.height], fill=(255, 0, 0, 100))
    if right < img.width:
        draw.rectangle([right, 0, img.width, img.height], fill=(255, 0, 0, 100))
    
    # Draw green box around content area
    draw.rectangle([left, top, right, bottom], outline=(0, 255, 0, 255), width=3)
    
    if output_path:
        vis_img.save(output_path)
        print(f"Saved visualization to: {output_path}")
    
    return vis_img


def compute_multi_scale_hash(img):
    """
    Compute hashes at multiple scales for better accuracy.
    """
    hashes = {
        'phash': imagehash.phash(img),
        'avg_hash': imagehash.average_hash(img),
        'dhash': imagehash.dhash(img),
        'whash': imagehash.whash(img)
    }
    return hashes


def compare_images_smart(img1_data, img2_data, threshold=8):
    """
    Compare two images using multiple metrics after smart cropping.
    """
    # Compare each hash type
    distances = {}
    for hash_type in img1_data['hashes'].keys():
        dist = abs(img1_data['hashes'][hash_type] - img2_data['hashes'][hash_type])
        distances[hash_type] = dist
    
    # Average distance
    avg_distance = sum(distances.values()) / len(distances)
    
    # Compute histogram similarity
    hist_sim = compute_histogram_similarity(
        img1_data['cropped'], 
        img2_data['cropped']
    )
    
    # Decision: duplicate if any hash is very similar OR histogram is very similar
    is_duplicate = (
        min(distances.values()) <= threshold or 
        hist_sim >= 0.95
    )
    
    return {
        'is_duplicate': is_duplicate,
        'avg_distance': avg_distance,
        'min_distance': min(distances.values()),
        'distances': distances,
        'histogram_similarity': hist_sim
    }


def compute_histogram_similarity(img1, img2):
    """
    Compare images based on color histogram.
    """
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    
    # Compute histograms
    hist1 = np.array(img1.histogram())
    hist2 = np.array(img2.histogram())
    
    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    
    # Compute correlation
    correlation = np.corrcoef(hist1, hist2)[0, 1]
    
    return correlation


def find_duplicates_with_smart_crop(image_dir, 
                                     crop_method='auto',
                                     threshold=8,
                                     save_crops=False,
                                     debug=False):
    """
    Find duplicates using smart video conference cropping.
    
    Args:
        image_dir: Directory containing screenshots
        crop_method: 'auto', 'conservative', 'aggressive', or 'sides'
        threshold: Perceptual hash distance threshold (0-15)
        save_crops: Save cropped versions to see what's being compared
        debug: Print detailed information
    """
    print(f"=== SMART VIDEO CONFERENCE DEDUPLICATION ===")
    print(f"Crop method: {crop_method}")
    print(f"Threshold: {threshold}\n")
    
    # Load and process all images
    images_data = []
    crop_dir = Path(image_dir) / 'cropped_for_comparison' if save_crops else None
    
    if save_crops and crop_dir:
        crop_dir.mkdir(exist_ok=True)
        print(f"Saving cropped images to: {crop_dir}\n")
    
    for img_path in sorted(Path(image_dir).rglob('*')):
        # Skip output directories
        if any(x in str(img_path) for x in ['crop_visualizations', 'cropped_for_comparison', 'report']):
            continue

        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            try:
                if debug:
                    print(f"Processing: {img_path.name}")
                
                img = Image.open(img_path)
                
                # Smart crop
                cropped = smart_crop_video_conference(img, method=crop_method, debug=debug)
                
                # Save cropped version if requested
                if save_crops and crop_dir:
                    crop_path = crop_dir / f"cropped_{img_path.name}"
                    cropped.save(crop_path)
                
                # Compute multiple hashes
                hashes = compute_multi_scale_hash(cropped)
                
                images_data.append({
                    'path': str(img_path),
                    'name': img_path.name,
                    'original': img,
                    'cropped': cropped,
                    'hashes': hashes,
                    'original_size': img.size,
                    'cropped_size': cropped.size
                })
                
                if debug:
                    orig_w, orig_h = img.size
                    crop_w, crop_h = cropped.size
                    print(f"  Original: {orig_w}x{orig_h}")
                    print(f"  Cropped: {crop_w}x{crop_h} ({crop_w/orig_w*100:.1f}% x {crop_h/orig_h*100:.1f}%)")
                    print()
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Loaded {len(images_data)} images\n")
    
    # Find duplicates
    print("Comparing images...\n")
    duplicates = []
    
    for i in range(len(images_data)):
        for j in range(i + 1, len(images_data)):
            comparison = compare_images_smart(
                images_data[i], 
                images_data[j], 
                threshold
            )
            
            if comparison['is_duplicate']:
                duplicates.append({
                    'image1': images_data[i]['name'],
                    'image2': images_data[j]['name'],
                    'path1': images_data[i]['path'],
                    'path2': images_data[j]['path'],
                    **comparison
                })
    
    # Display results
    print(f"Found {len(duplicates)} duplicate pairs:\n")
    
    for idx, dup in enumerate(duplicates, 1):
        print(f"Pair {idx}:")
        print(f"  Image 1: {dup['image1']}")
        print(f"  Image 2: {dup['image2']}")
        print(f"  Average distance: {dup['avg_distance']:.1f}")
        print(f"  Min distance: {dup['min_distance']}")
        print(f"  Histogram similarity: {dup['histogram_similarity']:.3f}")
        print(f"  Hash distances: {', '.join(f'{k}={v}' for k, v in dup['distances'].items())}")
        print()
    
    return duplicates, images_data


def create_comparison_report(image_dir, output_html='comparison_report.html'):
    """
    Create an HTML report showing original vs cropped images side by side.
    """
    html_parts = [
        '<html><head><style>',
        'body { font-family: Arial; margin: 20px; }',
        '.comparison { margin: 20px 0; border: 1px solid #ccc; padding: 10px; }',
        'img { max-width: 45%; margin: 5px; border: 1px solid #999; }',
        'h2 { color: #333; }',
        '</style></head><body>',
        '<h1>Video Conference Screenshot Crop Comparison</h1>'
    ]
    
    for img_path in sorted(Path(image_dir).rglob('*')):
        # Skip output directories
        if any(x in str(img_path) for x in ['crop_visualizations', 'cropped_for_comparison', 'report']):
            continue
            
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            img = Image.open(img_path)
            
            # Create cropped version
            cropped = smart_crop_video_conference(img, method='auto')
            
            # Save both
            orig_name = f"report_orig_{img_path.name}"
            crop_name = f"report_crop_{img_path.name}"
            
            report_dir = Path(image_dir) / 'report'
            report_dir.mkdir(exist_ok=True)
            
            img.save(report_dir / orig_name)
            cropped.save(report_dir / crop_name)
            
            html_parts.append(f'<div class="comparison">')
            html_parts.append(f'<h2>{img_path.name}</h2>')
            html_parts.append(f'<img src="report/{orig_name}" alt="Original">')
            html_parts.append(f'<img src="report/{crop_name}" alt="Cropped">')
            html_parts.append(f'<p>Original: {img.size[0]}x{img.size[1]} → Cropped: {cropped.size[0]}x{cropped.size[1]}</p>')
            html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    
    with open(Path(image_dir) / output_html, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print(f"Created comparison report: {image_dir}/{output_html}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Video Conference Screenshot Deduplication")
    parser.add_argument("image_dir", help="Directory containing screenshots")
    parser.add_argument("--method", default="auto", choices=['auto', 'conservative', 'aggressive', 'sides'], help="Cropping method")
    parser.add_argument("--threshold", type=int, default=8, help="Duplicate threshold")
    parser.add_argument("--no-report", action="store_true", help="Skip generating HTML report")
    
    args = parser.parse_args()
    
    IMAGE_DIR = args.image_dir
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory not found: {IMAGE_DIR}")
        exit(1)
    
    print("SMART VIDEO CONFERENCE SCREENSHOT DEDUPLICATION")
    print("="*70 + "\n")
    
    # Step 1: Visualize what will be cropped (optional)
    print("Step 1: Creating crop visualizations...")
    vis_dir = Path(IMAGE_DIR) / 'crop_visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Get first few images
    images = sorted(list(Path(IMAGE_DIR).rglob('*.png'))) + sorted(list(Path(IMAGE_DIR).rglob('*.jpg')))
    
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        exit(1)
        
    for img_path in images[:3]:  # First 3 images
        vis_path = vis_dir / f"vis_{img_path.name}"
        visualize_crop_regions(img_path, vis_path, method=args.method)
    
    print(f"Check {vis_dir} to see what will be cropped\n")
    
    # Step 2: Find duplicates with smart cropping
    print("\nStep 2: Finding duplicates with smart cropping...")
    duplicates, images_data = find_duplicates_with_smart_crop(
        IMAGE_DIR,
        crop_method=args.method,
        threshold=args.threshold,
        save_crops=True,  # Save cropped versions for inspection
        debug=True
    )
    
    # Step 3: Create comparison report
    if not args.no_report:
        print("\nStep 3: Creating comparison report...")
        create_comparison_report(IMAGE_DIR)
    
    print("\n" + "="*70)
    print("\nKEY FEATURES:")
    print("  ✓ Automatic UI detection (finds dark bars, low-variance regions)")
    print("  ✓ Multi-scale hashing (phash, average, dhash, whash)")
    print("  ✓ Histogram similarity comparison")
    print("  ✓ Crop visualizations (red = removed, green = kept)")
    print("  ✓ Comparison report (side-by-side original vs cropped)")
    print("  ✓ Saves cropped versions for manual inspection")
    print("\nInstall: pip install pillow imagehash numpy scikit-learn")
