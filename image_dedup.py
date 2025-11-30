"""
Smart Video Conference Screenshot Deduplication
Automatically detects and crops out video conference UI artifacts
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFilter
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
                                     threshold=12,  # Increased from 8
                                     histogram_threshold=0.90,  # Lowered from 0.95
                                     use_blur=True,  # NEW: Add blur preprocessing
                                     downscale=True,  # NEW: Downscale before comparison
                                     save_crops=False,
                                     debug=False):
    """
    Find duplicates using smart video conference cropping.
    OPTIMIZED FOR VIDEO SCREENSHOTS - more lenient matching.
    
    Args:
        image_dir: Directory containing screenshots
        crop_method: 'auto', 'conservative', 'aggressive', or 'sides'
        threshold: Perceptual hash distance (12-15 recommended for video)
        histogram_threshold: Color similarity (0.88-0.92 for video)
        use_blur: Apply Gaussian blur to reduce noise
        downscale: Resize before comparison for more forgiving matching
        save_crops: Save cropped versions to see what's being compared
        debug: Print detailed information
    """
    
    print(f"=== SMART VIDEO CONFERENCE DEDUPLICATION (OPTIMIZED) ===")
    print(f"Crop method: {crop_method}")
    print(f"Threshold: {threshold} (12-15 recommended for video)")
    print(f"Histogram threshold: {histogram_threshold}")
    print(f"Use blur: {use_blur}")
    print(f"Downscale: {downscale}\n")
    
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
                
                # NEW: Apply preprocessing for better video screenshot matching
                processed = cropped.copy()
                
                # Apply blur to reduce compression artifacts
                if use_blur:
                    processed = processed.filter(ImageFilter.GaussianBlur(radius=2))
                
                # Downscale to make small differences negligible
                if downscale:
                    w, h = processed.size
                    processed = processed.resize((w//2, h//2), Image.LANCZOS)
                
                # Save cropped version if requested
                if save_crops and crop_dir:
                    crop_path = crop_dir / f"cropped_{img_path.name}"
                    cropped.save(crop_path)
                    if use_blur or downscale:
                        processed_path = crop_dir / f"processed_{img_path.name}"
                        processed.save(processed_path)
                
                # Compute multiple hashes on processed image
                hashes = compute_multi_scale_hash(processed)
                
                images_data.append({
                    'path': str(img_path),
                    'name': img_path.name,
                    'original': img,
                    'cropped': cropped,
                    'processed': processed,  # NEW: Store processed version
                    'hashes': hashes,
                    'original_size': img.size,
                    'cropped_size': cropped.size,
                    'processed_size': processed.size
                })
                
                if debug:
                    orig_w, orig_h = img.size
                    crop_w, crop_h = cropped.size
                    proc_w, proc_h = processed.size
                    print(f"  Original: {orig_w}x{orig_h}")
                    print(f"  Cropped: {crop_w}x{crop_h}")
                    print(f"  Processed: {proc_w}x{proc_h}")
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


def keep_best_from_duplicates(duplicates, keep_strategy='first'):
    """
    Analyze duplicate pairs and recommend which images to keep/remove.
    """
    # Build adjacency list of duplicates
    adj = defaultdict(set)
    for dup in duplicates:
        adj[dup['image1']].add(dup['image2'])
        adj[dup['image2']].add(dup['image1'])
    
    # Find connected components (groups of duplicates)
    seen = set()
    groups = []
    
    for img in adj:
        if img not in seen:
            component = set()
            stack = [img]
            while stack:
                node = stack.pop()
                if node not in seen:
                    seen.add(node)
                    component.add(node)
                    stack.extend(adj[node] - seen)
            groups.append(list(component))
            
    recommendations = []
    for group in groups:
        # Sort by name to be deterministic
        group.sort()
        
        if keep_strategy == 'first':
            keep = group[0]
            remove = group[1:]
        elif keep_strategy == 'last':
            keep = group[-1]
            remove = group[:-1]
        else:
            keep = group[0]
            remove = group[1:]
            
        recommendations.append({
            'keep': keep,
            'remove': remove
        })
        
    return recommendations


def create_duplicate_pairs_report(duplicates, output_dir, filename='duplicates_report.html'):
    """
    Create an HTML report showing detected duplicate pairs side-by-side.
    """
    html_parts = [
        '<html><head><style>',
        'body { font-family: Arial; margin: 20px; background: #f5f5f5; }',
        '.pair { background: white; margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.images { display: flex; justify-content: space-around; align-items: center; }',
        '.image-container { text-align: center; width: 45%; }',
        'img { max-width: 100%; height: auto; border: 1px solid #eee; }',
        '.metrics { margin-top: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; font-size: 0.9em; }',
        '.metric-bad { color: #d32f2f; font-weight: bold; }',
        '.metric-good { color: #388e3c; font-weight: bold; }',
        'h1 { color: #333; }',
        'h3 { margin-top: 0; color: #555; }',
        '</style></head><body>',
        f'<h1>Duplicate Pairs Report ({len(duplicates)} pairs)</h1>'
    ]
    
    # Sort duplicates by similarity (lower distance is better)
    sorted_dups = sorted(duplicates, key=lambda x: x['avg_distance'])
    
    for i, dup in enumerate(sorted_dups, 1):
        # Calculate relative path for images
        try:
            # Try to make paths relative to the report file
            rel_path1 = os.path.relpath(dup['path1'], output_dir)
            rel_path2 = os.path.relpath(dup['path2'], output_dir)
        except ValueError:
            # Fallback to absolute paths if on different drives
            rel_path1 = Path(dup['path1']).as_uri()
            rel_path2 = Path(dup['path2']).as_uri()
            
        html_parts.append(f'<div class="pair">')
        html_parts.append(f'<h3>Pair #{i}: {dup["image1"]} vs {dup["image2"]}</h3>')
        
        html_parts.append('<div class="images">')
        html_parts.append(f'<div class="image-container"><img src="{rel_path1}" alt="{dup["image1"]}"><p>{dup["image1"]}</p></div>')
        html_parts.append(f'<div class="image-container"><img src="{rel_path2}" alt="{dup["image2"]}"><p>{dup["image2"]}</p></div>')
        html_parts.append('</div>')
        
        html_parts.append('<div class="metrics">')
        html_parts.append(f'<strong>Average Distance:</strong> <span class="metric-good">{dup["avg_distance"]:.2f}</span> (Lower is more similar)<br>')
        html_parts.append(f'<strong>Histogram Similarity:</strong> <span class="metric-good">{dup["histogram_similarity"]:.3f}</span> (Higher is more similar)<br>')
        html_parts.append(f'<strong>Hash Distances:</strong> {", ".join(f"{k}={v}" for k, v in dup["distances"].items())}')
        html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    
    report_path = output_dir / filename
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print(f"Created duplicate pairs report: {report_path}")


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Smart Video Conference Screenshot Deduplication Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python image_dedup.py /path/to/screenshots

  # Compare all modes (generates 3 reports)
  python image_dedup.py /path/to/screenshots --compare-modes

  # Custom settings
  python image_dedup.py /path/to/screenshots --mode lenient --report
        """
    )
    
    # Required arguments
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing video conference screenshots'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['lenient', 'moderate', 'strict'],
        default='moderate',
        help='Detection sensitivity mode (default: moderate)'
    )
    
    parser.add_argument(
        '--compare-modes',
        action='store_true',
        help='Run all 3 modes and generate separate reports for comparison'
    )
    
    # Advanced settings (override mode)
    parser.add_argument(
        '--threshold',
        type=int,
        help='Perceptual hash threshold (8-20, higher=more lenient)'
    )
    
    parser.add_argument(
        '--histogram-threshold',
        type=float,
        help='Histogram similarity threshold (0.0-1.0, lower=more lenient)'
    )
    
    parser.add_argument(
        '--crop-method',
        type=str,
        choices=['auto', 'conservative', 'aggressive', 'sides'],
        default='auto',
        help='Method for cropping UI regions (default: auto)'
    )
    
    parser.add_argument(
        '--blur',
        action='store_true',
        help='Apply blur to reduce noise (recommended for video)'
    )
    
    parser.add_argument(
        '--no-blur',
        action='store_true',
        help='Disable blur preprocessing'
    )
    
    parser.add_argument(
        '--downscale',
        action='store_true',
        help='Downscale images before comparison'
    )
    
    parser.add_argument(
        '--no-downscale',
        action='store_true',
        help='Disable downscaling'
    )
    
    # Output options
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save cropped images for inspection'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization showing crop regions'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML comparison report'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print detailed debugging information'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory for output files (default: same as input)'
    )
    
    return parser.parse_args()


def get_settings_from_mode(mode):
    """
    Get preset settings based on mode.
    """
    modes = {
        'lenient': {
            'threshold': 15,
            'histogram_threshold': 0.88,
            'use_blur': True,
            'downscale': True,
            'description': 'Catches all visually similar images (recommended for video)'
        },
        'moderate': {
            'threshold': 12,
            'histogram_threshold': 0.92,
            'use_blur': True,
            'downscale': False,
            'description': 'Balanced accuracy and recall (default)'
        },
        'strict': {
            'threshold': 8,
            'histogram_threshold': 0.95,
            'use_blur': False,
            'downscale': False,
            'description': 'Only near-identical images'
        }
    }
    return modes.get(mode, modes['moderate'])


def run_deduplication(image_dir, output_dir, mode, args, report_suffix=''):
    """
    Run a single deduplication pass.
    """
    # Get settings from mode
    settings = get_settings_from_mode(mode)
    
    # Override with custom arguments if provided
    threshold = args.threshold if args.threshold else settings['threshold']
    histogram_threshold = args.histogram_threshold if args.histogram_threshold else settings['histogram_threshold']
    
    # Handle blur flags
    if args.blur:
        use_blur = True
    elif args.no_blur:
        use_blur = False
    else:
        use_blur = settings['use_blur']
    
    # Handle downscale flags
    if args.downscale:
        downscale = True
    elif args.no_downscale:
        downscale = False
    else:
        downscale = settings['downscale']
        
    print(f"\nRunning Mode: {mode.upper()}")
    print(f"Settings: Threshold={threshold}, Hist={histogram_threshold}, Blur={use_blur}, Downscale={downscale}")
    
    duplicates, images_data = find_duplicates_with_smart_crop(
        image_dir,
        crop_method=args.crop_method,
        threshold=threshold,
        histogram_threshold=histogram_threshold,
        use_blur=use_blur,
        downscale=downscale,
        save_crops=args.save_crops,
        debug=args.debug
    )
    
    # Create duplicate pairs report
    report_name = f'duplicates_report_{mode}{report_suffix}.html'
    create_duplicate_pairs_report(duplicates, output_dir, report_name)
    
    return duplicates


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate directory
    IMAGE_DIR = Path(args.directory)
    if not IMAGE_DIR.exists():
        print(f"Error: Directory '{IMAGE_DIR}' does not exist")
        sys.exit(1)
    
    if not IMAGE_DIR.is_dir():
        print(f"Error: '{IMAGE_DIR}' is not a directory")
        sys.exit(1)
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else IMAGE_DIR
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("SMART VIDEO CONFERENCE SCREENSHOT DEDUPLICATION")
    print("="*70 + "\n")
    
    if args.compare_modes:
        print("COMPARING ALL MODES (Strict, Moderate, Lenient)")
        modes = ['strict', 'moderate', 'lenient']
        results = {}
        
        for mode in modes:
            duplicates = run_deduplication(IMAGE_DIR, output_dir, mode, args)
            results[mode] = len(duplicates)
            
        print("\n" + "="*70)
        print("COMPARISON RESULTS:")
        for mode in modes:
            print(f"  {mode.upper()}: Found {results[mode]} duplicate pairs")
        print(f"\nCheck the generated HTML reports in {output_dir}")
        
    else:
        # Single run
        run_deduplication(IMAGE_DIR, output_dir, args.mode, args)
        
        # Create crop comparison report if requested (only for single run)
        if args.report:
            create_comparison_report(IMAGE_DIR)
    
    print("\n" + "=" * 70)
    print("\nDone!")
