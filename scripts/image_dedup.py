"""
Smart Video Conference Screenshot Deduplication
Automatically detects and crops out video conference UI artifacts
Supports: Perceptual hashing, Histogram comparison, and Deep Learning
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

# Deep learning imports (optional)
try:
    import torch
    from torchvision import models, transforms
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


def analyze_image_content(img, debug=False):
    """
    Analyze image to detect if it has minimal content (blank or nearly blank).
    Uses center region analysis to avoid browser UI interference.
    """
    # Downscale for speed
    max_size = 640
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Convert once
    img_rgb = img.convert('RGB')
    img_array = np.array(img_rgb)
    gray = np.array(img.convert('L'))
    
    height, width = img_array.shape[:2]
    total_pixels = height * width
    
    # Analyze center region (excludes top/bottom 10% for browser UI)
    top_margin = int(height * 0.10)
    bottom_margin = int(height * 0.90)
    center_gray = gray[top_margin:bottom_margin, :]
    center_array = img_array[top_margin:bottom_margin, :, :]
    
    # Calculate metrics
    variance_full = np.var(img_array)
    variance_center = np.var(center_array)
    std_dev_full = np.std(gray)
    std_dev_center = np.std(center_gray)
    mean_brightness = np.mean(gray)
    mean_brightness_center = np.mean(center_gray)
    
    # Color diversity (histogram-based)
    hist_r = np.histogram(img_array[:,:,0], bins=32)[0]
    hist_g = np.histogram(img_array[:,:,1], bins=32)[0]
    hist_b = np.histogram(img_array[:,:,2], bins=32)[0]
    color_bins_used = np.sum(hist_r > 0) + np.sum(hist_g > 0) + np.sum(hist_b > 0)
    color_diversity = color_bins_used / 96
    
    # Entropy
    gray_hist = np.histogram(gray, bins=256)[0]
    gray_hist = gray_hist[gray_hist > 0]
    gray_hist = gray_hist / gray_hist.sum()
    entropy = -np.sum(gray_hist * np.log2(gray_hist))
    
    # Dominant intensity
    gray_hist_full = np.histogram(gray.flatten(), bins=256)[0]
    max_intensity_percentage = np.max(gray_hist_full) / total_pixels * 100
    
    # Store metrics
    metrics = {
        'variance': variance_full,
        'variance_center': variance_center,
        'std_dev': std_dev_full,
        'std_dev_center': std_dev_center,
        'mean_brightness': mean_brightness,
        'mean_brightness_center': mean_brightness_center,
        'entropy': entropy,
        'color_diversity': color_diversity,
        'dominant_intensity_percentage': max_intensity_percentage,
        'edge_density': std_dev_full / 255.0
    }
    
    # Determine if blank
    blank_score = 0
    blank_reasons = []
    
    # Check 1: Center variance
    if variance_center < 500:
        blank_score += 1
        blank_reasons.append(f"Low variance in center ({variance_center:.1f})")
    
    # Check 2: Center std dev
    if std_dev_center < 15:
        blank_score += 1
        blank_reasons.append(f"Low std deviation in center ({std_dev_center:.2f})")
    
    # Check 3: Entropy - WORTH 2 POINTS (most reliable)
    if entropy < 3.0:
        blank_score += 2
        blank_reasons.append(f"Very low entropy ({entropy:.2f})")
    
    # Check 4: Dominant intensity
    if max_intensity_percentage > 85:
        blank_score += 1
        blank_reasons.append(f"Dominated by single intensity ({max_intensity_percentage:.1f}%)")
    
    # Check 5: Extreme brightness in center
    if mean_brightness_center > 235 or mean_brightness_center < 30:
        blank_score += 1
        blank_reasons.append(f"Extreme brightness in center ({mean_brightness_center:.1f})")
    
    # Check 6: Color diversity
    if color_diversity < 0.30:
        blank_score += 1
        blank_reasons.append(f"Low color diversity ({color_diversity:.2f})")
    
    # Protection: Don't flag code/text editors as blank
    has_significant_content = (
        (variance_center > 2500 and std_dev_center > 35) or
        variance_center > 5000
    )
    
    # Final decision
    if blank_score >= 2 and not has_significant_content:
        is_blank = True
    else:
        is_blank = False
        blank_reasons = []
    
    metrics['is_blank'] = is_blank
    metrics['blank_reasons'] = blank_reasons
    metrics['blank_score'] = blank_score
    
    # Calculate content score
    score = 0
    score += min(25, variance_center / 100)
    score += min(25, std_dev_center * 5)
    score += min(25, entropy * 3)
    score += color_diversity * 25
    metrics['content_score'] = score
    
    if debug:
        print(f"  Content Analysis:")
        print(f"    Full Image:")
        print(f"      Variance: {variance_full:.2f}, Std Dev: {std_dev_full:.2f}")
        print(f"      Brightness: {mean_brightness:.1f}")
        print(f"    Center Region (excludes top/bottom 10%):")
        print(f"      Variance: {variance_center:.2f}, Std Dev: {std_dev_center:.2f}")
        print(f"      Brightness: {mean_brightness_center:.1f}")
        print(f"    Overall:")
        print(f"      Entropy: {entropy:.2f}")
        print(f"      Color diversity: {color_diversity:.2f}")
        print(f"      Dominant intensity: {max_intensity_percentage:.1f}%")
        print(f"    Detection:")
        print(f"      Blank indicators: {blank_score}/6")
        print(f"      Has significant content: {has_significant_content}")
        print(f"      blank_score >= 2: {blank_score >= 2}")
        print(f"      FINAL is_blank: {is_blank}")
        print(f"      Content score: {score:.2f}/100")
        if is_blank:
            print(f"    ⚠️  BLANK: {', '.join(blank_reasons)}")
        else:
            print(f"    ✓ Has content")
    
    return metrics


def get_image_files(image_dir, exclude_folders=None):
    """Get list of image files, excluding output folders."""
    if exclude_folders is None:
        exclude_folders = ['cropped_for_comparison', 'crop_visualizations', 'report', 'processed', '__pycache__']
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    for img_path in sorted(Path(image_dir).rglob('*')):
        if any(excluded in img_path.parts for excluded in exclude_folders):
            continue
        if img_path.suffix.lower() in image_extensions:
            image_files.append(img_path)
    
    return image_files


def find_blank_images(image_dir, content_threshold=25, save_report=True, debug=False):
    """Find images with minimal content."""
    import time
    
    print(f"=== BLANK/MINIMAL IMAGE DETECTION ===")
    print(f"Content threshold: {content_threshold}")
    print(f"Detection: 2+ indicators OR (entropy < 3.0 AND not significant content)\n")
    
    image_paths = get_image_files(image_dir)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], []
    
    print(f"Found {len(image_paths)} images to analyze\n")
    
    blank_images = []
    all_images = []
    start_time = time.time()
    
    for idx, img_path in enumerate(image_paths, 1):
        try:
            if debug:
                print(f"[{idx}/{len(image_paths)}] Analyzing: {img_path.name}")
            elif idx % 10 == 0:
                print(f"  Progress: {idx}/{len(image_paths)}")
            
            img = Image.open(img_path)
            metrics = analyze_image_content(img, debug=debug)
            
            image_data = {
                'path': str(img_path),
                'name': img_path.name,
                'size': img.size,
                'file_size': img_path.stat().st_size,
                **metrics
            }
            
            all_images.append(image_data)
            
            if metrics['is_blank'] or metrics['content_score'] < content_threshold:
                blank_images.append(image_data)
                if not debug:
                    print(f"⚠️  BLANK: {img_path.name} (score: {metrics['content_score']:.1f}, indicators: {metrics['blank_score']}/6)")
                    for reason in metrics['blank_reasons'][:3]:
                        print(f"     → {reason}")
            
            if debug:
                print()
            
        except Exception as e:
            print(f"Error: {img_path}: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(all_images)} images analyzed in {elapsed:.1f}s")
    print(f"Blank/minimal: {len(blank_images)} ({len(blank_images)/len(all_images)*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if blank_images:
        print("Blank images found:")
        for img in blank_images:
            print(f"  • {img['name']} (score: {img['content_score']:.1f}, indicators: {img['blank_score']}/6)")
    else:
        print("No blank images found. Lowest 5 scores:")
        for idx, img in enumerate(sorted(all_images, key=lambda x: x['content_score'])[:5], 1):
            print(f"  {idx}. {img['name']}")
            print(f"     Score: {img['content_score']:.1f}, Indicators: {img['blank_score']}/6")
            print(f"     Entropy: {img['entropy']:.2f}, Variance(ctr): {img['variance_center']:.1f}")
    
    # Generate HTML report
    if save_report and all_images:
        report_path = Path(image_dir) / 'blank_images_report.html'
        print(f"\nGenerating HTML report: {report_path}")
        try:
            create_blank_images_report(all_images, blank_images, report_path, content_threshold)
            print(f"✓ Report saved ({report_path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return blank_images, all_images


def create_blank_images_report(all_images, blank_images, output_path, threshold):
    """Create HTML report with image previews."""
    import base64
    from io import BytesIO
    
    def img_to_base64(path, max_w=800):
        try:
            img = Image.open(path)
            if img.width > max_w:
                ratio = max_w / img.width
                img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except:
            return None
    
    html = ['<!DOCTYPE html><html><head><meta charset="UTF-8">']
    html.append('<style>')
    html.append('body{font-family:Arial;margin:20px;background:#f5f5f5}')
    html.append('.container{max-width:1400px;margin:0 auto;background:white;padding:20px}')
    html.append('h1{color:#333;border-bottom:3px solid #e74c3c}')
    html.append('.blank{border:2px solid #e74c3c;background:#ffebee;padding:15px;margin:15px 0;border-radius:5px}')
    html.append('.good{border:2px solid #4caf50;background:#f1f8f4;padding:15px;margin:15px 0;border-radius:5px}')
    html.append('.btn{background:#2196F3;color:white;border:none;padding:8px 16px;cursor:pointer;border-radius:3px}')
    html.append('.btn:hover{background:#1976D2}')
    html.append('.hidden{display:none}')
    html.append('img{max-width:100%;border:1px solid #ccc;margin:10px 0}')
    html.append('.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}')
    html.append('.metric{background:white;padding:10px;border-radius:3px;border:1px solid #ddd}')
    html.append('</style>')
    html.append('<script>function toggle(id){var e=document.getElementById(id);var b=document.getElementById("b"+id);e.classList.toggle("hidden");b.textContent=e.classList.contains("hidden")?"Show":"Hide";}</script>')
    html.append('</head><body><div class="container">')
    html.append(f'<h1>Blank Image Detection Report</h1>')
    html.append(f'<p>Total: {len(all_images)} | Blank: {len(blank_images)} ({len(blank_images)/max(len(all_images),1)*100:.1f}%)</p>')
    
    if blank_images:
        html.append('<h2 style="color:#e74c3c">Blank Images (Delete These)</h2>')
        for i, img in enumerate(sorted(blank_images, key=lambda x: x['content_score']), 1):
            html.append(f'<div class="blank"><h3>{i}. {img["name"]}</h3>')
            b64 = img_to_base64(img['path'])
            if b64:
                html.append(f'<button class="btn" id="b{i}" onclick="toggle(\'{i}\')">Show</button>')
                html.append(f'<div id="{i}" class="hidden"><img src="{b64}"></div>')
            html.append(f'<p><b>Score:</b> {img["content_score"]:.1f}/100 | <b>Indicators:</b> {img["blank_score"]}/6</p>')
            html.append('<div class="metrics">')
            html.append(f'<div class="metric">Variance(ctr): {img["variance_center"]:.0f}</div>')
            html.append(f'<div class="metric">Entropy: {img["entropy"]:.2f}</div>')
            html.append(f'<div class="metric">Brightness(ctr): {img["mean_brightness_center"]:.0f}</div>')
            html.append('</div>')
            if img['blank_reasons']:
                html.append(f'<p><b>Why:</b> {"; ".join(img["blank_reasons"])}</p>')
            html.append('</div>')
    
    good = [img for img in all_images if img not in blank_images]
    if good:
        html.append('<h2 style="color:#4caf50">Lowest-Scoring (Not Blank)</h2>')
        for i, img in enumerate(sorted(good, key=lambda x: x['content_score'])[:10], 1):
            html.append(f'<div class="good"><h3>{i}. {img["name"]}</h3>')
            b64 = img_to_base64(img['path'])
            if b64:
                html.append(f'<button class="btn" id="bg{i}" onclick="toggle(\'g{i}\')">Show</button>')
                html.append(f'<div id="g{i}" class="hidden"><img src="{b64}"></div>')
            html.append(f'<p><b>Score:</b> {img["content_score"]:.1f}/100 | <b>Indicators:</b> {img["blank_score"]}/6</p>')
            html.append('<div class="metrics">')
            html.append(f'<div class="metric">Variance(ctr): {img["variance_center"]:.0f}</div>')
            html.append(f'<div class="metric">Entropy: {img["entropy"]:.2f}</div>')
            html.append(f'<div class="metric">StdDev(ctr): {img["std_dev_center"]:.1f}</div>')
            html.append('</div></div>')
    
    html.append('</div></body></html>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html))


def detect_ui_regions(img, debug=False):
    """Detect video conference UI regions to crop."""
    img_array = np.array(img.convert('RGB'))
    height, width = img_array.shape[:2]
    
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


def smart_crop_video_conference(img, method='auto', debug=False):
    """Intelligently crop video conference artifacts."""
    width, height = img.size
    
    if method == 'auto':
        crop_box = detect_ui_regions(img, debug)
        return img.crop(crop_box)
    elif method == 'conservative':
        return img.crop((0, int(height * 0.05), width, int(height * 0.90)))
    elif method == 'aggressive':
        return img.crop((0, int(height * 0.08), width, int(height * 0.85)))
    else:
        return img


def compute_multi_scale_hash(img):
    """Compute multiple hash types."""
    return {
        'phash': imagehash.phash(img),
        'avg_hash': imagehash.average_hash(img),
        'dhash': imagehash.dhash(img),
        'whash': imagehash.whash(img)
    }


def compute_histogram_similarity(img1, img2):
    """Compare images using histogram correlation."""
    hist1 = np.array(img1.convert('RGB').histogram())
    hist2 = np.array(img2.convert('RGB').histogram())
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    return np.corrcoef(hist1, hist2)[0, 1]


def compare_images_smart(img1_data, img2_data, threshold=12, hist_thresh=0.90):
    """Compare two images using multiple metrics."""
    img1 = img1_data.get('processed', img1_data['cropped'])
    img2 = img2_data.get('processed', img2_data['cropped'])
    
    distances = {k: abs(img1_data['hashes'][k] - img2_data['hashes'][k]) 
                 for k in img1_data['hashes'].keys()}
    
    min_dist = min(distances.values())
    avg_dist = sum(distances.values()) / len(distances)
    hist_sim = compute_histogram_similarity(img1, img2)
    
    is_dup = (min_dist <= threshold or hist_sim >= hist_thresh or 
              (avg_dist <= threshold + 3 and hist_sim >= hist_thresh - 0.05))
    
    return {
        'is_duplicate': is_dup,
        'min_distance': min_dist,
        'avg_distance': avg_dist,
        'histogram_similarity': hist_sim,
        'distances': distances
    }


def find_duplicates_with_smart_crop(image_dir, crop_method='auto', threshold=12, 
                                     histogram_threshold=0.90, use_blur=True, 
                                     downscale=True, save_crops=False, debug=False):
    """Find duplicates using smart cropping."""
    print(f"=== DUPLICATE DETECTION ===")
    print(f"Threshold: {threshold}, Blur: {use_blur}, Downscale: {downscale}\n")
    
    image_paths = get_image_files(image_dir)
    if not image_paths:
        return [], []
    
    print(f"Processing {len(image_paths)} images...\n")
    
    images_data = []
    crop_dir = Path(image_dir) / 'cropped_for_comparison' if save_crops else None
    if save_crops and crop_dir:
        crop_dir.mkdir(exist_ok=True)
    
    for idx, img_path in enumerate(image_paths, 1):
        try:
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(image_paths)}")
            
            img = Image.open(img_path)
            cropped = smart_crop_video_conference(img, method=crop_method)
            processed = cropped.copy()
            
            if use_blur:
                processed = processed.filter(ImageFilter.GaussianBlur(radius=2))
            if downscale:
                w, h = processed.size
                processed = processed.resize((w//2, h//2), Image.LANCZOS)
            
            crop_path_str = None
            if save_crops and crop_dir:
                crop_path = crop_dir / f"cropped_{img_path.name}"
                cropped.save(crop_path)
                crop_path_str = str(crop_path)
            
            hashes = compute_multi_scale_hash(processed)
            
            images_data.append({
                'path': str(img_path),
                'name': img_path.name,
                'cropped': cropped,
                'processed': processed,
                'hashes': hashes,
                'crop_path': crop_path_str
            })
        except Exception as e:
            print(f"Error: {img_path}: {e}")
    
    print(f"\n✓ Loaded {len(images_data)} images\n")
    print("Comparing images...\n")
    
    duplicates = []
    for i in range(len(images_data)):
        for j in range(i + 1, len(images_data)):
            comp = compare_images_smart(images_data[i], images_data[j], threshold, histogram_threshold)
            if comp['is_duplicate']:
                duplicates.append({
                    'image1': images_data[i]['name'],
                    'image2': images_data[j]['name'],
                    'path1': images_data[i]['path'],
                    'path2': images_data[j]['path'],
                    'crop_path1': images_data[i]['crop_path'],
                    'crop_path2': images_data[j]['crop_path'],
                    **comp
                })
    
    print(f"Found {len(duplicates)} duplicate pairs\n")
    
    for idx, dup in enumerate(duplicates, 1):
        print(f"Pair {idx}: {dup['image1']} ↔ {dup['image2']}")
        print(f"  Distance: {dup['min_distance']}, Histogram: {dup['histogram_similarity']:.3f}\n")
    
    return duplicates, images_data


def keep_best_from_duplicates(duplicates, keep_strategy='first'):
    """Recommend which images to keep."""
    groups = []
    for dup in duplicates:
        img1, img2 = dup['image1'], dup['image2']
        found = None
        for g in groups:
            if img1 in g or img2 in g:
                found = g
                break
        if found:
            found.update([img1, img2])
        else:
            groups.append({img1, img2})
    
    recs = []
    for idx, group in enumerate(groups, 1):
        group_list = sorted(list(group))
        keep = group_list[0]
        remove = group_list[1:]
        print(f"Group {idx}: KEEP {keep}, REMOVE {', '.join(remove)}")
        recs.append({'keep': keep, 'remove': remove})
    
    return recs


    return recs


def create_duplicate_pairs_report(duplicates, output_dir, filename='duplicates_report.html'):
    """
    Create an HTML report showing detected duplicate pairs side-by-side.
    Uses cropped images if available.
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
        '.note { color: #666; font-style: italic; font-size: 0.9em; margin-bottom: 10px; }',
        '</style></head><body>',
        f'<h1>Duplicate Pairs Report ({len(duplicates)} pairs)</h1>',
        '<p class="note">Note: Showing cropped images used for comparison. If cropped images are not available, original images are shown.</p>'
    ]
    
    # Sort duplicates by similarity (lower distance is better)
    sorted_dups = sorted(duplicates, key=lambda x: x['avg_distance'])
    
    for i, dup in enumerate(sorted_dups, 1):
        # Determine which paths to use (cropped if available, else original)
        path1 = dup.get('crop_path1') or dup['path1']
        path2 = dup.get('crop_path2') or dup['path2']
        is_cropped = bool(dup.get('crop_path1') and dup.get('crop_path2'))
        
        # Calculate relative path for images
        try:
            # Try to make paths relative to the report file
            rel_path1 = os.path.relpath(path1, output_dir)
            rel_path2 = os.path.relpath(path2, output_dir)
        except ValueError:
            # Fallback to absolute paths if on different drives
            rel_path1 = Path(path1).as_uri()
            rel_path2 = Path(path2).as_uri()
            
        html_parts.append(f'<div class="pair">')
        html_parts.append(f'<h3>Pair #{i}: {dup["image1"]} vs {dup["image2"]}</h3>')
        
        html_parts.append('<div class="images">')
        html_parts.append(f'<div class="image-container"><img src="{rel_path1}" alt="{dup["image1"]}"><p>{dup["image1"]}{" (Cropped)" if is_cropped else ""}</p></div>')
        html_parts.append(f'<div class="image-container"><img src="{rel_path2}" alt="{dup["image2"]}"><p>{dup["image2"]}{" (Cropped)" if is_cropped else ""}</p></div>')
        html_parts.append('</div>')
        
        html_parts.append('<div class="metrics">')
        html_parts.append(f'<strong>Average Distance:</strong> <span class="metric-good">{dup["avg_distance"]:.2f}</span> (Lower is more similar)<br>')
        html_parts.append(f'<strong>Histogram Similarity:</strong> <span class="metric-good">{dup["histogram_similarity"]:.3f}</span> (Higher is more similar)<br>')
        html_parts.append(f'<strong>Hash Distances:</strong> {", ".join(f"{k}={v}" for k, v in dup["distances"].items())}')
        html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    
    report_path = Path(output_dir) / filename
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    print(f"Created duplicate pairs report: {report_path}")


def run_deduplication(image_dir, output_dir, mode, args, report_suffix=''):
    """
    Run a single deduplication pass.
    """
    # Get settings from mode
    settings = get_settings_from_mode(mode)
    
    # Override with custom arguments if provided
    threshold = args.threshold if args.threshold else settings['threshold']
    histogram_threshold = settings.get('histogram_threshold', 0.92)
    
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
    
    # Force save_crops to True for reporting
    save_crops = True
    
    duplicates, images_data = find_duplicates_with_smart_crop(
        image_dir,
        crop_method=args.crop_method,
        threshold=threshold,
        histogram_threshold=histogram_threshold,
        use_blur=use_blur,
        downscale=downscale,
        save_crops=save_crops,
        debug=args.debug
    )
    
    # Create duplicate pairs report
    report_name = f'duplicates_report_{mode}{report_suffix}.html'
    create_duplicate_pairs_report(duplicates, output_dir, report_name)
    
    return duplicates


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Deduplication Tool')
    parser.add_argument('directory', type=str)
    parser.add_argument('--mode', type=str, 
                       choices=['lenient', 'moderate', 'strict', 'detect-blank', 'compare-all'],
                       default='moderate')
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--content-threshold', type=int, default=25)
    parser.add_argument('--crop-method', type=str, default='auto')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--no-blur', action='store_true')
    parser.add_argument('--downscale', action='store_true')
    parser.add_argument('--save-crops', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--detect-blank-first', action='store_true')
    
    return parser.parse_args()


def get_settings_from_mode(mode):
    """Get preset settings."""
    modes = {
        'lenient': {'threshold': 15, 'histogram_threshold': 0.88, 'use_blur': True, 'downscale': True},
        'moderate': {'threshold': 12, 'histogram_threshold': 0.92, 'use_blur': True, 'downscale': False},
        'strict': {'threshold': 8, 'histogram_threshold': 0.95, 'use_blur': False, 'downscale': False}
    }
    return modes.get(mode, modes['moderate'])


if __name__ == "__main__":
    args = parse_arguments()
    
    IMAGE_DIR = Path(args.directory)
    if not IMAGE_DIR.exists() or not IMAGE_DIR.is_dir():
        print(f"Error: Invalid directory")
        sys.exit(1)
    
    images = get_image_files(IMAGE_DIR)
    if not images:
        print(f"No images found")
        sys.exit(1)
    
    print(f"Found {len(images)} images in {IMAGE_DIR}\n")
    
    settings = get_settings_from_mode(args.mode)
    threshold = args.threshold or settings.get('threshold', 12)
    
    use_blur = args.blur or (not args.no_blur and settings.get('use_blur', True))
    downscale = args.downscale or settings.get('downscale', False)
    
    # Detect blank images
    if args.mode == 'detect-blank' or args.detect_blank_first:
        blank_images, all_imgs = find_blank_images(IMAGE_DIR, args.content_threshold, True, args.debug)
        
        if args.mode == 'detect-blank':
            print("\nDone! Check 'blank_images_report.html'")
            sys.exit(0)
    
    # Compare all modes
    if args.mode == 'compare-all':
        print("COMPARING ALL MODES (Strict, Moderate, Lenient)")
        modes = ['strict', 'moderate', 'lenient']
        results = {}
        
        for mode in modes:
            duplicates = run_deduplication(IMAGE_DIR, IMAGE_DIR, mode, args)
            results[mode] = len(duplicates)
            
        print("\n" + "="*70)
        print("COMPARISON RESULTS:")
        for mode in modes:
            print(f"  {mode.upper()}: Found {results[mode]} duplicate pairs")
        print(f"\nCheck the generated HTML reports in {IMAGE_DIR}")
        
    # Single run
    else:
        run_deduplication(IMAGE_DIR, IMAGE_DIR, args.mode, args)
        
        # We don't need to print recommendations here as run_deduplication generates the report
        # But we can print a summary if we want
    
    print("\nDone!")