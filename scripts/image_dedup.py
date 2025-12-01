"""
Smart Video Conference Screenshot Deduplication
Automatically detects and crops out video conference UI artifacts
Supports: Perceptual hashing, Histogram comparison, and Deep Learning
"""

import os
import sys
import argparse
import json
import numpy as np
import multiprocessing
from functools import partial

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
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFilter
import imagehash
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# OCR import (optional)
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


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
    
    # Analyze center region (excludes 10% margin on all sides for browser/video UI)
    top_margin = int(height * 0.10)
    bottom_margin = int(height * 0.90)
    left_margin = int(width * 0.10)
    right_margin = int(width * 0.90)
    
    center_gray = gray[top_margin:bottom_margin, left_margin:right_margin]
    center_array = img_array[top_margin:bottom_margin, left_margin:right_margin, :]
    
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
    
    # Check 7: Non-Slide Detection (Camera feeds, photos, complex backgrounds)
    # Refined based on user feedback to target specific low-intensity camera feeds (e.g. 043, 032, 085)
    # These specific examples have:
    # - Dominant intensity < 10% (043=8.2%, 032=8.2%)
    # - Color diversity = 1.00 (Very high)
    # - OR Very low entropy (< 2.0) like 085
    
    is_likely_nonslide = False
    
        
    
    #if is_likely_nonslide:
    #    blank_score += 3
    
    # Check 7: Non-Slide Detection (Camera feeds, photos, complex backgrounds)
    # Aggressive check: Low intensity + High diversity
    # We rely on OCR to save any valid slides that get flagged here.
    # REFINED: Only flag if intensity is VERY low (< 10%) and diversity is VERY high (> 0.9)
    if max_intensity_percentage < 10 and color_diversity > 0.9:
        blank_score += 3
        blank_reasons.append(f"Likely non-slide (Low intensity {max_intensity_percentage:.1f}%, Diversity {color_diversity:.2f})")
    
    # Protection: Don't flag code/text editors as blank
    # Editors usually have high variance/std_dev but also high dominant intensity (background)
    # So we only protect if dominant intensity is reasonable
    has_significant_content = (
        (variance_center > 2000 and std_dev_center > 35) or
        variance_center > 5000
    )
    
    # Override protection if it looks like a camera feed
    # (OCR will be the final judge)
    if "Likely non-slide" in str(blank_reasons):
        has_significant_content = False

    # Final decision
    # FORCE BLANK if entropy is very low (User Request)
    if entropy < 3.0:
        is_blank = True
        blank_reasons.append(f"FORCE BLANK: Entropy {entropy:.2f} < 3.0")
    elif blank_score >= 2 and not has_significant_content:
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
    
    # Penalize non-slides in the score
    if is_likely_nonslide:
        score = max(0, score - 50)
        
    metrics['content_score'] = score
    
    if debug:
        print(f"  Content Analysis:")
        print(f"    Full Image:")
        print(f"      Variance: {variance_full:.2f}, Std Dev: {std_dev_full:.2f}")
        print(f"      Brightness: {mean_brightness:.1f}")
        print(f"    Center Region (excludes 10% margin):")
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
            print(f"    ⚠️  BLANK/NON-SLIDE: {', '.join(blank_reasons)}")
        else:
            print(f"    ✓ Has content")
    
    return metrics


def check_text_content(img, reader):
    """
    Check for text content using OCR.
    Returns (has_text, text_count, confidence)
    """
    if not reader:
        return False, 0, 0.0, []
        
    try:
        # Convert to bytes for EasyOCR
        img_np = np.array(img)
        result = reader.readtext(img_np)
        
        # Filter for confident text
        valid_text = [text for bbox, text, conf in result if conf > 0.4 and len(text.strip()) > 1]
        text_count = len(valid_text)
        avg_conf = sum(conf for _, _, conf in result) / len(result) if result else 0
        
        return text_count > 0, text_count, avg_conf, valid_text
    except Exception as e:
        print(f"OCR Error: {e}")
        return False, 0, 0.0, []


def check_graphics_content(img, entropy=None, debug_path=None):
    """
    Check for significant graphics (charts, diagrams) using line detection.
    Focuses on center region to avoid UI borders.
    Returns (has_graphics, score, reason)
    """
    try:
        # Crop to center 60% (20% margins) to avoid border lines/UI
        w, h = img.size
        left, top, right, bottom = int(w*0.20), int(h*0.20), int(w*0.80), int(h*0.80)
        crop = img.crop((left, top, right, bottom))
        
        # Save debug visualization if requested
        if debug_path:
            from PIL import ImageDraw
            try:
                # Create a visualization: Original with box + Cropped
                vis = Image.new('RGB', (w + (right-left), h), (255, 255, 255))
                
                # Draw box on original
                orig_with_box = img.copy()
                draw = ImageDraw.Draw(orig_with_box)
                draw.rectangle([left, top, right, bottom], outline="red", width=5)
                vis.paste(orig_with_box, (0, 0))
                
                # Paste crop
                vis.paste(crop, (w, top)) # Align vertically with the crop area
                
                vis.save(debug_path)
            except Exception as e:
                print(f"Failed to save debug crop: {e}")

        # Safety check: If entropy is extremely low (< 2.0), it's likely a blank screen
        # with artifacts (like a player bar) that look like lines. Don't trust it.
        if entropy is not None and entropy < 2.0:
            return False, 0, f"Ignored lines due to low entropy ({entropy:.2f})"

        # Convert PIL to OpenCV format
        img_np = np.array(crop.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Detection
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Line Detection (Hough)
        # distinct lines often indicate charts/diagrams
        # Min line length 100px to avoid noise/artifacts
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        # Criteria for "Graphics"
        # - Significant straight lines (> 1) implies artificial structure (charts)
        # User requested threshold reduction to 1
        
        is_graphic = False
        reason = []
        
        if line_count > 1:
            is_graphic = True
            reason.append(f"Found {line_count} straight lines (Chart/Grid)")
        else:
            reason.append(f"Only {line_count} lines found (Threshold > 1)")
            
        return is_graphic, line_count, ", ".join(reason)
        
    except Exception as e:
        print(f"Graphics Check Error: {e}")
        return False, 0, ""


def get_image_files(image_dir, exclude_folders=None, recursive=False):
    """Get list of image files, excluding output folders."""
    if exclude_folders is None:
        exclude_folders = ['cropped_for_comparison', 'crop_visualizations', 'report', 'processed', '__pycache__', 'debug_crops']
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    # Use glob('*') by default (non-recursive) to avoid picking up output folders
    # Only use rglob if recursive=True
    iterator = Path(image_dir).rglob('*') if recursive else Path(image_dir).glob('*')
    
    for img_path in sorted(iterator):
        if not img_path.is_file():
            continue
            
        # Check excludes (important even for non-recursive if folders are in the list)
        if any(excluded in img_path.parts for excluded in exclude_folders):
            continue
            
        if img_path.suffix.lower() in image_extensions:
            image_files.append(img_path)
    
    return image_files


def process_image_for_blank(args):
    """Worker function for parallel blank image detection."""
    img_path, debug = args
    try:
        img = Image.open(img_path)
        metrics = analyze_image_content(img, debug=debug)
        
        return {
            'path': str(img_path),
            'name': img_path.name,
            'size': img.size,
            'file_size': img_path.stat().st_size,
            **metrics
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def find_blank_images(image_dir, content_threshold=25, debug=False):
    """
    Find blank or near-blank images in a directory.
    Uses content analysis, OCR, and graphics detection.
    """
    import time
    
    image_dir = Path(image_dir)
    output_dir = image_dir.parent
    
    print(f"=== BLANK/MINIMAL IMAGE DETECTION ===")
    print(f"Content threshold: {content_threshold}")
    print(f"Detection: 2+ indicators OR (entropy < 3.0 AND not significant content)\n")
    
    image_paths = get_image_files(image_dir)
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return [], []
    
    print(f"Found {len(image_paths)} images to analyze\n")
    
    # Initialize OCR if requested
    ocr_reader = None
    if OCR_AVAILABLE:
        print("Initializing OCR engine (EasyOCR)...")
        try:
            # Suppress verbose output
            import logging
            logging.getLogger('easyocr').setLevel(logging.ERROR)
            ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            print("✓ OCR initialized")
        except Exception as e:
            print(f"⚠ OCR initialization failed: {e}")
    else:
        print("⚠ EasyOCR not installed. Skipping OCR verification.")

    blank_images = []
    all_images = []
    start_time = time.time()
    
    # Parallel Processing for Metrics Calculation
    print(f"Analyzing {len(image_paths)} images in parallel...")
    pool_args = [(p, debug) for p in image_paths]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_image_for_blank, pool_args)
    
    # Filter out failed results
    results = [r for r in results if r is not None]
    
    # Sequential Post-Processing (OCR/Graphics) - Hard to parallelize due to GPU/File IO
    # But metrics calculation is the heavy part for CPU.
    
    for idx, image_data in enumerate(results, 1):
        try:
            img_path = Path(image_data['path'])
            
            if debug:
                print(f"[{idx}/{len(results)}] Checking: {image_data['name']}")
            elif idx % 50 == 0:
                print(f"  Post-processing: {idx}/{len(results)}")
            
            all_images.append(image_data)
            
            metrics = image_data # It's already flattened
            
            if metrics['is_blank'] or metrics['content_score'] < content_threshold:
                # OCR Verification for borderline cases
                is_confirmed_blank = True
                ocr_text = []
                
                if ocr_reader:
                    # We need to re-open image for OCR/Graphics check
                    # This adds IO overhead but saves memory vs passing images around
                    img = Image.open(img_path)
                    
                    # Run OCR on the center crop to verify
                    w, h = img.size
                    crop = img.crop((int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)))
                    has_text, count, conf, texts = check_text_content(crop, ocr_reader)
                    
                    if has_text:
                        # If meaningful text is found, it's likely a slide!
                        if count >= 3: # Threshold: at least 3 words/blocks
                            is_confirmed_blank = False
                            metrics['is_blank'] = False
                            metrics['blank_reasons'].append(f"SAVED by OCR: Found {count} text blocks")
                            # Boost score to save it
                            metrics['content_score'] = max(metrics['content_score'], 50)
                            ocr_text = texts
                
                # Graphics Verification
                debug_crop_path = None
                has_graphics = False
                graph_reason = ""
                
                # Only check graphics if still blank (optimization)
                if is_confirmed_blank:
                    img = Image.open(img_path) # Re-open
                    if debug:
                        debug_dir = output_dir / "debug_crops"
                        debug_dir.mkdir(exist_ok=True)
                        debug_crop_path = debug_dir / f"{img_path.stem}_crop.jpg"
                    
                    has_graphics, graph_score, graph_reason = check_graphics_content(img, metrics['entropy'], debug_crop_path)
                    image_data['graphics_info'] = graph_reason
                    image_data['debug_crop_path'] = str(debug_crop_path) if debug_crop_path else None
                    
                    if has_graphics:
                        is_confirmed_blank = False
                        metrics['is_blank'] = False
                        metrics['blank_reasons'].append(f"SAVED by Graphics: {graph_reason}")
                        metrics['content_score'] = max(metrics['content_score'], 50)
                
                if is_confirmed_blank:
                    blank_images.append(image_data)
                    if not debug and idx % 10 == 0:
                         # Print only occasionally to reduce spam
                         pass
                else:
                    # It was saved
                    pass
            
            image_data['ocr_text'] = ocr_text if 'ocr_text' in locals() else []
            if 'graphics_info' not in image_data:
                 image_data['graphics_info'] = ""
            
        except Exception as e:
            print(f"Error post-processing {img_path}: {e}")
    
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
            html.append(f'<div class="metric">Color Div: {img["color_diversity"]:.2f}</div>')
            html.append(f'<div class="metric">Dom. Int: {img["dominant_intensity_percentage"]:.1f}%</div>')
            html.append(f'<div class="metric">StdDev(ctr): {img["std_dev_center"]:.1f}</div>')
            html.append('</div>')
            if img.get('ocr_text'):
                html.append(f'<p style="color:#2e7d32"><b>OCR Found:</b> {len(img["ocr_text"])} blocks (e.g. "{img["ocr_text"][0] if img["ocr_text"] else ""}")</p>')
            if img.get('graphics_info'):
                html.append(f'<p style="color:#1976D2"><b>Graphics:</b> {img["graphics_info"]}</p>')
            
            # Debug Crop Visualization
            if img.get('debug_crop_path') and os.path.exists(img['debug_crop_path']):
                crop_b64 = img_to_base64(img['debug_crop_path'])
                if crop_b64:
                    html.append(f'<div style="margin-top:10px"><button class="btn" onclick="toggle(\'dc{i}\')">Show Graphics Debug</button>')
                    html.append(f'<div id="dc{i}" class="hidden" style="margin-top:5px"><img src="{crop_b64}" style="max-width:100%; border:1px solid #ccc"></div></div>')

            if img.get('blank_reasons'):
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
            html.append(f'<div class="metric">Brightness(ctr): {img["mean_brightness_center"]:.0f}</div>')
            html.append(f'<div class="metric">Color Div: {img["color_diversity"]:.2f}</div>')
            html.append(f'<div class="metric">Dom. Int: {img["dominant_intensity_percentage"]:.1f}%</div>')
            html.append(f'<div class="metric">StdDev(ctr): {img["std_dev_center"]:.1f}</div>')
            html.append('</div>')
            if img.get('ocr_text'):
                html.append(f'<p style="color:#2e7d32"><b>OCR Found:</b> {len(img["ocr_text"])} blocks</p>')
            if img.get('graphics_info'):
                html.append(f'<p style="color:#1976D2"><b>Graphics:</b> {img["graphics_info"]}</p>')
            if img.get('blank_reasons'):
                html.append(f'<p><b>Why:</b> {"; ".join(img["blank_reasons"])}</p>')
            html.append('</div>')
    
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


def get_crop_box(img, method='auto', margin=0.10, debug=False):
    """Get the crop box (left, top, right, bottom) for the given method."""
    width, height = img.size
    
    if method == 'auto':
        return detect_ui_regions(img, debug)
    elif method == 'content_aware':
        # Crop to center region based on margin (default 20% -> 0.20)
        left = int(width * margin)
        top = int(height * margin)
        right = int(width * (1 - margin))
        bottom = int(height * (1 - margin))
        return (left, top, right, bottom)
    elif method == 'conservative':
        return (0, int(height * 0.05), width, int(height * 0.90))
    elif method == 'aggressive':
        return (0, int(height * 0.08), width, int(height * 0.85))
    else:
        return (0, 0, width, height)


def smart_crop_video_conference(img, method='auto', margin=0.10, debug=False):
    """Intelligently crop video conference artifacts."""
    crop_box = get_crop_box(img, method, margin, debug)
    return img.crop(crop_box)


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


def get_processed_image(img_data):
    """Lazy load and process image if not in memory."""
    if 'processed' in img_data and img_data['processed'] is not None:
        return img_data['processed']
    
    # Reload from crop path (faster than original) or original path
    path = img_data.get('crop_path') or img_data['path']
    img = Image.open(path)
    
    # We need to apply the same transforms (blur/downscale)
    # But we don't know the flags here! 
    # We should probably store the flags in img_data or assume standard settings?
    # Actually, for histogram comparison, exact blur/downscale matters less than consistency.
    # But let's try to be consistent.
    # For now, let's just use the loaded image (maybe resize if it's huge).
    
    # Optimization: Cache it
    img_data['processed'] = img
    return img

def compare_images_smart(img1_data, img2_data, threshold=12, hist_thresh=0.90):
    """Compare two images using multiple metrics."""
    img1 = get_processed_image(img1_data)
    img2 = get_processed_image(img2_data)
    
    distances = {k: abs(img1_data['hashes'][k] - img2_data['hashes'][k]) 
                 for k in img1_data['hashes'].keys()}
    
    min_dist = min(distances.values())
    avg_dist = sum(distances.values()) / len(distances)
    hist_sim = compute_histogram_similarity(img1, img2)
    
    # Entropy-Aware Threshold Adjustment
    # Calculate entropy for both images (using grayscale)
    def get_entropy(img):
        gray = np.array(img.convert('L'))
        hist = np.histogram(gray, bins=256)[0]
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist))

    entropy1 = get_entropy(img1)
    entropy2 = get_entropy(img2)
    avg_entropy = (entropy1 + entropy2) / 2
    
    # Adjust threshold based on entropy
    # Low entropy (simple slides) -> Stricter (Lower threshold)
    # High entropy (complex photos) -> Lenient (Higher threshold)
    adjusted_threshold = threshold
    if avg_entropy < 4.0:
        adjusted_threshold = max(threshold - 3, 5) # Stricter for simple images
    elif avg_entropy > 7.0:
        adjusted_threshold = threshold + 3 # More lenient for complex images
        
    # Logic Refinement:
    # 1. Strong Match: Min distance is very low AND histogram is decent
    # 2. Histogram Match: Histogram is very high (colors are identical)
    # 3. Average Match: All hashes agree reasonably well
    
    is_min_match = min_dist <= adjusted_threshold
    
    # Safety: If relying on min_dist (single hash) and average is high (hashes disagree),
    # we MUST have strong histogram confirmation.
    if is_min_match and avg_dist > adjusted_threshold:
        if hist_sim < 0.85:
            is_min_match = False
            
    is_hist_match = hist_sim >= hist_thresh
    is_avg_match = avg_dist <= adjusted_threshold + 3 and hist_sim >= hist_thresh - 0.05
    
    
    is_dup = is_min_match or is_hist_match or is_avg_match
    
    match_reasons = []
    if is_min_match: match_reasons.append("Hash")
    if is_hist_match: match_reasons.append("Hist")
    if is_avg_match: match_reasons.append("AvgHash")
    
    return {
        'is_duplicate': is_dup,
        'score': min_dist, # Renamed from min_distance
        'avg_score': avg_dist, # Renamed from avg_distance
        'histogram_similarity': hist_sim,
        'distances': distances,
        'entropy': avg_entropy,
        'adjusted_threshold': adjusted_threshold,
        'reason': "+".join(match_reasons) if is_dup else "None"
    }


def process_image_for_dedup(args):
    """Worker function for parallel deduplication processing."""
    img_path, blur_enabled, downscale, crop_dir = args
    try:
        img = Image.open(img_path)
        cropped = smart_crop_video_conference(img, method='auto') # Always auto crop
        processed = cropped.copy()
        
        if blur_enabled:
            processed = processed.filter(ImageFilter.GaussianBlur(radius=2))
        if downscale:
            w, h = processed.size
            processed = processed.resize((w//2, h//2), Image.LANCZOS)
        
        crop_path_str = None
        if crop_dir:
            crop_path = crop_dir / f"cropped_{img_path.name}"
            cropped.save(crop_path)
            crop_path_str = str(crop_path)
        
        hashes = compute_multi_scale_hash(processed)
        
        # We can't return Image objects (not picklable/too large), so we return data
        # We'll reload/reprocess images only if needed for detailed comparison?
        # Actually, we need 'processed' for histogram comparison later.
        # But passing images back from subprocess is slow.
        # Better to return hashes and path, and reload if needed?
        # Or just return the small processed image as array?
        # Let's return the hashes and path. We can reload for histogram if needed.
        # Wait, histogram comparison needs the image.
        # Let's return the processed image converted to array (picklable).
        
        return {
            'path': str(img_path),
            'name': img_path.name,
            'hashes': hashes,
            'crop_path': crop_path_str,
            # 'processed_array': np.array(processed) # Optional if we want to avoid reload
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def find_duplicates_with_smart_crop(image_dir, threshold=10, blur_enabled=True, downscale=False, sequential=False, file_list=None):
    """
    Find duplicate images using smart cropping and perceptual hashing.
    """
    print("=== DUPLICATE DETECTION ===")
    print(f"Threshold: {threshold}, Blur: {blur_enabled}, Downscale: {downscale}")
    print(f"Method: {'Sequential (Optimized)' if sequential else 'Pairwise (O(N^2))'}")
    
    if file_list:
        image_paths = file_list
        print(f"Processing {len(image_paths)} provided images...")
    else:
        image_paths = get_image_files(image_dir)
        if not image_paths:
            return [], []
        print(f"Processing {len(image_paths)} images...\n")
    
    images_data = []
    crop_dir = Path(image_dir) / 'cropped_for_comparison' # Always save crops for combined report
    crop_dir.mkdir(exist_ok=True)
    
    # Parallel Processing
    print(f"Computing hashes for {len(image_paths)} images in parallel...")
    pool_args = [(p, blur_enabled, downscale, crop_dir) for p in image_paths]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(process_image_for_dedup, pool_args)
    
    # Filter and Reconstruct
    for res in results:
        if res:
            # We need to reload the image for histogram comparison later?
            # Or we can modify compare_images_smart to handle missing 'processed' image
            # by reloading it on demand.
            # For now, let's reload it here to keep compatibility with existing code structure
            # This is a trade-off. Parallel hashing saves time. Reloading adds some IO.
            # But since we have 3600 images, keeping all in memory is bad anyway.
            # Let's modify compare_images_smart to reload if needed.
            # But wait, compare_images_smart expects 'processed' or 'cropped' key.
            
            # Let's lazily load in the loop?
            # Actually, for the sequential loop, we only need 2 images in memory at a time.
            # For pairwise, we need random access.
            
            # Let's just store the path and reload in compare_images_smart?
            # No, compare_images_smart takes a dict.
            
            # I'll add a 'lazy_load' flag or handle it.
            # For now, let's just add the path and let the comparison logic handle it.
            # I will modify compare_images_smart to load from path if image object is missing.
            
            images_data.append(res)

    print(f"\n✓ Loaded {len(images_data)} images\n")
    print("Comparing images...\n")
    
    duplicates = []
    
    if sequential:
        # Optimized Sequential Logic: O(N)
        # Compare current image against a reference. 
        # If duplicate -> mark as duplicate of reference, continue.
        # If different -> new image becomes reference.
        if len(images_data) > 0:
            current_ref_idx = 0
            for i in range(1, len(images_data)):
                # Use a fixed histogram threshold for this function, e.g., 0.90
                comp = compare_images_smart(images_data[current_ref_idx], images_data[i], threshold, hist_thresh=0.90)
                
                if comp['is_duplicate']:
                    # It is a duplicate of the current reference
                    print(f"Pair {len(duplicates)+1}: {images_data[current_ref_idx]['name']} ↔ {images_data[i]['name']}")
                    print(f"  Reason: {comp['reason']} | Dist: {comp['score']}, Hist: {comp['histogram_similarity']:.3f}")
                    
                    duplicates.append({
                        'image1': images_data[current_ref_idx]['name'],
                        'image2': images_data[i]['name'],
                        'path1': images_data[current_ref_idx]['path'],
                        'path2': images_data[i]['path'],
                        'crop_path1': images_data[current_ref_idx]['crop_path'],
                        'crop_path2': images_data[i]['crop_path'],
                        **comp
                    })
                else:
                    # It is different, so it becomes the new reference
                    current_ref_idx = i
                    
        print(f"Found {len(duplicates)} duplicate pairs (Sequential Scan)\n")
        
    else:
        # Standard All-vs-All Logic: O(N^2)
        for i in range(len(images_data)):
            for j in range(i + 1, len(images_data)):
                # Use a fixed histogram threshold for this function, e.g., 0.90
                comp = compare_images_smart(images_data[i], images_data[j], threshold, hist_thresh=0.90)
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
        print(f"Found {len(duplicates)} duplicate pairs (out of {len(images_data) * (len(images_data)-1) // 2} possible pairs)\n")
    
    print(f"Found {len(duplicates)} duplicate pairs (out of {len(images_data) * (len(images_data)-1) // 2} possible pairs)\n")
    
    for idx, dup in enumerate(duplicates, 1):
        print(f"Pair {idx}: {dup['image1']} ↔ {dup['image2']}")
        print(f"  Distance: {dup['score']}, Histogram: {dup['histogram_similarity']:.3f}\n")
    
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


def create_duplicate_pairs_report(duplicates, output_dir, filename='duplicates_report.html', sequential=False):
    """
    Create an HTML report showing detected duplicate pairs.
    If sequential=True, groups duplicates by the original reference image.
    """
    html_parts = [
        '<html><head><style>',
        'body { font-family: Arial; margin: 20px; background: #f5f5f5; }',
        '.pair { background: white; margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.group { background: white; margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.images { display: flex; justify-content: space-around; align-items: center; }',
        '.group-images { display: flex; flex-wrap: wrap; gap: 15px; }',
        '.image-container { text-align: center; width: 45%; }',
        '.ref-container { text-align: center; width: 300px; border: 2px solid #2196F3; padding: 10px; background: #e3f2fd; }',
        '.dup-container { text-align: center; width: 250px; border: 1px solid #f44336; padding: 10px; background: #ffebee; }',
        'img { max-width: 100%; height: auto; border: 1px solid #eee; }',
        '.metrics { margin-top: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; font-size: 0.8em; text-align: left; }',
        '.metric-bad { color: #d32f2f; font-weight: bold; }',
        '.metric-good { color: #388e3c; font-weight: bold; }',
        'h1 { color: #333; }',
        'h3 { margin-top: 0; color: #555; }',
        '.note { color: #666; font-style: italic; font-size: 0.9em; margin-bottom: 10px; }',
        '</style></head><body>',
        f'<h1>Duplicate Report ({len(duplicates)} matches found)</h1>',
        '<p class="note">Note: Showing cropped images used for comparison. If cropped images are not available, original images are shown.</p>'
    ]

    if sequential:
        # Group by Original (image1)
        from collections import defaultdict
        groups = defaultdict(list)
        for dup in duplicates:
            groups[dup['image1']].append(dup)
            
        html_parts.append(f'<h2>Sequential Duplicate Groups ({len(groups)} unique slides found)</h2>')
        
        for ref_name, dups in groups.items():
            # Get paths from the first duplicate entry
            ref_path = dups[0].get('crop_path1') or dups[0]['path1']
            try:
                rel_ref_path = os.path.relpath(ref_path, output_dir)
            except ValueError:
                rel_ref_path = Path(ref_path).as_uri()
            
            # Create verification string
            dup_names = [d['image2'] for d in dups]
            verify_str = f"{ref_name}, {', '.join(dup_names)}"
            
            html_parts.append(f'<div style="background:#333; color:#fff; padding:10px; border-radius:4px; margin-bottom:15px; font-family:monospace; font-size:0.8em; overflow-x:auto;">')
            html_parts.append(f'<strong>Verify this group:</strong><br>')
            html_parts.append(f'python scripts/image_dedup.py "{output_dir}" --verify "{verify_str}" --mode moderate')
            html_parts.append(f'</div>')
            
            html_parts.append('<div class="group-images">')
            
            # Original Image
            html_parts.append(f'<div class="ref-container"><strong>ORIGINAL (KEEP)</strong><br><img src="{rel_ref_path}"><br>{ref_name}</div>')
            
            # Duplicate Images
            for dup in dups:
                dup_path = dup.get('crop_path2') or dup['path2']
                try:
                    rel_dup_path = os.path.relpath(dup_path, output_dir)
                except ValueError:
                    rel_dup_path = Path(dup_path).as_uri()
                    
                html_parts.append(f'<div class="dup-container"><strong>DUPLICATE (REMOVE)</strong><br><img src="{rel_dup_path}"><br>{dup["image2"]}')
                html_parts.append('<div class="metrics">')
                html_parts.append(f'Dist: {dup["min_distance"]} | Hist: {dup["histogram_similarity"]:.3f}<br>')
                html_parts.append(f'Hash: {", ".join(f"{k}={v}" for k, v in dup["distances"].items())}')
                html_parts.append('</div></div>')
                
            html_parts.append('</div></div>')
            
    else:
        # Standard Pairwise Report
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


def organize_files(image_dir, all_results, blanks, mode='moderate'):
    """
    Copy files to organized folders: unique, duplicates, blanks.
    Does NOT delete original files.
    """
    import shutil
    
    print(f"\n=== ORGANIZING FILES (Mode: {mode.upper()}) ===")
    
    # Setup directories (inside images folder)
    base_dir = image_dir / f"organized_{mode}"
    unique_dir = base_dir / "unique"
    dup_dir = base_dir / "duplicates"
    blank_dir = base_dir / "blanks"
    
    for d in [unique_dir, dup_dir, blank_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Identify all files
    all_files = list(get_image_files(image_dir))
    all_filenames = {f.name for f in all_files}
    
    # Identify Blanks
    blank_names = {b['name'] for b in blanks}
    
    # Identify Duplicates (based on selected mode)
    duplicates = all_results.get(mode, [])
    dup_names = {d['image2'] for d in duplicates} # image2 is always the duplicate to remove
    
    # Identify Unique (Everything else)
    # Unique = All - Blanks - Duplicates
    unique_names = all_filenames - blank_names - dup_names
    
    print(f"  Total Images: {len(all_files)}")
    print(f"  Blanks: {len(blank_names)}")
    print(f"  Duplicates: {len(dup_names)}")
    print(f"  Unique: {len(unique_names)}")
    
    # Copy Files
    print("  Copying files...")
    
    for f in all_files:
        if f.name in blank_names:
            shutil.copy2(f, blank_dir / f.name)
        elif f.name in dup_names:
            shutil.copy2(f, dup_dir / f.name)
        else:
            shutil.copy2(f, unique_dir / f.name)
            
    print(f"✓ Files organized in: {base_dir}")


def create_combined_report(all_results, output_dir, report_path, sequential=False, blanks=None):
    """Create a combined HTML report with tabs for each mode and blanks."""
    
    # Get all images for unique list calculation
    import json
    all_files = sorted(list(get_image_files(output_dir)))
    all_filenames = [f.name for f in all_files]
    blank_names = {b['name'] for b in blanks} if blanks else set()
    
    html = [
        '<!DOCTYPE html><html><head><meta charset="UTF-8">',
        '<style>',
        'body { font-family: Arial; margin: 20px; background: #f5f5f5; }',
        '.tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; border-radius: 5px 5px 0 0; }',
        '.tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }',
        '.tab button:hover { background-color: #ddd; }',
        '.tab button.active { background-color: #2196F3; color: white; }',
        '.tabcontent { display: none; padding: 20px; border: 1px solid #ccc; border-top: none; background: white; border-radius: 0 0 5px 5px; }',
        '.group { background: white; margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
        '.group-images { display: flex; flex-wrap: wrap; gap: 15px; }',
        '.ref-container { text-align: center; width: 300px; border: 2px solid #2196F3; padding: 10px; background: #e3f2fd; }',
        '.dup-container { text-align: center; width: 250px; border: 1px solid #f44336; padding: 10px; background: #ffebee; }',
        '.blank-container { text-align: center; width: 250px; border: 1px solid #9e9e9e; padding: 10px; background: #eeeeee; display: inline-block; margin: 10px; vertical-align: top; }',
        'img { max-width: 100%; height: auto; border: 1px solid #eee; }',
        '.metrics { margin-top: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; font-size: 0.8em; text-align: left; }',
        '.action-bar { position: sticky; top: 0; background: #333; color: white; padding: 15px; z-index: 100; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }',
        '.btn { background: #2196F3; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 4px; font-size: 16px; }',
        '.btn:hover { background: #1976D2; }',
        '.checkbox-wrapper { margin-top: 5px; font-weight: bold; }',
        '.radio-wrapper { margin-top: 5px; font-size: 0.9em; color: #333; }',
        '.storyline-section { margin-top: 40px; border-top: 3px solid #333; padding-top: 20px; }',
        '.storyline-grid { display: flex; flex-wrap: wrap; gap: 10px; padding: 20px; }',
        '.card { border: 4px solid #ccc; border-radius: 8px; overflow: hidden; background: white; transition: all 0.2s; position: relative; display: flex; flex-direction: column; }',
        '.card.keep { border-color: #4CAF50; opacity: 1.0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }',
        '.card.discard { border-color: #f44336; opacity: 1.0; background: #ffebee; }',
        '.card.blank { border-color: #E91E63; opacity: 1.0; background: #FCE4EC; }',
        '.card:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); opacity: 1.0; z-index: 10; }',
        '.card-img { flex-grow: 1; overflow: hidden; cursor: pointer; }',
        '.card-img img { width: 100%; height: 100%; object-fit: cover; }',
        '.card-footer { padding: 8px; font-size: 0.8em; border-top: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; font-weight: bold; }',
        '.keep .card-footer { background: #e8f5e9; color: #2e7d32; }',
        '.discard .card-footer { background: #ffebee; color: #c62828; }',
        '.blank .card-footer { background: #FCE4EC; color: #C2185B; }',
        '.status-badge { padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; text-transform: uppercase; border: 1px solid rgba(0,0,0,0.1); }',
        '.keep .status-badge { background: #c8e6c9; color: #2e7d32; }',
        '.discard .status-badge { background: #ffcdd2; color: #c62828; }',
        '.blank .status-badge { background: #F8BBD0; color: #C2185B; }',
        '.toolbar { position: sticky; top: 0; background: #333; color: white; padding: 10px 20px; z-index: 100; display: flex; gap: 20px; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.2); flex-wrap: wrap; }',
        '.toolbar-group { display: flex; align-items: center; gap: 10px; }',
        'input[type=range] { width: 150px; }',
        '/* Modal styles */',
        '.modal { display: none; position: fixed; z-index: 2000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.9); }',
        '.modal-content { margin: auto; display: block; max-width: 90%; max-height: 90vh; }',
        '.close { position: absolute; top: 15px; right: 35px; color: #f1f1f1; font-size: 40px; font-weight: bold; transition: 0.3s; cursor: pointer; }',
        '.close:hover, .close:focus { color: #bbb; text-decoration: none; cursor: pointer; }',
        '</style>',
        '<script>',
        'function openModal(src) {',
        '   var modal = document.getElementById("imgModal");',
        '   var modalImg = document.getElementById("img01");',
        '   modal.style.display = "block";',
        '   modalImg.src = src;',
        '}',
        'function closeModal() {',
        '   document.getElementById("imgModal").style.display = "none";',
        '}',
        'function openMode(evt, modeName) {',
        '  var i, tabcontent, tablinks;',
        '  tabcontent = document.getElementsByClassName("tabcontent");',
        '  for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }',
        '  tablinks = document.getElementsByClassName("tablinks");',
        '  for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }',
        '  document.getElementById(modeName).style.display = "block";',
        '  evt.currentTarget.className += " active";',
        '}',
        'function toggleCard(footer) {',
        '   // Toggle between keep and discard',
        '   var card = footer.closest(".card");',
        '   var checkbox = card.querySelector("input[type=checkbox]");',
        '   checkbox.checked = !checkbox.checked;',
        '   updateCardVisuals(card, checkbox.checked);',
        '}',
        'function updateCardVisuals(card, isKept) {',
        '   var badge = card.querySelector(".status-badge");',
        '   if (isKept) {',
        '       card.classList.remove("discard", "blank");',
        '       card.classList.add("keep");',
        '       badge.innerText = "KEEP";',
        '   } else {',
        '       card.classList.remove("keep");',
        '       // Determine if it was blank or duplicate based on original class or data attribute',
        '       // For simplicity, we just use "discard" style for all removed items',
        '       card.classList.add("discard");',
        '       badge.innerText = "DISCARD";',
        '   }',
        '   updateCounts();',
        '}',
        'function updateGridSize(val) {',
        '   var size = val + "px";',
        '   document.documentElement.style.setProperty("--card-width", size);',
        '   var cards = document.querySelectorAll(".card");',
        '   cards.forEach(c => c.style.width = size);',
        '}',
        'function toggleDiscarded(show) {',
        '   var discards = document.querySelectorAll(".card.discard, .card.blank");',
        '   discards.forEach(d => d.style.display = show ? "flex" : "none");',
        '}',
        'function updateCounts() {',
        '   // Update counts in toolbar if we add them',
        '}',
        'function generateScript(mode) {',
        '   var container = document.getElementById(mode);',
        '   var cards = container.querySelectorAll(".card");',
        '   var commands = [];',
        '   var baseDir = "organized_" + mode;',
        '   commands.push("@echo off");',
        '   commands.push("mkdir \\"" + baseDir + "\\\\unique\\"");',
        '   commands.push("mkdir \\"" + baseDir + "\\\\duplicates\\"");',
        '   commands.push("mkdir \\"" + baseDir + "\\\\blanks\\"");',
        '   ',
        '   cards.forEach(function(card) {',
        '       var checkbox = card.querySelector("input[type=checkbox]");',
        '       var file = checkbox.value;',
        '       var isKept = checkbox.checked;',
        '       var originalType = card.dataset.type; // duplicate, blank, unique',
        '       ',
        '       if (isKept) {',
        '           commands.push("copy \\"" + file + "\\" \\"" + baseDir + "\\\\unique\\\\" + file + "\\"");',
        '       } else {',
        '           // It is discarded. Move to appropriate folder.',
        '           var target = (originalType === "blank") ? "blanks" : "duplicates";',
        '           commands.push("copy \\"" + file + "\\" \\"" + baseDir + "\\\\" + target + "\\\\" + file + "\\"");',
        '       }',
        '   });',
        '   ',
        '   var text = commands.join("\\n");',
        '   var blob = new Blob([text], { type: "text/plain" });',
        '   var a = document.createElement("a");',
        '   a.href = URL.createObjectURL(blob);',
        '   a.download = "move_files_" + mode + ".bat";',
        '   a.click();',
        '}',
        '</script>',

        '</head><body>',
        '<h1>Deduplication Comparison Report</h1>',
        '<div class="tab">'
    ]
    
    modes = list(all_results.keys())
    for i, mode in enumerate(modes):
        active = ' class="tablinks active"' if i == 0 else ' class="tablinks"'
        html.append(f'<button{active} onclick="openMode(event, \'{mode}\')">{mode.upper()} ({len(all_results[mode])})</button>')
    
    # Blanks tab removed as blanks are now integrated into the main grid
        
    html.append('</div>')
    
    # Mode Tabs
    for i, mode in enumerate(modes):
        duplicates = all_results[mode]
        dup_names = {d['image2'] for d in duplicates}
        
        # Build Flat List of All Images with Status
        # 1. Start with all files
        # 2. Mark Blanks
        # 3. Mark Duplicates
        
        grid_items = []
        for filename in all_filenames:
            status = 'keep'
            reason = 'Unique'
            original_type = 'unique'
            
            if filename in blank_names:
                status = 'blank'
                reason = 'Blank'
                original_type = 'blank'
            elif filename in dup_names:
                status = 'discard'
                reason = 'Duplicate'
                original_type = 'duplicate'
                
            # Get path
            try:
                path = output_dir / filename
                rel_path = os.path.relpath(path, output_dir)
            except:
                rel_path = Path(path).as_uri()
                
            grid_items.append({
                'name': filename,
                'status': status,
                'reason': reason,
                'type': original_type,
                'src': rel_path
            })
            
        display = 'block' if i == 0 else 'none'
        html.append(f'<div id="{mode}" class="tabcontent" style="display: {display}">')
        
        # Toolbar
        html.append(f'<div class="toolbar">')
        html.append(f'<div class="toolbar-group"><strong>{mode.upper()}</strong></div>')
        html.append(f'<div class="toolbar-group"><label>Size:</label><input type="range" min="100" max="400" value="200" oninput="updateGridSize(this.value)"></div>')
        html.append(f'<div class="toolbar-group"><label><input type="checkbox" checked onchange="toggleDiscarded(this.checked)"> Show Discarded</label></div>')
        html.append(f'<div class="toolbar-group"><button class="btn" onclick="generateScript(\'{mode}\')">Download Move Script</button></div>')
        html.append(f'</div>')
        
        # Grid
        html.append(f'<div class="storyline-grid">')
        for item in grid_items:
            is_checked = "checked" if item['status'] == 'keep' else ""
            badge_text = "KEEP" if item['status'] == 'keep' else item['reason'].upper()
            
            html.append(f'<div class="card {item["status"]}" data-type="{item["type"]}" style="width: 200px;">')
            html.append(f'<div class="card-img"><img src="{item["src"]}" onclick="openModal(this.src)"></div>')
            html.append(f'<div class="card-footer" onclick="toggleCard(this)" style="cursor: pointer;" title="Click to Toggle Keep/Discard">')
            html.append(f'<span>{item["name"]}</span>')
            html.append(f'<span class="status-badge">{badge_text}</span>')
            html.append(f'<input type="checkbox" value="{item["name"]}" {is_checked} style="display:none">') # Hidden checkbox
            html.append(f'</div></div>')
            
        html.append(f'</div></div>')
        
    # Modal HTML
    html.append('<div id="imgModal" class="modal">')
    html.append('<span class="close" onclick="closeModal()">&times;</span>')
    html.append('<img class="modal-content" id="img01">')
    html.append('</div>')
        
    html.append('</body></html>')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    
    print(f"Created combined report: {report_path}")


def run_deduplication(image_dir, output_dir, mode, args, report_suffix=''):
    """
    Run a single deduplication pass.
    """
    # Get settings from mode
    settings = get_settings_from_mode(mode)
    
    # Override with custom arguments if provided
    threshold = args.threshold if args.threshold else settings['threshold']
    histogram_threshold = settings.get('histogram_threshold', 0.90) # Default to 0.90
    
    # Handle sequential flag
    sequential = args.sequential
    
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
    
    duplicates, images_data = find_duplicates_with_smart_crop(
        image_dir,
        threshold=threshold,
        blur_enabled=use_blur,
        downscale=downscale,
        sequential=sequential
    )
    
    # Create duplicate pairs report
    report_name = f'duplicates_report_{mode}{report_suffix}.html'
    create_duplicate_pairs_report(duplicates, output_dir, report_name, sequential=sequential)
    
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
    parser.add_argument('--crop-margin', type=float, default=0.20, help="Margin for content_aware crop (0.20 = 20%)")
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--no-blur', action='store_true')
    parser.add_argument('--downscale', action='store_true')
    parser.add_argument('--no-downscale', action='store_true')
    parser.add_argument('--save-crops', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--detect-blank-first', action='store_true')
    parser.add_argument('--sequential', action='store_true', help="Use optimized sequential comparison (faster, assumes sorted)")
    parser.add_argument('--verify', type=str, help="Verify specific group (e.g. 'img1.png, img2.png, ...')")
    
    return parser.parse_args()


def verify_group_all_modes(image_dir, group_string, args):
    """
    Analyze a group of images across ALL modes (Strict, Moderate, Lenient) 
    and generate a combined HTML report.
    """
    try:
        # Parse group string
        if ' vs ' in group_string:
            parts = group_string.split(' vs ')
        else:
            parts = group_string.split(',')
            
        image_names = [p.strip() for p in parts if p.strip()]
        
        if len(image_names) < 2:
            print(f"Error: Need at least 2 images to verify. Got: {image_names}")
            return
            
        ref_name = image_names[0]
        others = image_names[1:]
        
        print(f"\n=== VERIFYING GROUP ACROSS ALL MODES ===")
        print(f"Reference: {ref_name}")
        print(f"Comparing against: {', '.join(others)}")
        
        # Setup HTML
        html = [
            '<!DOCTYPE html><html><head><meta charset="UTF-8">',
            '<style>',
            'body { font-family: Arial; margin: 20px; background: #f5f5f5; }',
            '.mode-section { background: white; margin: 30px 0; border: 1px solid #ccc; border-radius: 8px; overflow: hidden; }',
            '.mode-header { background: #333; color: white; padding: 15px; }',
            '.mode-header h2 { margin: 0; }',
            '.settings { font-family: monospace; color: #ccc; margin-top: 5px; }',
            '.comparison { display: flex; padding: 20px; border-bottom: 1px solid #eee; align-items: flex-start; }',
            '.ref-col { width: 400px; text-align: center; border-right: 2px solid #eee; padding-right: 20px; margin-right: 20px; }',
            '.dup-col { width: 400px; text-align: center; }',
            '.metrics-col { flex: 1; padding-left: 20px; }',
            'img { max-width: 100%; border: 1px solid #ddd; margin-bottom: 5px; }',
            '.vis-label { font-size: 0.8em; color: #666; margin-bottom: 10px; }',
            '.status { font-weight: bold; padding: 5px 10px; border-radius: 4px; display: inline-block; margin-bottom: 10px; }',
            '.status-dup { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }',
            '.status-unique { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }',
            '.metric-row { margin: 5px 0; font-family: monospace; }',
            '</style>',
            '</head><body>',
            f'<h1>Verification Analysis Report</h1>',
            f'<p><strong>Group:</strong> {group_string}</p>'
        ]
        
        modes = ['strict', 'moderate', 'lenient']
        debug_dir = Path(image_dir) / "debug_verify"
        debug_dir.mkdir(exist_ok=True)
        
        for mode in modes:
            print(f"\nRunning {mode.upper()} analysis...")
            
            # Get settings
            settings = get_settings_from_mode(mode)
            threshold = args.threshold if args.threshold else settings['threshold']
            use_blur = args.blur or (not args.no_blur and settings.get('use_blur', True))
            downscale = args.downscale or settings.get('downscale', False)
            hist_thresh = settings.get('histogram_threshold', 0.90) # Default to 0.90
            
            html.append(f'<div class="mode-section">')
            html.append(f'<div class="mode-header"><h2>{mode.upper()} MODE</h2>')
            html.append(f'<div class="settings">Threshold: {threshold} | Blur: {use_blur} | Downscale: {downscale} | Hist Thresh: {hist_thresh}</div></div>')
            
            # Helper to process image
            def process_one(p, prefix):
                img = Image.open(p)
                
                # Get crop box and create visualization
                crop_box = get_crop_box(img, method=args.crop_method, margin=args.crop_margin)
                
                # Visualization: Original with Red Box
                vis_img = img.copy()
                draw = ImageDraw.Draw(vis_img)
                draw.rectangle(crop_box, outline="red", width=5)
                vis_save_name = f"{prefix}_vis_{p.name}"
                vis_img.save(debug_dir / vis_save_name)
                
                # Perform Crop & Process
                cropped = img.crop(crop_box)
                proc = cropped.copy()
                if use_blur: proc = proc.filter(ImageFilter.GaussianBlur(radius=2))
                if downscale: proc = proc.resize((proc.width//2, proc.height//2), Image.LANCZOS)
                hashes = compute_multi_scale_hash(proc)
                
                # Calculate Entropy
                gray = np.array(proc.convert('L'))
                hist = np.histogram(gray, bins=256)[0]
                hist = hist[hist > 0]
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist))
                
                # Save processed debug image
                save_name = f"{prefix}_{p.name}"
                proc.save(debug_dir / save_name)
                
                return {
                    'name': p.name, 
                    'cropped': cropped,
                    'processed': proc, 
                    'hashes': hashes,
                    'entropy': entropy,
                    'debug_path': str(debug_dir / save_name),
                    'vis_path': str(debug_dir / vis_save_name)
                }

            ref_path = Path(image_dir) / ref_name
            if not ref_path.exists():
                print(f"Skipping mode {mode}: Ref not found")
                continue
                
            ref_data = process_one(ref_path, f"{mode}_ref")
            
            # Get relative path for HTML
            try:
                rel_ref_path = os.path.relpath(ref_data['debug_path'], image_dir)
                rel_ref_vis = os.path.relpath(ref_data['vis_path'], image_dir)
            except:
                rel_ref_path = Path(ref_data['debug_path']).as_uri()
                rel_ref_vis = Path(ref_data['vis_path']).as_uri()
            
            for other_name in others:
                other_path = Path(image_dir) / other_name
                if not other_path.exists():
                    continue
                    
                other_data = process_one(other_path, f"{mode}_dup")
                
                # Compare
                comp = compare_images_smart(ref_data, other_data, threshold, hist_thresh)
                
                # Get relative path
                try:
                    rel_other_path = os.path.relpath(other_data['debug_path'], image_dir)
                    rel_other_vis = os.path.relpath(other_data['vis_path'], image_dir)
                except:
                    rel_other_path = Path(other_data['debug_path']).as_uri()
                    rel_other_vis = Path(other_data['vis_path']).as_uri()
                
                # Calculate Advanced Metrics (Use Raw Cropped images for consistency)
                ssim_score = compute_ssim_score(ref_data['cropped'], other_data['cropped'])
                dl_score = compute_deep_similarity(ref_data['cropped'], other_data['cropped'])
                
                # Add to HTML
                status_class = "status-dup" if comp['is_duplicate'] else "status-unique"
                status_text = "DUPLICATE" if comp['is_duplicate'] else "UNIQUE"
                
                html.append(f'<div class="comparison">')
                html.append(f'<div class="ref-col"><strong>Reference</strong><br><img src="{rel_ref_vis}"><div class="vis-label">Original with Crop Box</div><img src="{rel_ref_path}"><div class="vis-label">Processed Input (Entropy: {ref_data["entropy"]:.2f})</div>{ref_name}</div>')
                html.append(f'<div class="dup-col"><strong>Comparison</strong><br><img src="{rel_other_vis}"><div class="vis-label">Original with Crop Box</div><img src="{rel_other_path}"><div class="vis-label">Processed Input (Entropy: {other_data["entropy"]:.2f})</div>{other_name}</div>')
                html.append(f'<div class="metrics-col">')
                html.append(f'<div class="{status_class}">{status_text}</div>')
                html.append(f'<div class="metric-row">Avg Distance: <strong>{comp["avg_distance"]:.2f}</strong> (Limit {comp["adjusted_threshold"]})</div>')
                html.append(f'<div class="metric-row">Min Distance: <strong>{comp["min_distance"]}</strong> (Limit {comp["adjusted_threshold"]})</div>')
                html.append(f'<div class="metric-row">Hist Similarity: <strong>{comp["histogram_similarity"]:.4f}</strong> (Target {hist_thresh})</div>')
                html.append(f'<div class="metric-row">Entropy: <strong>{comp["entropy"]:.2f}</strong></div>')
                html.append(f'<hr>')
                html.append(f'<div class="metric-row" style="color:blue"><strong>Raw SSIM: {ssim_score:.4f}</strong> (Target > 0.95)</div>')
                html.append(f'<div class="metric-row" style="color:purple"><strong>Raw ResNet: {dl_score:.4f}</strong> (Target > 0.98)</div>')
                html.append(f'<hr>')
                html.append(f'<div class="metric-row">Hash Distances:</div>')
                for k, v in comp['distances'].items():
                    html.append(f'<div class="metric-row">&nbsp;&nbsp;{k}: {v}</div>')
                html.append(f'</div></div>')
                
            html.append('</div>')
            
        html.append('</body></html>')
        
        report_path = Path(image_dir) / "verification_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
            
        print(f"\nGenerated verification report: {report_path}")
        
    except Exception as e:
        print(f"Error verifying group: {e}")
        import traceback
        traceback.print_exc()


def get_settings_from_mode(mode):
    """Get preset settings."""
    modes = {
        'lenient': {'threshold': 15, 'histogram_threshold': 0.88, 'use_blur': True, 'downscale': True},
        'moderate': {'threshold': 12, 'histogram_threshold': 0.90, 'use_blur': True, 'downscale': False}, # Changed hist_thresh to 0.90
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
    
    # Verify specific pair
    if args.verify:
        verify_group_all_modes(IMAGE_DIR, args.verify, args)
        sys.exit(0)
    
    # Detect blank images
    if args.mode == 'detect-blank' or args.detect_blank_first:
        blank_images, all_imgs = find_blank_images(IMAGE_DIR, args.content_threshold, args.debug)
        
        if args.mode == 'detect-blank':
            # Generate report
            report_path = IMAGE_DIR / "blank_images_report.html"
            create_blank_images_report(all_imgs, blank_images, report_path, args.content_threshold)
            print("\nDone! Check 'blank_images_report.html'")
            sys.exit(0)
    
    # Compare all modes
    elif args.mode == 'compare-all':
        print(f"Found {len(images)} images in {IMAGE_DIR}")
        
        # 1. Detect Blanks First
        print("\n=== 1. DETECTING BLANK SLIDES ===")
        blanks, _ = find_blank_images(IMAGE_DIR) # find_blank_images returns (blank_images, all_images_data)
        blank_paths = {Path(b['path']) for b in blanks} # Convert to Path objects for comparison
        print(f"Found {len(blanks)} blank images. Excluding them from deduplication.")
        
        # 2. Filter images for deduplication
        # We only want to compare non-blank images
        all_image_paths = sorted(list(get_image_files(IMAGE_DIR))) # Get all image paths as Path objects
        non_blank_files = [f for f in all_image_paths if f not in blank_paths]
        print(f"Proceeding with {len(non_blank_files)} images for deduplication.")

        # 3. Run Moderate Mode Only
        print("\n=== 2. RUNNING DEDUPLICATION (MODERATE) ===")
        
        # Setup Moderate params
        threshold = 12
        blur = True
        downscale = False 
        
        print(f"Settings: Threshold={threshold}, Blur={blur}, Downscale={downscale}")
        
        duplicates, _ = find_duplicates_with_smart_crop(
            IMAGE_DIR, 
            threshold=threshold, 
            blur_enabled=blur, 
            downscale=downscale,
            sequential=True, # Always sequential as requested
            file_list=non_blank_files # Passing the filtered list
        )
        
        all_results = {'moderate': duplicates}
        
        # 4. Save JSON Results (for Web App)
        json_results = {
            'blanks': [b['name'] for b in blanks],
            'duplicates': duplicates,
            'all_files': [p.name for p in all_image_paths]
        }
        json_path = IMAGE_DIR / "dedup_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, cls=NumpyEncoder)
        print(f"Saved JSON results to: {json_path}")

        # 5. Generate Report
        report_path = IMAGE_DIR / "duplicates_report_combined.html"
        create_combined_report(all_results, IMAGE_DIR, report_path, sequential=True, blanks=blanks)
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print(f"  BLANKS:     {len(blanks)}")
        print(f"  DUPLICATES: {len(duplicates)}")
        print(f"  UNIQUE:     {len(all_image_paths) - len(blanks) - len(duplicates)}")
        print(f"\nCheck the report: {report_path}")
        print("Ready for curation in the web interface.")
        
    # Single run
    else:
        run_deduplication(IMAGE_DIR, IMAGE_DIR, args.mode, args)
        
        # We don't need to print recommendations here as run_deduplication generates the report
        # But we can print a summary if we want
    
    print("\nDone!")