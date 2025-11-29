import cv2
import numpy as np
import glob
import os
import time
import shutil
from utils import parse_image_timestamp
from similarity import calculate_similarity

def detect_dark_borders(img, border_threshold=30, border_width_ratio=0.1):
    """Detect dark borders around image edges (typical of video conference UI)"""
    h, w = img.shape[:2]
    border_width = int(min(h, w) * border_width_ratio)
    
    # Check top, bottom, left, right borders
    top_border = img[0:border_width, :]
    bottom_border = img[h-border_width:h, :]
    left_border = img[:, 0:border_width]
    right_border = img[:, w-border_width:w]
    
    borders = [top_border, bottom_border, left_border, right_border]
    border_brightness = [np.mean(border) for border in borders]
    
    # Check if borders are significantly darker than center
    center_region = img[border_width:h-border_width, border_width:w-border_width]
    center_brightness = np.mean(center_region)
    
    dark_borders = sum(1 for b in border_brightness if b < border_threshold)
    border_darkness_ratio = np.mean([(center_brightness - b) / center_brightness if center_brightness > 0 else 0 
                                     for b in border_brightness])
    
    return {
        "has_dark_borders": dark_borders >= 2 and border_darkness_ratio > 0.3,
        "dark_border_count": dark_borders,
        "border_darkness_ratio": border_darkness_ratio,
        "center_brightness": center_brightness,
        "border_brightness": border_brightness
    }


def analyze_image_content(img_path):
    """Analyze image for relevance and content type"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return {"relevant": False, "reason": "Could not read image", "type": "unknown"}
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Check for black/blank screen
        mean_brightness = np.mean(gray)
        if mean_brightness < 20:
            return {"relevant": False, "reason": "Black/blank screen", "type": "blank", "brightness": mean_brightness}
        
        # Check for very bright/white screen
        if mean_brightness > 240:
            return {"relevant": False, "reason": "White/blank screen", "type": "blank", "brightness": mean_brightness}
        
        # Detect dark borders (video conference UI often has dark borders)
        border_info = detect_dark_borders(gray)
        
        # Focus on center region for content analysis (ignore borders)
        border_width = int(min(h, w) * 0.1)
        center_region = gray[border_width:h-border_width, border_width:w-border_width]
        center_h, center_w = center_region.shape
        
        if center_h < 10 or center_w < 10:
            center_region = gray  # Fallback if center is too small
        
        # Check center region for low information content
        center_variance = np.var(center_region)
        if center_variance < 100:
            return {"relevant": False, "reason": "Low information content (uniform center)", "type": "low_info", "variance": center_variance}
        
        # Calculate overall variance
        variance = np.var(gray)
        
        # Analyze center region for content (ignore borders/UI)
        center_edges = cv2.Canny(center_region, 50, 150)
        center_edge_density = np.sum(center_edges > 0) / (center_h * center_w) if center_h > 0 and center_w > 0 else 0
        
        # Check corners of CENTER region (not full image) for UI elements
        ch, cw = center_region.shape
        center_corner_regions = [
            center_region[0:ch//4, 0:cw//4],  # Top-left
            center_region[0:ch//4, 3*cw//4:cw],  # Top-right
            center_region[3*ch//4:ch, 0:cw//4],  # Bottom-left
            center_region[3*ch//4:ch, 3*cw//4:cw]  # Bottom-right
        ]
        
        center_corner_variance = np.mean([np.var(region) for region in center_corner_regions])
        center_core_variance = np.var(center_region[ch//4:3*ch//4, cw//4:3*cw//4])
        
        # Check for faces in CENTER region only (ignore faces in borders/UI)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Detect faces in full image but filter by location
        all_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Count faces that are in the center region (not in borders)
        center_faces = []
        for (x, y, fw, fh) in all_faces:
            face_center_x = x + fw // 2
            face_center_y = y + fh // 2
            # Check if face center is in the center region (excluding borders)
            if (border_width < face_center_x < w - border_width and 
                border_width < face_center_y < h - border_width):
                center_faces.append((x, y, fw, fh))
        
        face_count = len(center_faces)
        face_coverage = sum([x[2] * x[3] for x in center_faces]) / (center_h * center_w) if face_count > 0 and center_h > 0 and center_w > 0 else 0
        
        # Calculate entropy for center region
        center_hist = cv2.calcHist([center_region], [0], None, [256], [0, 256])
        center_entropy = -np.sum([p * np.log2(p + 1e-10) for p in center_hist / (center_h * center_w) if p > 0])
        
        # Calculate overall entropy for comparison
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_entropy = -np.sum([p * np.log2(p + 1e-10) for p in hist / (h * w) if p > 0])
        
        # Improved video conference detection - focus on center content quality
        is_video_conference = False
        reasons = []
        confidence_score = 0
        
        # Criterion 1: Dark borders + multiple faces in center (strong video conference indicator)
        if border_info["has_dark_borders"] and face_count >= 2:
            confidence_score += 4
            reasons.append(f"Dark borders + {face_count} faces in center")
        
        # Criterion 2: Very low center content quality with faces (pure video conference, no slide)
        if center_variance < 3000 and center_entropy < 4.0 and face_count >= 2:
            confidence_score += 3
            reasons.append(f"Very low center content quality (variance: {center_variance:.0f}, entropy: {center_entropy:.2f}) with {face_count} faces")
        
        # Criterion 3: Multiple small faces in center with low information (grid layout)
        if face_count >= 3:
            avg_face_size = np.mean([x[2] * x[3] for x in center_faces]) if face_count > 0 else 0
            if avg_face_size < (center_h * center_w * 0.08) and center_variance < 5000:  # Very small faces
                confidence_score += 2
                reasons.append(f"{face_count} very small faces in grid layout")
        
        # Criterion 4: Dark borders + low center content (UI overlay, but check if center has actual content)
        if border_info["has_dark_borders"] and center_variance < 4000 and center_entropy < 4.5:
            # But if center still has reasonable content, it might be a slide with UI overlay
            if center_variance > 2000:  # Has some content
                confidence_score += 1  # Lower confidence - might be slide with UI
                reasons.append("Dark borders with moderate center content (possible slide with UI overlay)")
            else:
                confidence_score += 3  # High confidence - pure video conference
                reasons.append("Dark borders with very low center content")
        
        # Criterion 5: High face coverage in center (faces dominate the image)
        if face_coverage > 0.4 and face_count >= 2:
            confidence_score += 2
            reasons.append(f"High face coverage ({face_coverage:.1%}) in center")
        
        # Only flag as video conference if confidence is high (>= 5) AND center content is poor
        # This prevents flagging slides that happen to have some UI elements
        if confidence_score >= 5 and center_variance < 5000:
            is_video_conference = True
        elif confidence_score >= 6:  # Very high confidence regardless
            is_video_conference = True
        
        if is_video_conference:
            return {
                "relevant": False,
                "reason": f"Possible video conference UI: {', '.join(reasons)}",
                "type": "video_conference",
                "confidence_score": confidence_score,
                "center_variance": center_variance,
                "center_entropy": center_entropy,
                "face_count": face_count,
                "face_coverage": face_coverage,
                "has_dark_borders": border_info["has_dark_borders"],
                "border_info": border_info
            }
        
        # If we have dark borders but good center content, it's likely a slide with UI overlay
        slide_with_ui = border_info["has_dark_borders"] and center_variance > 2000
        
        return {
            "relevant": True,
            "reason": "Appears to be slide/content" + (" (with UI overlay)" if slide_with_ui else ""),
            "type": "slide" + ("_with_ui" if slide_with_ui else ""),
            "brightness": mean_brightness,
            "variance": variance,
            "center_variance": center_variance,
            "center_entropy": center_entropy,
            "entropy": hist_entropy,
            "face_count": face_count,
            "has_dark_borders": border_info["has_dark_borders"],
            "border_info": border_info
        }
    except Exception as e:
        return {"relevant": False, "reason": f"Analysis error: {e}", "type": "error"}


def analyze_images_comprehensive(images_folder, similarity_threshold=0.95, output_folder=None, move_duplicates=False, similarity_method='ssim'):
    """Comprehensive analysis of images: duplicates, relevance, and generate report"""
    print("\n" + "="*60)
    print(f"COMPREHENSIVE IMAGE ANALYSIS (Method: {similarity_method})")
    print("="*60)
    
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    if len(image_files) <= 1:
        print("Not enough images to analyze")
        return None
    
    # Parse timestamps and create image data
    image_data = []
    for img_file in image_files:
        timestamp = parse_image_timestamp(os.path.basename(img_file), return_minutes=True)
        if timestamp is not None:
            image_data.append((float(timestamp), img_file, os.path.basename(img_file)))
    
    if not image_data:
        print("No valid images found")
        return None
    
    # Sort by timestamp
    image_data.sort(key=lambda x: x[0])
    
    print(f"\nAnalyzing {len(image_data)} images...")
    print("This may take a few minutes...\n")
    
    # Analyze each image
    image_analyses = {}
    duplicate_groups = []
    irrelevant_images = []
    
    # First pass: analyze content relevance
    print("Step 1: Analyzing image content and relevance...")
    for idx, (timestamp, path, filename) in enumerate(image_data):
        print(f"  Analyzing {filename} ({idx+1}/{len(image_data)})...", end='\r')
        analysis = analyze_image_content(path)
        image_analyses[filename] = {
            "path": path,
            "timestamp": timestamp,
            "analysis": analysis
        }
        
        if not analysis.get("relevant", True):
            irrelevant_images.append({
                "filename": filename,
                "timestamp": timestamp,
                "reason": analysis.get("reason", "Unknown"),
                "type": analysis.get("type", "unknown")
            })
    print(f"\n  Found {len(irrelevant_images)} potentially irrelevant images")
    
    # Second pass: find duplicates (improved grouping)
    print("\nStep 2: Finding duplicate images...")
    checked_pairs = 0
    similarity_matrix = {}  # Store all similarities
    
    # First, calculate similarities for all pairs (within reasonable time window)
    for i in range(len(image_data)):
        filename1 = image_data[i][2]
        path1 = image_data[i][1]
        time1 = image_data[i][0]
        
        # Compare with subsequent images (extend window to catch all duplicates)
        for j in range(i + 1, len(image_data)):
            filename2 = image_data[j][2]
            path2 = image_data[j][1]
            time2 = image_data[j][0]
            
            time_diff_seconds = (time2 - time1) * 60
            
            # Check within 10 minutes (to catch duplicates even if far apart)
            if time_diff_seconds > 600:
                continue
            
            checked_pairs += 1
            if checked_pairs % 20 == 0:
                print(f"  Checked {checked_pairs} pairs...", end='\r')
            
            try:
                img1 = cv2.imread(path1)
                img2 = cv2.imread(path2)
                if img1 is not None and img2 is not None:
                    similarity = calculate_similarity(img1, img2, method=similarity_method)
                    if similarity and similarity > similarity_threshold:
                        # Store similarity
                        key = tuple(sorted([filename1, filename2]))
                        similarity_matrix[key] = {
                            "img1": filename1,
                            "img2": filename2,
                            "similarity": similarity,
                            "time1": time1,
                            "time2": time2
                        }
            except Exception as e:
                pass
    
    print(f"\n  Checked {checked_pairs} image pairs")
    
    # Group duplicates using union-find approach
    # Create groups for all similar pairs
    image_to_group = {}
    groups = []
    
    for key, data in similarity_matrix.items():
            img1 = data["img1"]
            img2 = data["img2"]
            similarity = data["similarity"]
            
            # Find or create groups
            group1_idx = image_to_group.get(img1)
            group2_idx = image_to_group.get(img2)
            
            if group1_idx is None and group2_idx is None:
                # Create new group - use earlier timestamp as primary
                primary = img1 if data["time1"] <= data["time2"] else img2
                new_group = {
                    "primary": primary,
                    "images": [img1, img2],
                    "similarities": {img1: 1.0, img2: similarity},
                    "timestamps": {img1: data["time1"], img2: data["time2"]}
                }
                groups.append(new_group)
                group_idx = len(groups) - 1
                image_to_group[img1] = group_idx
                image_to_group[img2] = group_idx
            elif group1_idx is not None and group2_idx is None:
                # Add img2 to existing group
                if img2 not in groups[group1_idx]["images"]:
                    groups[group1_idx]["images"].append(img2)
                    groups[group1_idx]["similarities"][img2] = similarity
                    groups[group1_idx]["timestamps"][img2] = data["time2"]
                    image_to_group[img2] = group1_idx
                    # Update primary if img2 is earlier
                    if data["time2"] < groups[group1_idx]["timestamps"][groups[group1_idx]["primary"]]:
                        groups[group1_idx]["primary"] = img2
            elif group1_idx is None and group2_idx is not None:
                # Add img1 to existing group
                if img1 not in groups[group2_idx]["images"]:
                    groups[group2_idx]["images"].append(img1)
                    groups[group2_idx]["similarities"][img1] = similarity
                    groups[group2_idx]["timestamps"][img1] = data["time1"]
                    image_to_group[img1] = group2_idx
                    # Update primary if img1 is earlier
                    if data["time1"] < groups[group2_idx]["timestamps"][groups[group2_idx]["primary"]]:
                        groups[group2_idx]["primary"] = img1
            else:
                # Both in groups - merge groups if different
                if group1_idx != group2_idx:
                    # Merge smaller group into larger group
                    if len(groups[group1_idx]["images"]) >= len(groups[group2_idx]["images"]):
                        target_group = group1_idx
                        source_group = group2_idx
                    else:
                        target_group = group2_idx
                        source_group = group1_idx
                    
                    # Merge source into target
                    for img in groups[source_group]["images"]:
                        if img not in groups[target_group]["images"]:
                            groups[target_group]["images"].append(img)
                            groups[target_group]["similarities"][img] = groups[source_group]["similarities"].get(img, 1.0)
                            groups[target_group]["timestamps"][img] = groups[source_group]["timestamps"].get(img, 0)
                        image_to_group[img] = target_group
                    
                    # Update primary to earliest timestamp
                    earliest = min(groups[target_group]["timestamps"].items(), key=lambda x: x[1])
                    groups[target_group]["primary"] = earliest[0]
                    groups[source_group] = None  # Mark for removal
    
    # Remove None groups and ensure primary is set correctly
    duplicate_groups = []
    for group in groups:
        if group is not None and len(group["images"]) > 1:
            # Ensure primary is set to earliest timestamp
            earliest = min(group["timestamps"].items(), key=lambda x: x[1])
            group["primary"] = earliest[0]
            # Sort images by timestamp for better presentation
            group["images"] = sorted(group["images"], key=lambda x: group["timestamps"].get(x, 0))
            duplicate_groups.append(group)
    
    print(f"\n  Checked {checked_pairs} image pairs")
    print(f"  Found {len(duplicate_groups)} duplicate groups")
    
    # Move duplicates if requested
    if move_duplicates and duplicate_groups:
        print("\nStep 3: Moving duplicates to review folders...")
        duplicates_review_folder = os.path.join(output_folder, "duplicates_review")
        os.makedirs(duplicates_review_folder, exist_ok=True)
        
        moved_count = 0
        for group in duplicate_groups:
            primary_name = group['primary']
            # Create folder for this group
            group_folder = os.path.join(duplicates_review_folder, os.path.splitext(primary_name)[0])
            os.makedirs(group_folder, exist_ok=True)
            
            # Create info file
            info_path = os.path.join(group_folder, "info.txt")
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Duplicate Group for {primary_name}\n")
                f.write("="*50 + "\n")
                f.write(f"Primary Image: {primary_name}\n")
                f.write(f"Timestamp: {image_analyses[primary_name]['timestamp']:.2f} minutes\n")
                f.write("="*50 + "\n\n")
                f.write("Moved Duplicates:\n")
            
            # Move duplicates
            for img_name in group['images']:
                if img_name != primary_name:
                    src_path = image_analyses[img_name]['path']
                    dst_path = os.path.join(group_folder, img_name)
                    
                    try:
                        # Move file
                        shutil.move(src_path, dst_path)
                        moved_count += 1
                        
                        # Update info file
                        similarity = group['similarities'].get(img_name, 0)
                        timestamp = image_analyses[img_name]['timestamp']
                        with open(info_path, 'a', encoding='utf-8') as f:
                            f.write(f"- {img_name}\n")
                            f.write(f"  Timestamp: {timestamp:.2f} minutes\n")
                            f.write(f"  Similarity: {similarity:.3f}\n")
                            f.write(f"  Original Path: {src_path}\n\n")
                            
                    except Exception as e:
                        print(f"Error moving {img_name}: {e}")
        
        print(f"  Moved {moved_count} duplicate images to {duplicates_review_folder}")
    
    # Generate report
    if output_folder is None:
        output_folder = os.path.dirname(images_folder)
    
    report_path = os.path.join(output_folder, "image_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IMAGE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Images Analyzed: {len(image_data)}\n")
        f.write(f"Similarity Threshold: {similarity_threshold}\n\n")
        
        # Summary
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Potentially Irrelevant Images: {len(irrelevant_images)}\n")
        f.write(f"Duplicate Groups Found: {len(duplicate_groups)}\n")
        f.write(f"Total Images Flagged: {len(irrelevant_images) + sum(len(g['images'])-1 for g in duplicate_groups)}\n\n")
        
        # Irrelevant images section
        if irrelevant_images:
            f.write("="*80 + "\n")
            f.write("POTENTIALLY IRRELEVANT IMAGES\n")
            f.write("="*80 + "\n\n")
            f.write("These images may not contain relevant slide content:\n\n")
            for img in irrelevant_images:
                f.write(f"  {img['filename']}\n")
                f.write(f"    Timestamp: {img['timestamp']:.2f} minutes\n")
                f.write(f"    Reason: {img['reason']}\n")
                f.write(f"    Type: {img['type']}\n")
                f.write(f"    Recommendation: REVIEW - May be safe to remove\n\n")
        
        # Duplicate groups section
        if duplicate_groups:
            f.write("="*80 + "\n")
            f.write("DUPLICATE IMAGE GROUPS\n")
            f.write("="*80 + "\n\n")
            f.write("These images are visually similar and may be duplicates:\n\n")
            
            for idx, group in enumerate(duplicate_groups, 1):
                f.write(f"Group {idx}:\n")
                f.write(f"  Primary (recommended to keep): {group['primary']}\n")
                f.write(f"  Timestamp: {image_analyses[group['primary']]['timestamp']:.2f} minutes\n")
                f.write(f"  Duplicates:\n")
                for img_name in group['images']:
                    if img_name != group['primary']:
                        similarity = group['similarities'].get(img_name, 0)
                        timestamp = image_analyses[img_name]['timestamp']
                        f.write(f"    - {img_name}\n")
                        f.write(f"      Timestamp: {timestamp:.2f} minutes\n")
                        f.write(f"      Similarity: {similarity:.3f}\n")
                        f.write(f"      Recommendation: REVIEW - Consider removing if duplicate\n")
                f.write("\n")
        
        # Detailed analysis
        f.write("="*80 + "\n")
        f.write("DETAILED IMAGE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        for filename, data in sorted(image_analyses.items(), key=lambda x: x[1]['timestamp']):
            f.write(f"{filename}\n")
            f.write(f"  Timestamp: {data['timestamp']:.2f} minutes\n")
            analysis = data['analysis']
            f.write(f"  Relevant: {analysis.get('relevant', 'Unknown')}\n")
            f.write(f"  Type: {analysis.get('type', 'unknown')}\n")
            f.write(f"  Reason: {analysis.get('reason', 'N/A')}\n")
            if 'brightness' in analysis:
                f.write(f"  Brightness: {analysis['brightness']:.1f}\n")
            if 'variance' in analysis:
                f.write(f"  Variance: {analysis['variance']:.1f}\n")
            if 'entropy' in analysis:
                f.write(f"  Entropy: {analysis['entropy']:.2f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        f.write("1. Review all flagged images manually before removing\n")
        f.write("2. For duplicate groups, keep the PRIMARY image (first occurrence)\n")
        f.write("3. For irrelevant images, verify they don't contain important content\n")
        f.write("4. Consider the context - some 'irrelevant' images might be important\n")
        f.write("5. After review, you can manually delete flagged images\n\n")
        f.write("="*80 + "\n")
    
    print(f"\n" + "="*60)
    print(f"Analysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"  - {len(irrelevant_images)} potentially irrelevant images")
    print(f"  - {len(duplicate_groups)} duplicate groups")
    print("="*60)
    
    return {
        "report_path": report_path,
        "irrelevant": irrelevant_images,
        "duplicates": duplicate_groups,
        "total_flagged": len(irrelevant_images) + sum(len(g['images'])-1 for g in duplicate_groups)
    }
