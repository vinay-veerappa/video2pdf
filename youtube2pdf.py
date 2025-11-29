#!/usr/bin/env python3
"""
YouTube Video to PDF Slides Converter
Based on miavisc approach with YouTube URL support
"""

import os
import time
import cv2
import imutils
import shutil
import img2pdf
import glob
import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from skimage.metrics import structural_similarity as ssim
import browser_cookie3
import tempfile
import base64

# Constants
OUTPUT_DIR = "./output"
FRAME_RATE = 6  # frames per second to process
WARMUP = FRAME_RATE  # initial frames to skip
FGBG_HISTORY = FRAME_RATE * 15  # frames in background model
VAR_THRESHOLD = 16  # Mahalanobis distance threshold
DETECT_SHADOWS = False
MIN_PERCENT = 0.1  # min % diff to detect motion stopped
MAX_PERCENT = 0.3  # max % diff to detect motion
SIMILARITY_THRESHOLD = 0.95  # SSIM threshold for duplicate detection
MIN_TIME_BETWEEN_CAPTURES = 0  # Minimum seconds between captures (0 = disabled by default)
MAX_SIMILARITY_COMPARISONS = 5  # Compare with last N images


def is_youtube_url(url):
    """Check if the input is a YouTube URL"""
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtu\.be/',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def sanitize_filename(filename):
    """Sanitize filename for filesystem"""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces
    filename = re.sub(r'\s+', ' ', filename).strip()
    # Limit length for Windows (max 255 chars for filename, but we'll use 100 to be safe)
    if len(filename) > 100:
        filename = filename[:100]
    filename = filename.rstrip('. ')
    return filename


def get_youtube_cookies():
    """Extract YouTube cookies from browser"""
    print("Attempting to extract cookies from browser...")
    cookies_file = os.path.join(tempfile.gettempdir(), 'youtube_cookies.txt')
    
    try:
        # Try Chrome first, then Edge, then Firefox
        cj = None
        try:
            cj = browser_cookie3.chrome(domain_name='.youtube.com')
            print("Found cookies in Chrome")
        except:
            try:
                cj = browser_cookie3.edge(domain_name='.youtube.com')
                print("Found cookies in Edge")
            except:
                try:
                    cj = browser_cookie3.firefox(domain_name='.youtube.com')
                    print("Found cookies in Firefox")
                except:
                    pass
        
        if cj:
            # Write Netscape format cookies
            with open(cookies_file, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in cj:
                    # Convert to Netscape format
                    # domain, flag, path, secure, expiration, name, value
                    domain = cookie.domain
                    flag = "TRUE" if domain.startswith('.') else "FALSE"
                    path = cookie.path
                    secure = "TRUE" if cookie.secure else "FALSE"
                    expiration = str(int(cookie.expires)) if cookie.expires else "0"
                    name = cookie.name
                    value = cookie.value
                    f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiration}\t{name}\t{value}\n")
            return cookies_file
    except Exception as e:
        print(f"Warning: Could not extract cookies: {e}")
    
    return None


def download_youtube_video(url, output_dir, cookies_path=None):
    """Download YouTube video using yt-dlp"""
    print(f"Downloading video from YouTube: {url}")
    
    # Check if yt-dlp is available
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("yt-dlp is not installed. Please install it with: pip install yt-dlp")
    
    # Create temp directory for downloads
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Use yt-dlp to download video
        # Download best quality video in mp4 format
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "-o", os.path.join(temp_dir, "%(title)s.%(ext)s"),
        ]
        
        # Add cookies if available
        if cookies_path and os.path.exists(cookies_path):
            print(f"Using cookies from: {cookies_path}")
            cmd.extend(["--cookies", cookies_path])
        else:
            # Try automatic extraction if no specific file provided
            auto_cookies = get_youtube_cookies()
            if auto_cookies:
                print(f"Using extracted browser cookies: {auto_cookies}")
                cmd.extend(["--cookies", auto_cookies])
            
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the downloaded file (prefer mp4, but accept any video format)
        video_extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi', '*.mov']
        downloaded_files = []
        for ext in video_extensions:
            downloaded_files.extend(glob.glob(os.path.join(temp_dir, ext)))
        
        if not downloaded_files:
            raise Exception("Video file not found after download")
        
        video_path = downloaded_files[0]
        print(f"Video downloaded successfully: {video_path}")
        return video_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        print(f"Error downloading video: {error_msg}")
        raise Exception(f"Failed to download video. Make sure the URL is valid and accessible.")
    except Exception as e:
        print(f"Error: {e}")
        raise


def get_video_title(url):
    """Extract video title from YouTube URL"""
    try:
        # Extract video ID from URL to get single video
        video_id = None
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0].split('?')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
        
        # Use video ID directly if we can extract it
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--get-title",
            "--no-playlist",
            "--no-warnings",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        title = result.stdout.strip().split('\n')[0]  # Get only first line
        return title if title else "youtube_video"
    except Exception as e:
        print(f"Warning: Could not get video title: {e}")
        return "youtube_video"


def download_youtube_transcript(url, output_folder, lang='en', prefer_auto=False, cookies_path=None):
    """Download transcript/subtitles from YouTube video using yt-dlp"""
    print(f"\nDownloading transcript from YouTube video...")
    
    try:
        # Extract video ID from URL to get single video
        video_id = None
        if 'watch?v=' in url:
            video_id = url.split('watch?v=')[1].split('&')[0].split('?')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0].split('&')[0]
        
        # Use video ID directly if we can extract it
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Build command for downloading subtitles
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "--no-warnings",
            "--skip-download",  # Don't download video, just subtitles
            "--sub-lang", lang,
            "--sub-format", "vtt",  # WebVTT format (can also use 'srt', 'json3')
            "-o", os.path.join(output_folder, "%(title)s.%(ext)s"),
        ]
        
        # Add cookies if available
        if cookies_path and os.path.exists(cookies_path):
            cmd.extend(["--cookies", cookies_path])
        
        # Prefer auto-generated subtitles if requested
        if prefer_auto:
            cmd.append("--write-auto-subs")
        else:
            cmd.append("--write-subs")
            # Also try auto-subs as fallback
            cmd.append("--write-auto-subs")
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find downloaded subtitle files
        subtitle_files = []
        for ext in ['*.vtt', '*.srt', '*.json3']:
            subtitle_files.extend(glob.glob(os.path.join(output_folder, ext)))
        
        if subtitle_files:
            # Get the most recent subtitle file
            subtitle_file = max(subtitle_files, key=os.path.getmtime)
            print(f"Transcript downloaded: {subtitle_file}")
            
            # Also create a plain text version for easier reading (with timestamps)
            transcript_txt_path = os.path.join(output_folder, "transcript.txt")
            convert_vtt_to_txt(subtitle_file, transcript_txt_path, keep_timestamps=True)
            
            return subtitle_file, transcript_txt_path
        else:
            print("Warning: No transcript file found. The video may not have subtitles available.")
            return None, None
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout
        print(f"Warning: Could not download transcript: {error_msg}")
        # Try with auto-generated subtitles as fallback
        if not prefer_auto:
            print("Trying auto-generated subtitles...")
            return download_youtube_transcript(url, output_folder, lang, prefer_auto=True, cookies_path=cookies_path)
        return None, None
    except Exception as e:
        print(f"Warning: Error downloading transcript: {e}")
        return None, None


def convert_vtt_to_txt(vtt_file, txt_file, keep_timestamps=True):
    """Convert VTT subtitle file to plain text format with optional timestamps"""
    try:
        import re
        import html
        
        with open(vtt_file, 'r', encoding='utf-8') as f:
            vtt_content = f.read()
        
        lines = vtt_content.split('\n')
        text_lines = []
        current_timestamp = None
        current_text_block = []  # Collect all text for current timestamp
        seen_blocks = set()  # To avoid duplicate blocks
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines, headers, and metadata
            if not line or line.startswith('WEBVTT') or \
               line.startswith('Kind:') or line.startswith('Language:') or \
               line.startswith('NOTE') or line.startswith('align:') or \
               line.startswith('position:'):
                i += 1
                continue
            
            # Extract timestamp from lines like "00:00:01.189 --> 00:00:01.199"
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if timestamp_match:
                # Process previous block if exists
                if current_timestamp and current_text_block:
                    # Combine all text lines for this timestamp
                    combined_text = ' '.join(current_text_block)
                    combined_text = ' '.join(combined_text.split())  # Normalize whitespace
                    
                    if combined_text:
                        # Check if we already have text for this exact timestamp
                        # If so, keep the longer/more complete version
                        existing_line = None
                        existing_idx = None
                        for idx, existing in enumerate(text_lines):
                            if existing.startswith(f"[{current_timestamp}]"):
                                existing_line = existing
                                existing_idx = idx
                                break
                        
                        if existing_line:
                            # Compare lengths - keep the longer one
                            existing_text = existing_line.replace(f"[{current_timestamp}] ", "")
                            if len(combined_text) > len(existing_text):
                                # Replace with longer version
                                if keep_timestamps:
                                    text_lines[existing_idx] = f"[{current_timestamp}] {combined_text}"
                                else:
                                    text_lines[existing_idx] = combined_text
                        else:
                            # New timestamp, add it
                            # Create unique key to avoid exact duplicates
                            unique_key = f"{current_timestamp}:{combined_text.lower()}"
                            if unique_key not in seen_blocks:
                                seen_blocks.add(unique_key)
                                
                                # Add timestamp if requested
                                if keep_timestamps:
                                    formatted_line = f"[{current_timestamp}] {combined_text}"
                                else:
                                    formatted_line = combined_text
                                
                                text_lines.append(formatted_line)
                                
                                # Clear seen_blocks periodically
                                if len(seen_blocks) > 200:
                                    seen_blocks = set(list(seen_blocks)[-100:])
                
                # Start new block with new timestamp
                current_timestamp = timestamp_match.group(1)
                # Convert to readable format: 00:00:01.189 -> 00:00:01
                if current_timestamp:
                    time_parts = current_timestamp.split('.')
                    readable_time = time_parts[0]  # Keep HH:MM:SS format
                    current_timestamp = readable_time
                current_text_block = []
                i += 1
                continue
            
            # Process text lines
            if line and not line.startswith('00:') and not line.isdigit():
                # Remove VTT formatting tags like <00:00:01.439><c>text</c>
                cleaned = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}><c>', '', line)
                cleaned = re.sub(r'</c>', '', cleaned)
                cleaned = re.sub(r'<c>', '', cleaned)
                
                # Decode HTML entities
                cleaned = html.unescape(cleaned)
                
                # Remove extra whitespace
                cleaned = ' '.join(cleaned.split())
                
                # Add to current text block if not empty
                if cleaned:
                    current_text_block.append(cleaned)
            
            i += 1
        
        # Process final block
        if current_timestamp and current_text_block:
            combined_text = ' '.join(current_text_block)
            combined_text = ' '.join(combined_text.split())
            if combined_text:
                unique_key = f"{current_timestamp}:{combined_text.lower()}"
                if unique_key not in seen_blocks:
                    if keep_timestamps:
                        formatted_line = f"[{current_timestamp}] {combined_text}"
                    else:
                        formatted_line = combined_text
                    text_lines.append(formatted_line)
        
        # Write to text file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_lines))
        
        print(f"Plain text transcript created: {txt_file} ({len(text_lines)} lines)")
        return txt_file
    except Exception as e:
        print(f"Warning: Could not convert VTT to text: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_frames(video_path, frame_rate):
    """Generator function to return frames from video at specified frame rate"""
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'Unable to open file {video_path}')

    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vs.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {total_duration/60:.2f} minutes")
    print(f"Processing at {frame_rate} frames per second")

    frame_time = 0
    frame_count = 0

    while True:
        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        frame_time += 1 / frame_rate

        (_, frame) = vs.read()
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()


def calculate_similarity(img1, img2):
    """Calculate structural similarity between two images"""
    # Resize images to same size for comparison
    img1_resized = cv2.resize(img1, (300, 200))
    img2_resized = cv2.resize(img2, (300, 200))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM (data_range=255 for uint8 images)
    similarity = ssim(gray1, gray2, data_range=255)
    return similarity


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


def detect_unique_screenshots(video_path, output_folder_screenshot_path, 
                              frame_rate, min_percent, max_percent,
                              use_similarity=True, similarity_threshold=0.95,
                              min_time_interval=MIN_TIME_BETWEEN_CAPTURES,
                              save_duplicates_path=None):
    """Detect and save unique screenshots from video with improved duplicate detection"""
    # Calculate derived parameters
    warmup = frame_rate
    fgbg_history = frame_rate * 15
    
    # Initialize background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=fgbg_history, 
        varThreshold=VAR_THRESHOLD,
        detectShadows=DETECT_SHADOWS
    )

    captured = False
    start_time = time.time()
    (W, H) = (None, None)
    screenshots_count = 0
    previous_frames = []  # Store last N frames for comparison
    previous_hashes = []  # Store hashes of last N frames
    last_capture_time = None  # Track time of last capture

    for frame_count, frame_time, frame in get_frames(video_path, frame_rate):
        orig = frame.copy()
        frame_resized = imutils.resize(frame, width=600)
        mask = fgbg.apply(frame_resized)

        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # Compute percentage of foreground
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # Motion detection logic
        if p_diff < min_percent and not captured and frame_count > warmup:
            # Check minimum time interval (if enabled)
            if min_time_interval > 0 and last_capture_time is not None:
                time_since_last = (frame_time - last_capture_time) * 60  # Convert to seconds
                if time_since_last < min_time_interval:
                    continue  # Skip if too soon after last capture
            
            # Additional similarity check to avoid duplicates
            should_save = True
            
            # 1. Fast check using dHash
            current_hash = dhash(orig)
            if use_similarity and previous_hashes:
                for prev_hash in previous_hashes:
                    dist = calculate_hamming_distance(current_hash, prev_hash)
                    if dist <= 5:  # Threshold for "very similar" (0-5 bits different out of 64)
                        should_save = False
                        # print(f"Frame {frame_count} skipped (dHash distance: {dist})")
                        break
            
            # 2. Detailed check using SSIM (only if dHash didn't flag it)
            if should_save and use_similarity and previous_frames:
                # Compare with last N frames
                max_similarity = 0
                for prev_frame in previous_frames:
                    similarity = calculate_similarity(orig, prev_frame)
                    if similarity and similarity > max_similarity:
                        max_similarity = similarity
                
                if max_similarity > similarity_threshold:
                    should_save = False
                    print(f"Frame {frame_count} skipped (similarity: {max_similarity:.3f}, "
                          f"time: {frame_time/60:.2f} min)")
                    
                    # Save duplicate if requested
                    if save_duplicates_path:
                        filename = f"{screenshots_count:03}_{round(frame_time/60, 2)}_DUPLICATE.png"
                        path = os.path.join(save_duplicates_path, filename)
                        cv2.imwrite(path, orig)

            if should_save:
                captured = True
                filename = f"{screenshots_count:03}_{round(frame_time/60, 2)}.png"
                path = os.path.join(output_folder_screenshot_path, filename)
                print(f"Saving {path} (frame {frame_count}, time: {frame_time/60:.2f} min)")
                cv2.imwrite(path, orig)
                
                # Update tracking
                last_capture_time = frame_time
                previous_frames.append(orig.copy())
                previous_hashes.append(current_hash)
                
                # Keep only last N frames for comparison
                if len(previous_frames) > MAX_SIMILARITY_COMPARISONS:
                    previous_frames.pop(0)
                    previous_hashes.pop(0)
                
                screenshots_count += 1

        elif captured and p_diff >= max_percent:
            captured = False

    elapsed_time = time.time() - start_time
    print(f'\n{screenshots_count} unique screenshots captured!')
    print(f'Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')
    return screenshots_count


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


def analyze_images_comprehensive(images_folder, similarity_threshold=0.95, output_folder=None, move_duplicates=False):
    """Comprehensive analysis of images: duplicates, relevance, and generate report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE IMAGE ANALYSIS")
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
                    similarity = calculate_similarity(img1, img2)
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


def initialize_output_folder(video_name, output_dir):
    """Create output folder structure"""
    # Sanitize video name for folder
    safe_name = sanitize_filename(video_name)
    if not safe_name:
        safe_name = "youtube_video"
    
    output_folder = os.path.join(output_dir, safe_name)
    
    # Create main output folder (create parent directories if needed)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create images subfolder
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    print(f'Output folder initialized: {output_folder}')
    return output_folder, images_folder


def convert_screenshots_to_pdf(images_folder, output_folder, video_name):
    """Convert screenshots to PDF"""
    safe_name = sanitize_filename(video_name)
    output_pdf_path = os.path.join(output_folder, f"{safe_name}.pdf")
    
    # Get all PNG files sorted
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    
    if not image_files:
        print("No images found to convert to PDF")
        return None
    
    print(f'Converting {len(image_files)} images to PDF...')
    
    try:
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_files))
        print(f'PDF created successfully: {output_pdf_path}')
        return output_pdf_path
    except Exception as e:
        print(f'Error creating PDF: {e}')
        raise


def parse_image_timestamp(filename, return_minutes=False):
    """Parse timestamp from image filename (format: 042_34.16.png -> 34.16 minutes or HH:MM:SS)"""
    try:
        # Extract the timestamp part (e.g., "34.16" from "042_34.16.png")
        match = re.search(r'_(\d+\.\d+)\.png', filename)
        if match:
            minutes = float(match.group(1))
            if return_minutes:
                return minutes
            # Convert to seconds
            total_seconds = int(minutes * 60)
            # Convert to HH:MM:SS format
            hours = total_seconds // 3600
            mins = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
    except:
        pass
    return None


def clean_transcript_text(text_or_lines):
    """Clean and improve transcript text with advanced duplicate removal for overlapping captions"""
    try:
        # If input is a single string, split it, otherwise assume it's a list of lines
        if isinstance(text_or_lines, str):
            lines = text_or_lines.split('\n')
        else:
            lines = text_or_lines

        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove timestamp if present at start of line
            line = re.sub(r'^\[\d{2}:\d{2}:\d{2}\]\s*', '', line)
            
            # Basic cleanup
            line = re.sub(r'\s+', ' ', line)
            line = line.replace('>>', '')
            line = line.replace('[laughter]', '')
            line = line.replace('[music]', '')
            line = line.replace('[applause]', '')
            line = line.strip()
            
            if not line:
                continue

            # Check for overlap with previous line
            # Many VTT/SRT files repeat the end of the previous line in the new line
            # Example:
            # Line 1: "Hello world this is"
            # Line 2: "world this is a test"
            
            # If the new line starts with the end of the previous line
            overlap_found = False
            
            # Check for exact subset
            if line in prev_line:
                continue # Skip completely redundant lines
                
            if prev_line in line:
                # If previous line is fully contained in current line, prefer current line
                # But we've already added prev_line. 
                # In a streaming context we might replace, but here we append.
                # Let's just keep the new content.
                if len(cleaned_lines) > 0:
                    cleaned_lines.pop()
                cleaned_lines.append(line)
                prev_line = line
                continue

            # Check for partial overlap (suffix of prev == prefix of curr)
            # We check from largest possible overlap down to a minimum threshold
            min_overlap = 10 # Minimum characters to consider an overlap
            max_check = min(len(prev_line), len(line))
            
            best_overlap = 0
            for i in range(max_check, min_overlap - 1, -1):
                suffix = prev_line[-i:]
                prefix = line[:i]
                if suffix == prefix:
                    best_overlap = i
                    break
            
            if best_overlap > 0:
                # Append only the new part
                new_part = line[best_overlap:].strip()
                if new_part:
                    cleaned_lines.append(new_part)
                    prev_line = line # Update prev_line to the FULL current line for next comparison
            else:
                # No overlap, just append
                cleaned_lines.append(line)
                prev_line = line

        # Join all parts
        full_text = ' '.join(cleaned_lines)
        
        # Now perform sentence splitting and paragraphing on the cleaner text
        sentences = re.split(r'([.!?]+)\s+', full_text)
        
        # Reconstruct sentences (split keeps delimiters)
        final_sentences = []
        current_sent = ""
        for part in sentences:
            if re.match(r'[.!?]+', part):
                current_sent += part
                final_sentences.append(current_sent.strip())
                current_sent = ""
            else:
                current_sent += part
        if current_sent:
            final_sentences.append(current_sent.strip())

        # Post-processing sentences
        processed_sentences = []
        seen_sentences = set()
        
        for sentence in final_sentences:
            if not sentence or len(sentence) < 2:
                continue
                
            # Remove duplicates (case-insensitive)
            sentence_lower = sentence.lower()
            if sentence_lower in seen_sentences:
                continue
            seen_sentences.add(sentence_lower)
            
            # Basic fixes
            sentence = sentence.replace(' cuz ', ' because ')
            sentence = sentence.replace(' u ', ' you ')
            sentence = sentence.replace(' ur ', ' your ')
            sentence = sentence.replace(' im ', ' I\'m ')
            sentence = sentence.replace(' dont ', ' don\'t ')
            sentence = sentence.replace(' cant ', ' can\'t ')
            sentence = sentence.replace(' wont ', ' won\'t ')
            sentence = sentence.replace(' its ', ' it\'s ')
            sentence = sentence.replace(' thats ', ' that\'s ')
            
            # Capitalize
            if sentence and not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
                
            processed_sentences.append(sentence)
            
        # Group into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for sentence in processed_sentences:
            current_paragraph.append(sentence)
            if len(current_paragraph) >= 4 or len(sentence) > 200:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return '\n\n'.join(paragraphs)

    except Exception as e:
        print(f"Warning: Error cleaning transcript: {e}")
        if isinstance(text_or_lines, list):
            return ' '.join(text_or_lines)
        return str(text_or_lines)
        return text


def create_markdown_report(images_folder, transcript_file, output_folder, video_name, embed_images=False):
    """Create a structured Markdown report for NotebookLM"""
    print("\nCreating Markdown report for NotebookLM...")
    
    try:
        # Parse transcript
        transcript_entries = []
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    timestamp = match.group(1)
                    text = match.group(2)
                    transcript_entries.append((timestamp, text))
        
        # Get all images
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
        image_data = []
        for img_file in image_files:
            timestamp = parse_image_timestamp(os.path.basename(img_file))
            if timestamp:
                image_data.append((timestamp, img_file))
        
        if not image_data:
            print("No images found for Markdown report")
            return None
            
        # Create Markdown file
        safe_name = sanitize_filename(video_name)
        output_md = os.path.join(output_folder, f"{safe_name}_notebooklm.md")
        
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(f"# {video_name}\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d')}\n")
            f.write(f"**Source:** Video Presentation\n\n")
            f.write("---\n\n")
            
            image_idx = 0
            
            # Helper to convert timestamp to seconds
            def timestamp_to_seconds(ts):
                parts = ts.split(':')
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
            while image_idx < len(image_data):
                img_timestamp, img_path = image_data[image_idx]
                img_filename = os.path.basename(img_path)
                
                # Determine time range for this slide
                current_seconds = timestamp_to_seconds(img_timestamp)
                
                # Find next slide timestamp to define the range
                if image_idx + 1 < len(image_data):
                    next_timestamp, _ = image_data[image_idx + 1]
                    next_seconds = timestamp_to_seconds(next_timestamp)
                else:
                    next_seconds = current_seconds + 300 # Assume last slide lasts 5 mins max
                
                # Collect transcript for this time range
                slide_transcript = []
                for ts, text in transcript_entries:
                    ts_seconds = timestamp_to_seconds(ts)
                    # Include text that starts after this slide appears, up until the next slide
                    if current_seconds <= ts_seconds < next_seconds:
                        slide_transcript.append(text)
                
                # Write Slide Section
                f.write(f"## Slide {image_idx + 1} ({img_timestamp})\n\n")
                
                if embed_images:
                    # Embed image as base64
                    try:
                        with open(img_path, "rb") as img_f:
                            encoded_string = base64.b64encode(img_f.read()).decode('utf-8')
                        f.write(f"![Slide {image_idx + 1}](data:image/png;base64,{encoded_string})\n\n")
                    except Exception as e:
                        print(f"Warning: Could not embed image {img_filename}: {e}")
                        rel_path = os.path.join("images", img_filename).replace("\\", "/")
                        f.write(f"![Slide {image_idx + 1}]({rel_path})\n\n")
                else:
                    # Note: NotebookLM might not render local images, but having the reference helps context
                    # We use a relative path
                    rel_path = os.path.join("images", img_filename).replace("\\", "/")
                    f.write(f"![Slide {image_idx + 1}]({rel_path})\n\n")
                
                # Write Transcript
                if slide_transcript:
                    # Pass list of lines directly to the new cleaning function
                    cleaned_text = clean_transcript_text(slide_transcript)
                    f.write("### Transcript\n\n")
                    f.write(f"{cleaned_text}\n\n")
                else:
                    f.write("*No speech detected for this slide.*\n\n")
                
                f.write("---\n\n")
                image_idx += 1
                
        print(f"Markdown report created: {output_md}")
        return output_md
        
    except Exception as e:
        print(f"Error creating Markdown report: {e}")
        import traceback
        traceback.print_exc()
        return None


def sync_images_with_transcript(images_folder, transcript_file, output_folder):
    """Sync images with transcript and create a combined document"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
        print("\nSyncing images with transcript and creating combined document...")
        
        # Parse transcript
        transcript_entries = []
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Extract timestamp and text: [00:00:01] text
                match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    timestamp = match.group(1)
                    text = match.group(2)
                    transcript_entries.append((timestamp, text))
        
        # Get all images with their timestamps
        image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
        image_data = []
        for img_file in image_files:
            timestamp = parse_image_timestamp(os.path.basename(img_file))
            if timestamp:
                image_data.append((timestamp, img_file))
        
        if not image_data:
            print("No images found to sync")
            return None
        
        # Create PDF
        output_pdf = os.path.join(output_folder, "combined_slides_with_transcript.pdf")
        doc = SimpleDocTemplate(output_pdf, pagesize=letter)
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#333333',
            spaceAfter=12,
            alignment=TA_CENTER
        )
        timestamp_style = ParagraphStyle(
            'Timestamp',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666666',
            spaceAfter=6,
            alignment=TA_LEFT
        )
        text_style = ParagraphStyle(
            'TranscriptText',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=12,
            alignment=TA_LEFT
        )
        
        # Match images with transcript
        image_idx = 0
        transcript_idx = 0
        
        while image_idx < len(image_data):
            img_timestamp, img_path = image_data[image_idx]
            
            # Find transcript entries that match this image's timestamp
            # Look for transcript entries within 30 seconds of the image
            matching_transcript = []
            
            # Convert timestamps to seconds for comparison
            def timestamp_to_seconds(ts):
                parts = ts.split(':')
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            
            img_seconds = timestamp_to_seconds(img_timestamp)
            
            # Find transcript entries near this timestamp
            for ts, text in transcript_entries:
                ts_seconds = timestamp_to_seconds(ts)
                if abs(ts_seconds - img_seconds) <= 30:  # Within 30 seconds
                    matching_transcript.append((ts, text))
            
            # Add image
            try:
                img = Image(img_path, width=6*inch, height=4.5*inch)
                img.drawHeight = 4.5*inch
                img.drawWidth = 6*inch
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                print(f"Warning: Could not add image {img_path}: {e}")
            
            # Add matching transcript text
            if matching_transcript:
                # Clean and format transcript
                transcript_texts = [text for _, text in matching_transcript]
                combined_text = ' '.join(transcript_texts)
                cleaned_text = clean_transcript_text(combined_text)
                
                if cleaned_text:
                    story.append(Paragraph(f"<b>Transcript ({img_timestamp}):</b>", timestamp_style))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(cleaned_text.replace('\n', '<br/>'), text_style))
                    story.append(Spacer(1, 0.3*inch))
            
            story.append(PageBreak())
            image_idx += 1
        
        # Build PDF
        doc.build(story)
        print(f"Combined PDF created: {output_pdf}")
        return output_pdf
        
    except ImportError as e:
        print(f"Error: Missing required library. Please install: pip install reportlab pyspellchecker")
        print(f"Error details: {e}")
        return None
    except Exception as e:
        print(f"Error creating combined document: {e}")
        import traceback
        traceback.print_exc()
        return None


def cleanup_temp_files(output_dir):
    """Clean up temporary files"""
    temp_dir = os.path.join(output_dir, "temp")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up")
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YouTube video or local video file to PDF slides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube2pdf.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python youtube2pdf.py "./input/video.mp4"
  python youtube2pdf.py "https://youtu.be/VIDEO_ID" --frame-rate 10 --similarity-threshold 0.98
  python youtube2pdf.py "https://youtu.be/VIDEO_ID" --download-transcript
  python youtube2pdf.py "https://youtu.be/VIDEO_ID" --download-transcript --create-combined
  python youtube2pdf.py "https://youtu.be/VIDEO_ID" --download-transcript --clean-transcript --create-combined
        """
    )
    
    parser.add_argument(
        "input",
        help="YouTube URL or path to local video file"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "-fr", "--frame-rate",
        type=int,
        default=FRAME_RATE,
        help=f"Frames per second to process (default: {FRAME_RATE})"
    )
    
    parser.add_argument(
        "-st", "--similarity-threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for duplicate detection 0-1 (default: {SIMILARITY_THRESHOLD})"
    )
    
    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Disable similarity-based duplicate detection"
    )
    
    parser.add_argument(
        "--min-percent",
        type=float,
        default=MIN_PERCENT,
        help=f"Min %% diff to detect motion stopped (default: {MIN_PERCENT})"
    )
    
    parser.add_argument(
        "--max-percent",
        type=float,
        default=MAX_PERCENT,
        help=f"Max %% diff to detect motion (default: {MAX_PERCENT})"
    )
    
    parser.add_argument(
        "--download-transcript",
        action="store_true",
        help="Download transcript/subtitles from YouTube video (YouTube URLs only)"
    )
    
    parser.add_argument(
        "--transcript-lang",
        type=str,
        default="en",
        help="Language code for transcript (default: 'en'). Use 'auto' for auto-generated subtitles"
    )
    
    parser.add_argument(
        "--prefer-auto-subs",
        action="store_true",
        help="Prefer auto-generated subtitles over manual subtitles"
    )
    
    parser.add_argument(
        "--create-combined",
        action="store_true",
        help="Create a combined PDF with images and synchronized transcript"
    )
    
    parser.add_argument(
        "--create-markdown",
        action="store_true",
        help="Create a Markdown report suitable for NotebookLM"
    )
    
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed images as Base64 in the Markdown report (WARNING: Creates very large files)"
    )
    
    parser.add_argument(
        "--clean-transcript",
        action="store_true",
        help="Create a cleaned transcript file with paragraphs and spell checking"
    )
    
    parser.add_argument(
        "--min-time-interval",
        type=int,
        default=MIN_TIME_BETWEEN_CAPTURES,
        help=f"Minimum seconds between image captures (default: {MIN_TIME_BETWEEN_CAPTURES}, 0 = disabled)"
    )
    
    parser.add_argument(
        "--post-process",
        action="store_true",
        help="Analyze images and generate report for duplicates and irrelevant content (does not auto-delete)"
    )
    
    parser.add_argument(
        "--save-duplicates",
        action="store_true",
        help="Save detected duplicates to a 'duplicates' folder instead of discarding them"
    )
    
    parser.add_argument(
        "--cookies",
        type=str,
        help="Path to cookies file (Netscape format) for YouTube authentication"
    )
    
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip video download and image extraction, only run post-processing on existing images"
    )
    
    args = parser.parse_args()
    
    # Get parameters from arguments
    frame_rate = args.frame_rate
    min_percent = args.min_percent
    max_percent = args.max_percent
    similarity_threshold = args.similarity_threshold
    
    # Ensure output directory exists
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    input_source = args.input
    video_path = None
    video_name = None
    
    try:
        # Check if input is YouTube URL or local file
        is_youtube = is_youtube_url(input_source)
        if is_youtube:
            print("Detected YouTube URL")
            video_name = get_video_title(input_source)
            if not args.skip_extraction:
                video_path = download_youtube_video(input_source, output_dir, cookies_path=args.cookies)
            else:
                print("Skipping video download (--skip-extraction)")
        else:
            print("Detected local video file")
            video_name = Path(input_source).stem
            if not args.skip_extraction:
                if not os.path.exists(input_source):
                    raise FileNotFoundError(f"Video file not found: {input_source}")
                video_path = input_source
            else:
                print("Skipping video check (--skip-extraction)")
        
        print(f"\nProcessing video: {video_name}")
        if video_path:
            print(f"Video path: {video_path}\n")
        
        # Initialize output folders
        output_folder, images_folder = initialize_output_folder(video_name, output_dir)
        
        # Create duplicates folder if requested
        duplicates_folder = None
        if args.save_duplicates:
            duplicates_folder = os.path.join(output_folder, "duplicates")
            os.makedirs(duplicates_folder, exist_ok=True)
            print(f"Duplicates will be saved to: {duplicates_folder}")
        
        # Download transcript if requested and it's a YouTube URL
        transcript_vtt = None
        transcript_txt = None
        if args.download_transcript and is_youtube and not args.skip_extraction:
            transcript_vtt, transcript_txt = download_youtube_transcript(
                input_source,
                output_folder,
                lang=args.transcript_lang,
                prefer_auto=args.prefer_auto_subs,
                cookies_path=args.cookies
            )
        elif args.download_transcript and not is_youtube:
            print("Warning: Transcript download is only available for YouTube URLs. Skipping transcript download.")
        
        # If we skipped download or didn't request it, try to find existing transcript
        if not transcript_txt:
            possible_transcripts = glob.glob(os.path.join(output_folder, "*.txt"))
            for t in possible_transcripts:
                filename = os.path.basename(t)
                if "cleaned" not in filename and "report" not in filename and "info" not in filename:
                    transcript_txt = t
                    print(f"Found existing transcript: {transcript_txt}")
                    break

        # Detect unique screenshots
        if not args.skip_extraction:
            screenshots_count = detect_unique_screenshots(
                video_path,
                images_folder,
                frame_rate=frame_rate,
                min_percent=min_percent,
                max_percent=max_percent,
                use_similarity=not args.no_similarity,
                similarity_threshold=similarity_threshold,
                min_time_interval=args.min_time_interval,
                save_duplicates_path=duplicates_folder
            )
        else:
            # Count existing images
            existing_images = glob.glob(os.path.join(images_folder, "*.png"))
            screenshots_count = len(existing_images)
            print(f"Skipping extraction. Found {screenshots_count} existing images.")
        
        if screenshots_count == 0:
            print("No screenshots were captured (or found). Exiting.")
            return
        
        # Analyze images if requested
        if args.post_process:
            analysis_result = analyze_images_comprehensive(
                images_folder,
                similarity_threshold=similarity_threshold,
                output_folder=output_folder,
                move_duplicates=args.save_duplicates
            )
            if analysis_result:
                print(f"\nAnalysis Summary:")
                print(f"  - {analysis_result['total_flagged']} images flagged for review")
                print(f"  - {len(analysis_result['irrelevant'])} potentially irrelevant")
                print(f"  - {len(analysis_result['duplicates'])} duplicate groups")
                print(f"\nPlease review the report: {analysis_result['report_path']}")
        
        # Convert to PDF
        pdf_path = convert_screenshots_to_pdf(images_folder, output_folder, video_name)
        
        # Create cleaned transcript if requested
        cleaned_transcript_path = None
        if args.clean_transcript and transcript_txt:
            print("\nCreating cleaned transcript...")
            try:
                cleaned_transcript_path = os.path.join(output_folder, "transcript_cleaned.txt")
                with open(transcript_txt, 'r', encoding='utf-8') as f:
                    transcript_content = f.read()
                
                # Extract all text (remove timestamps for cleaning)
                text_only = re.sub(r'\[\d{2}:\d{2}:\d{2}\]\s*', '', transcript_content)
                cleaned_text = clean_transcript_text(text_only)
                
                with open(cleaned_transcript_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                print(f"Cleaned transcript saved to: {cleaned_transcript_path}")
            except Exception as e:
                print(f"Warning: Could not create cleaned transcript: {e}")
        
        # Create combined PDF if requested
        combined_pdf_path = None
        if args.create_combined and transcript_txt:
            combined_pdf_path = sync_images_with_transcript(
                images_folder, 
                transcript_txt, 
                output_folder
            )
            
        # Create Markdown report if requested
        markdown_path = None
        if args.create_markdown and transcript_txt:
            markdown_path = create_markdown_report(
                images_folder,
                transcript_txt,
                output_folder,
                video_name,
                embed_images=args.embed_images
            )
        elif args.create_markdown and not transcript_txt:
            print("Warning: Cannot create Markdown report without transcript.")
        
        # Cleanup temp files
        cleanup_temp_files(output_dir)
        
        print("\n" + "="*60)
        print("Conversion completed successfully!")
        print(f"Images saved to: {images_folder}")
        print(f"PDF saved to: {pdf_path}")
        if transcript_txt:
            print(f"Transcript (text) saved to: {transcript_txt}")
        if transcript_vtt:
            print(f"Transcript (VTT) saved to: {transcript_vtt}")
        if cleaned_transcript_path:
            print(f"Cleaned transcript saved to: {cleaned_transcript_path}")
        if combined_pdf_path:
            print(f"Combined PDF (images + transcript) saved to: {combined_pdf_path}")
        if markdown_path:
            print(f"NotebookLM Markdown Report saved to: {markdown_path}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

