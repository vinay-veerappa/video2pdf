import os
import sys
import json
import shutil
import threading
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Import our existing logic
from config import OUTPUT_DIR
from utils import is_youtube_url, get_video_id
from main import main as run_pipeline_cli 
from main import process_video_workflow
from scripts import image_dedup
from pdf_generator import create_pdf_from_data, create_docx_from_data
from utils import parse_image_timestamp, sanitize_filename
import glob
import re
import datetime

def detect_drawn_crop_box(img):
    """
    Detects a drawn rectangle (Red, Green, or Blue) in the image
    and returns (x, y, w, h). Returns None if no clear box is found.
    """
    try:
        import cv2
        import numpy as np
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Color ranges for common "markup" colors (Red, Green, Blue)
        # Red is tricky as it wraps around 0/180
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        
        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_g = cv2.inRange(hsv, lower_green, upper_green)
        mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_r1, mask_r2)
        mask = cv2.bitwise_or(mask, mask_g)
        mask = cv2.bitwise_or(mask, mask_b)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour by area
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100: # Minimum area noise filter
                x, y, w, h = cv2.boundingRect(c)
                return x, y, w, h
    except Exception as e:
        print(f"Error in detect_drawn_crop_box: {e}")
    return None


app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = OUTPUT_DIR

# Global state to track progress (simple version)
# In production, use Redis or a database
JOBS = {} 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list_videos')
def list_videos():
    """List existing video projects and their status."""
    output_dir = app.config['OUTPUT_FOLDER']
    projects = []
    
    if os.path.exists(output_dir):
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                # Check status
                has_video = os.path.exists(os.path.join(path, "video")) and len(os.listdir(os.path.join(path, "video"))) > 0
                
                images_dir = os.path.join(path, "images")
                has_images = os.path.exists(images_dir) and len(glob.glob(os.path.join(images_dir, "*.png"))) > 0
                
                dedup_json = os.path.join(images_dir, "dedup_results.json")
                has_dedup = os.path.exists(dedup_json)
                
                projects.append({
                    'id': name,
                    'name': name,
                    'has_video': has_video,
                    'has_images': has_images,
                    'has_dedup': has_dedup,
                    'has_transcript': any(
                        len(glob.glob(os.path.join(path, "transcripts", f"*{ext}"))) > 0 or 
                        len(glob.glob(os.path.join(path, f"*{ext}"))) > 0 
                        for ext in ['.txt', '.vtt', '.srt', '.json', '.xml']
                    )
                })
    
    return jsonify(projects)

@app.route('/process', methods=['POST'])
def process_video():
    data = request.form
    url = data.get('url')
    existing_video_id = data.get('existing_video_id')
    
    # Flags
    skip_download = data.get('skip_download') == 'true'
    skip_extraction = data.get('skip_extraction') == 'true'
    skip_deduplication = data.get('skip_deduplication') == 'true'
    auto_crop = data.get('auto_crop') == 'true'
    download_transcript = data.get('download_transcript') == 'true'
    
    if not url and not existing_video_id:
        return jsonify({'error': 'No URL or Existing Video provided'}), 400
        
    # Start processing in a background thread
    job_id = "job_" + str(len(JOBS) + 1)
    JOBS[job_id] = {'status': 'processing', 'log': [], 'message': 'Starting...', 'percent': 0}
    
    thread = threading.Thread(target=run_processing_task, args=(job_id, url, existing_video_id, skip_download, skip_extraction, skip_deduplication, download_transcript, auto_crop))
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def job_status(job_id):
    return jsonify(JOBS.get(job_id, {'status': 'unknown'}))

from main import process_video_workflow
from scripts import image_dedup

from main import process_video_workflow
from scripts import image_dedup

def run_processing_task(job_id, url, existing_video_id=None, skip_download=False, skip_extraction=False, skip_deduplication=False, download_transcript=True, auto_crop=False):
    """
    Run the extraction and deduplication pipeline.
    """
    try:
        JOBS[job_id]['percent'] = 0
        JOBS[job_id]['message'] = "Starting process..."
        
        def progress_callback(data):
            JOBS[job_id].update(data)

        # Check for cookies.txt in root
        cookies_path = None
        if os.path.exists('cookies.txt'):
            cookies_path = os.path.abspath('cookies.txt')
            print(f"Found cookies.txt at {cookies_path}")
        
        # 1. Run Extraction (or skip if requested)
        # If existing_video_id is provided, we construct the path manually or let main.py handle it?
        # main.py expects a URL or a file path.
        # If we have existing_video_id, we need to find the video file.
        
        input_source = url
        if existing_video_id:
            # Find the video file in the output directory
            video_dir = os.path.join(app.config['OUTPUT_FOLDER'], existing_video_id, "video")
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, "*.*"))
                if videos:
                    input_source = videos[0] # Use the local file path
                    print(f"Using existing video file: {input_source}")
                    
                    # If we need to download transcript, we need the original URL
                    if download_transcript:
                        metadata_path = os.path.join(app.config['OUTPUT_FOLDER'], existing_video_id, "metadata.txt")
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        if line.startswith("url:"):
                                            original_url = line.replace("url:", "").strip()
                                            if original_url:
                                                print(f"Found original URL in metadata: {original_url}")
                                                # Use URL as input source to allow transcript download
                                                # main.py will skip video download if skip_extraction is True
                                                input_source = original_url
                                            break
                            except Exception as e:
                                print(f"Error reading metadata: {e}")
                else:
                    # Maybe only images exist?
                    # If we skip extraction, we don't strictly need the video file if images are there.
                    # But process_video_workflow might need it for naming.
                    pass
            
            if not input_source:
                 # Fallback to just using the folder name if we are skipping everything up to deduplication
                 print(f"Warning: Could not find video file for {existing_video_id}. Assuming resume/skip_extraction.")
                 input_source = existing_video_id

        result = process_video_workflow(
            input_source, 
            output_dir=app.config['OUTPUT_FOLDER'],
            progress_callback=progress_callback,
            cookies=cookies_path,
            skip_download=skip_download,
            skip_extraction=skip_extraction,
            download_transcript=download_transcript
        )
        
        images_folder = result['images_folder']
        video_name = result['video_name']
        
        # 1.5 Auto-Crop and Reference Logic
        if auto_crop:
            JOBS[job_id]['status'] = 'cropping'
            JOBS[job_id]['message'] = 'Processing crop and references...'
            JOBS[job_id]['percent'] = 60
            
            print("Auto-cropping requested. Moving originals to raw/...")
            raw_dir = os.path.join(images_folder, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            
            # Move all pngs to raw
            pngs = glob.glob(os.path.join(images_folder, "*.png"))
            if pngs:
                for png in pngs:
                    if not os.path.exists(os.path.join(raw_dir, os.path.basename(png))):
                        shutil.move(png, os.path.join(raw_dir, os.path.basename(png)))
                    else:
                        try: os.remove(png)
                        except: pass
                
                # --- STRATEGY DETERMINATION ---
                workspace_root = os.path.dirname(os.path.abspath(__file__))
                crop_ref_dir = os.path.join(workspace_root, 'references', 'crop')
                ignore_ref_dir = os.path.join(workspace_root, 'references', 'ignore')
                config_path = os.path.join(workspace_root, 'global_crop_config.json')
                
                # 1. Look for Reference Crop Image
                crop_ratios = None
                crop_mode = "auto"
                
                if os.path.exists(crop_ref_dir):
                    refs = glob.glob(os.path.join(crop_ref_dir, "*"))
                    if refs:
                         # Use the first valid image to determine crop
                         try:
                             print(f"Analyzing reference crop image: {refs[0]}")
                             from scripts import optimize_images
                             import cv2
                             ref_img = cv2.imread(refs[0])
                             if ref_img is not None:
                                 # A. Check for Drawn Box First
                                 drawn_box = detect_drawn_crop_box(ref_img)
                                 H, W = ref_img.shape[:2]
                                 
                                 if drawn_box:
                                     x, y, w, h = drawn_box
                                     print(f"Detected DRAWN MARKER: {x},{y} {w}x{h}")
                                     crop_ratios = [x/W, y/H, w/W, h/H]
                                     crop_mode = "reference_drawn"
                                 else:
                                     print("No drawn marker detected, using content auto-detection...")
                                     x, y, w, h = optimize_images.get_content_bbox(ref_img)
                                     crop_ratios = [x/W, y/H, w/W, h/H]
                                     crop_mode = "reference_auto"
                                     
                                 print(f"Reference crop determined: {crop_ratios}")
                         except Exception as e:
                             print(f"Error processing reference crop: {e}")

                # 2. If no reference, check specific config
                if not crop_ratios and os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            crop_ratios = json.load(f).get('ratios')
                        if crop_ratios: crop_mode = "config"
                    except: pass
                
                # --- EXECUTION ---
                
                # Prepare processor
                from scripts import optimize_images
                raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
                
                # Pre-load ignore images if any
                ignore_hashes = []
                if os.path.exists(ignore_ref_dir):
                    from scripts import image_dedup
                    from PIL import Image
                    import imagehash
                    
                    ign_files = glob.glob(os.path.join(ignore_ref_dir, "*"))
                    print(f"Loading {len(ign_files)} ignore reference images...")
                    for ign in ign_files:
                        try:
                            # We use perceptual hash for robustness
                            with Image.open(ign) as i:
                                h = imagehash.phash(i)
                                ignore_hashes.append(h)
                        except: pass
                
                print(f"Processing {len(raw_files)} images with mode={crop_mode}...")
                
                for rf in raw_files:
                    fname = os.path.basename(rf)
                    out_path = os.path.join(images_folder, fname)
                    
                    # 1. CROP
                    if crop_ratios:
                        # Apply specific crop
                        shutil.copy2(rf, out_path)
                        apply_global_crop_to_image(out_path, crop_ratios)
                    else:
                        # Auto crop (per image or union? process_images does union by default if we ask)
                        # But here we are iterating one by one. 
                        # To use optimize_images's union logic, we should call it on the directory.
                        pass # We will handle auto-batch below if no ratios
                        
                    # 2. FILTER (Ignore List)
                    if ignore_hashes and os.path.exists(out_path):
                        try:
                            from PIL import Image
                            import imagehash
                            with Image.open(out_path) as i:
                                curr_h = imagehash.phash(i)
                                # Check distance
                                is_ignored = False
                                for ref_h in ignore_hashes:
                                    if curr_h - ref_h < 10: # Threshold 10 is fairly strict for phash
                                        print(f"Ignored {fname} (matched reference)")
                                        is_ignored = True
                                        break
                                if is_ignored:
                                    os.remove(out_path)
                                    continue # Skip header/footer check if ignored
                        except: pass
                
                # If we didn't have ratios, we haven't cropped yet. Run batch auto-crop now.
                # Note: If we had ratios, we already copied and cropped above.
                if not crop_ratios:
                    # Filter existing in destination from Ignore List first?
                    # No, we haven't moved them yet.
                    # We need to run process_images on RAW, but output to IMAGES.
                    # But optimize_images doesn't support the 'ignore list' internally.
                    # Let's run standard process, then post-filter.
                    
                    optimize_images.process_images(input_dir=raw_dir, output_dir=images_folder, crop=True)
                    
                    # Post-filter for ignores
                    if ignore_hashes:
                        processed = glob.glob(os.path.join(images_folder, "*.png"))
                        from PIL import Image
                        import imagehash
                        for p in processed:
                            try:
                                with Image.open(p) as i:
                                    curr_h = imagehash.phash(i)
                                    for ref_h in ignore_hashes:
                                        if curr_h - ref_h < 10:
                                            os.remove(p)
                                            print(f"Ignored {os.path.basename(p)} (matched reference)")
                                            break
                            except: pass

                print("Auto-cropping and referencing complete.")
            else:
                print("No images found to crop.")
        
        
        JOBS[job_id]['status'] = 'analyzing'
        JOBS[job_id]['message'] = 'Starting image analysis and deduplication...'
        JOBS[job_id]['percent'] = 75
        
        if not skip_deduplication:
            # We use the 'compare-all' mode logic from image_dedup.py
            # But we need to call it programmatically.
            
            # We can use subprocess or import. Since we are in the same env, import is better.
            # However, image_dedup.py is designed as a script. 
            # Let's use subprocess to be safe and consistent with CLI.
            import subprocess
            
            print(f"Running deduplication on {images_folder}...")
            JOBS[job_id]['message'] = 'Analyzing images for duplicates and blanks...'
            
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "scripts", "image_dedup.py"),
                images_folder,
                "--mode", "compare-all",
                "--sequential",
                "--crop-method", "content_aware",
                "--crop-margin", "0.20",
                "--skip-blanks"
            ]
            
            subprocess.run(cmd, check=True)
            
            JOBS[job_id]['message'] = 'Deduplication complete!'
            JOBS[job_id]['percent'] = 95
        else:
            # Check if we can reuse existing results
            json_path = os.path.join(images_folder, "dedup_results.json")
            if os.path.exists(json_path):
                 print(f"Skipping deduplication. Reusing existing results at {json_path}")
                 JOBS[job_id]['message'] = 'Deduplication skipped - reusing existing results...'
                 JOBS[job_id]['percent'] = 95
            else:
                 print("Skipping deduplication as requested. No existing results found.")
                 JOBS[job_id]['message'] = 'Deduplication skipped - generating dummy results...'
                 JOBS[job_id]['percent'] = 85
                 # Generate dummy dedup_results.json so curation page works
                 # We treat all images as unique/kept for now, or just list them.
                 # curate.html expects: { 'blanks': [], 'duplicates': [], 'all_files': [...] }
                 all_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(images_folder, "*.png"))])
                 dummy_results = {
                     'blanks': [],
                     'duplicates': [],
                     'all_files': all_files
                 }
                 with open(json_path, 'w') as f:
                     json.dump(dummy_results, f)
                 print(f"Created dummy results at {json_path}")
                 JOBS[job_id]['percent'] = 95
        
        JOBS[job_id]['status'] = 'ready_for_curation'
        JOBS[job_id]['message'] = 'Ready for curation!'
        JOBS[job_id]['percent'] = 100
        JOBS[job_id]['video_id'] = video_name # This is the folder name
        
    except Exception as e:
        print(f"Job failed: {e}")
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)

@app.route('/curate/<video_id>')
def curate(video_id):
    # Flask automatically URL-decodes route parameters
    # Load the JSON results
    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_id, "images")
    json_path = os.path.join(images_dir, "dedup_results.json")
    
    if not os.path.exists(json_path):
        # Fallback: Generate dummy data if missing
        # This handles cases where deduplication was skipped or failed, but images exist.
        glob_pattern = os.path.join(images_dir, "*.png")
        all_files = sorted([os.path.basename(f) for f in glob.glob(glob_pattern)])
        
        if not all_files:
             return "Error: No images found for this video.", 404
             
        data = {
            'blanks': [],
            'duplicates': [],
            'all_files': all_files
        }
        # Optionally save it to avoid re-generating
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            pass # If we can't write, just use the in-memory data
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
    return render_template('curate.html', video_id=video_id, data=data)

@app.route('/image/<video_id>/<path:filename>')
def serve_image(video_id, filename):
    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_id, "images")
    return send_from_directory(images_dir, filename)

@app.route('/image_raw/<video_id>/<path:filename>')
def serve_image_raw(video_id, filename):
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_dir = os.path.join(output_folder, "images")
    raw_dir = os.path.join(images_dir, "raw")
    
    # Try to serve from raw first (original uncropped)
    if os.path.exists(os.path.join(raw_dir, filename)):
        return send_from_directory(raw_dir, filename)
    
    # Fallback to current images (if no raw exists yet)
    return send_from_directory(images_dir, filename)

@app.route('/save_curation', methods=['POST'])
def save_curation():
    data = request.json
    video_id = data.get('video_id')
    keeps = set(data.get('keeps', []))
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'}), 400
        
    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_id, "images")
    base_dir = os.path.join(images_dir, "organized_moderate")
    
    unique_dir = os.path.join(base_dir, "unique")
    duplicates_dir = os.path.join(base_dir, "duplicates")
    blanks_dir = os.path.join(base_dir, "blanks")
    
    os.makedirs(unique_dir, exist_ok=True)
    os.makedirs(duplicates_dir, exist_ok=True)
    os.makedirs(blanks_dir, exist_ok=True)
    
    # Load original results to know what was blank vs duplicate
    json_path = os.path.join(images_dir, "dedup_results.json")
    with open(json_path, 'r') as f:
        orig_data = json.load(f)
        
    blanks = set(orig_data['blanks'])
    all_files = orig_data['all_files']
    
    # Move files
    count_kept = 0
    for filename in all_files:
        src = os.path.join(images_dir, filename)
        if not os.path.exists(src): continue
        
        if filename in keeps:
            dst = os.path.join(unique_dir, filename)
            shutil.copy2(src, dst)
            count_kept += 1
        else:
            # Discarded
            if filename in blanks:
                dst = os.path.join(blanks_dir, filename)
            else:
                dst = os.path.join(duplicates_dir, filename)
            shutil.copy2(src, dst)
            
    return jsonify({'status': 'success', 'kept': count_kept})

def prepare_slides_data(video_id):
    """
    Sync curated images with transcript and return data for editing.
    """
    print(f"DEBUG: prepare_slides_data for {video_id}")
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_folder = os.path.join(output_folder, "images", "organized_moderate", "unique")
    
    # If curated folder doesn't exist, fallback to main images folder (if user skipped curation)
    if not os.path.exists(images_folder):
        print(f"DEBUG: Curated folder {images_folder} not found, checking fallback...")
        images_folder = os.path.join(output_folder, "images")
        
    print(f"DEBUG: Using images_folder: {images_folder}")
        
    # Find transcript
    transcript_file = None
    possible_transcripts = []
    for ext in ['*.txt', '*.vtt', '*.srt', '*.json', '*.xml']:
        possible_transcripts.extend(glob.glob(os.path.join(output_folder, "transcripts", ext)))
        possible_transcripts.extend(glob.glob(os.path.join(output_folder, ext)))
    
    for t in possible_transcripts:
        if "cleaned" not in t and "report" not in t and "metadata" not in t:
            transcript_file = t
            break
            
    print(f"DEBUG: Found transcript file: {transcript_file}")
            
    if not transcript_file or not os.path.exists(images_folder):
        print("DEBUG: Missing transcript or images folder.")
        return []

    # Parse transcript
    transcript_entries = []
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                if match:
                    transcript_entries.append((match.group(1), match.group(2)))
    except Exception as e:
        print(f"DEBUG: Error parsing transcript: {e}")
        pass
        
    print(f"DEBUG: Parsed {len(transcript_entries)} transcript entries.")

    # Get images
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    print(f"DEBUG: Found {len(image_files)} images.")
    slides = []
    
    # Helper to parse timestamp from filename (e.g., "001_12.5.png" -> 12.5)
    def get_time(fname):
        try:
            return float(os.path.basename(fname).split('_')[1].rsplit('.', 1)[0])
        except:
            return 0.0
            
    # Calculate relative path for URL generation
    # images_folder is absolute path to where images were found
    # We serve images from /image/<video_id>/<path> which maps to output/<video_id>/images/<path>
    images_root = os.path.join(output_folder, "images")
    rel_path = os.path.relpath(images_folder, images_root)
    if rel_path == ".":
        rel_path = ""
    # Ensure forward slashes for URL
    rel_path = rel_path.replace("\\", "/")
            
    # Sort files by timestamp
    sorted_files = sorted(image_files, key=get_time)
    
    # Parse transcript timestamps to seconds
    parsed_transcript = []
    for ts, text in transcript_entries:
        parts = ts.split(':')
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        parsed_transcript.append({'start': seconds, 'text': text})
    
    # Detect if image timestamps are in minutes or seconds
    time_multiplier = 1.0
    if sorted_files and parsed_transcript:
        max_img_time = get_time(sorted_files[-1])
        max_trans_time = parsed_transcript[-1]['start']
        
        # If max image time is significantly smaller than transcript time, 
        # and multiplying by 60 brings it closer, assume minutes.
        # We check if the error with *60 is smaller than error with *1
        err_seconds = abs(max_img_time - max_trans_time)
        err_minutes = abs(max_img_time * 60 - max_trans_time)
        
        if err_minutes < err_seconds:
            time_multiplier = 60.0
            
    for i, img_path in enumerate(sorted_files):
        filename = os.path.basename(img_path)
        current_time = get_time(img_path) * time_multiplier
        
        # Determine end time: start of next slide or end of video
        if i < len(sorted_files) - 1:
            next_time = get_time(sorted_files[i+1]) * time_multiplier
        else:
            next_time = float('inf')
            
        # Filter transcript entries
        slide_text_parts = []
        for entry in parsed_transcript:
            # simple logic: if entry start time is >= current slide time and < next slide time
            if current_time <= entry['start'] < next_time:
                slide_text_parts.append(entry['text'])
        
        # Construct URL
        if rel_path:
            img_url = f"/image/{video_id}/{rel_path}/{filename}"
        else:
            img_url = f"/image/{video_id}/{filename}"

        slides.append({
            'image': filename,
            'image_url': img_url,
            'image_path': img_path,
            'timestamp': str(datetime.timedelta(seconds=int(current_time))),
            'text': " ".join(slide_text_parts)
        })
        
    return slides

def prepare_grouped_data(video_id):
    """
    Groups linear slides into 5-minute segments for V2 editing interface.
    """
    slides = prepare_slides_data(video_id)
    if not slides:
        return []
        
    groups = []
    current_group = {
        'start_time': 0,
        'end_time': 300, # 5 mins
        'label': '00:00 - 05:00',
        'slides': []
    }
    
    # Sort just in case
    # Timestamp in slides is string "0:00:12" or similar. We need seconds.
    # Luckily prepare_slides_data calculated seconds but didn't store exactly.
    # But files are sorted by time.
    
    def parsestr(t):
        # "0:00:12" -> seconds
        try:
            parts = t.split(':')
            return int(parts[0])*3600 + int(parts[1])*60 + int(float(parts[2]))
        except: return 0
        
    for slide in slides:
        ts = parsestr(slide['timestamp'])
        
        # Check if belongs to next group
        while ts >= current_group['end_time']:
            # Push current
            groups.append(current_group)
            
            # Make new
            next_start = current_group['end_time']
            next_end = next_start + 300
            
            # Format label
            s_min = next_start // 60
            e_min = next_end // 60
            
            current_group = {
                'start_time': next_start,
                'end_time': next_end,
                'label': f"{s_min:02d}:00 - {e_min:02d}:00",
                'slides': []
            }
            
        current_group['slides'].append(slide)
        
    # Process into final format
    final_groups = []
    
    # Process the last group first into a list of raw groups
    all_raw_groups = []
    
    # We need to re-loop because the previous logic creates 'groups' list + 'current_group'
    # Let's just fix the loop logic slightly to flush the last one
    if current_group['slides']:
        groups.append(current_group)
        
    for g in groups:
        # Aggregate text
        combined_text = " ".join([s['text'] for s in g['slides'] if s['text']])
        
        # Format images
        formatted_images = []
        for s in g['slides']:
            formatted_images.append({
                'url': s['image_url'],
                'removed': False,
                # Keep other metadata if needed
                'original_path': s['image_path'],
                'timestamp': s['timestamp']
            })
            
        final_groups.append({
            'time_range': g['label'],
            'text': combined_text,
            'images': formatted_images,
            'start_time': g['start_time'],
            'end_time': g['end_time']
        })
        
    return final_groups
        
    print(f"DEBUG: Returning {len(groups)} groups.")
    return groups

@app.route('/curate_v2/<video_id>')
def curate_v2(video_id):
    """
    New grid-based curation interface.
    """
    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_id, "images")
    json_path = os.path.join(images_dir, "dedup_results.json")
    
    if not os.path.exists(json_path):
        # Fallback logic same as v1
        glob_pattern = os.path.join(images_dir, "*.png")
        all_files = sorted([os.path.basename(f) for f in glob.glob(glob_pattern)])
        
        if not all_files:
             return "Error: No images found for this video.", 404
             
        data = {
            'blanks': [],
            'duplicates': [],
            'all_files': all_files
        }
        try:
            with open(json_path, 'w') as f:
                json.dump(data, f)
        except: pass
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
    # Calculate already_curated_count
    # This is useful to show if the user is resuming a session
    unique_dir = os.path.join(images_dir, "organized_moderate", "unique")
    if os.path.exists(unique_dir):
        # Count only files that are in 'all_files' (to avoid counting random junk)
        # or just count pngs.
        curated_files = glob.glob(os.path.join(unique_dir, "*.png"))
        data['already_curated_count'] = len(curated_files)
    else:
        data['already_curated_count'] = 0
        
    return render_template('curate_v2.html', video_id=video_id, data=data)

@app.route('/edit_v2/<video_id>')
def edit_v2(video_id):
    """
    New split-screen editing interface (Text Left, Image Grid Right).
    """
    groups = prepare_grouped_data(video_id)
    return render_template('edit_v2.html', video_id=video_id, groups=groups)

@app.route('/edit/<video_id>')
def edit_transcript(video_id):
    slides = prepare_slides_data(video_id)
    return render_template('edit.html', video_id=video_id, slides=slides)

@app.route('/generate', methods=['POST'])
def generate_pdf():
    data = request.json
    video_id = data.get('video_id')
    slides = data.get('slides')
    
    if not video_id or not slides:
        return jsonify({'error': 'Missing data'}), 400
        
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    
    # Reconstruct absolute paths for images because frontend only sends filename
    # Actually, we passed 'image' (filename) to frontend.
    # We need to find where they are.
    # They should be in organized_moderate/unique or images/
    
    # Let's check where they are based on prepare_slides_data logic
    images_folder = os.path.join(output_folder, "images", "organized_moderate", "unique")
    if not os.path.exists(images_folder):
        images_folder = os.path.join(output_folder, "images")
        
    # Update slides with full paths
    for slide in slides:
        slide['image_path'] = os.path.join(images_folder, slide['image'])
        
    # Generate PDF
    pdf_path = os.path.join(output_folder, f"{video_id}_final.pdf")
    pdf_result = create_pdf_from_data(slides, pdf_path)
    
    return jsonify({'status': 'success', 'pdf': pdf_path})

@app.route('/save_grouped_data', methods=['POST'])
def save_grouped_data():
    """
    Handle checking out from V2 Edit Interface.
    Flattens groups back into slides and generates documents.
    """
    try:
        data = request.json
        video_id = data.get('video_id')
        groups = data.get('groups')
        
        if not video_id or not groups:
            return jsonify({'error': 'Missing data'}), 400
            
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
        
        # Determine strict deduplication or standard
        # Actually we just use the images provided by frontend (which respect removals)
        
        # Flatten groups into linear slides
        final_slides = []
        
        for g in groups:
            # We want to associate the group text with the FIRST image of the group
            # Or distribute it? For now, let's put it on the first visible image.
            
            group_text = g.get('text', '').strip()
            first_image_found = False
            
            for img in g.get('images', []):
                if img.get('removed'):
                    continue
                    
                # Reconstruct full path
                # img['url'] is like /image/video_id/path/to/file.png
                # We need the absolute path.
                # Actually we can just use the 'original_path' if we passed it back
                # Or reconstruct from filename if strictly standard structure.
                
                # In prepare_grouped_data we passed 'original_path'.
                if 'original_path' in img:
                    abs_path = img['original_path']
                else:
                    # Fallback (unsafe if path structure changed)
                    fname = os.path.basename(img['url'])
                    abs_path = os.path.join(output_folder, "images", "organized_moderate", "unique", fname)
                    if not os.path.exists(abs_path): # Check standard folder
                         abs_path = os.path.join(output_folder, "images", fname)
                
                slide = {
                    'image_path': abs_path,
                    'image': os.path.basename(abs_path),
                    'timestamp': img.get('timestamp', '00:00:00'),
                    'text': group_text if not first_image_found else "" # Only attach text to first image of group
                }
                
                final_slides.append(slide)
                if not first_image_found:
                    first_image_found = True
                    
        if not final_slides:
            return jsonify({'error': "No images selected!"}), 400

        # Generate PDF
        pdf_path = os.path.join(output_folder, f"{video_id}_final.pdf")
        
        # Import dynamically
        from pdf_generator import create_pdf_from_data, create_docx_from_data
        create_pdf_from_data(final_slides, pdf_path)
        
        # Generate DOCX
        docx_path = os.path.join(output_folder, f"{video_id}_final.docx")
        try:
             create_docx_from_data(final_slides, docx_path)
        except Exception as e:
             print(f"DOCX Generation failed: {e}")
             docx_path = None
        
        return jsonify({
            'status': 'success', 
            'pdf_path': pdf_path,
            'docx_path': docx_path
        })
        
    except Exception as e:
        print(f"Error in save_grouped_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/open_folder/<video_id>')
def open_folder(video_id):
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    # Ensure absolute path for explorer
    output_folder = os.path.abspath(output_folder)
    
    if os.path.exists(output_folder):
        try:
            # Force new window with explorer /n
            import subprocess
            # Try converting to backslashes explicitly just in case
            output_folder_fixed = output_folder.replace('/', '\\')
            # Use string command with quotes to handle spaces correctly
            subprocess.Popen(f'explorer /n, "{output_folder_fixed}"')
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Folder not found'}), 404

@app.route('/cleanup/<video_id>', methods=['POST'])
def cleanup_files(video_id):
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_dir = os.path.join(output_folder, "images")
    
    if not os.path.exists(output_folder):
        return jsonify({'error': 'Folder not found'}), 404
        
    files_to_delete = []
    
    # 1. Raw images in images/ root
    # files_to_delete.extend(glob.glob(os.path.join(images_dir, "*.png")))
    # User requested to KEEP raw images for now.
    
    # 2. Duplicates
    dupes_dir = os.path.join(images_dir, "organized_moderate", "duplicates")
    if os.path.exists(dupes_dir):
        files_to_delete.extend(glob.glob(os.path.join(dupes_dir, "*")))
        
    # 3. Blanks
    blanks_dir = os.path.join(images_dir, "organized_moderate", "blanks")
    if os.path.exists(blanks_dir):
        files_to_delete.extend(glob.glob(os.path.join(blanks_dir, "*")))
        
    # 4. Cropped for comparison folder
    cropped_dir = os.path.join(images_dir, "cropped_for_comparison")
    if os.path.exists(cropped_dir):
        # We handle directories differently, but let's just list all files in it for the count
        # or just remove the tree and count it as 1 "item" or try to count files.
        # Let's count files for stats then remove tree.
        cropped_files = glob.glob(os.path.join(cropped_dir, "*"))
        files_to_delete.extend(cropped_files)
        # We also need to remove the directory itself later
        
    # 5. Report file
    report_file = os.path.join(images_dir, "duplicates_report_combined.html")
    if os.path.exists(report_file):
        files_to_delete.append(report_file)
        
    deleted_count = 0
    reclaimed_bytes = 0
    
    for f in files_to_delete:
        try:
            if os.path.isfile(f):
                size = os.path.getsize(f)
                os.remove(f)
                reclaimed_bytes += size
                deleted_count += 1
        except Exception as e:
            print(f"Error deleting {f}: {e}")
            
    # Remove the cropped directory if it exists and is empty (or just try rmdir)
    if os.path.exists(cropped_dir):
        try:
            os.rmdir(cropped_dir) # Should be empty now
        except:
            import shutil
            shutil.rmtree(cropped_dir, ignore_errors=True)
            
    # Format size
    def format_size(size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"
        
    return jsonify({
        'status': 'success',
        'deleted_count': deleted_count,
        'reclaimed_space': format_size(reclaimed_bytes)
    })

@app.route('/delete_slide', methods=['POST'])
def delete_slide():
    data = request.json
    video_id = data.get('video_id')
    image_filename = data.get('image')
    
    if not video_id or not image_filename:
        return jsonify({'error': 'Missing data'}), 400
        
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    
    # Try to find the image in likely locations
    possible_paths = [
        os.path.join(output_folder, "images", "organized_moderate", "unique", image_filename),
        os.path.join(output_folder, "images", image_filename)
    ]
    
    deleted = False
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Option 1: Delete permanently
                # os.remove(path)
                
                # Option 2: Move to a "trash" folder (safer)
                trash_dir = os.path.join(output_folder, "images", "trash")
                os.makedirs(trash_dir, exist_ok=True)
                shutil.move(path, os.path.join(trash_dir, image_filename))
                
                deleted = True
                print(f"Moved slide to trash: {path}")
                break
            except Exception as e:
                print(f"Error deleting slide {path}: {e}")
                return jsonify({'error': str(e)}), 500
                
    if deleted:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/re_dedupe', methods=['POST'])
def re_dedupe():
    data = request.json
    video_id = data.get('video_id')
    crop_box = data.get('crop_box') # [x, y, width, height]
    
    if not video_id or not crop_box:
        return jsonify({'error': 'Missing data'}), 400
        
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_dir = os.path.join(output_folder, "images")
    raw_dir = os.path.join(images_dir, "raw")
    
    if not os.path.exists(images_dir):
        return jsonify({'error': 'Images folder not found'}), 404
        
    try:
        # 1. Backup current images to raw if raw doesn't exist (safety net)
        # If raw exists, it means we have the original original.
        # If raw doesn't exist, this is the first crop, so current images are original.
        if not os.path.exists(raw_dir):
            print("Creating backup of original images in raw/...")
            os.makedirs(raw_dir, exist_ok=True)
            current_images = glob.glob(os.path.join(images_dir, "*.png"))
            for img in current_images:
                shutil.copy2(img, os.path.join(raw_dir, os.path.basename(img)))
                
        # 2. Apply Crop IN-PLACE to current images
        # User requested iterative cropping on current images.
        from PIL import Image
        x, y, w, h = crop_box
        # Ensure integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        print(f"Applying crop to current images: x={x}, y={y}, w={w}, h={h}")
        
        images_to_process = glob.glob(os.path.join(images_dir, "*.png"))
        for img_path in images_to_process:
            try:
                with Image.open(img_path) as img:
                    # Crop
                    # Box is (left, upper, right, lower)
                    cropped_img = img.crop((x, y, x + w, y + h))
                    cropped_img.save(img_path)
            except Exception as e:
                print(f"Error cropping {img_path}: {e}")
                
        # 3. Run Deduplication (Optional)
        skip_dedupe = data.get('skip_dedupe', False)
        
        if not skip_dedupe:
            import subprocess
            print("Running deduplication on cropped images...")
            
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "scripts", "image_dedup.py"),
                images_dir,
                "--mode", "compare-all",
                "--sequential",
                "--crop-method", "content_aware", 
                "--crop-margin", "0.10",
                "--skip-blanks"
            ]
            
            subprocess.run(cmd, check=True)
        else:
            print("Skipping deduplication as requested.")
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        print(f"Re-dedupe failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_crop_config', methods=['POST'])
def save_crop_config():
    """
    Saves a global crop configuration (ratios) to be used by the background process.
    """
    try:
        data = request.json
        ratios = data.get('ratios') # [left, top, width, height] as ratios 0.0-1.0
        
        if not ratios or len(ratios) != 4:
            return jsonify({'error': 'Invalid crop data'}), 400
            
        config_path = os.path.join(app.config['OUTPUT_FOLDER'], 'global_crop_config.json')
        
        # We can also save to project root if we want it to persist across sessions/restarts more reliably
        # For now, saving to OUTPUT_FOLDER root (which seems to conform to the user's workspace structure)
        # Actually user workspace is likely c:\Users\vinay\video2pdf
        # app.config['OUTPUT_FOLDER'] is usually user_workspace/output or similar.
        # Let's save to the main workspace root so it applies to ALL new videos.
        workspace_root = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(workspace_root, 'global_crop_config.json')
        
        with open(config_path, 'w') as f:
            json.dump({'ratios': ratios}, f)
            
        return jsonify({'status': 'success', 'path': config_path})
    except Exception as e:
        print(f"Error saving crop config: {e}")
        return jsonify({'error': str(e)}), 500

def apply_global_crop_to_image(img_path, ratios):
    """
    Helper to apply crop ratios to an image file.
    """
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size
            
            # ratios: [left, top, width, height]
            rx, ry, rw, rh = ratios
            print(f"DEBUG CROP RATIOS: rx={rx:.4f}, ry={ry:.4f}, rw={rw:.4f}, rh={rh:.4f}")
            
            x = int(rx * w)
            y = int(ry * h)
            cw = int(rw * w)
            ch = int(rh * h)
            print(f"DEBUG CROP PIXELS: x={x}, y={y}, w={cw}, h={ch} (Image: {w}x{h})")
            
            # Bounds check
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            cw = max(1, min(cw, w - x))
            ch = max(1, min(ch, h - y))
            
            cropped = img.crop((x, y, x+cw, y+ch))
            cropped.save(img_path)
            return True
    except Exception as e:
        print(f"Failed to auto-crop {img_path}: {e}")
        return False
        
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, use_reloader=False, port=5000)
