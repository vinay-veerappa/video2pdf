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
from main import main as run_pipeline_cli # We might need to refactor main.py to be importable as a function
from main import process_video_workflow
from scripts import image_dedup
from pdf_generator import create_pdf_from_data
from utils import parse_image_timestamp, sanitize_filename
import glob
import re

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
                    'has_dedup': has_dedup
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
    
    if not url and not existing_video_id:
        return jsonify({'error': 'No URL or Existing Video provided'}), 400
        
    # Start processing in a background thread
    job_id = "job_" + str(len(JOBS) + 1)
    JOBS[job_id] = {'status': 'processing', 'log': []}
    
    thread = threading.Thread(target=run_processing_task, args=(job_id, url, existing_video_id, skip_download, skip_extraction))
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def job_status(job_id):
    return jsonify(JOBS.get(job_id, {'status': 'unknown'}))

from main import process_video_workflow
from scripts import image_dedup

from main import process_video_workflow
from scripts import image_dedup

def run_processing_task(job_id, url, existing_video_id=None, skip_download=False, skip_extraction=False):
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
                else:
                    # Maybe only images exist?
                    # If we skip extraction, we don't strictly need the video file if images are there.
                    # But process_video_workflow might need it for naming.
                    pass
            
            if not input_source:
                 # Fallback to just using the folder name if we are skipping everything up to deduplication
                 # But process_video_workflow needs a source.
                 # Let's assume the user wants to resume.
                 pass

        result = process_video_workflow(
            input_source, 
            output_dir=app.config['OUTPUT_FOLDER'],
            progress_callback=progress_callback,
            cookies=cookies_path,
            skip_download=skip_download,
            skip_extraction=skip_extraction
        )
        
        images_folder = result['images_folder']
        video_name = result['video_name']
        
        JOBS[job_id]['status'] = 'analyzing'
        
        # 2. Run Deduplication (Curator Grid Generation)
        # We use the 'compare-all' mode logic from image_dedup.py
        # But we need to call it programmatically.
        
        # We can use subprocess or import. Since we are in the same env, import is better.
        # However, image_dedup.py is designed as a script. 
        # Let's use subprocess to be safe and consistent with CLI.
        import subprocess
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "scripts", "image_dedup.py"),
            images_folder,
            "--mode", "compare-all",
            "--sequential",
            "--crop-method", "content_aware",
            "--crop-margin", "0.20"
        ]
        
        subprocess.run(cmd, check=True)
        
        JOBS[job_id]['status'] = 'ready_for_curation'
        JOBS[job_id]['video_id'] = video_name # This is the folder name
        
    except Exception as e:
        print(f"Job failed: {e}")
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)

@app.route('/curate/<video_id>')
def curate(video_id):
    # Load the JSON results
    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_id, "images")
    json_path = os.path.join(images_dir, "dedup_results.json")
    
    if not os.path.exists(json_path):
        return "Error: Deduplication results not found.", 404
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    return render_template('curate.html', video_id=video_id, data=data)

@app.route('/image/<video_id>/<filename>')
def serve_image(video_id, filename):
    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], video_id, "images")
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
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_folder = os.path.join(output_folder, "images", "organized_moderate", "unique")
    
    # If curated folder doesn't exist, fallback to main images folder (if user skipped curation)
    if not os.path.exists(images_folder):
        images_folder = os.path.join(output_folder, "images")
        
    # Find transcript
    transcript_file = None
    possible_transcripts = glob.glob(os.path.join(output_folder, "transcripts", "*.txt"))
    possible_transcripts.extend(glob.glob(os.path.join(output_folder, "*.txt")))
    
    for t in possible_transcripts:
        if "cleaned" not in t and "report" not in t and "metadata" not in t:
            transcript_file = t
            break
            
    if not transcript_file or not os.path.exists(images_folder):
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
    except:
        pass

    # Get images
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    slides = []
    
    def timestamp_to_seconds(ts):
        parts = ts.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    for img_path in image_files:
        filename = os.path.basename(img_path)
        timestamp = parse_image_timestamp(filename)
        
        text_segments = []
        if timestamp and transcript_entries:
            img_seconds = timestamp_to_seconds(timestamp)
            
            # Find text within window (e.g. -5s to +30s)
            # Or better: between this image and the next image?
            # For now, let's use the simple window logic from pdf_generator
            for ts, text in transcript_entries:
                ts_seconds = timestamp_to_seconds(ts)
                if abs(ts_seconds - img_seconds) <= 30:
                    text_segments.append(text)
        
        slides.append({
            'image': filename,
            'image_path': img_path, # Absolute path for backend
            'timestamp': timestamp,
            'text': ' '.join(text_segments)
        })
        
    return slides

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
    result = create_pdf_from_data(slides, pdf_path)
    
    if result:
        return jsonify({'status': 'success', 'path': result})
    else:
        return jsonify({'error': 'Failed to generate PDF'}), 500

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
