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

@app.route('/image/<video_id>/<path:filename>')
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
    
    # Helper to parse timestamp from filename (e.g., "001_12.5.png" -> 12.5)
    def get_time(fname):
        try:
            return float(os.path.basename(fname).split('_')[1].rsplit('.', 1)[0])
        except:
            return 0.0
            
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
        
        slides.append({
            'image': filename,
            'image_path': img_path,
            'timestamp': str(datetime.timedelta(seconds=int(current_time))),
            'text': " ".join(slide_text_parts)
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
    pdf_result = create_pdf_from_data(slides, pdf_path)
    
    # Generate DOCX
    docx_path = os.path.join(output_folder, f"{video_id}_final.docx")
    docx_result = create_docx_from_data(slides, docx_path)
    
    if pdf_result or docx_result:
        return jsonify({
            'status': 'success', 
            'pdf_path': pdf_result,
            'docx_path': docx_result
        })
    else:
        return jsonify({'error': 'Failed to generate documents'}), 500

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
        
    deleted_count = 0
    reclaimed_bytes = 0
    
    for f in files_to_delete:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            reclaimed_bytes += size
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {f}: {e}")
            
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

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
