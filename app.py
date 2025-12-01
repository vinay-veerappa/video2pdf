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

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = OUTPUT_DIR

# Global state to track progress (simple version)
# In production, use Redis or a database
JOBS = {} 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    data = request.form
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
        
    # Start processing in a background thread
    job_id = "job_" + str(len(JOBS) + 1)
    JOBS[job_id] = {'status': 'processing', 'log': []}
    
    thread = threading.Thread(target=run_processing_task, args=(job_id, url))
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def job_status(job_id):
    return jsonify(JOBS.get(job_id, {'status': 'unknown'}))

from main import process_video_workflow
from scripts import image_dedup

def run_processing_task(job_id, url):
    """
    Run the extraction and deduplication pipeline.
    """
    try:
        JOBS[job_id]['status'] = 'starting'
        JOBS[job_id]['percent'] = 0
        JOBS[job_id]['message'] = "Starting process..."
        
        def progress_callback(data):
            JOBS[job_id].update(data)
        
        # 1. Run Extraction
        result = process_video_workflow(
            url, 
            output_dir=app.config['OUTPUT_FOLDER'],
            download_transcript=True,
            optimize_images=True, # Ensure we get optimized images
            progress_callback=progress_callback
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

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
