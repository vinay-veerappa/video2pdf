def prepare_grouped_data(video_id, interval_minutes=5):
    """
    Groups curated images and transcript by time intervals (e.g., 5 min).
    """
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_folder = os.path.join(output_folder, "images", "organized_moderate", "unique")
    
    # Fallback to main images folder if curated is empty
    if not os.path.exists(images_folder) or not any(fname.endswith('.png') for fname in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, fname))):
        images_folder = os.path.join(output_folder, "images")
        
    # Reuse transcript finding logic (simplifying for brevity, better to refactor helper)
    transcript_file = None
    transcript_dir = os.path.join(output_folder, "transcripts")
    
    priority_list = [
        os.path.join(transcript_dir, f"{video_id}_ollama_condensed.txt"),
        os.path.join(transcript_dir, f"{video_id}_ollama_clean.txt"),
        os.path.join(transcript_dir, f"{video_id}_cleaned.txt"),
        os.path.join(transcript_dir, "transcript_cleaned.txt"), 
        os.path.join(transcript_dir, f"{video_id}.txt"),
        os.path.join(transcript_dir, "transcript.txt"),
        os.path.join(output_folder, f"{video_id}.txt"),
        os.path.join(output_folder, "transcript.txt")
    ]
    
    for p in priority_list:
        if os.path.exists(p):
            transcript_file = p
            break
            
    transcript_name = os.path.basename(transcript_file) if transcript_file else "None"
    
    # Parse transcript
    transcript_entries = []
    if transcript_file:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s*(.+)', line)
                    if match:
                        transcript_entries.append((match.group(1), match.group(2)))
        except: pass
        
    # Get images
    image_files = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    
    # Helper to parse timestamp
    def get_time(fname):
        try:
            parts = os.path.basename(fname).split('_')
            if len(parts) > 1:
                timestamp_str = parts[1].rsplit('.', 1)[0]
                return float(timestamp_str)
            return float(os.path.basename(fname).rsplit('.', 1)[0])
        except:
            return 0.0

    # Sort files
    sorted_files = sorted(image_files, key=get_time)
    
    # Determine Minutes vs Seconds for images
    time_multiplier = 1.0
    parsed_transcript = []
    for ts, text in transcript_entries:
        parts = ts.split(':')
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        parsed_transcript.append({'start': seconds, 'text': text})
        
    if sorted_files and parsed_transcript:
        max_img_time = get_time(sorted_files[-1])
        max_trans_time = parsed_transcript[-1]['start']
        if abs(max_img_time * 60 - max_trans_time) < abs(max_img_time - max_trans_time):
            time_multiplier = 60.0

    # Grouping Logic
    interval_seconds = interval_minutes * 60
    groups = {}
    
    # 1. Initialize Groups based on Transcript Duration (roughly)
    # Or just create them dynamically.
    
    # Assign Images to Groups
    processed_images = []
    
    # URL Helper
    images_root = os.path.join(output_folder, "images")
    rel_path = os.path.relpath(images_folder, images_root).replace("\\", "/")
    if rel_path == ".": rel_path = ""
    
    for img_path in sorted_files:
        filename = os.path.basename(img_path)
        t = get_time(img_path) * time_multiplier
        group_idx = int(t // interval_seconds)
        
        if group_idx not in groups:
            groups[group_idx] = {'start': group_idx * interval_seconds, 'images': [], 'text_parts': []}
            
        img_url = f"/image/{video_id}/{rel_path}/{filename}" if rel_path else f"/image/{video_id}/{filename}"
        
        groups[group_idx]['images'].append({
            'filename': filename,
            'url': img_url,
            'timestamp': str(datetime.timedelta(seconds=int(t)))
        })

    # Assign Transcript to Groups
    for entry in parsed_transcript:
        t = entry['start']
        group_idx = int(t // interval_seconds)
        if group_idx not in groups:
             groups[group_idx] = {'start': group_idx * interval_seconds, 'images': [], 'text_parts': []}
        groups[group_idx]['text_parts'].append(f"[{entry['start']}] {entry['text']}") # Keep simpler format or reformat
        
        # Actually, let's just append the text.
        # We might want to strip the raw timestamp for the textarea if we want "clean" text, 
        # but keeping it allows the user to see flow. The prompt asked for "text added below".
        # Let's keep the raw text mostly.
    
    # Format for Frontend
    final_groups = []
    sorted_indices = sorted(groups.keys())
    
    for idx in sorted_indices:
        g = groups[idx]
        start_time = str(datetime.timedelta(seconds=g['start']))
        end_time = str(datetime.timedelta(seconds=g['start'] + interval_seconds))
        
        # Combine text parts
        # If we have [seconds], maybe convert back to HH:MM:SS for display
        combined_text = []
        for entry in parsed_transcript:
             if g['start'] <= entry['start'] < g['start'] + interval_seconds:
                  formatted_ts = str(datetime.timedelta(seconds=entry['start']))
                  combined_text.append(f"[{formatted_ts}] {entry['text']}")
        
        final_groups.append({
            'index': idx,
            'time_range': f"{start_time} - {end_time}",
            'images': g['images'],
            'text': "\n\n".join(combined_text)
        })
        
    return final_groups, transcript_name

@app.route('/edit_v2/<path:video_id>')
def edit_v2(video_id):
    groups, transcript_name = prepare_grouped_data(video_id)
    return render_template('edit_v2.html', video_id=video_id, groups=groups, transcript_name=transcript_name)
    
