@app.route('/apply_crop', methods=['POST'])
def apply_crop():
    """
    Apply a manual crop to all images in the video project.
    Expects JSON: { video_id, crop_box: [x, y, w, h] }
    """
    data = request.json
    video_id = data.get('video_id')
    crop_box = data.get('crop_box') # [x, y, w, h]
    
    if not video_id or not crop_box:
        return jsonify({'error': 'Missing data'}), 400
        
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], video_id)
    images_dir = os.path.join(output_folder, "images", "organized_moderate", "unique")
    
    # Fallback if unique folder is empty
    if not os.path.exists(images_dir) or not glob.glob(os.path.join(images_dir, "*.png")):
         images_dir = os.path.join(output_folder, "images")
    
    if not os.path.exists(images_dir):
        return jsonify({'error': 'Images folder not found'}), 404
        
    try:
        from PIL import Image
        x, y, w, h = [int(v) for v in crop_box]
        
        print(f"Applying manual crop to {images_dir}: {x},{y},{w},{h}")
        
        images = glob.glob(os.path.join(images_dir, "*.png"))
        count = 0
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    # Validate crop box vs image size
                    iw, ih = img.size
                    
                    # Ensure within bounds
                    cx = max(0, min(x, iw))
                    cy = max(0, min(y, ih))
                    cw = max(1, min(w, iw - cx))
                    ch = max(1, min(h, ih - cy))
                    
                    cropped = img.crop((cx, cy, cx+cw, cy+ch))
                    cropped.save(img_path)
                    count += 1
            except Exception as e:
                print(f"Error cropping {img_path}: {e}")
                
        return jsonify({'status': 'success', 'count': count})
        
    except Exception as e:
        print(f"Crop failed: {e}")
        return jsonify({'error': str(e)}), 500
