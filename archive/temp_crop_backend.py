
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
            
            x = int(rx * w)
            y = int(ry * h)
            cw = int(rw * w)
            ch = int(rh * h)
            
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
