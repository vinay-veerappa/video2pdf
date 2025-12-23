import os
import sys
import argparse
import base64
import requests
import json
import re
from PIL import Image
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def encode_image(image_path, resize_dim=512):
    """Resize and encode image to base64 to speed up local inference."""
    try:
        img = Image.open(image_path)
        # Resize to max dimension to reduce token count/processing time
        if max(img.size) > resize_dim:
            ratio = resize_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def get_crop_coordinates_local(image_path, model="qwen3-vl:latest"):
    """
    Get crop coordinates using a simplified text prompt to avoid JSON strictness issues.
    """
    print(f"Requesting crop for {os.path.basename(image_path)} using {model}...")
    img_b64 = encode_image(image_path)
    if not img_b64:
        return None

    # Simplified prompt that asks for a list, not JSON
    prompt = """
    Look at this trading chart. I need to crop it to keep only the main chart area.
    Identify the boundaries (0-100 scale) to remove:
    1. Top toolbar
    2. Left toolbar
    3. Right sidebar/watchlist
    4. Bottom date axis (keep if relevant, otherwise crop)

    Provide the crop box in this EXACT format:
    ymin: [number]
    xmin: [number]
    ymax: [number]
    xmax: [number]

    Do not explain. Just give the numbers.
    """

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result.get('response', '').strip()
        print("\n--- RAW MODEL OUTPUT ---")
        print(text)
        print("------------------------")
        
        return parse_coordinates(text)

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def parse_coordinates(text):
    """Parse the ymin/xmin/ymax/xmax from text response."""
    coords = {}
    try:
        # Regex to find "key: value" or "key=value"
        patterns = {
            'ymin': r'ymin[:=]\s*(\d+)',
            'xmin': r'xmin[:=]\s*(\d+)',
            'ymax': r'ymax[:=]\s*(\d+)',
            'xmax': r'xmax[:=]\s*(\d+)'
        }
        
        for key, pat in patterns.items():
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                # Normalize to 0-1 if it looks like 0-100
                if val > 1:
                    val = val / 100.0
                coords[key] = val
        
        if len(coords) == 4:
            return coords
        else:
            print(f"Incomplete coordinates found: {coords}")
            return None
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--model", default="qwen3-vl:latest", help="Ollama Vision Model")
    args = parser.parse_args()
    
    crop = get_crop_coordinates_local(args.image, args.model)
    if crop:
        print("\nParsed Crop Box:")
        print(json.dumps(crop, indent=2))
        
        # Verify strictness (e.g. if crop is too aggressive or too minimal)
        width = crop['xmax'] - crop['xmin']
        height = crop['ymax'] - crop['ymin']
        print(f"Crop Area: {width*100:.1f}% width, {height*100:.1f}% height")
