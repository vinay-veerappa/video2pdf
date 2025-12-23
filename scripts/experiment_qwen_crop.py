import os
import sys
import base64
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ollama_utils import OllamaClient

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_qwen_crop_coordinates(image_path, model="qwen3-vl"):
    client = OllamaClient()
    
    print(f"Requesting crop coordinates for {Path(image_path).name} using {model}...")
    img_base64 = encode_image(image_path)
    
    prompt = """
Identify the main central content area of this slide, specifically the chart and the primary presentation content. 
I want to CROP OUT the TradingView menus, sidebars, top toolbars, and any other distracting UI elements.

Return ONLY a JSON object with the following normalized coordinates (0.0 to 1.0):
- x_min: Left edge of content
- y_min: Top edge of content
- x_max: Right edge of content
- y_max: Bottom edge of content

Example format: {"x_min": 0.05, "y_min": 0.1, "x_max": 0.95, "y_max": 0.85}
"""
    
    url = f"{client.base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_base64]
            }
        ],
        "stream": False
    }
    
    try:
        import requests
        response = requests.post(url, json=payload)
        response.raise_for_status()
        res = response.json()
        raw_response = res.get("message", {}).get("content", "").strip()
        coords = json.loads(raw_response)
        return coords
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        if 'raw_response' in locals():
            print(f"Raw response: {raw_response}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get crop coordinates from Qwen-VL")
    parser.add_argument("image", help="Path to image file")
    
    args = parser.parse_args()
    coords = get_qwen_crop_coordinates(args.image)
    if coords:
        print("\n--- RECOMMENDED CROP ---")
        print(json.dumps(coords, indent=4))
    else:
        print("Failed to get coordinates.")
