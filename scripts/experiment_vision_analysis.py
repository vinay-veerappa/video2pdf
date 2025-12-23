import os
import sys
import base64
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ollama_utils import OllamaClient

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_slide_vision(image_path, model="qwen3-vl"):
    client = OllamaClient()
    
    print(f"Analyzing {Path(image_path).name} using {model}...")
    
    # Qwen-VL in Ollama usually expects a different format for multimodal
    # but the API allows passing images in the 'images' field of a generate/chat request.
    img_base64 = encode_image(image_path)
    
    prompt = """
Describe this slide in detail. 
Identify the main topic, any visible charts, key text points, and the overall layout.
This info will be used to differentiate this slide from others in a presentation.
Be concise but technical.
"""
    
    url = f"{client.base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_base64],
        "stream": False
    }
    
    try:
        import requests
        response = requests.post(url, json=payload)
        response.raise_for_status()
        res = response.json()
        description = res.get("response", "").strip()
        return description
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze slide using Vision Model (Ollama)")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="qwen3-vl", help="Ollama model to use")
    
    args = parser.parse_args()
    desc = analyze_slide_vision(args.image, args.model)
    print("\n--- SLIDE DESCRIPTION ---")
    print(desc)
    print("-------------------------")
