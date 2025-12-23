import os
import sys
import argparse
import time
from pathlib import Path
from google.genai import errors

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gemini_utils import GeminiClient

def test_gemini_crop(image_path):
    client = GeminiClient()
    import time
    
    print(f"Requesting intelligent crop for {Path(image_path).name} using Gemini...")
    retries = 3
    while retries > 0:
        try:
            raw_res = client.get_crop_coordinates(image_path)
            print("\n--- GEMINI RAW RESPONSE ---")
            print(raw_res)
            print("---------------------------")
            
            # Try to parse JSON if it returned it
            if "```json" in raw_res:
                raw_res = raw_res.split("```json")[1].split("```")[0].strip()
            elif "{" in raw_res:
                import re
                match = re.search(r"\{.*\}", raw_res, re.DOTALL)
                if match:
                    raw_res = match.group(0)
            
            coords = json.loads(raw_res)
            return coords
        except errors.ClientError as e:
            if e.code == 429:
                print(f"  Rate limited. Waiting 60s... (Retries left: {retries-1})")
                time.sleep(60)
                retries -= 1
            else:
                print(f"Error extracting coordinates: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini Intelligent Cropping")
    parser.add_argument("image", help="Path to image file")
    
    args = parser.parse_args()
    coords = test_gemini_crop(args.image)
    if coords:
        print("\n--- EXTRACTED COORDINATES ---")
        print(json.dumps(coords, indent=4))
