import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import json

class GeminiClient:
    def __init__(self, model_name="models/gemini-2.5-flash"):
        # Load .env from project root
        root_dir = Path(__file__).parent.parent
        load_dotenv(root_dir / ".env")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate_content(self, prompt, image_path=None):
        contents = [prompt]
        if image_path:
            # Handle image loading - pass PIL Image directly
            img = Image.open(image_path)
            contents.append(img)
        
        # New API call structure
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        return response.text

    def get_crop_coordinates(self, image_path):
        prompt = """
        Identify the main central content area of this trading chart slide. 
        I want to CROP OUT:
        - The TradingView toolbars (left and top)
        - The TradingView sidebars (right)
        - The bottom status bar/time axis if it's redundant.
        - Any browser tabs or window borders.

        Return ONLY a JSON object with normalized coordinates [0, 1000] for a 2D bounding box:
        {
          "box_2d": [ymin, xmin, ymax, xmax]
        }
        Example: {"box_2d": [100, 50, 900, 950]}
        """
        
        # Configure JSON response schema
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        img = Image.open(image_path)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, img],
            config=config
        )
        return response.text

    def clean_transcript(self, text):
        prompt = f"""
        You are an expert technical editor. Convert the following transcript segment into CRISP TECHNICAL NOTES.

        STRICT INSTRUCTIONS:
        1. REMOVE ALL BANTER, small talk, jokes, and introductory pleasantries.
        2. EXTRACT the core technical points, rules, and market observations.
        3. PRESERVE THE TIMESTAMPS [HH:MM:SS] for each significant point.
        4. If a block of text is just conversation/banter, discard it.
        5. Use bullet points for clarify.
        6. Preserve all technical terms (Liquidity, Order Block, RSI, FVG, etc.).

        TRANSCRIPT SEGMENT:
        {text}

        CRISP TECHNICAL NOTES:
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

if __name__ == "__main__":
    # Test connection
    try:
        client = GeminiClient()
        print("Gemini API (google-genai) connection successful.")
    except Exception as e:
        print(f"Connection failed: {e}")
