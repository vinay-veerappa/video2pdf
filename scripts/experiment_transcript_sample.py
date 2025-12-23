import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ollama_utils import OllamaClient

def clean_transcript_sample(input_file, output_file, model="llama3", limit=5000):
    client = OllamaClient()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read(limit)

    print(f"Cleaning SAMPLE of transcript using {model}...")
    
    prompt = f"""
You are an expert editor. Clean up the following segment of a trading bootcamp transcript.

RULES:
1. Remove verbal fillers (um, uh, you know).
2. Fix grammar and spelling.
3. Preserve all technical terms (Liquidity, Order Block, etc.).
4. Stay detailed, do not summarize.

TRANSCRIPT SEGMENT:
{content}

CLEANED VERSION:
"""
    try:
        res = client.generate(model, prompt)
        cleaned_content = res.get("response", "").strip()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("--- ORIGINAL SAMPLE ---\n")
            f.write(content)
            f.write("\n\n--- CLEANED VERSION ---\n")
            f.write(cleaned_content)
        
        print(f"Sample cleaning complete. Results in: {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python experiment_transcript_sample.py <input> <output>")
        sys.exit(1)
    clean_transcript_sample(sys.argv[1], sys.argv[2])
