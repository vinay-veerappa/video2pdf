import os
import sys
import argparse
import time
from pathlib import Path
from google.genai import errors

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.gemini_utils import GeminiClient

def clean_transcript_gemini(input_file, output_file, model="gemini-2.0-flash-exp"):
    client = GeminiClient(model_name=model)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Generating CRISP NOTES using Gemini...")
    
    # Smaller chunks to avoid token-per-minute limits on free tier
    chunk_size = 8000 
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    refined_content = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        retries = 3
        while retries > 0:
            try:
                notes = client.clean_transcript(chunk)
                refined_content.append(notes)
                # Pause to avoid rate limits
                print("  Waiting 60s for next chunk...")
                time.sleep(60)
                break
            except errors.ClientError as e:
                print(f"ClientError processing chunk {i+1}: {e}")
                refined_content.append(f"[Error processing chunk {i+1}]")
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                refined_content.append(f"[Error processing chunk {i+1}]")
                break
        
        # Stop after 1 chunk for testing
        print("Stopping after 1 chunk as requested.")
        break


    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(refined_content))
    
    print(f"Crisp notes saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate crisp notes from transcript using Gemini")
    parser.add_argument("input", help="Input transcript file")
    parser.add_argument("output", help="Output notes file")
    
    args = parser.parse_args()
    clean_transcript_gemini(args.input, args.output)
