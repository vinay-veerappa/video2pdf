import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ollama_utils import OllamaClient

def generate_crisp_notes(input_file, output_file, model="llama3"):
    client = OllamaClient()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Generating CRISP NOTES using {model}...")
    
    chunk_size = 5000 
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    refined_content = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        prompt = f"""
You are an expert technical writer. Convert the following transcript segment into CRISP TECHNICAL NOTES.

STRICT INSTRUCTIONS:
1. REMOVE ALL BANTER, small talk, jokes, and introductory pleasantries.
2. EXTRACT the core technical points, rules, and market observations.
3. PRESERVE THE TIMESTAMPS [HH:MM:SS] for each significant point.
4. If a block of text is just conversation/banter, discard it.
5. Use bullet points for clarify.
6. Preserve all technical terms (MAE, MFE,Liquidity, Order Block, RSI, FVG, etc.).

TRANSCRIPT SEGMENT:
{chunk}

CRISP TECHNICAL NOTES:
"""
        try:
            res = client.generate(model, prompt)
            notes = res.get("response", "").strip()
            refined_content.append(notes)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            refined_content.append(f"[Error processing chunk {i+1}]")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(refined_content))
    
    print(f"Crisp notes saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate crisp notes from transcript")
    parser.add_argument("input", help="Input transcript file")
    parser.add_argument("output", help="Output notes file")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    
    args = parser.parse_args()
    generate_crisp_notes(args.input, args.output, args.model)
