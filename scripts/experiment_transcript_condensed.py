import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ollama_utils import OllamaClient

def clean_transcript_llm(input_file, output_file, model="gemini-3-flash-preview"):
    client = OllamaClient()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Cleaning transcript using {model}...")
    
    # Simple chunking logic (Ollama has context limits, though llama3 is 8k)
    # For a 76kb file (approx 15k-20k words), we should chunk.
    chunk_size = 4000 # characters
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    
    cleaned_content = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...")
        prompt = f"""
Role: You are an expert Technical Editor and Market Analyst. Your goal is Lossless Condensation of a trading transcript into a high-fidelity descriptive  study guide.

Objective: Rewrite the transcript while preserving every specific numerical value, "If/Then" logic chain, and statistical correlation. You must not simplify technical jargon or omit specific price levels, as these are critical for the student's data-set entry.

STRICT RULES:
NO INFORMATION LOSS: Do not summarize or generalize. If a mentor describes a specific sequence (e.g., "rejected the 09:00 mid, breached 10 basis points, and failed the 3-hour line"), every one of those specific anchors must remain in the final text.
NUMERICAL FIDELITY: Retain all percentages (e.g., 73% Asia probability), basis points, risk-to-reward ratios, and specific time-of-day references (e.g., 09:30, 13:00, 15:00).
PRESERVE LOGIC CHAINS: Every "If/Then" statement must be preserved in full. If the mentor says, "If price breaches X, then Y becomes the target," do not shorten this to "Y is the target."
CHRONOLOGICAL DATA FLOW: Maintain the hierarchy of analysis:
HTF Context: (Monthly/Quarterly/Weekly) 
Session Probabilities: (Asia/London/Pre-market) 
Live Execution: (09:30 Open through PM session) 
Candle Science: (Projections for the following day) 

Remove all filler words (um, uh, you know) and off-topic banter/jokes, unnecessary pauses and irrelevant conversations 
consolidate related text into a single paragraph.

STRICT TEXT REPLACEMENTS:
"MAE" -> "MAE"
"MFE" -> "MFE"
"FVG" -> "Fair Value Gap (FVG)"
"MA" -> "MAE"
"MF" -> "MFE"
"Dogee" -> "Doji"
"braker" -> "Breaker"
"DMP" -> "DNP" (Directional Net Price)
"MAMF" -> "MAE MFE"
FORMATTING: * Use timestamps [HH:MM:SS] to start paragraphs. 
NO METALANGUAGE: Do NOT include "Here are the notes", "Glossary:", "Summary:", or introductory/concluding remarks.
OUTPUT ONLY THE CLEANED PARAGRAPHS.

TRANSCRIPT SEGMENT:
{chunk}

CLEANED NOTES:
"""
        try:
            res = client.generate(model, prompt)
            cleaned_chunk = res.get("response", "").strip()
            # Post-processing to remove common model verbalization if it leaks
            lines = cleaned_chunk.split('\n')
            filtered_lines = [line for line in lines if not line.strip().startswith(("Here is", "Sure", "These are", "**TERMINOLOGY", "Glossary"))]
            cleaned_chunk = "\n".join(filtered_lines)
            cleaned_content.append(cleaned_chunk)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            cleaned_content.append(chunk) # Fallback to original

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(cleaned_content))
    
    print(f"Cleaned transcript saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean transcript using Ollama (Condensed)")
    parser.add_argument("input", help="Input transcript file")
    parser.add_argument("output", help="Output cleaned transcript file")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    
    args = parser.parse_args()
    clean_transcript_llm(args.input, args.output, args.model)
