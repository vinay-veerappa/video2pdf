import os
import sys
import time
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from gemini_utils import GeminiClient

SYSTEM_PROMPT = """You are an expert trading educator creating detailed study notes from live trading session transcripts by The Currency Merchant (TCM / Kish). These are recordings of his Patreon live calls where he teaches ICT-based concepts with his own unique framework.

Your task: Read the entire transcript carefully and produce comprehensive, detailed study notes in Markdown format that a trader can use for studying and live trading reference.

CRITICAL INSTRUCTIONS — DO NOT SKIP ANY OF THESE:
1. DO NOT lose any specific information — especially concrete numbers, price levels, time references, pip counts, or specific rules stated
2. Capture ALL "nuggets" — specific tips, tricks, or observations about what to expect from price action. These are the most valuable parts.
3. When he states a RULE or PRINCIPLE, quote it exactly or near-exactly and mark it clearly with bold or blockquote
4. Capture the REASONING behind each concept — not just "what" but "why"
5. All times he references are UTC-5 (NOT Eastern Time — he explicitly uses UTC-5 and recommends setting TradingView to Lima timezone)
6. Note when a concept appears to be introduced for the FIRST TIME vs being reinforced from earlier sessions
7. Capture any trade management details — stop placement logic, partial profit levels, position sizing comments, break-even rules
8. Note any FILTERS or CONDITIONS — when NOT to trade, what reduces risk, what increases conviction
9. If he discusses specific instruments, note which ones (NQ, ES, US30, AU, GU, Gold, DXY, etc.)
10. If there's student Q&A, capture the key questions and his answers — these often contain critical clarifications
11. If he takes or describes a specific trade, capture the full setup: entry logic, stop logic, target logic, management
12. Pay special attention to statements like "never," "always," "on principle," "rule," "remember" — these are often core rules

OUTPUT FORMAT — use this structure:

# [Date from filename] — [Session Type]
**Type:** [Live Session / Review / Weekend Review / Backtesting / Teaching] | **Bias:** [Bullish/Bearish/Neutral/Mixed] | **Instrument:** [NQ/ES/etc.]

## Session Summary
[2-4 sentences on what this session covered and the main takeaway]

## Key Concepts Taught

### [Concept Name 1]
[Detailed explanation with quotes for key rules. Include the WHY, not just the WHAT.]

### [Concept Name 2]
[Continue for each distinct concept discussed]

## Specific Price Action Observations
[Concrete observations about how price behaved — specific levels, reactions, patterns, what worked and what didn't]

## Trade Setups Discussed
[Any specific trades taken or analyzed — entry, stop, target, outcome, management decisions]

## Trading Psychology / Risk Notes
[Comments about psychology, risk management, discipline, when to sit out]

## Forward-Looking
[What he expects for tomorrow/this week, levels to watch, upcoming catalysts]

## Actionable Rules & Takeaways
[Numbered list of concrete, actionable trading rules extracted from THIS session]

---

KEY TCM TERMINOLOGY — recognize and use these correctly:
- ONS = Overnight Session (4:00 AM - 8:15 AM UTC-5)
- EO = Engineered Origin (the first open price after a delivery objective is met and price closes outside the range)
- DO = Delivery Origin (the first open price after a delivery objective has been met)
- SO = Swing Origin (the open price of the order block that initiates a price swing)
- CSD / CISD = Change in State of Delivery (the shift from buy-side to sell-side delivery or vice versa)
- OLR = Opposing Liquidity Run (absorbing liquidity opposing the order flow)
- DOL = Draw on Liquidity (the target/objective)
- FVG / FEG = Fair Value Gap (imbalance — 3-candle pattern with gap between candle 1 and 3)
- BPR = Balanced Price Range (3 passes through a range)
- MSS = Market Structure Shift (internal change within a range)
- BSL = Buy-Side Liquidity (above highs)
- SSL = Sell-Side Liquidity (below lows)
- True Day = 4:00 AM to 4:00 PM UTC-5
- Profile 1-6 = ONS daily delivery profiles
- Dealing Range = price swing that takes liquidity from both sides
- Sponsored Leg = big candle body closing outside a range showing commitment to leave
- Book Making = Accumulation → Failure Swing → OLR → CSD
- Rebalance Macro = 11:00 AM - 1:30 PM UTC-5
- Lunch/Launch Macro = 12:45 PM - 1:45 PM UTC-5
- Offset Macro = 9:45 AM - 10:00 AM UTC-5
- Settlement Check = 1:45 PM - 2:45/3:15 PM UTC-5
- Institutional levels = Big figure (XX,000), Mid-figure (XX,500), Quarter (XX,250 / XX,750)
- OpEx = Options Expiry
- 3-hour cycle = 8:00-11:00 with 36-minute blocks
- Judas swing = first fake move (typically 8:15-9:45)

Be thorough. These notes will be the trader's primary study reference."""

def process_transcript(transcript_path, output_path, client):
    """Process a single transcript file and generate notes."""
    print(f"\nProcessing: {transcript_path.name}")
    
    # Read transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read()
    
    # Check if transcript is too short (likely failed transcription)
    if len(transcript_text) < 500:
        print(f"  [SKIP] Skipping - transcript too short ({len(transcript_text)} chars)")
        return False
    
    # Create prompt
    prompt = f"{SYSTEM_PROMPT}\n\nTranscript filename: {transcript_path.stem}\n\nTranscript:\n{transcript_text}"
    
    try:
        # Generate notes using Gemini
        notes = client.generate_content(prompt)
        
        # Save notes
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notes)
        
        print(f"  [OK] Notes created: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    base_dir = Path("C:/ICT_Videos/TCM")
    
    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = GeminiClient(model_name="models/gemini-2.5-flash")
    
    # Find all transcript files (excluding 2023 and existing notes)
    transcript_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip 2023 directory
        if '2023' in root:
            continue
        # Only process transcripts folders
        if 'transcripts' not in root.lower():
            continue
        # Skip Notes folders
        if 'Notes' in root or 'notes' in root:
            continue
            
        for file in files:
            if file.endswith('.txt'):
                transcript_files.append(Path(root) / file)
    
    print(f"Found {len(transcript_files)} transcripts to process\n")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for transcript_path in sorted(transcript_files):
        # Create output path (same location, in Notes subfolder)
        notes_dir = transcript_path.parent / "Notes"
        output_path = notes_dir / f"{transcript_path.stem}.md"
        
        # Skip if notes already exist
        if output_path.exists():
            print(f"\nSkipping (exists): {transcript_path.name}")
            skipped += 1
            continue
        
        # Process transcript
        success = process_transcript(transcript_path, output_path, client)
        
        if success:
            processed += 1
        else:
            failed += 1
        
        # Rate limiting - wait between API calls
        time.sleep(3)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(transcript_files)}")

if __name__ == "__main__":
    main()
