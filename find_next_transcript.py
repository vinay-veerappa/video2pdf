import os
from pathlib import Path

def find_next_transcript():
    """Find the next transcript that needs notes created."""
    base_dir = Path("C:/ICT_Videos/TCM")
    
    # Find all transcript files (excluding 2023)
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
            
        for file in sorted(files):
            if file.endswith('.txt'):
                transcript_path = Path(root) / file
                
                # Check if notes already exist
                notes_dir = transcript_path.parent / "Notes"
                notes_path = notes_dir / f"{transcript_path.stem}.md"
                
                if not notes_path.exists():
                    # Found one that needs processing
                    return transcript_path
    
    return None

if __name__ == "__main__":
    next_file = find_next_transcript()
    if next_file:
        print(f"Next transcript to process:")
        print(f"  File: {next_file.name}")
        print(f"  Path: {next_file}")
        print(f"\nTo save notes, create:")
        notes_path = next_file.parent / "Notes" / f"{next_file.stem}.md"
        print(f"  {notes_path}")
    else:
        print("All transcripts have notes!")
