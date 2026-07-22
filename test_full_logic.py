
import os
import sys
from transcript import download_youtube_transcript

# Add current dir to path
sys.path.append(os.getcwd())

def test_download():
    url = "https://www.youtube.com/watch?v=JwUPH5XscMk"
    output_dir = "test_transcripts"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing download_youtube_transcript for {url}...")
    res, txt_path = download_youtube_transcript(url, output_dir, output_filename="api_check_test")
    
    if txt_path and os.path.exists(txt_path):
        print(f"SUCCESS! Transcript saved to {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            print("First 200 chars:")
            print(f.read(200))
    else:
        print("FAILED to download transcript.")

if __name__ == "__main__":
    test_download()
