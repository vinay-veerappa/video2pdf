
from youtube_transcript_api import YouTubeTranscriptApi
import sys

def test_api(video_id):
    try:
        print(f"Attempting to fetch transcript for: {video_id}")
        # Match transcript.py structure
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        
        # Prefer manual English
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
            print("Found manual English transcript.")
        except:
            transcript = transcript_list.find_generated_transcript(['en'])
            print("Found auto-generated English transcript.")
            
        data = transcript.fetch()
        print(f"SUCCESS: Retrieved {len(data)} fragments.")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test with a known TCM video
    test_api("JwUPH5XscMk")
