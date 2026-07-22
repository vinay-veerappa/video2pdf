
from youtube_transcript_api import YouTubeTranscriptApi
import sys

def test_api(video_id):
    try:
        print(f"Attempting to fetch transcript for: {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print("SUCCESS: Transcript retrieved!")
        # Print first few lines
        for line in transcript[:3]:
            print(f"  {line['text']}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    # Test with a known TCM video
    test_api("JwUPH5XscMk")
