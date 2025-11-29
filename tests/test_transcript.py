import sys
import os
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcript import clean_transcript_text

class TestTranscript(unittest.TestCase):
    def test_clean_transcript_basic(self):
        text = "hello world"
        cleaned = clean_transcript_text(text)
        self.assertEqual(cleaned, "Hello world")

    def test_clean_transcript_overlap(self):
        lines = [
            "This is a test sentence",
            "test sentence that overlaps"
        ]
        cleaned = clean_transcript_text(lines)
        self.assertEqual(cleaned, "This is a test sentence that overlaps")

    def test_clean_transcript_junk(self):
        lines = [
            "Um, hello.",
            "Can you hear me?",
            "This is real content."
        ]
        cleaned = clean_transcript_text(lines)
        self.assertIn("This is real content", cleaned)
        self.assertNotIn("can you hear me", cleaned.lower())

if __name__ == '__main__':
    unittest.main()
