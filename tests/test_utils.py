import sys
import os
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import is_youtube_url, sanitize_filename, parse_image_timestamp

class TestUtils(unittest.TestCase):
    def test_is_youtube_url(self):
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_youtube_url("https://youtu.be/dQw4w9WgXcQ"))
        self.assertFalse(is_youtube_url("https://google.com"))

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("test: file?.mp4"), "test_ file_.mp4")

    def test_parse_image_timestamp(self):
        self.assertEqual(parse_image_timestamp("001_1.5.png"), "00:01:30")
        self.assertEqual(parse_image_timestamp("001_1.5.png", return_minutes=True), 1.5)

if __name__ == '__main__':
    unittest.main()
