import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor import detect_unique_screenshots

class TestExtractor(unittest.TestCase):
    @patch('subprocess.run')
    @patch('shutil.rmtree')
    @patch('os.makedirs')
    @patch('glob.glob')
    @patch('shutil.move')
    @patch('os.path.exists')
    def test_detect_unique_screenshots(self, mock_exists, mock_move, mock_glob, mock_makedirs, mock_rmtree, mock_subprocess):
        # Setup mocks
        mock_exists.return_value = False # Output folder doesn't exist initially
        mock_glob.return_value = ['temp/00001.png', 'temp/00002.png']
        
        # Mock subprocess result for ffmpeg
        mock_result = MagicMock()
        mock_result.stderr = "n: 0 pts_time:0.0 pos: 0\nn: 1 pts_time:1.5 pos: 1"
        mock_subprocess.return_value = mock_result
        
        output_folder = 'output/images'
        
        count = detect_unique_screenshots('video.mp4', output_folder, frame_rate=1, min_percent=0, max_percent=100)
        
        self.assertEqual(count, 2)
        
        # Verify ffmpeg command
        args, _ = mock_subprocess.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], 'ffmpeg')
        self.assertIn('-skip_frame', cmd)
        self.assertIn('nokey', cmd)
        
        # Verify moves
        self.assertEqual(mock_move.call_count, 2)

if __name__ == '__main__':
    unittest.main()
