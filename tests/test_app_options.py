import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module # Import the module to patch functions in it

class TestAppOptions(unittest.TestCase):
    def setUp(self):
        self.app = app_module.app.test_client()
        self.app.testing = True
        # Mock the JOBS dictionary in app to avoid key errors
        app_module.JOBS['job_1'] = {'status': 'processing', 'log': [], 'message': 'Starting...', 'percent': 0}

    @patch.object(app_module, 'process_video_workflow')
    @patch('subprocess.run')
    def test_skip_download(self, mock_subprocess, mock_workflow):
        # Setup mock return
        mock_workflow.return_value = {'images_folder': 'test_images', 'video_name': 'test_video'}
        
        # Call function
        app_module.run_processing_task('job_1', 'http://youtube.com/watch?v=123', skip_download=True)
        
        # Verify
        mock_workflow.assert_called_once()
        args, kwargs = mock_workflow.call_args
        self.assertTrue(kwargs.get('skip_download'))

    @patch.object(app_module, 'process_video_workflow')
    @patch('subprocess.run')
    def test_skip_extraction(self, mock_subprocess, mock_workflow):
        mock_workflow.return_value = {'images_folder': 'test_images', 'video_name': 'test_video'}
        
        app_module.run_processing_task('job_1', 'http://youtube.com/watch?v=123', skip_extraction=True)
        
        mock_workflow.assert_called_once()
        args, kwargs = mock_workflow.call_args
        self.assertTrue(kwargs.get('skip_extraction'))

    @patch.object(app_module, 'process_video_workflow')
    @patch('subprocess.run')
    def test_download_transcript_false(self, mock_subprocess, mock_workflow):
        mock_workflow.return_value = {'images_folder': 'test_images', 'video_name': 'test_video'}
        
        # download_transcript=False
        app_module.run_processing_task('job_1', 'http://youtube.com/watch?v=123', download_transcript=False)
        
        mock_workflow.assert_called_once()
        args, kwargs = mock_workflow.call_args
        self.assertFalse(kwargs.get('download_transcript'))

    @patch.object(app_module, 'process_video_workflow')
    @patch('subprocess.run')
    def test_download_transcript_true(self, mock_subprocess, mock_workflow):
        mock_workflow.return_value = {'images_folder': 'test_images', 'video_name': 'test_video'}
        
        # download_transcript=True
        app_module.run_processing_task('job_1', 'http://youtube.com/watch?v=123', download_transcript=True)
        
        mock_workflow.assert_called_once()
        args, kwargs = mock_workflow.call_args
        self.assertTrue(kwargs.get('download_transcript'))

    @patch.object(app_module, 'process_video_workflow')
    @patch('subprocess.run')
    @patch('app.glob.glob')
    @patch('builtins.open', new_callable=mock_open)
    @patch('app.json.dump')
    def test_skip_deduplication(self, mock_json, mock_file, mock_glob, mock_subprocess, mock_workflow):
        mock_workflow.return_value = {'images_folder': 'test_images', 'video_name': 'test_video'}
        mock_glob.return_value = ['img1.png', 'img2.png']
        
        app_module.run_processing_task('job_1', 'http://youtube.com/watch?v=123', skip_deduplication=True)
        
        # Verify subprocess (dedup script) was NOT called
        mock_subprocess.assert_not_called()
        
        # Verify dummy results were created
        mock_json.assert_called_once()
        # Check if json.dump was called with the expected structure
        args, _ = mock_json.call_args
        data = args[0]
        self.assertIn('blanks', data)
        self.assertIn('duplicates', data)
        self.assertIn('all_files', data)

    @patch.object(app_module, 'process_video_workflow')
    @patch('subprocess.run')
    def test_run_deduplication(self, mock_subprocess, mock_workflow):
        mock_workflow.return_value = {'images_folder': 'test_images', 'video_name': 'test_video'}
        
        app_module.run_processing_task('job_1', 'http://youtube.com/watch?v=123', skip_deduplication=False)
        
        # Verify subprocess WAS called
        mock_subprocess.assert_called_once()

if __name__ == '__main__':
    unittest.main()
