import unittest
from unittest.mock import patch, MagicMock
import os
import json
import sys

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, JOBS

class TestApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['OUTPUT_FOLDER'] = 'test_output'
        self.client = app.test_client()
        
    def test_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Video to PDF', response.data)

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('glob.glob')
    def test_list_videos(self, mock_glob, mock_isdir, mock_exists, mock_listdir):
        mock_exists.return_value = True
        mock_listdir.return_value = ['Project1']
        mock_isdir.return_value = True
        mock_glob.return_value = ['img1.png']
        
        response = self.client.get('/list_videos')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'Project1')
        self.assertTrue(data[0]['has_images'])

    @patch('threading.Thread')
    def test_process_video(self, mock_thread):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        response = self.client.post('/process', data={
            'url': 'http://example.com/video',
            'skip_download': 'false',
            'skip_extraction': 'false',
            'skip_deduplication': 'false'
        })
        
        if response.status_code != 200:
            print(f"Process Video Failed: {response.data}")
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('job_id', data)
        self.assertTrue(mock_thread_instance.start.called)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"all_files": ["img1.png"], "blanks": [], "duplicates": []}')
    def test_curate_existing_json(self, mock_file, mock_exists):
        mock_exists.return_value = True
        
        response = self.client.get('/curate/Project1')
        if response.status_code != 200:
            print(f"Curate Existing Failed: {response.data}")
            
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Curate Slides', response.data)

    @patch('os.path.exists')
    @patch('glob.glob')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_curate_missing_json(self, mock_file, mock_glob, mock_exists):
        # Simulate missing json but existing images
        mock_exists.side_effect = lambda p: p.endswith('.png') or 'images' in p
        mock_glob.return_value = ['path/to/img1.png']
        
        data = {
            'video_id': 'Project1',
            'keeps': ['img1.png']
        }
        
        response = self.client.post('/save_curation', 
                                  data=json.dumps(data),
                                  content_type='application/json')
                                  
        self.assertEqual(response.status_code, 200)
        self.assertTrue(mock_copy.called)

    @patch('app.create_pdf_from_data')
    @patch('app.create_docx_from_data')
    @patch('os.path.exists')
    def test_generate_documents(self, mock_exists, mock_docx, mock_pdf):
        mock_exists.return_value = True
        mock_pdf.return_value = 'output.pdf'
        mock_docx.return_value = 'output.docx'
        
        data = {
            'video_id': 'Project1',
            'slides': [{'image': 'img1.png', 'text': 'Slide 1'}]
        }
        
        response = self.client.post('/generate',
                                  data=json.dumps(data),
                                  content_type='application/json')
                                  
        self.assertEqual(response.status_code, 200)
        resp_data = json.loads(response.data)
        self.assertEqual(resp_data['pdf_path'], 'output.pdf')
        self.assertEqual(resp_data['docx_path'], 'output.docx')

if __name__ == '__main__':
    unittest.main()
