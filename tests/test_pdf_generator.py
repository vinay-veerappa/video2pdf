import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_generator import create_pdf_from_data, create_docx_from_data

class TestPdfGenerator(unittest.TestCase):
    @patch('pdf_generator.SimpleDocTemplate')
    @patch('pdf_generator.Image')
    @patch('PIL.Image.open')
    def test_create_pdf_from_data(self, mock_pil_open, mock_rl_image, mock_doc_template):
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.width = 100
        mock_img.height = 100
        mock_pil_open.return_value.__enter__.return_value = mock_img
        
        slides_data = [
            {'image_path': 'img1.png', 'text': 'Slide 1', 'timestamp': '00:01:00'}
        ]
        
        output_path = 'output.pdf'
        result = create_pdf_from_data(slides_data, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(mock_doc_template.called)
        # Verify build was called
        mock_doc_template.return_value.build.assert_called_once()

    @patch('pdf_generator.Document')
    @patch('PIL.Image.open')
    def test_create_docx_from_data(self, mock_pil_open, mock_document):
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.width = 100
        mock_img.height = 100
        mock_pil_open.return_value.__enter__.return_value = mock_img
        
        slides_data = [
            {'image_path': 'img1.png', 'text': 'Slide 1', 'timestamp': '00:01:00'}
        ]
        
        output_path = 'output.docx'
        result = create_docx_from_data(slides_data, output_path)
        
        self.assertEqual(result, output_path)
        self.assertTrue(mock_document.called)
        # Verify save was called
        mock_document.return_value.save.assert_called_with(output_path)

if __name__ == '__main__':
    unittest.main()
