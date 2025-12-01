import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import image_dedup

class TestImageDedup(unittest.TestCase):
    def test_compute_histogram_similarity(self):
        # Create two identical images
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='red')
        
        sim = image_dedup.compute_histogram_similarity(img1, img2)
        self.assertAlmostEqual(sim, 1.0, places=4)
        
        # Create different images
        img3 = Image.new('RGB', (100, 100), color='blue')
        sim_diff = image_dedup.compute_histogram_similarity(img1, img3)
        self.assertLess(sim_diff, 1.0)

    @patch('scripts.image_dedup.get_processed_image')
    @patch('scripts.image_dedup.compute_histogram_similarity')
    def test_compare_images_smart(self, mock_hist, mock_get_img):
        # Mock dependencies
        mock_get_img.return_value = Image.new('RGB', (10, 10))
        
        img1_data = {'hashes': {'phash': 10, 'dhash': 20}, 'entropy': 5.0}
        img2_data = {'hashes': {'phash': 12, 'dhash': 22}, 'entropy': 5.0} # diff 2
        
        # Case 1: Duplicate (High Hist, Low Dist)
        mock_hist.return_value = 0.95
        result = image_dedup.compare_images_smart(img1_data, img2_data, threshold=10)
        self.assertTrue(result['is_duplicate'])
        
        # Case 2: Non-Duplicate (Low Hist, High Dist)
        img3_data = {'hashes': {'phash': 50, 'dhash': 60}, 'entropy': 5.0} # diff 40
        mock_hist.return_value = 0.5 # Low similarity
        result_diff = image_dedup.compare_images_smart(img1_data, img3_data, threshold=10)
        self.assertFalse(result_diff['is_duplicate'])

    @patch('pathlib.Path.mkdir')
    @patch('scripts.image_dedup.get_image_files')
    @patch('multiprocessing.Pool')
    def test_find_duplicates_with_smart_crop(self, mock_pool, mock_get_files, mock_mkdir):
        # Mock file list
        mock_get_files.return_value = [MagicMock(), MagicMock()]
        
        # Mock pool map results
        # Return dummy data for 2 images
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        
        mock_pool_instance.map.return_value = [
            {
                'path': 'img1.png', 'name': 'img1.png', 
                'hashes': {'phash': 10}, 'entropy': 5.0,
                'crop_path': None, 'vis_path': None, 'debug_path': None
            },
            {
                'path': 'img2.png', 'name': 'img2.png', 
                'hashes': {'phash': 10}, 'entropy': 5.0, # Identical hash
                'crop_path': None, 'vis_path': None, 'debug_path': None
            }
        ]
        
        # Mock compare_images_smart inside the function? 
        # Actually find_duplicates calls compare_images_smart directly.
        # We can patch it or rely on the real one.
        # Let's patch it to simplify
        with patch('scripts.image_dedup.compare_images_smart') as mock_compare:
            mock_compare.return_value = {
                'is_duplicate': True, 'score': 0, 'avg_score': 0, 
                'histogram_similarity': 1.0, 'distances': {}, 
                'entropy': 5.0, 'adjusted_threshold': 10, 'reason': 'Test'
            }
            
            duplicates, _ = image_dedup.find_duplicates_with_smart_crop('dummy_dir', sequential=True)
            
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(duplicates[0]['image1'], 'img1.png')
            self.assertEqual(duplicates[0]['image2'], 'img2.png')

if __name__ == '__main__':
    unittest.main()
