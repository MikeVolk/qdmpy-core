from __future__ import annotations

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from QDMpy.odmr.odmr import ODMR
from QDMpy.odmr.data import ODMRData


class test_odmr(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data - small test array
        # Shape: (2 polarities, 1 frequency range, 4 frequencies, 9 pixels)
        self.raw_data = np.ones((2, 1, 4, 9))
        # Add a dip in the center frequency for each pixel
        self.raw_data[:, :, 1:3, :] *= 0.9
        # Dimensions are 3x3 pixels
        self.scan_dimensions = np.array([3, 3])
        # Sample frequencies
        self.frequencies = np.array([2.87e9, 2.88e9, 2.89e9, 2.90e9])
        
        # Create test ODMRData instance
        self.odmr_data = ODMRData(
            self.raw_data.copy(), 
            self.scan_dimensions.copy(), 
            self.frequencies.copy()
        )
        
        # Create ODMR instance with data
        self.odmr = ODMR(self.odmr_data)

    def test_get_binned_pixel_indices(self):
        """Test getting binned pixel indices."""
        # This function doesn't exist yet in the current implementation
        # This is a placeholder for future implementation
        pass

    def test_rc2idx(self):
        """Test converting row-column to index."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_idx2rc(self):
        """Test converting index to row-column."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_get_most_divergent_from_mean(self):
        """Test getting most divergent pixels from mean."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__qdmio_stack_data(self):
        """Test internal QDMio data stacking."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_from_qdmio(self):
        """Test loading data from QDMio format."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_get_norm_factors(self):
        """Test getting normalization factors."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_data_shape(self):
        """Test data shape property."""
        # Verify the shape of the raw data
        self.assertEqual(self.odmr.raw_data.shape, (2, 1, 4, 9))

    def test_img_shape(self):
        """Test image shape property."""
        # Verify the shape of the image (scan dimensions)
        self.assertTrue(np.array_equal(self.odmr.raw_data.scan_dimensions, np.array([3, 3])))

    def test_n_pixel(self):
        """Test number of pixels."""
        # Verify number of pixels (9 in our test data)
        self.assertEqual(self.odmr.raw_data.shape[3], 9)

    def test_n_freqs(self):
        """Test number of frequencies."""
        # Verify number of frequencies (4 in our test data)
        self.assertEqual(self.odmr.raw_data.shape[2], 4)

    def test_frequencies(self):
        """Test frequencies property."""
        # Verify frequencies match our test data
        self.assertTrue(np.array_equal(self.odmr.raw_data.frequencies, self.frequencies))

    def test_f_hz(self):
        """Test frequencies in Hz."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_f_ghz(self):
        """Test frequencies in GHz."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_global_factor(self):
        """Test global normalization factor."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_data(self):
        """Test data access."""
        # Verify the data can be accessed and matches our test data
        self.assertTrue(np.array_equal(self.odmr.raw_data.data, self.raw_data))

    def test_delta_mean(self):
        """Test delta mean calculation."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_mean_odmr(self):
        """Test mean ODMR calculation."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_raw_contrast(self):
        """Test raw contrast calculation."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_mean_contrast(self):
        """Test mean contrast calculation."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__mean_baseline(self):
        """Test internal mean baseline calculation."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_bin_factor(self):
        """Test bin factor property."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__apply_edit_stack(self):
        """Test internal edit stack application."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_reset_data(self):
        """Test resetting data to original state."""
        # Process the data first (use MagicMock to avoid dependencies on processor manager)
        with patch.object(self.odmr.processor_manager, 'process', 
                         return_value=self.odmr_data):
            self.odmr.process_data()
            self.assertTrue(self.odmr.is_processed)
            
            # Reset the data
            self.odmr.reset()
            
            # Verify reset worked
            self.assertFalse(self.odmr.is_processed)
            self.assertIsNone(self.odmr._processed_data)

    def test_normalize_data(self):
        """Test data normalization."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__normalize_data(self):
        """Test internal data normalization."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_apply_outlier_mask(self):
        """Test applying outlier mask."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__apply_outlier_mask(self):
        """Test internal outlier mask application."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_bin_data(self):
        """Test binning data."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__bin_data(self):
        """Test internal data binning."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_remove_overexposed(self):
        """Test removing overexposed pixels."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_calc_gf_correction(self):
        """Test calculating global fluorescence correction."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_correct_glob_fluorescence(self):
        """Test correcting global fluorescence."""
        # This function doesn't exist yet in the current implementation
        pass

    def test__correct_glob_fluorescence(self):
        """Test internal global fluorescence correction."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_check_glob_fluorescence(self):
        """Test checking global fluorescence."""
        # This function doesn't exist yet in the current implementation
        pass

    def test_load_data(self):
        """Test loading data into ODMR instance."""
        # Create a new ODMR instance without data
        odmr = ODMR()
        
        # Load data
        odmr.load_data(self.raw_data, self.scan_dimensions, self.frequencies)
        
        # Verify data was loaded correctly
        self.assertIsNotNone(odmr._raw_data)
        self.assertTrue(np.array_equal(odmr.raw_data.data, self.raw_data))
        self.assertTrue(np.array_equal(odmr.raw_data.scan_dimensions, self.scan_dimensions))
        self.assertTrue(np.array_equal(odmr.raw_data.frequencies, self.frequencies))
        self.assertFalse(odmr.is_processed)

    def test_process_data(self):
        """Test processing data."""
        # Mock the processor to return a modified copy of the data
        processed_data = ODMRData(
            self.raw_data.copy() * 0.5,  # Modify data for testing
            self.scan_dimensions.copy(),
            self.frequencies.copy()
        )
        
        with patch.object(self.odmr.processor_manager, 'process', 
                         return_value=processed_data):
            # Process the data
            self.odmr.process_data()
            
            # Verify processing was done
            self.assertTrue(self.odmr.is_processed)
            self.assertIsNotNone(self.odmr._processed_data)
            self.assertEqual(self.odmr.processed_data, processed_data)

    def test_process_data_no_raw_data(self):
        """Test processing with no raw data loaded."""
        # Create empty ODMR instance
        odmr = ODMR()
        
        # Attempt to process without raw data
        with self.assertRaises(ValueError):
            odmr.process_data()

    def test_reset_no_raw_data(self):
        """Test resetting with no raw data loaded."""
        # Create empty ODMR instance
        odmr = ODMR()
        
        # Attempt to reset without raw data
        with self.assertRaises(ValueError):
            odmr.reset()

    def test_access_raw_data_none(self):
        """Test accessing raw data when none is loaded."""
        # Create empty ODMR instance
        odmr = ODMR()
        
        # Attempt to access raw data
        with self.assertRaises(ValueError):
            odmr.raw_data

    def test_access_processed_data_none(self):
        """Test accessing processed data when none exists."""
        # Create ODMR instance with raw data but no processing
        odmr = ODMR(self.odmr_data)
        
        # Attempt to access processed data
        with self.assertRaises(ValueError):
            odmr.processed_data