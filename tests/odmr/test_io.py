"""
Test module for QDMpy.odmr.io
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from QDMpy.odmr.io import BaseLoader, MatlabLoader


@pytest.fixture
def test_data_path() -> str:
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class TestBaseLoader:
    """Test class for BaseLoader."""

    def test_abstract_class(self):
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()


class TestMatlabLoader:
    """Test class for MatlabLoader."""

    def test_init(self, test_data_path):
        """Test initialization of MatlabLoader."""
        loader = MatlabLoader(data_folder=test_data_path)
        assert loader.data_folder == test_data_path

    def test_load(self, test_data_path):
        """Test load method with real data."""
        loader = MatlabLoader(data_folder=test_data_path)
        raw_data, img_shape, frequencies = loader.load()
        
        assert isinstance(raw_data, np.ndarray)
        assert isinstance(img_shape, np.ndarray)
        assert isinstance(frequencies, np.ndarray)
        
        # Validate dimensions
        assert len(img_shape) == 2  # Should have row and column dimensions
        assert len(frequencies.shape) == 1  # Should be a 1D array of frequencies

    def test_load_no_files(self):
        """Test load method with no valid files."""
        with patch('os.listdir', return_value=[]):
            loader = MatlabLoader(data_folder="/dummy/path")
            with pytest.raises(FileNotFoundError):
                loader.load()

    def test_process_mat_file_2stacks(self):
        """Test _process_mat_file with 2 image stacks."""
        # Mock data with 2 image stacks
        mock_data = {
            "imgStack1": np.ones((10, 10)),
            "imgStack2": np.ones((10, 10)) * 2,
        }
        
        result = MatlabLoader._process_mat_file(mock_data)
        assert result.shape == (2, 10, 10)
        assert np.array_equal(result[0], np.ones((10, 10)))
        assert np.array_equal(result[1], np.ones((10, 10)) * 2)

    def test_process_mat_file_4stacks(self):
        """Test _process_mat_file with 4 image stacks."""
        # Mock data with 4 image stacks
        mock_data = {
            "imgStack1": np.ones((5, 10)),
            "imgStack2": np.ones((5, 10)) * 2,
            "imgStack3": np.ones((5, 10)) * 3,
            "imgStack4": np.ones((5, 10)) * 4,
        }
        
        result = MatlabLoader._process_mat_file(mock_data)
        assert result.shape == (2, 10, 5)
        # First concatenated stack
        assert np.array_equal(result[0][:5], np.ones((5, 5)))
        assert np.array_equal(result[0][5:], np.ones((5, 5)) * 2)
        # Second concatenated stack
        assert np.array_equal(result[1][:5], np.ones((5, 5)) * 3)
        assert np.array_equal(result[1][5:], np.ones((5, 5)) * 4)

    def test_process_mat_file_unsupported(self):
        """Test _process_mat_file with unsupported number of stacks."""
        # Mock data with 3 image stacks (unsupported)
        mock_data = {
            "imgStack1": np.ones((10, 10)),
            "imgStack2": np.ones((10, 10)) * 2,
            "imgStack3": np.ones((10, 10)) * 3,
        }
        
        with pytest.raises(ValueError, match="Unsupported number of image stacks"):
            MatlabLoader._process_mat_file(mock_data)

    def test_missing_required_keys(self):
        """Test handling of missing required keys."""
        with patch('mat73.loadmat', side_effect=Exception("Test exception")), \
             patch('scipy.io.loadmat', return_value={"missing_keys": True}):
            
            loader = MatlabLoader(data_folder="/dummy/path")
            with patch('os.listdir', return_value=["run_00000.mat"]), \
                 patch('os.path.join', return_value="dummy_path"):
                with pytest.raises(ValueError, match="Missing required key"):
                    loader.load()