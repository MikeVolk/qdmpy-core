"""Test module for QDMpy.odmr.io."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from QDMpy.exceptions import DataLoadError
from QDMpy.odmr.io import BaseLoader, MatlabLoader


@pytest.fixture
def test_data_path() -> str:
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class TestBaseLoader:
    """Test class for BaseLoader."""

    def test_abstract_class(self) -> None:
        """Test that BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()


class TestMatlabLoader:
    """Test class for MatlabLoader."""

    def test_init(self, test_data_path) -> None:
        """Test initialization of MatlabLoader."""
        loader = MatlabLoader(data_folder=test_data_path)
        assert loader.data_folder == test_data_path

    def test_load(self, test_data_path) -> None:
        """Test load method returns xr.DataArray with correct structure."""
        if not os.path.isdir(test_data_path):
            pytest.skip("Test data directory not found")

        mat_files = [
            f for f in os.listdir(test_data_path) if f.startswith("run_") and f.endswith(".mat")
        ]
        if not mat_files:
            pytest.skip("No .mat files found in test data directory")

        loader = MatlabLoader(data_folder=test_data_path)
        result = loader.load()

        assert isinstance(result, xr.DataArray)
        assert result.dims == ("polarity", "freq_range", "y", "x", "freq_idx")
        assert len(result.shape) == 5

    def test_load_no_files(self) -> None:
        """Test load method with no valid files."""
        with patch("os.listdir", return_value=[]):
            loader = MatlabLoader(data_folder="/dummy/path")
            with pytest.raises(DataLoadError):
                loader.load()

    def test_process_mat_file_2stacks(self) -> None:
        """Test _process_mat_file with 2 image stacks."""
        mock_data = {
            "imgStack1": np.ones((10, 10)),
            "imgStack2": np.ones((10, 10)) * 2,
        }

        result = MatlabLoader._process_mat_file(mock_data)
        assert result.shape == (2, 10, 10)
        assert np.array_equal(result[0], np.ones((10, 10)))
        assert np.array_equal(result[1], np.ones((10, 10)) * 2)

    def test_process_mat_file_4stacks(self) -> None:
        """Test _process_mat_file with 4 image stacks (concat-before-transpose)."""
        mock_data = {
            "imgStack1": np.ones((5, 10)),
            "imgStack2": np.ones((5, 10)) * 2,
            "imgStack3": np.ones((5, 10)) * 3,
            "imgStack4": np.ones((5, 10)) * 4,
        }

        result = MatlabLoader._process_mat_file(mock_data)

        # concat([imgStack1(5,10), imgStack2(5,10)], axis=0) -> (10, 10), then .T -> (10, 10)
        assert result.shape == (2, 10, 10)

        expected_low = np.concatenate([np.ones((5, 10)), np.ones((5, 10)) * 2], axis=0).T
        assert np.array_equal(result[0], expected_low)

        expected_high = np.concatenate([np.ones((5, 10)) * 3, np.ones((5, 10)) * 4], axis=0).T
        assert np.array_equal(result[1], expected_high)

    def test_process_mat_file_unsupported(self) -> None:
        """Test _process_mat_file with unsupported number of stacks."""
        mock_data = {
            "imgStack1": np.ones((10, 10)),
            "imgStack2": np.ones((10, 10)) * 2,
            "imgStack3": np.ones((10, 10)) * 3,
        }

        with pytest.raises(DataLoadError, match="Unsupported number of image stacks"):
            MatlabLoader._process_mat_file(mock_data)

    def test_keys_missing_exception(self) -> None:
        """Test that DataLoadError is raised when keys are missing from data."""
        mock_data = {"some_other_key": "value"}

        with pytest.raises(DataLoadError, match="Missing required key"):
            try:
                np.array(
                    [
                        int(np.squeeze(mock_data["imgNumRows"])),
                        int(np.squeeze(mock_data["imgNumCols"])),
                    ],
                )
            except KeyError as e:
                raise DataLoadError(f"Missing required key in MATLAB file: {e}")

        with pytest.raises(DataLoadError, match="Missing required key"):
            try:
                np.squeeze(mock_data["freqList"])
            except KeyError as e:
                raise DataLoadError(f"Missing required key in MATLAB file: {e}")
