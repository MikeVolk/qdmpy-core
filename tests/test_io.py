"""Test module for QDMpy.io.

These tests cover the functions for file loading and handling in the io module.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from QDMpy.io import get_image, get_image_file, has_csv


class TestIO:
    """Test the io module functions."""

    def test_has_csv(self) -> None:
        """Test has_csv function."""
        # Empty list
        assert not has_csv([])

        # No CSV files
        assert not has_csv(["file.txt", "image.jpg"])

        # Contains CSV files
        assert has_csv(["file.csv", "image.jpg"])
        assert has_csv(["FILE.CSV", "image.jpg"])  # Case insensitive

    def test_get_image_file(self) -> None:
        """Test get_image_file function."""
        # Test with CSV files
        files = ["data.csv", "image.jpg"]
        assert get_image_file(files) == "data.csv"

        # Test with only JPG files
        files = ["image1.jpg", "image2.jpg"]
        assert get_image_file(files) == "image1.jpg"

        # Test with no suitable files
        with pytest.raises(ValueError):
            get_image_file(["file1.txt", "file2.txt"])

    def test_get_image(self) -> None:
        """Test get_image function with temporary files."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create a test CSV file
            csv_path = os.path.join(tmpdirname, "test.csv")
            test_data = np.array([[1, 2, 3], [4, 5, 6]])
            np.savetxt(csv_path, test_data, delimiter=",")

            # Test loading the CSV file
            img = get_image(tmpdirname, ["test.csv"])
            assert img.shape == (2, 3)
            assert np.array_equal(img, test_data)

            # Test with non-existent file
            with pytest.raises(ValueError):
                get_image(tmpdirname, ["nonexistent.csv"])


if __name__ == "__main__":
    pytest.main()
