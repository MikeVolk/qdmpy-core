from __future__ import annotations

import unittest

import numpy as np

from qdmpy.utils import (
    double_norm,
    idx2rc,
    millify,
    polyfit2d,
    rc2idx,
    rms,
)


class TestUtils(unittest.TestCase):
    """Tests for utility functions in QDMpy.utils."""

    def test_millify(self) -> None:
        """Test millify function for human-readable numbers."""
        # Test a range of values
        test_cases = [
            # Input, expected output
            (0.0, "0.0"),
            (1, "1.0"),
            (1000, "1.0 K"),
            (1234, "1.2 K"),
            (1000000, "1.0 M"),
            (1234567, "1.2 M"),
            (1000000000, "1.0 B"),
            (1234567890, "1.2 B"),
            (1000000000000, "1.0 T"),
            (1234567890123, "1.2 T"),
            # Test precision
            (1234, "1.23 K", 2),
            (1234, "1.234 K", 3),
            # Test negative numbers
            (-1234, "-1.2 K"),
        ]

        for test_case in test_cases:
            if len(test_case) == 2:
                input_val, expected = test_case
                result = millify(input_val)
            else:
                input_val, expected, precision = test_case
                result = millify(input_val, precision)

            assert result == expected

    def test_double_norm(self) -> None:
        """Test double norm function for array normalization."""
        # Create a test array with positive and negative values
        test_array = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=float)

        # Test with default settings
        result = double_norm(test_array)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert result.min() == 0
        assert result.max() == 1

    def test_idx2rc(self) -> None:
        """Test conversion from linear indices to row-column coordinates."""
        # Create a 3x4 array for testing
        shape = (3, 4)  # 3 rows, 4 columns

        # Test single index
        idx = 5  # Should be at row 1, column 1 (0-indexed)
        row, col = idx2rc(idx, shape)
        assert row == 1
        assert col == 1

        # Test multiple indices
        indices = [0, 5, 10]  # Should map to (0,0), (1,1), (2,2)
        rows, cols = idx2rc(indices, shape)
        assert np.array_equal(rows, np.array([0, 1, 2]))
        assert np.array_equal(cols, np.array([0, 1, 2]))

        # Test array input
        indices_array = np.array([0, 5, 10])
        rows, cols = idx2rc(indices_array, shape)
        assert np.array_equal(rows, np.array([0, 1, 2]))
        assert np.array_equal(cols, np.array([0, 1, 2]))

    def test_rc2idx(self) -> None:
        """Test conversion from row-column coordinates to linear indices."""
        # Create a 3x4 array for testing
        shape = (3, 4)  # 3 rows, 4 columns

        # Test single coordinate pair
        rc = np.array([[1], [1]])  # Row 1, column 1 should be index 5
        idx = rc2idx(rc, shape)
        assert idx[0] == 5

        # Test multiple coordinate pairs
        rc = np.array([[0, 1, 2], [0, 1, 2]])  # (0,0), (1,1), (2,2) should map to 0, 5, 10
        idx = rc2idx(rc, shape)
        assert np.array_equal(idx, np.array([0, 5, 10]))

    def test_polyfit2d(self) -> None:
        """Test 2D polynomial fitting."""
        # Create sample data
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])

        # Create a simple z surface: z = x + y
        xx, yy = np.meshgrid(x, y)
        z = xx + yy

        # Fit a polynomial
        solution, residuals, rank, singular_values = polyfit2d(x, y, z)

        # Check the shape of the solution (should be (kx+1)*(ky+1) = 16 for default kx=ky=3)
        assert len(solution) == 16

        # Check residuals (should be close to zero for perfect fit)
        self.assertAlmostEqual(np.sum(residuals), 0, places=10)

        # Test with order parameter
        solution, residuals, _rank, _singular_values = polyfit2d(x, y, z, order=1)

        # Check that higher order terms are excluded (coefficients should be close to [0, 1, 1, 0])
        expected_coeffs = np.zeros(16)
        expected_coeffs[0] = 0  # constant term
        expected_coeffs[1] = 1  # y term
        expected_coeffs[4] = 1  # x term

        # Reshape solution to match expected_coeffs and check relevant terms
        self.assertAlmostEqual(solution[0], 0, places=5)  # constant term
        self.assertAlmostEqual(solution[1], 1, places=5)  # y term
        self.assertAlmostEqual(solution[4], 1, places=5)  # x term

    def test_rms(self) -> None:
        """Test root mean square calculation."""
        # Test with all zeros
        data = np.zeros(10)
        result = rms(data)
        assert result == 0.0

        # Test with all ones
        data = np.ones(10)
        result = rms(data)
        assert result == 1.0

        # Test with known values
        data = np.array([1, 2, 3, 4, 5])
        # RMS = sqrt(mean([1², 2², 3², 4², 5²])) = sqrt((1+4+9+16+25)/5) = sqrt(55/5) = sqrt(11) ≈ 3.317
        expected = np.sqrt(np.mean(np.square(data)))
        result = rms(data)
        self.assertAlmostEqual(result, expected)

        # Test with negative values
        data = np.array([-1, -2, -3, -4, -5])
        # RMS should be the same as with positive values
        expected = np.sqrt(np.mean(np.square(data)))
        result = rms(data)
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
