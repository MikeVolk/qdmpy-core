"""Test module for QDMpy.odmr.validation."""

from __future__ import annotations

import numpy as np
import pytest

from qdmpy_core.exceptions import DataValidationError
from qdmpy_core.odmr._validators import NV_FREQ_MAX_GHZ, NV_FREQ_MIN_GHZ, validate_frequencies


class TestValidateFrequencies:
    """Tests for validate_frequencies utility."""

    def test_accepts_valid_1d_array(self) -> None:
        """Test that a valid 1D frequency array passes."""
        freq = np.linspace(2.85, 2.90, 50)
        validate_frequencies(freq)

    def test_accepts_valid_2d_array(self) -> None:
        """Test that a valid 2D frequency array passes."""
        freq = np.array(
            [
                np.linspace(2.85, 2.88, 50),
                np.linspace(2.88, 2.90, 50),
            ]
        )
        validate_frequencies(freq)

    def test_rejects_empty_array(self) -> None:
        """Test that an empty array raises DataValidationError."""
        with pytest.raises(DataValidationError, match="empty"):
            validate_frequencies(np.array([]))

    def test_rejects_nan_values(self) -> None:
        """Test that NaN values raise DataValidationError."""
        freq = np.linspace(2.85, 2.90, 50)
        freq[10] = np.nan
        with pytest.raises(DataValidationError, match="non-finite"):
            validate_frequencies(freq)

    def test_rejects_inf_values(self) -> None:
        """Test that infinite values raise DataValidationError."""
        freq = np.linspace(2.85, 2.90, 50)
        freq[0] = np.inf
        with pytest.raises(DataValidationError, match="non-finite"):
            validate_frequencies(freq)

    def test_rejects_non_monotonic(self) -> None:
        """Test that non-monotonic frequencies raise DataValidationError."""
        freq = np.array([2.85, 2.86, 2.84, 2.87])
        with pytest.raises(DataValidationError, match="monotonically increasing"):
            validate_frequencies(freq)

    def test_rejects_non_monotonic_2d_row(self) -> None:
        """Test that non-monotonic row in 2D array raises DataValidationError."""
        freq = np.array(
            [
                np.linspace(2.85, 2.88, 10),
                np.array([2.88, 2.89, 2.87, 2.90, 2.91, 2.92, 2.93, 2.94, 2.95, 2.96]),
            ]
        )
        with pytest.raises(DataValidationError, match="monotonically increasing"):
            validate_frequencies(freq)

    def test_warns_out_of_nv_range(self, capfd) -> None:
        """Test that out-of-range frequencies produce a warning (not error)."""
        freq = np.linspace(1.0, 1.5, 50)
        validate_frequencies(freq)

    def test_no_warning_in_nv_range(self) -> None:
        """Test that in-range frequencies produce no warning."""
        freq = np.linspace(NV_FREQ_MIN_GHZ + 0.1, NV_FREQ_MAX_GHZ - 0.1, 50)
        validate_frequencies(freq)

    def test_single_frequency_passes(self) -> None:
        """Test that a single frequency value passes monotonicity check."""
        validate_frequencies(np.array([2.87]))

    def test_boundary_nv_range(self) -> None:
        """Test frequencies exactly at NV range boundaries pass without warning."""
        freq = np.linspace(NV_FREQ_MIN_GHZ, NV_FREQ_MAX_GHZ, 50)
        validate_frequencies(freq)
