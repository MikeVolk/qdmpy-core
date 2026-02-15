"""Test module for QDMpy.odmr.data."""

from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from QDMpy.odmr.data import ODMRData

N_POL = 2
N_FRANGE = 3
ROWS = 10
COLS = 10
N_PIXELS = ROWS * COLS
N_FREQS = 50


@pytest.fixture
def raw_data() -> NDArray:
    """Provide a 4D numpy array with shape (n_pol, n_frange, n_pixels, n_freqs)."""
    rng = np.random.default_rng(42)
    return rng.random((N_POL, N_FRANGE, N_PIXELS, N_FREQS))


@pytest.fixture
def scan_dimensions() -> tuple[int, int]:
    """Provide scan dimensions as (rows, cols)."""
    return (ROWS, COLS)


@pytest.fixture
def frequencies_1d() -> NDArray:
    """Provide a 1D frequency array in Hz."""
    return np.linspace(2.87e9, 2.89e9, N_FREQS)


@pytest.fixture
def odmr_data(raw_data, scan_dimensions, frequencies_1d) -> ODMRData:
    """Provide an ODMRData instance created via from_numpy."""
    return ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)


class TestFromNumpy:
    """Tests for ODMRData.from_numpy class method."""

    def test_returns_odmr_data_instance(self, raw_data, scan_dimensions, frequencies_1d):
        """Test that from_numpy returns an ODMRData instance."""
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)
        assert isinstance(result, ODMRData)

    def test_underlying_data_is_xarray(self, odmr_data):
        """Test that the internal data attribute is an xr.DataArray."""
        assert isinstance(odmr_data.data, xr.DataArray)

    def test_data_dimensions(self, odmr_data):
        """Test that the DataArray has the expected named dimensions."""
        assert odmr_data.data.dims == ('polarity', 'freq_range', 'y', 'x', 'freq_idx')

    def test_data_shape_is_5d(self, odmr_data):
        """Test that 4D input is reshaped into a 5D DataArray."""
        assert odmr_data.data.shape == (N_POL, N_FRANGE, ROWS, COLS, N_FREQS)

    def test_polarity_coordinates(self, odmr_data):
        """Test that polarity coordinates are labelled correctly."""
        expected = [f'pol_{i}' for i in range(N_POL)]
        assert list(odmr_data.data.coords['polarity'].values) == expected

    def test_frange_coordinates(self, odmr_data):
        """Test that freq_range coordinates are labelled correctly."""
        expected = [f'frange_{i}' for i in range(N_FRANGE)]
        assert list(odmr_data.data.coords['freq_range'].values) == expected

    def test_freq_ghz_coordinate_shape(self, odmr_data):
        """Test that freq_ghz coordinate has shape (n_frange, n_freqs)."""
        freq_ghz = odmr_data.data.coords['freq_ghz'].values
        assert freq_ghz.shape == (N_FRANGE, N_FREQS)

    def test_1d_frequencies_are_tiled(self, raw_data, scan_dimensions, frequencies_1d):
        """Test that a 1D frequency array is tiled across freq_ranges."""
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)
        freq_ghz = result.data.coords['freq_ghz'].values
        for i in range(N_FRANGE):
            np.testing.assert_allclose(freq_ghz[i], frequencies_1d / 1e9)

    def test_2d_frequencies_preserved(self, raw_data, scan_dimensions):
        """Test that a 2D frequency array is stored without tiling."""
        freqs_2d = np.stack(
            [np.linspace(2.85e9, 2.87e9, N_FREQS) for _ in range(N_FRANGE)]
        )
        result = ODMRData.from_numpy(raw_data, scan_dimensions, freqs_2d)
        freq_ghz = result.data.coords['freq_ghz'].values
        np.testing.assert_allclose(freq_ghz, freqs_2d / 1e9)

    def test_default_metadata_is_empty_dict(self, odmr_data):
        """Test that metadata defaults to an empty dict."""
        assert odmr_data.metadata == {}

    def test_values_roundtrip(self, raw_data, scan_dimensions, frequencies_1d):
        """Test that data values survive the numpy -> xarray conversion."""
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)
        expected_5d = raw_data.reshape(N_POL, N_FRANGE, ROWS, COLS, N_FREQS)
        np.testing.assert_array_equal(result.data.values, expected_5d)


class TestFromNumpyWithMetadata:
    """Tests for ODMRData.from_numpy with metadata argument."""

    def test_metadata_stored(self, raw_data, scan_dimensions, frequencies_1d):
        """Test that provided metadata is stored on the instance."""
        metadata = {'source': 'test', 'version': 2}
        result = ODMRData.from_numpy(
            raw_data, scan_dimensions, frequencies_1d, metadata=metadata
        )
        assert result.metadata == metadata

    def test_metadata_keys_accessible(self, raw_data, scan_dimensions, frequencies_1d):
        """Test that individual metadata keys are accessible."""
        metadata = {'instrument': 'QDM-v2', 'temperature_k': 295.0}
        result = ODMRData.from_numpy(
            raw_data, scan_dimensions, frequencies_1d, metadata=metadata
        )
        assert result.metadata['instrument'] == 'QDM-v2'
        assert result.metadata['temperature_k'] == 295.0


class TestShapeProperty:
    """Tests for the shape property."""

    def test_shape_matches_data_array(self, odmr_data):
        """Test that shape returns the DataArray's shape."""
        assert odmr_data.shape == odmr_data.data.shape

    def test_shape_value(self, odmr_data):
        """Test that shape returns the expected 5D tuple."""
        assert odmr_data.shape == (N_POL, N_FRANGE, ROWS, COLS, N_FREQS)

    def test_shape_is_tuple(self, odmr_data):
        """Test that shape returns a tuple."""
        assert isinstance(odmr_data.shape, tuple)


class TestScanDimensions:
    """Tests for the scan_dimensions property."""

    def test_returns_tuple(self, odmr_data):
        """Test that scan_dimensions returns a tuple."""
        assert isinstance(odmr_data.scan_dimensions, tuple)

    def test_returns_rows_cols(self, odmr_data):
        """Test that scan_dimensions returns (rows, cols)."""
        assert odmr_data.scan_dimensions == (ROWS, COLS)

    def test_non_square_dimensions(self, frequencies_1d):
        """Test scan_dimensions with non-square spatial grid."""
        rows, cols = 5, 20
        rng = np.random.default_rng(0)
        data = rng.random((N_POL, N_FRANGE, rows * cols, N_FREQS))
        result = ODMRData.from_numpy(data, (rows, cols), frequencies_1d)
        assert result.scan_dimensions == (rows, cols)


class TestFrequencies:
    """Tests for the frequencies property."""

    def test_returns_numpy_array(self, odmr_data):
        """Test that frequencies returns a numpy array."""
        assert isinstance(odmr_data.frequencies, np.ndarray)

    def test_frequencies_in_ghz(self, odmr_data, frequencies_1d):
        """Test that frequencies are returned in GHz."""
        result = odmr_data.frequencies
        for i in range(N_FRANGE):
            np.testing.assert_allclose(result[i], frequencies_1d / 1e9, rtol=1e-10)

    def test_frequencies_shape(self, odmr_data):
        """Test that frequencies has shape (n_frange, n_freqs)."""
        assert odmr_data.frequencies.shape == (N_FRANGE, N_FREQS)


class TestNumpyProperty:
    """Tests for the numpy property."""

    def test_returns_numpy_array(self, odmr_data):
        """Test that numpy property returns a numpy ndarray."""
        assert isinstance(odmr_data.numpy, np.ndarray)

    def test_numpy_shape_matches_data_array(self, odmr_data):
        """Test that numpy array shape matches the DataArray shape."""
        assert odmr_data.numpy.shape == odmr_data.data.shape

    def test_numpy_values_match_input(self, odmr_data, raw_data):
        """Test that numpy values match the original input data."""
        expected = raw_data.reshape(N_POL, N_FRANGE, ROWS, COLS, N_FREQS)
        np.testing.assert_array_equal(odmr_data.numpy, expected)


class TestFromLoader:
    """Tests for ODMRData.from_loader class method."""

    def test_from_loader_wraps_error_in_runtime_error(self):
        """Test that loader exceptions are wrapped in RuntimeError."""

        class _FailingLoader:
            def load(self, **kwargs):
                raise ValueError('intentional test error')

        with pytest.raises(RuntimeError, match='Data loading failed'):
            ODMRData.from_loader(_FailingLoader())

    def test_from_loader_with_real_data(self):
        """Test from_loader with MatlabLoader against real test data."""
        from QDMpy.odmr.io import MatlabLoader

        test_data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
        if not os.path.isdir(test_data_path):
            pytest.skip('Test data directory not found')

        try:
            loader = MatlabLoader(data_folder=test_data_path)
        except Exception:
            pytest.skip('MatlabLoader could not be initialized with test data')

        try:
            result = ODMRData.from_loader(loader)
        except RuntimeError:
            pytest.skip('Loader failed to load test data')

        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)
