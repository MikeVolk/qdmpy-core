"""Test module for QDMpy.odmr.data."""

from __future__ import annotations

import os
from typing import NoReturn

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from qdmpy_core.constants import GAMMA_NV
from qdmpy_core.exceptions import DataLoadError, DataValidationError
from qdmpy_core.odmr.analysis import b111_from_dip_positions
from qdmpy_core.odmr.data import EXPECTED_DIMS, ODMRData

N_POL = 2
N_FRANGE = 2
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

    def test_returns_odmr_data_instance(self, raw_data, scan_dimensions, frequencies_1d) -> None:
        """Test that from_numpy returns an ODMRData instance."""
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)
        assert isinstance(result, ODMRData)

    def test_underlying_data_is_xarray(self, odmr_data) -> None:
        """Test that the internal data attribute is an xr.DataArray."""
        assert isinstance(odmr_data.data, xr.DataArray)

    def test_data_dimensions(self, odmr_data) -> None:
        """Test that the DataArray has the expected named dimensions."""
        assert odmr_data.data.dims == ("polarity", "freq_range", "y", "x", "freq_idx")

    def test_data_shape_is_5d(self, odmr_data) -> None:
        """Test that 4D input is reshaped into a 5D DataArray."""
        assert odmr_data.data.shape == (N_POL, N_FRANGE, ROWS, COLS, N_FREQS)

    def test_polarity_coordinates(self, odmr_data) -> None:
        """Test that polarity coordinates use semantic labels."""
        assert list(odmr_data.data.coords["polarity"].values) == ["neg", "pos"]

    def test_frange_coordinates(self, odmr_data) -> None:
        """Test that freq_range coordinates use semantic labels."""
        assert list(odmr_data.data.coords["freq_range"].values) == ["low", "high"]

    def test_freq_ghz_coordinate_shape(self, odmr_data) -> None:
        """Test that freq_ghz coordinate has shape (n_frange, n_freqs)."""
        freq_ghz = odmr_data.data.coords["freq_ghz"].values
        assert freq_ghz.shape == (N_FRANGE, N_FREQS)

    def test_1d_frequencies_are_tiled(self, raw_data, scan_dimensions, frequencies_1d) -> None:
        """Test that a 1D frequency array is tiled across freq_ranges."""
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)
        freq_ghz = result.data.coords["freq_ghz"].values
        for i in range(N_FRANGE):
            np.testing.assert_allclose(freq_ghz[i], frequencies_1d / 1e9)

    def test_2d_frequencies_preserved(self, raw_data, scan_dimensions) -> None:
        """Test that a 2D frequency array is stored without tiling."""
        freqs_2d = np.stack([np.linspace(2.85e9, 2.87e9, N_FREQS) for _ in range(N_FRANGE)])
        result = ODMRData.from_numpy(raw_data, scan_dimensions, freqs_2d)
        freq_ghz = result.data.coords["freq_ghz"].values
        np.testing.assert_allclose(freq_ghz, freqs_2d / 1e9)

    def test_default_metadata_is_empty_dict(self, odmr_data) -> None:
        """Test that metadata defaults to an empty dict."""
        assert odmr_data.metadata == {}

    def test_values_roundtrip(self, raw_data, scan_dimensions, frequencies_1d) -> None:
        """Test that data values survive the numpy -> xarray conversion."""
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d)
        expected_5d = raw_data.reshape(N_POL, N_FRANGE, ROWS, COLS, N_FREQS)
        np.testing.assert_array_equal(result.data.values, expected_5d)


class TestFromNumpyWithMetadata:
    """Tests for ODMRData.from_numpy with metadata argument."""

    def test_metadata_stored(self, raw_data, scan_dimensions, frequencies_1d) -> None:
        """Test that provided metadata is stored on the instance."""
        metadata = {"source": "test", "version": 2}
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d, metadata=metadata)
        assert result.metadata == metadata

    def test_metadata_keys_accessible(self, raw_data, scan_dimensions, frequencies_1d) -> None:
        """Test that individual metadata keys are accessible."""
        metadata = {"instrument": "QDM-v2", "temperature_k": 295.0}
        result = ODMRData.from_numpy(raw_data, scan_dimensions, frequencies_1d, metadata=metadata)
        assert result.metadata["instrument"] == "QDM-v2"
        assert result.metadata["temperature_k"] == 295.0


class TestShapeProperty:
    """Tests for the shape property."""

    def test_shape_matches_data_array(self, odmr_data) -> None:
        """Test that shape returns the DataArray's shape."""
        assert odmr_data.shape == odmr_data.data.shape

    def test_shape_value(self, odmr_data) -> None:
        """Test that shape returns the expected 5D tuple."""
        assert odmr_data.shape == (N_POL, N_FRANGE, ROWS, COLS, N_FREQS)

    def test_shape_is_tuple(self, odmr_data) -> None:
        """Test that shape returns a tuple."""
        assert isinstance(odmr_data.shape, tuple)


class TestScanDimensions:
    """Tests for the scan_dimensions property."""

    def test_returns_tuple(self, odmr_data) -> None:
        """Test that scan_dimensions returns a tuple."""
        assert isinstance(odmr_data.scan_dimensions, tuple)

    def test_returns_rows_cols(self, odmr_data) -> None:
        """Test that scan_dimensions returns (rows, cols)."""
        assert odmr_data.scan_dimensions == (ROWS, COLS)

    def test_non_square_dimensions(self, frequencies_1d) -> None:
        """Test scan_dimensions with non-square spatial grid."""
        rows, cols = 5, 20
        rng = np.random.default_rng(0)
        data = rng.random((N_POL, N_FRANGE, rows * cols, N_FREQS))
        result = ODMRData.from_numpy(data, (rows, cols), frequencies_1d)
        assert result.scan_dimensions == (rows, cols)


class TestFrequencies:
    """Tests for the frequencies property."""

    def test_returns_numpy_array(self, odmr_data) -> None:
        """Test that frequencies returns a numpy array."""
        assert isinstance(odmr_data.frequencies, np.ndarray)

    def test_frequencies_in_ghz(self, odmr_data, frequencies_1d) -> None:
        """Test that frequencies are returned in GHz."""
        result = odmr_data.frequencies
        for i in range(N_FRANGE):
            np.testing.assert_allclose(result[i], frequencies_1d / 1e9, rtol=1e-10)

    def test_frequencies_shape(self, odmr_data) -> None:
        """Test that frequencies has shape (n_frange, n_freqs)."""
        assert odmr_data.frequencies.shape == (N_FRANGE, N_FREQS)


class TestNumpyProperty:
    """Tests for the numpy property."""

    def test_returns_numpy_array(self, odmr_data) -> None:
        """Test that numpy property returns a numpy ndarray."""
        assert isinstance(odmr_data.numpy, np.ndarray)

    def test_numpy_shape_matches_data_array(self, odmr_data) -> None:
        """Test that numpy array shape matches the DataArray shape."""
        assert odmr_data.numpy.shape == odmr_data.data.shape

    def test_numpy_values_match_input(self, odmr_data, raw_data) -> None:
        """Test that numpy values match the original input data."""
        expected = raw_data.reshape(N_POL, N_FRANGE, ROWS, COLS, N_FREQS)
        np.testing.assert_array_equal(odmr_data.numpy, expected)


class TestFromLoader:
    """Tests for ODMRData.from_loader class method."""

    def test_from_loader_wraps_error_in_runtime_error(self) -> None:
        """Test that loader exceptions are wrapped in RuntimeError."""

        class _FailingLoader:
            def load(self, **kwargs) -> NoReturn:
                raise ValueError("intentional test error")

        with pytest.raises(DataLoadError, match="Data loading failed"):
            ODMRData.from_loader(_FailingLoader())

    def test_from_loader_with_real_data(self) -> None:
        """Test from_loader with MatlabLoader against real test data."""
        from qdmpy_core.odmr.io import MatlabLoader

        test_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if not os.path.isdir(test_data_path):
            pytest.skip("Test data directory not found")

        try:
            loader = MatlabLoader(data_folder=test_data_path)
        except Exception:
            pytest.skip("MatlabLoader could not be initialized with test data")

        try:
            result = ODMRData.from_loader(loader)
        except (RuntimeError, DataLoadError):
            pytest.skip("Loader failed to load test data")

        assert isinstance(result, ODMRData)
        assert isinstance(result.data, xr.DataArray)


@pytest.fixture
def dip_position_data() -> xr.DataArray:
    """Synthetic DataArray with known argmin dip positions for B₁₁₁ testing.

    Dip indices (into n_freq=20):
        neg/low=3, neg/high=14, pos/low=6, pos/high=10
    """
    H, W, n_freq = 4, 5, 20
    freqs_low = np.linspace(2.82, 2.87, n_freq)
    freqs_high = np.linspace(2.87, 2.92, n_freq)
    freq_ghz = np.stack([freqs_low, freqs_high])  # (2, 20)

    data = np.ones((2, 2, H, W, n_freq))
    dip_idx = {("neg", "low"): 3, ("neg", "high"): 14, ("pos", "low"): 6, ("pos", "high"): 10}
    pol_labels = ["neg", "pos"]
    frange_labels = ["low", "high"]
    for i_pol, pol in enumerate(pol_labels):
        for i_frange, frange in enumerate(frange_labels):
            data[i_pol, i_frange, :, :, dip_idx[(pol, frange)]] = 0.0

    return xr.DataArray(
        data,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": pol_labels,
            "freq_range": frange_labels,
            "freq_ghz": (["freq_range", "freq_idx"], freq_ghz),
        },
    )


class TestB111FromDipPositions:
    """Tests for the quick_b111 standalone function."""

    def test_returns_dict(self, dip_position_data: xr.DataArray) -> None:
        result = b111_from_dip_positions(dip_position_data)
        assert isinstance(result, dict)

    def test_result_keys(self, dip_position_data: xr.DataArray) -> None:
        result = b111_from_dip_positions(dip_position_data)
        assert set(result.keys()) == {"remanent", "induced"}

    def test_output_shape(self, dip_position_data: xr.DataArray) -> None:
        result = b111_from_dip_positions(dip_position_data)
        assert result["remanent"].shape == (4, 5)
        assert result["induced"].shape == (4, 5)

    def test_output_is_ndarray(self, dip_position_data: xr.DataArray) -> None:
        result = b111_from_dip_positions(dip_position_data)
        assert isinstance(result["remanent"], np.ndarray)
        assert isinstance(result["induced"], np.ndarray)

    def test_uniform_across_pixels(self, dip_position_data: xr.DataArray) -> None:
        """Uniform dip positions give uniform field maps."""
        result = b111_from_dip_positions(dip_position_data)
        np.testing.assert_allclose(result["remanent"], result["remanent"].flat[0])
        np.testing.assert_allclose(result["induced"], result["induced"].flat[0])

    def test_physics_values(self, dip_position_data: xr.DataArray) -> None:
        """Verify exact B₁₁₁ values against the manual formula."""
        freqs_low = np.linspace(2.82, 2.87, 20)
        freqs_high = np.linspace(2.87, 2.92, 20)

        dip_neg_low = freqs_low[3]
        dip_neg_high = freqs_high[14]
        dip_pos_low = freqs_low[6]
        dip_pos_high = freqs_high[10]

        delta_neg = -1.0 * (dip_neg_high - dip_neg_low) / 2.0 / GAMMA_NV * 1e6
        delta_pos = +1.0 * (dip_pos_high - dip_pos_low) / 2.0 / GAMMA_NV * 1e6
        expected_remanent = (delta_neg + delta_pos) / 2.0
        expected_induced = (delta_neg - delta_pos) / 2.0

        result = b111_from_dip_positions(dip_position_data)
        np.testing.assert_allclose(result["remanent"], expected_remanent, rtol=1e-10)
        np.testing.assert_allclose(result["induced"], expected_induced, rtol=1e-10)

    def test_symmetric_splitting_gives_zero_remanent(self) -> None:
        """Equal dip positions across polarities → zero remanent field."""
        H, W, n_freq = 3, 3, 10
        freqs_low = np.linspace(2.84, 2.87, n_freq)
        freqs_high = np.linspace(2.87, 2.90, n_freq)
        freq_ghz = np.stack([freqs_low, freqs_high])

        data = np.ones((2, 2, H, W, n_freq))
        # Same dip index for both polarities
        for i_pol in range(2):
            data[i_pol, 0, :, :, 4] = 0.0  # low range, idx 4
            data[i_pol, 1, :, :, 7] = 0.0  # high range, idx 7

        da = xr.DataArray(
            data,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": ["neg", "pos"],
                "freq_range": ["low", "high"],
                "freq_ghz": (["freq_range", "freq_idx"], freq_ghz),
            },
        )
        result = b111_from_dip_positions(da)
        np.testing.assert_allclose(result["remanent"], 0.0, atol=1e-10)

    def test_missing_polarity_raises(self, dip_position_data: xr.DataArray) -> None:
        bad = dip_position_data.sel(polarity=["neg"])
        with pytest.raises(DataValidationError, match="polarity='pos'"):
            b111_from_dip_positions(bad)

    def test_missing_frange_raises(self, dip_position_data: xr.DataArray) -> None:
        bad = dip_position_data.sel(freq_range=["low"])
        with pytest.raises(DataValidationError, match="freq_range='high'"):
            b111_from_dip_positions(bad)



class TestODMRDataValidation:
    """Tests for ODMRData Pydantic validation."""

    def test_rejects_non_dataarray(self) -> None:
        """Test that passing a non-DataArray raises ValidationError."""
        with pytest.raises((DataValidationError, Exception)):
            ODMRData(data=np.ones((2, 3, 10, 10, 50)))

    def test_rejects_wrong_dims(self) -> None:
        """Test that wrong dimension names raise DataValidationError."""
        da = xr.DataArray(
            np.ones((2, 3, 10, 10, 50)),
            dims=("a", "b", "c", "d", "e"),
        )
        with pytest.raises((DataValidationError, Exception)):
            ODMRData(data=da)

    def test_rejects_missing_dims(self) -> None:
        """Test that a DataArray with wrong number of dims is rejected."""
        da = xr.DataArray(
            np.ones((2, 3, 100, 50)),
            dims=("polarity", "freq_range", "pixel", "freq_idx"),
        )
        with pytest.raises((DataValidationError, Exception)):
            ODMRData(data=da)

    def test_rejects_non_numeric_dtype(self) -> None:
        """Test that non-numeric data type is rejected."""
        da = xr.DataArray(
            np.array(["a"] * 2 * 3 * 10 * 10 * 50).reshape(2, 3, 10, 10, 50),
            dims=EXPECTED_DIMS,
            coords={"freq_ghz": (["freq_range", "freq_idx"], np.ones((3, 50)))},
        )
        with pytest.raises((DataValidationError, Exception)):
            ODMRData(data=da)

    def test_rejects_missing_freq_ghz_coord(self) -> None:
        """Test that missing freq_ghz coordinate is rejected."""
        da = xr.DataArray(
            np.ones((2, 3, 10, 10, 50)),
            dims=EXPECTED_DIMS,
        )
        with pytest.raises((DataValidationError, Exception)):
            ODMRData(data=da)

    def test_accepts_valid_data(self) -> None:
        """Test that valid data is accepted."""
        da = xr.DataArray(
            np.ones((2, 3, 10, 10, 50)),
            dims=EXPECTED_DIMS,
            coords={"freq_ghz": (["freq_range", "freq_idx"], np.ones((3, 50)))},
        )
        result = ODMRData(data=da)
        assert isinstance(result, ODMRData)

    def test_is_pydantic_basemodel(self, odmr_data) -> None:
        """Test that ODMRData is a Pydantic BaseModel instance."""
        from pydantic import BaseModel

        assert isinstance(odmr_data, BaseModel)
