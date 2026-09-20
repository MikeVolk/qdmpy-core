"""Test module for QDMpy.odmr.io."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from scipy.io import savemat

from qdmpy.exceptions import DataLoadError
from qdmpy.odmr.io import BaseLoader, MatlabLoader

# ---------------------------------------------------------------------------
# Synthetic .mat fixtures
#
# tests/data/ is gitignored (the smallest real FOV is 61 MB, and pre-commit caps
# files at 1500 kB), so every real-data test here skips on CI. These synthetic
# files are what actually exercises MatlabLoader.load() in the pipeline.
#
# They are written with scipy.io.savemat, i.e. MATLAB v5. That is the same path
# the real fixtures take: mat73 raises on v5 and io.py falls back to
# scipy.io.loadmat, which is why the "loaded with mat73" debug line at io.py:123
# is uncovered even when real data is present.
# ---------------------------------------------------------------------------


def _write_mat(
    path: Path,
    *,
    rows: int = 4,
    cols: int = 3,
    n_freq: int = 12,
    keys_to_drop: tuple[str, ...] = (),
    split_freqs: bool = True,
) -> None:
    """Write one minimal run_*.mat file.

    Each image stack is filled so that the value of every pixel equals its row
    index. After the loader reshapes C-order into (rows, cols) and flips y, that
    makes the y-axis orientation directly assertable.

    Args:
        path: Destination .mat file.
        rows: Image rows (imgNumRows).
        cols: Image columns (imgNumCols).
        n_freq: Frequencies per range.
        keys_to_drop: Keys to omit, for exercising the missing-key branches.
        split_freqs: If True write a concatenated 2*n_freq freqList (the low/high
            split branch); if False write exactly n_freq (the tile branch).
    """
    n_pixels = rows * cols
    row_of_pixel = np.arange(n_pixels) // cols
    stack = np.tile(row_of_pixel.astype(float), (n_freq, 1))

    freq_hz = (
        np.linspace(2.82e9, 2.92e9, 2 * n_freq)
        if split_freqs
        else np.linspace(2.82e9, 2.92e9, n_freq)
    )

    payload: dict[str, object] = {
        "imgStack1": stack,
        "imgStack2": stack + 100.0,
        "imgNumRows": rows,
        "imgNumCols": cols,
        "freqList": freq_hz,
        "numFreqs": n_freq,
    }
    for key in keys_to_drop:
        payload.pop(key)
    savemat(str(path), payload)


@pytest.fixture
def synthetic_folder(tmp_path: Path) -> Path:
    """A two-polarity folder of synthetic .mat files."""
    _write_mat(tmp_path / "run_00000.mat")
    _write_mat(tmp_path / "run_00001.mat")
    return tmp_path


@pytest.fixture
def test_data_path() -> str:
    """Return the path to the test data directory."""
    # Use a cropped real-data fixture for faster loader tests.
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "real_fov18x_fov5838_x78y24",
    )


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

        # Value checks that hold for any real FOV. Deliberately no assertion on
        # the spatial shape: tests/data/ is gitignored, so the fixture on any
        # given machine is whatever happens to be there.
        assert np.issubdtype(result.dtype, np.floating)
        assert list(result.polarity.values) == ["neg", "pos"]
        assert list(result.freq_range.values) == ["low", "high"]
        freq_ghz = result.coords["freq_ghz"].values
        assert freq_ghz.min() > 2.5, "frequencies look like Hz, not GHz"
        assert freq_ghz.max() < 3.2, "frequencies look like Hz, not GHz"

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


class TestMatlabLoaderSynthetic:
    """End-to-end MatlabLoader.load() coverage that also runs on CI.

    Replaces the former test_keys_missing_exception, which re-implemented
    io.py's try/except inside its own body and asserted its own raise --
    MatlabLoader was never called, so it would have passed unchanged if the
    production error handling had been deleted outright.
    """

    ROWS, COLS, N_FREQ = 4, 3, 12

    def test_load_structure(self, synthetic_folder: Path) -> None:
        """Dims, sizes and coordinate presence for a two-polarity folder."""
        result = MatlabLoader(data_folder=str(synthetic_folder)).load()

        assert result.dims == ("polarity", "freq_range", "y", "x", "freq_idx")
        assert result.sizes["polarity"] == 2
        assert result.sizes["freq_range"] == 2
        assert result.sizes["y"] == self.ROWS
        assert result.sizes["x"] == self.COLS
        assert result.sizes["freq_idx"] == self.N_FREQ
        assert "freq_ghz" in result.coords

    def test_load_uses_canonical_coords(self, synthetic_folder: Path) -> None:
        """Loader emits the QEP-025 labels, not positional ones."""
        result = MatlabLoader(data_folder=str(synthetic_folder)).load()

        assert list(result.polarity.values) == ["neg", "pos"]
        assert list(result.freq_range.values) == ["low", "high"]

    def test_load_flips_y_axis(self, synthetic_folder: Path) -> None:
        """Row index must be inverted relative to the MATLAB pixel order.

        The instrument stores rows bottom-to-top while the camera images use
        top-to-bottom, so io.py flips y to align them. Each synthetic pixel
        holds its own MATLAB row index, so after the flip row y must hold
        value (rows - 1 - y). Getting this backwards mirrors every field map
        against the LED image, which is invisible in a structural assertion.
        """
        result = MatlabLoader(data_folder=str(synthetic_folder)).load()

        for y in range(self.ROWS):
            expected = self.ROWS - 1 - y
            assert np.all(result.values[0, 0, y, :, :] == expected)

    def test_load_converts_hz_to_ghz(self, synthetic_folder: Path) -> None:
        """FreqList is Hz on disk; the DataArray must be GHz."""
        result = MatlabLoader(data_folder=str(synthetic_folder)).load()

        freq_ghz = result.coords["freq_ghz"].values
        assert freq_ghz.shape == (2, self.N_FREQ)
        assert freq_ghz.min() > 2.5
        assert freq_ghz.max() < 3.2
        # low branch sits below the high branch
        assert freq_ghz[0].mean() < freq_ghz[1].mean()

    def test_load_splits_concatenated_freqlist(self, synthetic_folder: Path) -> None:
        """A 2*numFreqs freqList is split into distinct low/high branches."""
        result = MatlabLoader(data_folder=str(synthetic_folder)).load()

        freq_ghz = result.coords["freq_ghz"].values
        assert not np.allclose(freq_ghz[0], freq_ghz[1])

    def test_load_tiles_single_freqlist(self, tmp_path: Path) -> None:
        """A freqList of exactly numFreqs is tiled across both ranges."""
        _write_mat(tmp_path / "run_00000.mat", split_freqs=False)
        _write_mat(tmp_path / "run_00001.mat", split_freqs=False)

        result = MatlabLoader(data_folder=str(tmp_path)).load()

        freq_ghz = result.coords["freq_ghz"].values
        assert freq_ghz.shape == (2, self.N_FREQ)
        np.testing.assert_allclose(freq_ghz[0], freq_ghz[1])

    def test_load_single_file_adds_polarity_axis(self, tmp_path: Path) -> None:
        """One run file yields n_pol=1 rather than a dropped dimension."""
        _write_mat(tmp_path / "run_00000.mat")

        result = MatlabLoader(data_folder=str(tmp_path)).load()

        assert result.sizes["polarity"] == 1
        assert list(result.polarity.values) == ["neg"]

    @pytest.mark.parametrize("missing_key", ["imgNumRows", "imgNumCols", "freqList"])
    def test_load_missing_key_raises(self, tmp_path: Path, missing_key: str) -> None:
        """A missing required key surfaces as DataLoadError from load()."""
        _write_mat(tmp_path / "run_00000.mat", keys_to_drop=(missing_key,))

        loader = MatlabLoader(data_folder=str(tmp_path))
        with pytest.raises(DataLoadError, match="Missing required key"):
            loader.load()
