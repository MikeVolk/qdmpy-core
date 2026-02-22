"""Tests for QDMpy.load() and Measurement.from_folder()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import xarray as xr

import qdmpy_core
from qdmpy_core.measurement import Measurement
from qdmpy_core.odmr.data import ODMRData
from qdmpy_core.odmr.manager import ODMR


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_POL, N_FRANGE, H, W, N_FREQ = 2, 2, 8, 8, 20


def _make_xr_data() -> xr.DataArray:
    """Minimal xr.DataArray with correct dims for ODMRData."""
    rng = np.random.default_rng(42)
    arr = rng.random((N_POL, N_FRANGE, H, W, N_FREQ))
    freq_ghz = np.linspace(2.82, 2.92, N_FREQ)
    return xr.DataArray(
        arr,
        dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
        coords={
            'polarity': ['neg', 'pos'],
            'freq_range': ['low', 'high'],
            'freq_ghz': (['freq_range', 'freq_idx'], np.stack([freq_ghz, freq_ghz])),
        },
    )


@pytest.fixture
def mock_loader_data() -> xr.DataArray:
    return _make_xr_data()


@pytest.fixture
def patched_from_folder(tmp_path: Path, mock_loader_data: xr.DataArray):
    """Context manager that patches the heavy dependencies of from_folder.

    Yields a dict with the mock objects so individual tests can inspect calls.
    """
    odmr_data = ODMRData(data=mock_loader_data)
    odmr = ODMR(odmr_data)
    odmr.process_data()  # sets up processed_data with scan_dimensions

    class _Ctx:
        def __init__(self):
            self.mocks: dict = {}

        def __enter__(self):
            self._patches = [
                patch('QDMpy.odmr.io.MatlabLoader.load', return_value=mock_loader_data),
                patch('QDMpy.measurement.os.listdir', return_value=[]),
                patch(
                    'QDMpy.measurement.get_image',
                    side_effect=__import__('QDMpy.exceptions', fromlist=['DataLoadError']).DataLoadError('no image'),
                ),
            ]
            for p in self._patches:
                p.start()
            return self

        def __exit__(self, *args):
            for p in self._patches:
                p.stop()

    return _Ctx()


# ---------------------------------------------------------------------------
# QDMpy.load() smoke test
# ---------------------------------------------------------------------------


def test_load_is_callable() -> None:
    assert callable(QDMpy.load)


def test_load_in_all() -> None:
    assert 'load' in QDMpy.__all__


# ---------------------------------------------------------------------------
# Measurement.from_folder — processor pipeline
# ---------------------------------------------------------------------------


class TestFromFolderProcessors:
    """Verify the correct processors are added based on keyword arguments."""

    def _run(self, tmp_path: Path, **kwargs) -> list[str]:
        """Return list of processor type names added to the ODMR instance."""
        xr_data = _make_xr_data()

        added_processors: list[str] = []

        original_add = ODMR.processor_manager.fget  # type: ignore[attr-defined]

        class SpyManager:
            def __init__(self):
                self._processors: list = []

            def add_processor(self, p):
                added_processors.append(type(p).__name__)
                self._processors.append(p)

        with (
            patch('QDMpy.odmr.io.MatlabLoader.load', return_value=xr_data),
            patch('QDMpy.measurement.os.listdir', return_value=[]),
            patch(
                'QDMpy.measurement.get_image',
                side_effect=__import__(
                    'QDMpy.exceptions', fromlist=['DataLoadError']
                ).DataLoadError('no image'),
            ),
        ):
            m = Measurement.from_folder(tmp_path, **kwargs)

        return [type(p).__name__ for p in m.odmr.processed_data.__class__.__mro__], m

    def _processors_on(self, tmp_path: Path, **kwargs) -> tuple[list[str], Measurement]:
        """Return (processor_names, measurement) by inspecting the ODMR manager."""
        xr_data = _make_xr_data()
        with (
            patch('QDMpy.odmr.io.MatlabLoader.load', return_value=xr_data),
            patch('QDMpy.measurement.os.listdir', return_value=[]),
            patch(
                'QDMpy.measurement.get_image',
                side_effect=__import__(
                    'QDMpy.exceptions', fromlist=['DataLoadError']
                ).DataLoadError('no image'),
            ),
        ):
            m = Measurement.from_folder(tmp_path, **kwargs)
        processors = [type(p).__name__ for p in m.odmr.processor_manager.processors]
        return processors, m

    def test_no_processors_by_default_except_normalization(self, tmp_path: Path) -> None:
        procs, _ = self._processors_on(tmp_path)
        # Default: normalize=True, fluorescence_correction=0.2 → 2 processors
        assert 'NormalizationProcessor' in procs
        assert 'FluorescenceCorrectionProcessor' in procs

    def test_bin_factor_adds_binning_processor(self, tmp_path: Path) -> None:
        procs, _ = self._processors_on(tmp_path, bin_factor=2)
        assert 'BinningProcessor' in procs

    def test_bin_factor_1_skips_binning(self, tmp_path: Path) -> None:
        procs, _ = self._processors_on(tmp_path, bin_factor=1)
        assert 'BinningProcessor' not in procs

    def test_normalize_false_skips_normalization(self, tmp_path: Path) -> None:
        procs, _ = self._processors_on(tmp_path, normalize=False)
        assert 'NormalizationProcessor' not in procs

    def test_fluorescence_none_skips_correction(self, tmp_path: Path) -> None:
        procs, _ = self._processors_on(tmp_path, fluorescence_correction=None)
        assert 'FluorescenceCorrectionProcessor' not in procs

    def test_fluorescence_factor_is_passed(self, tmp_path: Path) -> None:
        _, m = self._processors_on(tmp_path, fluorescence_correction=0.5)
        flu_proc = next(
            p
            for p in m.odmr.processor_manager.processors
            if type(p).__name__ == 'FluorescenceCorrectionProcessor'
        )
        assert flu_proc.correction_factor == 0.5


# ---------------------------------------------------------------------------
# Measurement.from_folder — image handling
# ---------------------------------------------------------------------------


class TestFromFolderImages:
    def _make(self, tmp_path: Path, folder_files=None, get_image_side_effect=None):
        """Run from_folder with controllable image loading."""
        from qdmpy_core.exceptions import DataLoadError as DLE

        xr_data = _make_xr_data()
        folder_files = folder_files or []
        get_image_side_effect = get_image_side_effect or DLE('no image')

        with (
            patch('QDMpy.odmr.io.MatlabLoader.load', return_value=xr_data),
            patch('QDMpy.measurement.os.listdir', return_value=folder_files),
            patch('QDMpy.measurement.get_image', side_effect=get_image_side_effect),
        ):
            return Measurement.from_folder(
                tmp_path, normalize=False, fluorescence_correction=None
            )

    def test_missing_images_fall_back_to_zeros(self, tmp_path: Path) -> None:
        m = self._make(tmp_path)
        assert isinstance(m.light_image, np.ndarray)
        assert isinstance(m.laser_image, np.ndarray)
        assert np.all(m.light_image == 0)
        assert np.all(m.laser_image == 0)

    def test_fallback_image_shape_matches_scan_dimensions(self, tmp_path: Path) -> None:
        m = self._make(tmp_path)
        assert m.light_image.shape == m.odmr.processed_data.scan_dimensions

    def test_light_files_filtered_by_keyword(self, tmp_path: Path) -> None:
        """Only files with 'light' in name are passed to get_image for light."""
        from qdmpy_core.exceptions import DataLoadError as DLE

        xr_data = _make_xr_data()
        captured_calls: list = []

        def capture_get_image(folder, lst):
            captured_calls.append(list(lst))
            raise DLE('no image')

        folder_files = ['light_ref.jpg', 'laser_ref.jpg', 'run_00000.mat', 'other.txt']
        with (
            patch('QDMpy.odmr.io.MatlabLoader.load', return_value=xr_data),
            patch('QDMpy.measurement.os.listdir', return_value=folder_files),
            patch('QDMpy.measurement.get_image', side_effect=capture_get_image),
        ):
            Measurement.from_folder(tmp_path, normalize=False, fluorescence_correction=None)

        # First call = light (should only include 'light_ref.jpg')
        assert captured_calls[0] == ['light_ref.jpg']
        # Second call = laser (should only include 'laser_ref.jpg')
        assert captured_calls[1] == ['laser_ref.jpg']

    def test_found_image_is_used(self, tmp_path: Path) -> None:
        dummy_img = np.ones((H, W))

        call_count = [0]

        def get_image_alternating(folder, lst):
            call_count[0] += 1
            return dummy_img

        m = self._make(
            tmp_path,
            folder_files=['light_img.jpg', 'laser_img.jpg'],
            get_image_side_effect=get_image_alternating,
        )
        np.testing.assert_array_equal(m.light_image, dummy_img)
        np.testing.assert_array_equal(m.laser_image, dummy_img)


# ---------------------------------------------------------------------------
# Measurement.from_folder — configuration pass-through
# ---------------------------------------------------------------------------


class TestFromFolderConfig:
    def _make(self, tmp_path: Path, **kwargs) -> Measurement:
        from qdmpy_core.exceptions import DataLoadError as DLE

        xr_data = _make_xr_data()
        with (
            patch('QDMpy.odmr.io.MatlabLoader.load', return_value=xr_data),
            patch('QDMpy.measurement.os.listdir', return_value=[]),
            patch('QDMpy.measurement.get_image', side_effect=DLE('no image')),
        ):
            return Measurement.from_folder(tmp_path, **kwargs)

    def test_returns_measurement(self, tmp_path: Path) -> None:
        m = self._make(tmp_path)
        assert isinstance(m, Measurement)

    def test_pixel_spacing_passed_through(self, tmp_path: Path) -> None:
        m = self._make(tmp_path, pixel_spacing=2e-6)
        assert m.pixel_spacing == 2e-6

    def test_model_passed_through(self, tmp_path: Path) -> None:
        m = self._make(tmp_path, model='ESR14N')
        assert m._fit_model == 'ESR14N'

    def test_default_output_directory(self, tmp_path: Path) -> None:
        m = self._make(tmp_path)
        assert m.output_directory == tmp_path / 'results'

    def test_custom_output_directory(self, tmp_path: Path) -> None:
        custom = tmp_path / 'custom_out'
        m = self._make(tmp_path, output_directory=custom)
        assert m.output_directory == custom
