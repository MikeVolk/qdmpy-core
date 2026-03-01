"""Test module for qdmpy_core.measurement.

These tests cover the Measurement class, which encapsulates all data and processing
related to a single QDM (Quantum Diamond Microscope) measurement.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from qdmpy_core.exceptions import DataNotLoadedError, DependencyError
from qdmpy_core.fitting.result import FitResult
from qdmpy_core.measurement import Measurement
from qdmpy_core.odmr.data import ODMRData
from qdmpy_core.odmr.folding import FoldedODMR, FoldingSettings
from qdmpy_core.odmr.manager import ODMR
from qdmpy_core.odmr.processors import BinningProcessor
from qdmpy_core.result import QDMResult


@pytest.fixture
def sample_data():
    """Create sample data for ODMR testing."""
    data = np.random.random((2, 2, 100, 50))  # (n_pol, n_frange, n_pixels, n_freqs)
    scan_dimensions = (10, 10)  # 10x10 grid
    frequencies = np.linspace(2.87e9, 2.89e9, 50)
    return data, scan_dimensions, frequencies


@pytest.fixture
def sample_odmr_data(sample_data):
    """Create a sample ODMRData instance for testing."""
    data, scan_dimensions, frequencies = sample_data
    return ODMRData.from_numpy(data, scan_dimensions, frequencies)


@pytest.fixture
def sample_odmr(sample_odmr_data):
    """Create a sample ODMR instance for testing."""
    odmr = ODMR(sample_odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.process_data()
    return odmr


@pytest.fixture
def sample_images():
    """Create sample light and laser images for testing."""
    light_image = np.random.random((10, 10))
    laser_image = np.random.random((10, 10))
    return light_image, laser_image


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    return tmp_path


@pytest.fixture
def measurement(sample_odmr, sample_images, temp_output_dir):
    """Create a standard Measurement for testing."""
    light_image, laser_image = sample_images
    return Measurement(
        odmr=sample_odmr,
        light_image=light_image,
        laser_image=laser_image,
        output_directory=temp_output_dir,
    )


def _make_fit_result(model_name: str = 'ESR15N') -> FitResult:
    """Create a minimal FitResult for mocking fit() return values."""
    return FitResult(
        parameters={
            'center': np.random.random(25),
            'chi2': np.random.random(25),
            'states': np.zeros(25, dtype=int),
        },
        scan_dimensions=(5, 5),
        pixel_spacing=4e-6,
        model_name=model_name,
        metadata={'fit_timestamp': '2026-01-01', 'quality_metrics': {}},
    )


def _make_folded_odmr(n_pol: int = 2, ny: int = 5, nx: int = 5) -> FoldedODMR:
    """Build a minimal synthetic FoldedODMR object."""
    n_df = 20
    delta_f = np.linspace(0.005, 0.060, n_df)
    pol_labels = ['neg', 'pos'][:n_pol]
    spec = np.ones((n_pol, ny, nx, n_df), dtype=np.float32)

    folded_da = xr.DataArray(
        spec,
        dims=('polarity', 'y', 'x', 'freq_idx'),
        coords={'polarity': pol_labels, 'delta_f_ghz': ('freq_idx', delta_f)},
    )
    anti_da = xr.DataArray(
        np.zeros_like(spec),
        dims=('polarity', 'y', 'x', 'freq_idx'),
        coords={'polarity': pol_labels, 'delta_f_ghz': ('freq_idx', delta_f)},
    )
    d_zfs_da = xr.DataArray(
        np.full((n_pol, ny, nx), 2.870),
        dims=('polarity', 'y', 'x'),
        coords={'polarity': pol_labels},
    )
    fold_residual_da = xr.DataArray(
        np.zeros((n_pol, ny, nx)),
        dims=('polarity', 'y', 'x'),
        coords={'polarity': pol_labels},
    )
    return FoldedODMR(
        folded_spectrum=folded_da,
        antisymmetric_spectrum=anti_da,
        d_zfs_map=d_zfs_da,
        fold_residual=fold_residual_da,
        settings=FoldingSettings(),
    )


class TestMeasurement:
    """Test class for Measurement."""

    def test_init(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test initialization with standard parameters."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=4e-6,
            fit_model='auto',
        )

        assert measurement.odmr is sample_odmr
        assert np.array_equal(measurement.light_image, light_image)
        assert np.array_equal(measurement.laser_image, laser_image)
        assert isinstance(measurement.output_directory, Path)
        assert measurement.output_directory == temp_output_dir
        assert measurement.pixel_spacing == 4e-6
        assert measurement._fit_model == 'auto'
        assert isinstance(measurement.metadata, dict)
        assert len(measurement.metadata) == 0
        assert measurement._outliers is not None
        assert measurement._folded_odmr is None

    def test_init_with_unprocessed_odmr(
        self, sample_odmr_data, sample_images, temp_output_dir
    ) -> None:
        """Test initialization with an ODMR instance that hasn't been processed."""
        light_image, laser_image = sample_images
        odmr = ODMR(sample_odmr_data)

        measurement = Measurement(
            odmr=odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        assert measurement.odmr is odmr

    def test_init_with_no_odmr_data(self, sample_images, temp_output_dir) -> None:
        """Test initialization with an ODMR instance that has no data."""
        light_image, laser_image = sample_images
        empty_odmr = ODMR()

        with pytest.raises(DataNotLoadedError, match='ODMR instance has no raw data'):
            Measurement(
                odmr=empty_odmr,
                light_image=light_image,
                laser_image=laser_image,
                output_directory=temp_output_dir,
            )

    def test_string_representations(self, measurement) -> None:
        """Test the string representation methods."""
        str_repr = str(measurement)
        assert 'Measurement' in str_repr
        assert 'pixel_spacing' in str_repr

        repr_str = repr(measurement)
        assert 'Measurement' in repr_str
        assert 'light_image.shape' in repr_str

    def test_with_different_pixel_spacing(
        self, sample_odmr, sample_images, temp_output_dir
    ) -> None:
        """Test initialization with different pixel spacing values."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=1e-6,
        )
        assert measurement.pixel_spacing == 1e-6

    def test_with_string_output_directory(
        self, sample_odmr, sample_images, temp_output_dir
    ) -> None:
        """Test initialization with string output directory."""
        light_image, laser_image = sample_images
        output_dir_str = str(temp_output_dir)

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=output_dir_str,
        )
        assert isinstance(measurement.output_directory, Path)
        assert str(measurement.output_directory) == output_dir_str

    def test_metadata_dictionary(self, measurement) -> None:
        """Test that the metadata dictionary works as expected."""
        assert isinstance(measurement.metadata, dict)
        assert len(measurement.metadata) == 0

        measurement.metadata['test_key'] = 'test_value'
        assert measurement.metadata['test_key'] == 'test_value'

    def test_outliers_property(self, measurement, sample_odmr) -> None:
        """Test the _outliers attribute."""
        assert measurement._outliers is not None
        assert isinstance(measurement._outliers, np.ndarray)
        assert measurement._outliers.shape == sample_odmr.raw_data.shape
        assert measurement._outliers.dtype == bool

    def test_fit_model_attribute(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test the _fit_model attribute."""
        light_image, laser_image = sample_images
        m_auto = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        assert m_auto._fit_model == 'auto'

        m_14n = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model='ESR14N',
        )
        assert m_14n._fit_model == 'ESR14N'


class TestFitODMR:
    """Tests for Measurement.fit_odmr()."""

    def test_auto_model_detection(self, measurement) -> None:
        """Test fit_odmr with automatic model detection."""
        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result('ESR15N')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                result = measurement.fit_odmr()

            mock_fm_cls.assert_called_once()
            assert isinstance(result, QDMResult)
            assert isinstance(result.fit_result, FitResult)
            assert result.model_name == 'ESR15N'

    def test_specific_model(self, measurement) -> None:
        """Test fit_odmr with a specific model name."""
        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result('ESR14N')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                result = measurement.fit_odmr(model_name='ESR14N')

            _, kwargs = mock_fm_cls.call_args
            assert kwargs.get('model_name') == 'ESR14N'
            assert result.model_name == 'ESR14N'

    def test_no_processed_data(self, sample_odmr_data, sample_images, temp_output_dir) -> None:
        """Test fit_odmr with ODMR that has no processed data."""
        light_image, laser_image = sample_images
        odmr = ODMR(sample_odmr_data)
        m = Measurement(
            odmr=odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        with pytest.raises(DataNotLoadedError, match='ODMR data must be processed'):
            m.fit_odmr()

    def test_pixel_spacing_passed(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test that fit_odmr passes pixel_spacing to fit()."""
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=5e-6,
        )

        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            expected = _make_fit_result('ESRSINGLE')
            object.__setattr__(expected, 'pixel_spacing', 5e-6)
            mock_fm.fit.return_value = expected

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                m.fit_odmr()

            _, fit_kwargs = mock_fm.fit.call_args
            assert fit_kwargs.get('pixel_spacing') == 5e-6

    def test_metadata_preservation(self, measurement) -> None:
        """Test that fit_odmr returns a FitResult with metadata from fit()."""
        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result('ESRSINGLE')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                result = measurement.fit_odmr()

            assert 'fit_timestamp' in result.fit_result.metadata
            assert 'quality_metrics' in result.fit_result.metadata

    def test_fit_model_wiring(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test that _fit_model is used as default when model_name is None."""
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model='ESR14N',
        )

        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result('ESR14N')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                m.fit_odmr()  # no model_name arg

            _, kwargs = mock_fm_cls.call_args
            assert kwargs['model_name'] == 'ESR14N'


class TestValidateFitPrerequisites:
    """Tests for Measurement._validate_fit_prerequisites."""

    def test_no_processed_data(self, sample_odmr_data, sample_images, temp_output_dir) -> None:
        light_image, laser_image = sample_images
        odmr = ODMR(sample_odmr_data)
        m = Measurement(
            odmr=odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        with pytest.raises(DataNotLoadedError, match='ODMR data must be processed'):
            m._validate_fit_prerequisites()

    def test_no_pygpufit(self, measurement) -> None:
        with (
            patch('qdmpy_core.settings.is_pygpufit_available', return_value=False),
            pytest.raises(DependencyError, match='pyGpufit is required'),
        ):
            measurement._validate_fit_prerequisites()


class TestFoldedODMRProperty:
    """Tests for Measurement.folded_odmr property."""

    def test_raises_before_fold(self, measurement) -> None:
        """Accessing folded_odmr before fold_odmr() raises DataNotLoadedError."""
        with pytest.raises(DataNotLoadedError, match='fold_odmr'):
            _ = measurement.folded_odmr

    def test_returns_cached_after_fold(self, measurement) -> None:
        """After fold_odmr(), folded_odmr returns the cached FoldedODMR."""
        folded = _make_folded_odmr()
        measurement._folded_odmr = folded
        assert measurement.folded_odmr is folded


class TestFoldODMR:
    """Tests for Measurement.fold_odmr()."""

    def test_fold_creates_and_caches(self, measurement) -> None:
        """fold_odmr() creates a FoldedODMR and caches it."""
        mock_folded = _make_folded_odmr()

        with patch('qdmpy_core.measurement.SpectralFolder') as mock_sf_cls:
            mock_sf = mock_sf_cls.return_value
            mock_sf.fold.return_value = mock_folded

            result = measurement.fold_odmr()

            assert result is mock_folded
            assert measurement._folded_odmr is mock_folded
            assert measurement.folded_odmr is mock_folded

    def test_fold_passes_settings(self, measurement) -> None:
        """fold_odmr() passes custom settings to SpectralFolder."""
        settings = FoldingSettings(bin_factor=4, search_steps=101)
        mock_folded = _make_folded_odmr()

        with patch('qdmpy_core.measurement.SpectralFolder') as mock_sf_cls:
            mock_sf = mock_sf_cls.return_value
            mock_sf.fold.return_value = mock_folded

            measurement.fold_odmr(settings=settings)

            call_args = mock_sf_cls.call_args
            assert call_args.args[1] is settings

    def test_fold_uses_default_settings(self, measurement) -> None:
        """fold_odmr() uses default FoldingSettings when none provided."""
        mock_folded = _make_folded_odmr()

        with patch('qdmpy_core.measurement.SpectralFolder') as mock_sf_cls:
            mock_sf = mock_sf_cls.return_value
            mock_sf.fold.return_value = mock_folded

            measurement.fold_odmr()

            call_args = mock_sf_cls.call_args
            settings_arg = call_args.args[1]
            assert isinstance(settings_arg, FoldingSettings)
            assert settings_arg.bin_factor == 8  # default

    def test_fold_raises_without_processed_data(
        self, sample_odmr_data, sample_images, temp_output_dir
    ) -> None:
        """fold_odmr() raises DataNotLoadedError if data hasn't been processed."""
        light_image, laser_image = sample_images
        odmr = ODMR(sample_odmr_data)
        m = Measurement(
            odmr=odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        with pytest.raises(DataNotLoadedError, match='processed before folding'):
            m.fold_odmr()


class TestFitFoldedODMR:
    """Tests for Measurement.fit_folded_odmr()."""

    def test_uses_cached_folded(self, measurement) -> None:
        """fit_folded_odmr() uses cached folded data when no arg provided."""
        folded = _make_folded_odmr()
        measurement._folded_odmr = folded

        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result('ESRSINGLE+FOLDED')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                result = measurement.fit_folded_odmr()

            # Verify fit_folded was called with the cached folded data
            call_args = mock_fm.fit_folded.call_args
            assert call_args.args[0] is folded
            assert isinstance(result, QDMResult)

    def test_backward_compat_explicit_folded(self, measurement) -> None:
        """fit_folded_odmr(folded=...) still works for backward compat."""
        explicit_folded = _make_folded_odmr()

        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result('ESRSINGLE+FOLDED')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                result = measurement.fit_folded_odmr(folded=explicit_folded)

            call_args = mock_fm.fit_folded.call_args
            assert call_args.args[0] is explicit_folded
            assert isinstance(result, QDMResult)

    def test_raises_without_cached_or_explicit(self, measurement) -> None:
        """fit_folded_odmr() raises DataNotLoadedError with no folded data."""
        with (
            patch('qdmpy_core.settings.is_pygpufit_available', return_value=True),
            pytest.raises(DataNotLoadedError, match='fold_odmr'),
        ):
            measurement.fit_folded_odmr()

    def test_validates_gpu(self, measurement) -> None:
        """fit_folded_odmr() checks GPU availability."""
        measurement._folded_odmr = _make_folded_odmr()

        with (
            patch('qdmpy_core.settings.is_pygpufit_available', return_value=False),
            pytest.raises(DependencyError, match='pyGpufit is required'),
        ):
            measurement.fit_folded_odmr()

    def test_fit_model_wiring(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """fit_folded_odmr() uses _fit_model when model_name is None."""
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model='ESR14N',
        )
        m._folded_odmr = _make_folded_odmr()

        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result('ESR14N+FOLDED')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                m.fit_folded_odmr()

            _, kwargs = mock_fm_cls.call_args
            assert kwargs['model_name'] == 'ESR14N'

    def test_explicit_model_overrides_default(self, measurement) -> None:
        """Explicit model_name overrides the _fit_model default."""
        measurement._folded_odmr = _make_folded_odmr()

        with patch('qdmpy_core.fitting.manager.FitManager') as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result('ESRSINGLE+FOLDED')

            with patch('qdmpy_core.settings.is_pygpufit_available', return_value=True):
                measurement.fit_folded_odmr(model_name='ESRSINGLE')

            _, kwargs = mock_fm_cls.call_args
            assert kwargs['model_name'] == 'ESRSINGLE'
