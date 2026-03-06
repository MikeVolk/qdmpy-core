"""Test module for qdmpy.measurement.

These tests cover the Measurement class, which encapsulates all data and processing
related to a single QDM (Quantum Diamond Microscope) measurement.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from qdmpy.exceptions import DataNotLoadedError, DependencyError
from qdmpy.fitting.result import FitResult
from qdmpy.measurement import Measurement
from qdmpy.odmr.data import ODMRData
from qdmpy.odmr.folding import FoldedODMR, FoldingSettings
from qdmpy.odmr.manager import ODMR
from qdmpy.odmr.processors import BinningProcessor
from qdmpy.result import QDMResult


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


def _make_fit_result(model_name: str = "ESR15N") -> FitResult:
    """Create a minimal FitResult for mocking fit() return values."""
    return FitResult(
        parameters={
            "center": np.random.random(25),
            "chi2": np.random.random(25),
            "states": np.zeros(25, dtype=int),
        },
        scan_dimensions=(5, 5),
        pixel_spacing=4e-6,
        model_name=model_name,
        metadata={"fit_timestamp": "2026-01-01", "quality_metrics": {}},
    )


def _make_folded_odmr(n_pol: int = 2, ny: int = 5, nx: int = 5) -> FoldedODMR:
    """Build a minimal synthetic FoldedODMR object."""
    n_df = 20
    delta_f = np.linspace(0.005, 0.060, n_df)
    pol_labels = ["neg", "pos"][:n_pol]
    spec = np.ones((n_pol, ny, nx, n_df), dtype=np.float32)

    folded_da = xr.DataArray(
        spec,
        dims=("polarity", "y", "x", "freq_idx"),
        coords={"polarity": pol_labels, "delta_f_ghz": ("freq_idx", delta_f)},
    )
    anti_da = xr.DataArray(
        np.zeros_like(spec),
        dims=("polarity", "y", "x", "freq_idx"),
        coords={"polarity": pol_labels, "delta_f_ghz": ("freq_idx", delta_f)},
    )
    d_zfs_da = xr.DataArray(
        np.full((n_pol, ny, nx), 2.870),
        dims=("polarity", "y", "x"),
        coords={"polarity": pol_labels},
    )
    fold_residual_da = xr.DataArray(
        np.zeros((n_pol, ny, nx)),
        dims=("polarity", "y", "x"),
        coords={"polarity": pol_labels},
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
            fit_model="auto",
        )

        assert measurement.odmr is sample_odmr
        assert np.array_equal(measurement.light_image, light_image)
        assert np.array_equal(measurement.laser_image, laser_image)
        assert isinstance(measurement.output_directory, Path)
        assert measurement.output_directory == temp_output_dir
        assert measurement.pixel_spacing == 4e-6
        assert measurement._fit_model == "auto"
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

        with pytest.raises(DataNotLoadedError, match="ODMR instance has no raw data"):
            Measurement(
                odmr=empty_odmr,
                light_image=light_image,
                laser_image=laser_image,
                output_directory=temp_output_dir,
            )

    def test_string_representations(self, measurement) -> None:
        """Test the string representation methods."""
        str_repr = str(measurement)
        assert "Measurement" in str_repr
        assert "pixel_spacing" in str_repr

        repr_str = repr(measurement)
        assert "Measurement" in repr_str
        assert "light_image.shape" in repr_str

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

        measurement.metadata["test_key"] = "test_value"
        assert measurement.metadata["test_key"] == "test_value"

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
        assert m_auto._fit_model == "auto"

        m_14n = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",
        )
        assert m_14n._fit_model == "ESR14N"


class TestFitODMR:
    """Tests for Measurement.fit_odmr()."""

    def test_auto_model_detection(self, measurement) -> None:
        """Test fit_odmr with automatic model detection."""
        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result("ESR15N")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                result = measurement.fit_odmr()

            mock_fm_cls.assert_called_once()
            assert isinstance(result, QDMResult)
            assert isinstance(result.fit_result, FitResult)
            assert result.model_name == "ESR15N"

    def test_specific_model(self, measurement) -> None:
        """Test fit_odmr with a specific model name."""
        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result("ESR14N")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                result = measurement.fit_odmr(model_name="ESR14N")

            _, kwargs = mock_fm_cls.call_args
            assert kwargs.get("model_name") == "ESR14N"
            assert result.model_name == "ESR14N"

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
        with pytest.raises(DataNotLoadedError, match="ODMR data must be processed"):
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

        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            expected = _make_fit_result("ESRSINGLE")
            object.__setattr__(expected, "pixel_spacing", 5e-6)
            mock_fm.fit.return_value = expected

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                m.fit_odmr()

            _, fit_kwargs = mock_fm.fit.call_args
            assert fit_kwargs.get("pixel_spacing") == 5e-6

    def test_metadata_preservation(self, measurement) -> None:
        """Test that fit_odmr returns a FitResult with metadata from fit()."""
        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result("ESRSINGLE")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                result = measurement.fit_odmr()

            assert "fit_timestamp" in result.fit_result.metadata
            assert "quality_metrics" in result.fit_result.metadata

    def test_fit_model_wiring(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test that _fit_model is used as default when model_name is None."""
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",
        )

        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit.return_value = _make_fit_result("ESR14N")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                m.fit_odmr()  # no model_name arg

            _, kwargs = mock_fm_cls.call_args
            assert kwargs["model_name"] == "ESR14N"


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
        with pytest.raises(DataNotLoadedError, match="ODMR data must be processed"):
            m._validate_fit_prerequisites()

    def test_no_pygpufit(self, measurement) -> None:
        with (
            patch("qdmpy.settings.is_pygpufit_available", return_value=False),
            pytest.raises(DependencyError, match="pyGpufit is required"),
        ):
            measurement._validate_fit_prerequisites()


class TestFoldedODMRProperty:
    """Tests for Measurement.folded_odmr property."""

    def test_raises_before_fold(self, measurement) -> None:
        """Accessing folded_odmr before fold_odmr() raises DataNotLoadedError."""
        with pytest.raises(DataNotLoadedError, match="fold_odmr"):
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

        with patch("qdmpy.measurement.SpectralFolder") as mock_sf_cls:
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

        with patch("qdmpy.measurement.SpectralFolder") as mock_sf_cls:
            mock_sf = mock_sf_cls.return_value
            mock_sf.fold.return_value = mock_folded

            measurement.fold_odmr(settings=settings)

            call_args = mock_sf_cls.call_args
            assert call_args.args[1] is settings

    def test_fold_uses_default_settings(self, measurement) -> None:
        """fold_odmr() uses default FoldingSettings when none provided."""
        mock_folded = _make_folded_odmr()

        with patch("qdmpy.measurement.SpectralFolder") as mock_sf_cls:
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
        with pytest.raises(DataNotLoadedError, match="processed before folding"):
            m.fold_odmr()


class TestFitFoldedODMR:
    """Tests for Measurement.fit_folded_odmr()."""

    def test_uses_cached_folded(self, measurement) -> None:
        """fit_folded_odmr() uses cached folded data when no arg provided."""
        folded = _make_folded_odmr()
        measurement._folded_odmr = folded

        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result("ESRSINGLE+FOLDED")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                result = measurement.fit_folded_odmr()

            # Verify fit_folded was called with the cached folded data
            call_args = mock_fm.fit_folded.call_args
            assert call_args.args[0] is folded
            assert isinstance(result, QDMResult)

    def test_backward_compat_explicit_folded(self, measurement) -> None:
        """fit_folded_odmr(folded=...) still works for backward compat."""
        explicit_folded = _make_folded_odmr()

        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result("ESRSINGLE+FOLDED")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                result = measurement.fit_folded_odmr(folded=explicit_folded)

            call_args = mock_fm.fit_folded.call_args
            assert call_args.args[0] is explicit_folded
            assert isinstance(result, QDMResult)

    def test_raises_without_cached_or_explicit(self, measurement) -> None:
        """fit_folded_odmr() raises DataNotLoadedError with no folded data."""
        with (
            patch("qdmpy.settings.is_pygpufit_available", return_value=True),
            pytest.raises(DataNotLoadedError, match="fold_odmr"),
        ):
            measurement.fit_folded_odmr()

    def test_validates_gpu(self, measurement) -> None:
        """fit_folded_odmr() checks GPU availability."""
        measurement._folded_odmr = _make_folded_odmr()

        with (
            patch("qdmpy.settings.is_pygpufit_available", return_value=False),
            pytest.raises(DependencyError, match="pyGpufit is required"),
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
            fit_model="ESR14N",
        )
        m._folded_odmr = _make_folded_odmr()

        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result("ESR14N+FOLDED")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                m.fit_folded_odmr()

            _, kwargs = mock_fm_cls.call_args
            assert kwargs["model_name"] == "ESR14N"

    def test_explicit_model_overrides_default(self, measurement) -> None:
        """Explicit model_name overrides the _fit_model default."""
        measurement._folded_odmr = _make_folded_odmr()

        with patch("qdmpy.fitting.manager.FitManager") as mock_fm_cls:
            mock_fm = mock_fm_cls.return_value
            mock_fm.fit_folded.return_value = _make_fit_result("ESRSINGLE+FOLDED")

            with patch("qdmpy.settings.is_pygpufit_available", return_value=True):
                measurement.fit_folded_odmr(model_name="ESRSINGLE")

            _, kwargs = mock_fm_cls.call_args
            assert kwargs["model_name"] == "ESRSINGLE"


def _mock_from_folder(tmp_path, toml_content: str, **kwargs):
    """Helper: write toml_content to tmp_path/metadata.toml and call from_folder()."""
    (tmp_path / "metadata.toml").write_text(toml_content)
    mock_odmr_data = ODMRData.from_numpy(
        np.random.random((2, 2, 10, 10, 50)),
        (10, 10),
        np.linspace(2.87e9, 2.89e9, 50),
    )
    with (
        patch("qdmpy.odmr.io.MatlabLoader"),
        patch("qdmpy.odmr.data.ODMRData.from_loader", return_value=mock_odmr_data),
    ):
        return Measurement.from_folder(tmp_path, output_directory=tmp_path / "results", **kwargs)


class TestAcquisitionFallbacks:
    """Tests for [acquisition] section fallback behaviour in from_folder()."""

    def test_pixel_spacing_from_metadata(self, tmp_path) -> None:
        """[acquisition] pixel_spacing overrides the code default."""
        m = _mock_from_folder(tmp_path, "[acquisition]\npixel_spacing = 2.0e-6\n")
        assert m.pixel_spacing == pytest.approx(2.0e-6)

    def test_explicit_pixel_spacing_wins_over_metadata(self, tmp_path) -> None:
        """Explicit pixel_spacing keyword beats metadata.toml value."""
        m = _mock_from_folder(
            tmp_path,
            "[acquisition]\npixel_spacing = 2.0e-6\n",
            pixel_spacing=5e-6,
        )
        assert m.pixel_spacing == pytest.approx(5e-6)

    def test_model_from_metadata(self, tmp_path) -> None:
        """[acquisition] model sets _fit_model."""
        m = _mock_from_folder(tmp_path, '[acquisition]\nmodel = "ESR15N"\n')
        assert m._fit_model == "ESR15N"

    def test_explicit_model_wins_over_metadata(self, tmp_path) -> None:
        """Explicit model keyword beats metadata.toml value."""
        m = _mock_from_folder(tmp_path, '[acquisition]\nmodel = "ESR15N"\n', model="ESRSINGLE")
        assert m._fit_model == "ESRSINGLE"

    def test_code_defaults_when_no_acquisition_section(self, tmp_path) -> None:
        """Missing [acquisition] section falls back to code defaults."""
        m = _mock_from_folder(tmp_path, '[measurement]\nsample = "X"\n')
        assert m.pixel_spacing == pytest.approx(4e-6)
        assert m._fit_model == "auto"

    def test_fluorescence_correction_from_metadata(self, tmp_path) -> None:
        """[acquisition] fluorescence_correction is respected."""
        from qdmpy.odmr.processors import FluorescenceCorrectionProcessor

        m = _mock_from_folder(tmp_path, "[acquisition]\nfluorescence_correction = 0.5\n")
        fc_procs = [
            p
            for p in m.odmr.processor_manager.processors
            if isinstance(p, FluorescenceCorrectionProcessor)
        ]
        assert len(fc_procs) == 1
        assert fc_procs[0].correction_factor == pytest.approx(0.5)

    def test_explicit_none_disables_fluorescence_correction(self, tmp_path) -> None:
        """Passing fluorescence_correction=None skips the processor regardless of metadata."""
        from qdmpy.odmr.processors import FluorescenceCorrectionProcessor

        m = _mock_from_folder(
            tmp_path,
            "[acquisition]\nfluorescence_correction = 0.5\n",
            fluorescence_correction=None,
        )
        fc_procs = [
            p
            for p in m.odmr.processor_manager.processors
            if isinstance(p, FluorescenceCorrectionProcessor)
        ]
        assert len(fc_procs) == 0

    def test_measurement_section_stored_in_metadata(self, tmp_path) -> None:
        """[measurement] fields are accessible via measurement.metadata."""
        toml = (
            "[measurement]\n"
            'date = "2026-03-06"\n'
            'sample = "MIL2"\n'
            'subsample = "chipA"\n'
            'fov = "FOV1"\n'
            'operator = "alice"\n'
        )
        m = _mock_from_folder(tmp_path, toml)
        meas = m.metadata["measurement"]
        assert meas["date"] == "2026-03-06"
        assert meas["sample"] == "MIL2"
        assert meas["subsample"] == "chipA"
        assert meas["fov"] == "FOV1"
        assert meas["operator"] == "alice"

    def test_acquisition_section_also_in_metadata(self, tmp_path) -> None:
        """[acquisition] values are preserved in metadata alongside [measurement]."""
        toml = "[acquisition]\npixel_spacing = 2.0e-6\nbin_factor = 2\n"
        m = _mock_from_folder(tmp_path, toml)
        acq = m.metadata["acquisition"]
        assert acq["pixel_spacing"] == pytest.approx(2.0e-6)
        assert acq["bin_factor"] == 2


class TestMetadataTOML:
    """Tests for metadata.toml loading in from_folder()."""

    def test_metadata_toml_loaded_when_present(self, tmp_path) -> None:
        """Test that metadata.toml is loaded and populates measurement.metadata."""
        # Create a temporary folder with metadata.toml
        metadata_file = tmp_path / "metadata.toml"
        metadata_content = """
[experiment]
sample = "MIL2_FOV1"
operator = "Mike"
temperature_k = 295.0

[notes]
comment = "Test measurement"
"""
        metadata_file.write_text(metadata_content)

        # Mock the loader infrastructure
        with patch("qdmpy.odmr.io.MatlabLoader"):
            with patch("qdmpy.odmr.data.ODMRData.from_loader") as mock_from_loader:
                mock_odmr_data = ODMRData.from_numpy(
                    np.random.random((2, 2, 10, 10, 50)), (10, 10), np.linspace(2.87e9, 2.89e9, 50)
                )
                mock_from_loader.return_value = mock_odmr_data

                measurement = Measurement.from_folder(
                    tmp_path,
                    output_directory=tmp_path / "results",
                )

                assert "experiment" in measurement.metadata
                assert measurement.metadata["experiment"]["sample"] == "MIL2_FOV1"
                assert measurement.metadata["experiment"]["operator"] == "Mike"
                assert measurement.metadata["experiment"]["temperature_k"] == 295.0
                assert measurement.metadata["notes"]["comment"] == "Test measurement"

    def test_metadata_toml_missing_returns_empty_dict(self, tmp_path) -> None:
        """Test that missing metadata.toml returns empty metadata dict."""
        # Create a temporary folder WITHOUT metadata.toml
        with patch("qdmpy.odmr.io.MatlabLoader"):
            with patch("qdmpy.odmr.data.ODMRData.from_loader") as mock_from_loader:
                mock_odmr_data = ODMRData.from_numpy(
                    np.random.random((2, 2, 10, 10, 50)), (10, 10), np.linspace(2.87e9, 2.89e9, 50)
                )
                mock_from_loader.return_value = mock_odmr_data

                measurement = Measurement.from_folder(
                    tmp_path,
                    output_directory=tmp_path / "results",
                )

                assert measurement.metadata == {}

    def test_metadata_toml_with_nested_sections(self, tmp_path) -> None:
        """Test that nested TOML sections are preserved correctly."""
        metadata_file = tmp_path / "metadata.toml"
        metadata_content = """
[instrument]
microscope = "QDM-1"
laser_power = 250.5

[instrument.optics]
objective_na = 0.95
wavelength_nm = 532

[calibration]
date = "2026-01-15"
"""
        metadata_file.write_text(metadata_content)

        with patch("qdmpy.odmr.io.MatlabLoader"):
            with patch("qdmpy.odmr.data.ODMRData.from_loader") as mock_from_loader:
                mock_odmr_data = ODMRData.from_numpy(
                    np.random.random((2, 2, 10, 10, 50)), (10, 10), np.linspace(2.87e9, 2.89e9, 50)
                )
                mock_from_loader.return_value = mock_odmr_data

                measurement = Measurement.from_folder(
                    tmp_path,
                    output_directory=tmp_path / "results",
                )

                assert measurement.metadata["instrument"]["microscope"] == "QDM-1"
                assert measurement.metadata["instrument"]["laser_power"] == 250.5
                assert measurement.metadata["instrument"]["optics"]["objective_na"] == 0.95
                assert measurement.metadata["instrument"]["optics"]["wavelength_nm"] == 532
                assert measurement.metadata["calibration"]["date"] == "2026-01-15"

    def test_metadata_toml_malformed_returns_empty_dict(self, tmp_path) -> None:
        """Test that malformed TOML returns empty metadata dict without raising."""
        metadata_file = tmp_path / "metadata.toml"
        # Write invalid TOML (unclosed bracket)
        metadata_file.write_text("[experiment\nname = 'test'")

        with patch("qdmpy.odmr.io.MatlabLoader"):
            with patch("qdmpy.odmr.data.ODMRData.from_loader") as mock_from_loader:
                mock_odmr_data = ODMRData.from_numpy(
                    np.random.random((2, 2, 10, 10, 50)), (10, 10), np.linspace(2.87e9, 2.89e9, 50)
                )
                mock_from_loader.return_value = mock_odmr_data

                # Should not raise; should silently return empty metadata
                measurement = Measurement.from_folder(
                    tmp_path,
                    output_directory=tmp_path / "results",
                )

                assert measurement.metadata == {}
