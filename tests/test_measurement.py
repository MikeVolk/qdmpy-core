"""Test module for QDMpy.measurement.

These tests cover the Measurement class, which encapsulates all data and processing
related to a single QDM (Quantum Diamond Microscope) measurement.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from QDMpy.exceptions import DataNotLoadedError, DependencyError
from QDMpy.fitting.result import FitResult
from QDMpy.measurement import Measurement
from QDMpy.odmr.data import ODMRData
from QDMpy.odmr.manager import ODMR
from QDMpy.odmr.processors import BinningProcessor


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
        assert np.array_equal(measurement.light_image, light_image)
        assert np.array_equal(measurement.laser_image, laser_image)
        assert hasattr(measurement, "_outliers")

    def test_init_with_no_odmr_data(self, sample_images, temp_output_dir) -> None:
        """Test initialization with an ODMR instance that has no data."""
        light_image, laser_image = sample_images
        empty_odmr = ODMR()

        with pytest.raises(DataNotLoadedError) as excinfo:
            Measurement(
                odmr=empty_odmr,
                light_image=light_image,
                laser_image=laser_image,
                output_directory=temp_output_dir,
            )

        assert "ODMR instance has no raw data" in str(excinfo.value)

    def test_string_representations(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test the string representation methods."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        str_repr = str(measurement)
        assert "Measurement" in str_repr
        assert str(temp_output_dir) in str_repr
        assert "pixel_spacing" in str_repr

        repr_str = repr(measurement)
        assert "Measurement" in repr_str
        assert "light_image.shape" in repr_str
        assert "laser_image.shape" in repr_str

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
        assert measurement.pixel_spacing != 4e-6

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

    def test_metadata_dictionary(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test that the metadata dictionary works as expected."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        assert isinstance(measurement.metadata, dict)
        assert len(measurement.metadata) == 0

        measurement.metadata["test_key"] = "test_value"
        assert "test_key" in measurement.metadata
        assert measurement.metadata["test_key"] == "test_value"

        measurement.metadata.update({"another_key": 123})
        assert measurement.metadata["another_key"] == 123

    def test_outliers_property(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test the _outliers attribute."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        assert measurement._outliers is not None
        assert isinstance(measurement._outliers, np.ndarray)
        assert measurement._outliers.shape == sample_odmr.raw_data.shape
        assert measurement._outliers.dtype == bool

    def test_fit_model_attribute(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test the _fit_model attribute."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        assert measurement._fit_model == "auto"

        measurement2 = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",
        )
        assert measurement2._fit_model == "ESR14N"

    def _make_fit_result(self, model_name: str = "ESR15N") -> FitResult:
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

    def test_fit_odmr_auto_model_detection(
        self, sample_odmr, sample_images, temp_output_dir
    ) -> None:
        """Test fit_odmr with automatic model detection."""
        light_image, laser_image = sample_images

        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="auto",
        )

        with patch("QDMpy.fitting.guess.guess_model") as mock_guess:
            mock_model = type("MockModel", (), {"name": "ESR15N"})()
            mock_guess.return_value = mock_model

            with patch("QDMpy.fitting.manager.FitManager") as mock_fit_manager:
                mock_fit_instance = mock_fit_manager.return_value
                expected_result = self._make_fit_result("ESR15N")
                mock_fit_instance.fit.return_value = expected_result

                with patch("QDMpy.is_pygpufit_available", return_value=True):
                    result = measurement.fit_odmr()

                mock_fit_manager.assert_called_once()

                assert isinstance(result, FitResult)
                assert result.model_name == "ESR15N"

    def test_fit_odmr_specific_model(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test fit_odmr with a specific model name."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESR14N",
        )

        with patch("QDMpy.fitting.manager.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            expected_result = self._make_fit_result("ESR14N")
            mock_fit_instance.fit.return_value = expected_result

            with patch("QDMpy.is_pygpufit_available", return_value=True):
                result = measurement.fit_odmr(model_name="ESR14N")

            _, kwargs = mock_fit_manager.call_args
            assert kwargs.get("model_name") == "ESR14N"

            assert isinstance(result, FitResult)
            assert result.model_name == "ESR14N"

    def test_fit_odmr_no_processed_data(
        self, sample_odmr_data, sample_images, temp_output_dir
    ) -> None:
        """Test fit_odmr with ODMR that has no processed data."""
        light_image, laser_image = sample_images
        unprocessed_odmr = ODMR(sample_odmr_data)

        measurement = Measurement(
            odmr=unprocessed_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        with pytest.raises(DataNotLoadedError, match="ODMR data must be processed"):
            measurement.fit_odmr()

    def test_fit_odmr_data_extraction(self, sample_odmr, sample_images, temp_output_dir) -> None:
        """Test that fit_odmr passes pixel_spacing to fit()."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            pixel_spacing=5e-6,
        )

        with patch("QDMpy.fitting.manager.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            expected_result = self._make_fit_result("ESRSINGLE")
            # Override pixel_spacing to verify it was passed
            object.__setattr__(expected_result, "pixel_spacing", 5e-6)
            mock_fit_instance.fit.return_value = expected_result

            with patch("QDMpy.is_pygpufit_available", return_value=True):
                measurement.fit_odmr()

            _, fit_kwargs = mock_fit_instance.fit.call_args
            assert fit_kwargs.get("pixel_spacing") == 5e-6

    def test_fit_odmr_metadata_preservation(
        self, sample_odmr, sample_images, temp_output_dir
    ) -> None:
        """Test that fit_odmr returns a FitResult with metadata from fit()."""
        light_image, laser_image = sample_images
        measurement = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )

        with patch("QDMpy.fitting.manager.FitManager") as mock_fit_manager:
            mock_fit_instance = mock_fit_manager.return_value
            expected_result = self._make_fit_result("ESRSINGLE")
            mock_fit_instance.fit.return_value = expected_result

            with patch("QDMpy.is_pygpufit_available", return_value=True):
                result = measurement.fit_odmr()

            assert "fit_timestamp" in result.metadata
            assert "quality_metrics" in result.metadata


class TestDetectModel:
    """Tests for Measurement._detect_model."""

    def test_explicit_model_name(self, sample_odmr, sample_images, temp_output_dir) -> None:
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        assert m._detect_model("ESR14N") == "ESR14N"

    def test_auto_detect_success(self, sample_odmr, sample_images, temp_output_dir) -> None:
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        with patch("QDMpy.fitting.guess.guess_model") as mock_guess:
            mock_guess.return_value = type("M", (), {"name": "ESR15N"})()
            assert m._detect_model(None) == "ESR15N"

    def test_auto_detect_fallback(self, sample_odmr, sample_images, temp_output_dir) -> None:
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
            fit_model="ESRSINGLE",
        )
        with patch("QDMpy.fitting.guess.guess_model", side_effect=RuntimeError("fail")):
            assert m._detect_model(None) == "ESRSINGLE"


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

    def test_no_pygpufit(self, sample_odmr, sample_images, temp_output_dir) -> None:
        light_image, laser_image = sample_images
        m = Measurement(
            odmr=sample_odmr,
            light_image=light_image,
            laser_image=laser_image,
            output_directory=temp_output_dir,
        )
        with patch("QDMpy.is_pygpufit_available", return_value=False):
            with pytest.raises(DependencyError, match="pyGpufit is required"):
                m._validate_fit_prerequisites()
