"""Unit tests for the fit module.

This test suite provides comprehensive testing for the FitManager class and related
functionality in the QDMpy.fit module.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.typing import NDArray

from qdmpy.exceptions import (
    DataValidationError,
    ModelNotFoundError,
    ModelNotResolvedError,
    ParameterError,
)
from qdmpy.fitting.constraints import CONSTRAINT_TYPES, ConstraintManager
from qdmpy.fitting.guesser import ParameterGuesser
from qdmpy.fitting.manager import FitManager
from qdmpy.fitting.models import ESR14N, ESR15N, ESRSINGLE, Model, ModelRegistry
from qdmpy.fitting.result import FitResult
from qdmpy.settings import (
    FitSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)
from qdmpy.testing import FakeFitBackend

# Mock settings for tests (center/width values in GHz, matching default settings convention)
MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(
        estimator="LSE",
        max_number_iterations=100,
        tolerance=1e-6,
    ),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.8,
            center_max=2.9,
            center_type="FREE",
            width_min=0.001,
            width_max=0.01,
            width_type="FREE",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="FREE",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="FREE",
        )
    ),
)


def _make_xr_data(numpy_4d: np.ndarray) -> xr.DataArray:
    """Convert 4D numpy (n_pol, n_frange, n_pixel, n_freq) to 5D xr.DataArray.

    Assumes pixels can be arranged as a square grid.
    """
    n_pol, n_frange, n_pixel, n_freq = numpy_4d.shape
    side = int(np.sqrt(n_pixel))
    assert side * side == n_pixel, f"n_pixel={n_pixel} is not a perfect square"

    data_5d = numpy_4d.reshape(n_pol, n_frange, side, side, n_freq)
    freq_ghz = np.tile(np.linspace(2.87, 2.88, n_freq), (n_frange, 1))

    return xr.DataArray(
        data_5d,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": [f"pol_{i}" for i in range(n_pol)],
            "freq_range": [f"frange_{i}" for i in range(n_frange)],
            "freq_ghz": (["freq_range", "freq_idx"], freq_ghz),
        },
    )


@pytest.fixture
def sample_numpy_data():
    """Create sample 4D numpy data (n_pol, n_frange, n_pixel, n_freq)."""
    data = np.ones((2, 1, 4, 10))

    for pol in range(2):
        for pixel in range(4):
            center_idx = 5
            for i in range(10):
                contrast = 0.1 * (pixel + 1)
                width = 1.0 + 0.5 * pixel
                x = i - center_idx
                data[pol, 0, pixel, i] = 1.0 - contrast * (width**2 / (x**2 + width**2))

    return data


@pytest.fixture
def sample_data(sample_numpy_data):
    """Create sample xr.DataArray for testing the FitManager."""
    return _make_xr_data(sample_numpy_data)


@pytest.fixture
def sample_frequencies():
    """Create sample frequencies for testing."""
    return np.linspace(2.87, 2.88, 10)


class TestFitInitialization:
    """Test initialization of the FitManager class."""

    def test_init_default(self) -> None:
        """Test initialization with default model (ESR14N)."""
        fit = FitManager(settings=MOCK_SETTINGS)
        assert fit.model_name == "ESR14N"
        assert isinstance(fit.model, ESR14N)

    def test_init_auto_defers_model(self) -> None:
        """Test that auto mode leaves model unresolved until fit() is called."""
        fit = FitManager(model_name="auto", settings=MOCK_SETTINGS)
        assert fit.model is None
        assert fit.model_name == "auto"

    def test_init_with_specific_model(self) -> None:
        """Test initialization with a specific model."""
        fit = FitManager(model_name="ESR14N", settings=MOCK_SETTINGS)
        assert fit.model_name == "ESR14N"
        assert isinstance(fit.model, ESR14N)

    def test_init_with_invalid_model(self) -> None:
        """Test initialization with an invalid model name."""
        with pytest.raises(ModelNotFoundError) as excinfo:
            FitManager(model_name="INVALID_MODEL", settings=MOCK_SETTINGS)
        assert "Unknown model" in str(excinfo.value)

    def test_init_with_custom_constraints(self) -> None:
        """Test initialization with custom constraints."""
        constraints = {"center": {"vmin": 2.87, "vmax": 2.88, "constraint_type": "LOWER_UPPER"}}

        fit = FitManager(model_name="ESRSINGLE", constraints=constraints, settings=MOCK_SETTINGS)

        assert fit.constraints["center"].vmin == 2.87
        assert fit.constraints["center"].vmax == 2.88
        assert fit.constraints["center"].constraint_type == "LOWER_UPPER"

    def test_auto_mode_raises_on_constraints_access(self) -> None:
        """Test that accessing constraints in auto mode before fit() raises ModelNotResolvedError."""
        fit = FitManager(model_name="auto", settings=MOCK_SETTINGS)
        with pytest.raises(ModelNotResolvedError, match="not yet resolved"):
            _ = fit.constraints


class TestFitProperties:
    """Test property getters and setters of the FitManager class."""

    def test_model_name_property(self) -> None:
        """Test model_name property is read-only after construction."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        assert fit.model_name == "ESRSINGLE"

        # Model is immutable; attempting to change it raises AttributeError
        with pytest.raises(AttributeError, match="no setter"):
            fit.model_name = "ESR15N"

        # To use a different model, create a new FitManager
        fit2 = FitManager(model_name="ESR15N", settings=MOCK_SETTINGS)
        assert fit2.model_name == "ESR15N"
        assert isinstance(fit2.model, ESR15N)

    def test_parameter_names_raises_in_auto_mode(self) -> None:
        """Test that parameter_names raises ModelNotResolvedError in unresolved auto mode."""
        fit = FitManager(model_name="auto", settings=MOCK_SETTINGS)
        with pytest.raises(ModelNotResolvedError, match="not yet resolved"):
            _ = fit.parameter_names


class TestConstraintsMethods:
    """Test constraint-related methods of the FitManager class."""

    def test_set_constraints(self) -> None:
        """Test set_constraints method."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        fit.set_constraints("center", vmin=2.85, vmax=2.90, constraint_type="LOWER_UPPER")

        assert fit.constraints["center"].vmin == 2.85
        assert fit.constraints["center"].vmax == 2.90
        assert fit.constraints["center"].constraint_type == "LOWER_UPPER"

    def test_set_constraints_with_numeric_type(self) -> None:
        """Test set_constraints with numeric constraint type."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        fit.set_constraints("width", vmin=1e6, constraint_type=1)

        assert fit.constraints["width"].vmin == 1e6
        assert fit.constraints["width"].constraint_type == "LOWER"

    def test_set_constraints_invalid_type(self) -> None:
        """Test set_constraints with invalid constraint type."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)

        with pytest.raises(ParameterError):
            fit.set_constraints("center", constraint_type="INVALID_TYPE")

        with pytest.raises(ParameterError):
            fit.set_constraints("center", constraint_type=10)

    def test_set_free_constraints(self) -> None:
        """Test set_free_constraints method."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        fit.set_constraints("center", vmin=2.85, vmax=2.90, constraint_type="LOWER_UPPER")
        fit.set_constraints("width", vmin=0.001, constraint_type="LOWER")

        fit.set_free_constraints()

        for param in fit.parameter_names:
            assert fit.constraints[param].constraint_type == "FREE"

    def test_get_constraints_array(self) -> None:
        """Test get_constraints_array method."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        fit.set_constraints("center", vmin=2.85, vmax=2.90)
        fit.set_constraints("width", vmin=0.001, vmax=0.01)

        constraints_array = fit.get_constraints_array(2)

        # ESRSINGLE params: center, width, contrast, offset -> 4 params x 2 = 8 columns
        assert constraints_array.shape == (2, 8)

        # All values stay in GHz (no Hz conversion — QEP-018)
        expected_first_row = [
            fit.constraints["center"].vmin,
            fit.constraints["center"].vmax,
            fit.constraints["width"].vmin,
            fit.constraints["width"].vmax,
            fit.constraints["contrast"].vmin,
            fit.constraints["contrast"].vmax,
            fit.constraints["offset"].vmin,
            fit.constraints["offset"].vmax,
        ]
        assert_array_almost_equal(constraints_array[0], expected_first_row)
        assert_array_almost_equal(constraints_array[0], constraints_array[1])

    def test_get_constraint_types(self) -> None:
        """Test get_constraint_types method."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        model_params = fit.parameter_names

        for i, param in enumerate(model_params):
            constraint_type = CONSTRAINT_TYPES[i % len(CONSTRAINT_TYPES)]
            fit.set_constraints(param, constraint_type=constraint_type)

        constraint_types = fit.get_constraint_types()

        assert len(constraint_types) == len(model_params)
        used_types = set(constraint_types)
        assert len(used_types) > 0
        assert all(t in range(len(CONSTRAINT_TYPES)) for t in used_types)


def _make_test_model(
    params: list[str],
    param_types: dict[str, str],
    freq_params: list[str],
) -> Model:
    """Create a concrete Model subclass for ConstraintManager tests."""

    class _TestModel(Model):
        def __init__(self) -> None:
            super().__init__("TEST", 1, params)

        @property
        def parameter_types(self) -> dict[str, str]:
            return param_types

        @property
        def frequency_parameters(self) -> list[str]:
            return freq_params

        def func(self, x, parameters):
            return x

    return _TestModel()


class TestConstraintManager:
    """Test the ConstraintManager class."""

    def test_initialization(self) -> None:
        """Test initialization of the ConstraintManager."""
        model = _make_test_model(
            ["center", "width_0", "contrast", "offset"],
            {"center": "center", "width_0": "width", "contrast": "contrast", "offset": "offset"},
            ["center"],
        )
        settings = ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.8,
            center_max=2.9,
            center_type="FREE",
            width_min=0.001,
            width_max=0.01,
            width_type="LOWER",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="UPPER",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="LOWER_UPPER",
        )

        constraint_manager = ConstraintManager(model, settings)
        constraints = constraint_manager.get_constraints()
        assert len(constraints) == 4

        assert constraints["center"].vmin == 2.8
        assert constraints["center"].vmax == 2.9
        assert constraints["center"].constraint_type == "FREE"
        assert constraints["center"].unit == "GHz"

        assert constraints["width_0"].vmin == 0.001
        assert constraints["width_0"].vmax == 0.01
        assert constraints["width_0"].constraint_type == "LOWER"
        assert constraints["width_0"].unit == "a.u."

    def test_set_constraint(self) -> None:
        """Test setting constraints."""
        model = _make_test_model(
            ["center", "width_0", "contrast", "offset"],
            {"center": "center", "width_0": "width", "contrast": "contrast", "offset": "offset"},
            ["center"],
        )
        settings = ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.8,
            center_max=2.9,
            center_type="FREE",
            width_min=0.001,
            width_max=0.01,
            width_type="FREE",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="FREE",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="FREE",
        )

        constraint_manager = ConstraintManager(model, settings)
        constraint_manager.set_constraint(
            "center", vmin=2.85, vmax=2.88, constraint_type="LOWER_UPPER"
        )

        constraints = constraint_manager.get_constraints()
        assert constraints["center"].vmin == 2.85
        assert constraints["center"].vmax == 2.88
        assert constraints["center"].constraint_type == "LOWER_UPPER"

        constraint_manager.set_constraint("width_0", vmin=0.002)
        constraints = constraint_manager.get_constraints()
        assert constraints["width_0"].vmin == 0.002
        assert constraints["width_0"].vmax == 0.01
        assert constraints["width_0"].constraint_type == "FREE"

        with pytest.raises(ParameterError):
            constraint_manager.set_constraint("invalid_param", vmin=1.0)

        with pytest.raises(ParameterError):
            constraint_manager.set_constraint("center", constraint_type="INVALID")

    def test_to_array(self) -> None:
        """Test conversion to constraint array."""
        model = _make_test_model(
            ["contrast", "center", "width_0", "offset"],
            {"contrast": "contrast", "center": "center", "width_0": "width", "offset": "offset"},
            ["center"],
        )
        settings = ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.8,
            center_max=2.9,
            center_type="FREE",
            width_min=0.001,
            width_max=0.01,
            width_type="FREE",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="FREE",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="FREE",
        )

        constraint_manager = ConstraintManager(model, settings)
        model_params = model.parameter_names
        constraints_array = constraint_manager.to_array(2, model_params)

        assert constraints_array.shape == (2, 8)

        expected_first_row = [
            0.0,  # contrast_min
            1.0,  # contrast_max
            2.8,  # center_min (GHz — no Hz conversion, QEP-018)
            2.9,  # center_max
            0.001,  # width_min
            0.01,  # width_max
            -0.1,  # offset_min
            0.1,  # offset_max
        ]
        assert_array_almost_equal(constraints_array[0], expected_first_row)
        assert_array_almost_equal(constraints_array[0], constraints_array[1])

    def test_get_constraint_types(self) -> None:
        """Test getting constraint types as array."""
        model = _make_test_model(
            ["contrast", "center", "width_0", "offset"],
            {"contrast": "contrast", "center": "center", "width_0": "width", "offset": "offset"},
            ["center"],
        )
        settings = ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.8,
            center_max=2.9,
            center_type="LOWER",
            width_min=0.001,
            width_max=0.01,
            width_type="UPPER",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="FREE",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="LOWER_UPPER",
        )

        constraint_manager = ConstraintManager(model, settings)
        model_params = model.parameter_names
        constraint_types = constraint_manager.get_constraint_types(model_params)

        expected_types = [
            CONSTRAINT_TYPES.index("FREE"),
            CONSTRAINT_TYPES.index("LOWER"),
            CONSTRAINT_TYPES.index("UPPER"),
            CONSTRAINT_TYPES.index("LOWER_UPPER"),
        ]
        assert_array_equal(constraint_types, expected_types)


class TestFitting:
    """Test fitting methods of the FitManager class."""

    def test_fit_returns_fit_result(self, sample_data, sample_frequencies) -> None:
        """Test that fit() returns a FitResult."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        result = fit.fit(sample_data, sample_frequencies)
        assert isinstance(result, FitResult)

    def test_fit_reuse(self, sample_data, sample_frequencies) -> None:
        """Test that the same FitManager can be called twice with different data."""
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        result1 = fit.fit(sample_data, sample_frequencies)
        result2 = fit.fit(sample_data, sample_frequencies)
        assert isinstance(result1, FitResult)
        assert isinstance(result2, FitResult)
        assert result1 is not result2


class TestPipelineStages:
    """Unit tests for the extracted fit() pipeline stages (QEP-070 phase 3)."""

    def test_prepare_data_shapes(self, sample_data, sample_frequencies) -> None:
        from qdmpy.fitting.manager import _PreparedFitInputs

        prepared = FitManager._prepare_data(sample_data, sample_frequencies)
        assert isinstance(prepared, _PreparedFitInputs)
        assert prepared.flat_data.shape == (2, 1, 4, 10)
        assert prepared.freq_ghz.shape == (1, 10)
        assert prepared.scan_dimensions == (2, 2)
        assert prepared.n_pol == 2
        assert prepared.n_frange == 1
        assert prepared.n_pixel == 4
        assert prepared.n_freq == 10

    def test_resolve_model_already_set(self) -> None:
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        model = fit._resolve_model(np.zeros((2, 1, 4, 10)))
        assert model is fit.model

    def test_resolve_model_auto_detects(self, sample_data, sample_frequencies) -> None:
        fit = FitManager(model_name="auto", settings=MOCK_SETTINGS)
        resolved_model = ESRSINGLE()
        prepared = FitManager._prepare_data(sample_data, sample_frequencies)
        with patch("qdmpy.fitting.manager.guess_model", return_value=resolved_model):
            model = fit._resolve_model(prepared.flat_data)
        assert model is resolved_model
        assert fit.model is resolved_model

    def test_resolve_model_raises_if_unresolved(self) -> None:
        fit = FitManager(model_name="auto", settings=MOCK_SETTINGS)
        with (
            patch.object(fit, "_resolve_auto_model", return_value=None),
            pytest.raises(ModelNotResolvedError, match="Model must be set"),
        ):
            fit._resolve_model(np.zeros((2, 1, 4, 10)))

    def test_guess_parameters_shape(self, sample_data, sample_frequencies) -> None:
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
        prepared = FitManager._prepare_data(sample_data, sample_frequencies)
        range_data = prepared.flat_data[:, 0]
        range_freq = prepared.freq_ghz[0]
        initial = fit._guess_parameters(fit.model, range_data, range_freq)
        assert initial.shape == (prepared.n_pol, prepared.n_pixel, fit.model.n_parameters)

    def test_assemble_result_quality_metrics(self) -> None:
        from qdmpy.fitting.manager import _PreparedFitInputs, _RangeFitOutputs

        model = ModelRegistry.get("ESRSINGLE")
        n_frange, n_pol, n_pixel = 1, 2, 4
        raw = _RangeFitOutputs(
            params=np.ones((n_frange, n_pol, n_pixel, model.n_parameters), dtype=np.float32),
            states=np.zeros((n_frange, n_pol, n_pixel), dtype=np.int32),
            chi2=np.zeros((n_frange, n_pol, n_pixel), dtype=np.float32),
            iterations=np.ones((n_frange, n_pol, n_pixel), dtype=np.int32),
            exec_times=(0.5,),
        )
        prepared = _PreparedFitInputs(
            flat_data=np.zeros((n_pol, n_frange, n_pixel, 10)),
            freq_ghz=np.zeros((n_frange, 10)),
            scan_dimensions=(2, 2),
        )
        result = FitManager._assemble_result(raw, model, prepared, pixel_spacing=4e-6)
        assert result.metadata["quality_metrics"]["total_fit_time"] == pytest.approx(0.5)
        assert result.metadata["quality_metrics"]["convergence_rate"] == pytest.approx(1.0)
        assert result.scan_dimensions == (2, 2)

    def test_assemble_result_merges_extra_metadata(self) -> None:
        from qdmpy.fitting.manager import _PreparedFitInputs, _RangeFitOutputs

        model = ModelRegistry.get("ESRSINGLE")
        n_frange, n_pol, n_pixel = 1, 2, 4
        raw = _RangeFitOutputs(
            params=np.ones((n_frange, n_pol, n_pixel, model.n_parameters), dtype=np.float32),
            states=np.zeros((n_frange, n_pol, n_pixel), dtype=np.int32),
            chi2=np.zeros((n_frange, n_pol, n_pixel), dtype=np.float32),
            iterations=np.ones((n_frange, n_pol, n_pixel), dtype=np.int32),
            exec_times=(0.0,),
        )
        prepared = _PreparedFitInputs(
            flat_data=np.zeros((n_pol, n_frange, n_pixel, 10)),
            freq_ghz=np.zeros((n_frange, 10)),
            scan_dimensions=(2, 2),
        )
        result = FitManager._assemble_result(
            raw, model, prepared, pixel_spacing=4e-6, extra_metadata={"folded_fit": True}
        )
        assert result.metadata["folded_fit"] is True


def test_set_constraints_missing_param() -> None:
    """Test set_constraints with a missing parameter."""
    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)

    with pytest.raises(ParameterError) as excinfo:
        fit.set_constraints("non_existent_param", vmin=0, vmax=1)
    assert "Unknown parameter" in str(excinfo.value)


def test_constraint_manager_missing_settings() -> None:
    """Test ConstraintManager initialization with all settings provided."""
    model = _make_test_model(
        ["center", "width_0", "contrast", "offset"],
        {"center": "center", "width_0": "width", "contrast": "contrast", "offset": "offset"},
        ["center"],
    )
    settings = ModelConstraintsSettings(
        constraint_units="absolute_ghz",
        center_min=2.8,
        center_max=2.9,
        center_type="FREE",
        width_min=0.001,
        width_max=0.01,
        width_type="FREE",
        contrast_min=0.0,
        contrast_max=1.0,
        contrast_type="FREE",
        offset_min=-0.1,
        offset_max=0.1,
        offset_type="FREE",
    )

    cm = ConstraintManager(model, settings)
    assert len(cm.get_constraints()) == 4


def test_to_array_zero_pixels() -> None:
    """Test ConstraintManager.to_array with zero pixels."""
    model = _make_test_model(
        ["center", "width_0", "contrast", "offset"],
        {"center": "center", "width_0": "width", "contrast": "contrast", "offset": "offset"},
        ["center"],
    )
    settings = ModelConstraintsSettings(
        constraint_units="absolute_ghz",
        center_min=2.8,
        center_max=2.9,
        center_type="FREE",
        width_min=0.001,
        width_max=0.01,
        width_type="FREE",
        contrast_min=0.0,
        contrast_max=1.0,
        contrast_type="FREE",
        offset_min=-0.1,
        offset_max=0.1,
        offset_type="FREE",
    )
    constraint_manager = ConstraintManager(model, settings)

    model_params = model.parameter_names
    constraints_array = constraint_manager.to_array(0, model_params)
    assert constraints_array.shape == (0, len(model_params) * 2)


def test_get_constraints_returns_defensive_copy() -> None:
    """Mutating the dict returned by get_constraints() must not affect the manager."""
    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)

    snapshot = fit.constraints
    del snapshot["center"]

    assert "center" in fit.constraints


def test_constraints_to_array_matches_manager_to_array() -> None:
    """Module-level constraints_to_array() must match ConstraintManager.to_array()."""
    from qdmpy.fitting.constraints import constraints_to_array

    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
    params = fit.parameter_names

    expected = fit.get_constraints_array(3)
    actual = constraints_to_array(fit.constraints, 3, params)
    assert_array_equal(actual, expected)


def test_constraint_type_indices_matches_manager_get_constraint_types() -> None:
    """Module-level constraint_type_indices() must match ConstraintManager.get_constraint_types()."""
    from qdmpy.fitting.constraints import constraint_type_indices

    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS)
    params = fit.parameter_names

    expected = fit.get_constraint_types()
    actual = constraint_type_indices(fit.constraints, params)
    assert_array_equal(actual, expected)


def test_fit_frange_mocked(sample_data, sample_frequencies) -> None:
    """Test fit_frange via the injectable FakeFitBackend (no GPU required)."""
    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())

    # flat_data: (n_pol, n_frange, n_pixel, n_freq) -> per-range: (n_pol, n_pixel, n_freq)
    values = sample_data.values
    n_pol, n_frange = values.shape[0], values.shape[1]
    n_freq = values.shape[-1]
    flat_data = values.reshape(n_pol, n_frange, -1, n_freq)
    guesser = ParameterGuesser(fit.model, np.atleast_2d(sample_frequencies))
    initial_params = guesser.guess(flat_data)
    results = fit.fit_frange(flat_data[:, 0], sample_frequencies, initial_params[:, 0])
    assert len(results) == 5
    assert results[0].shape == (8, fit.n_parameter)


def test_set_free_constraints_complex_model() -> None:
    """Test set_free_constraints with a complex model."""
    fit = FitManager(model_name="ESR14N", settings=MOCK_SETTINGS)
    fit.set_constraints("center", vmin=2.85, vmax=2.90, constraint_type="LOWER_UPPER")

    fit.set_free_constraints()

    for param in fit.parameter_names:
        assert fit.constraints[param].constraint_type == "FREE"


class TestParameterGuesser:
    """Tests for the ParameterGuesser class."""

    def test_shape_correctness(self, sample_data, sample_frequencies) -> None:
        """Test that guess returns correctly shaped arrays for each model."""
        for model_name in ["ESRSINGLE", "ESR15N", "ESR14N"]:
            model = ModelRegistry.get(model_name)
            guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
            values = sample_data.values
            n_pol, n_frange = values.shape[0], values.shape[1]
            flat_data = values.reshape(n_pol, n_frange, -1, values.shape[-1])
            result = guesser.guess(flat_data)
            expected_shape = (2, 1, 4, model.n_parameters)
            assert result.shape == expected_shape, f"Failed for {model_name}"

    def test_caching_behavior(self, sample_data, sample_frequencies) -> None:
        """Test that repeated calls return the same cached object."""
        model = ModelRegistry.get("ESRSINGLE")
        guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
        values = sample_data.values
        n_pol, n_frange = values.shape[0], values.shape[1]
        flat_data = values.reshape(n_pol, n_frange, -1, values.shape[-1])
        first = guesser.guess(flat_data)
        second = guesser.guess(flat_data)
        assert first is second

    def test_reset_clears_cache(self, sample_data, sample_frequencies) -> None:
        """Test that reset clears the cache so next call recomputes."""
        model = ModelRegistry.get("ESRSINGLE")
        guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
        values = sample_data.values
        n_pol, n_frange = values.shape[0], values.shape[1]
        flat_data = values.reshape(n_pol, n_frange, -1, values.shape[-1])
        first = guesser.guess(flat_data)
        guesser.reset()
        assert guesser._cache is None
        second = guesser.guess(flat_data)
        assert first is not second
        assert_array_equal(first, second)


class TestFitManagerValidation:
    """Tests for FitManager input validation (now in fit() call)."""

    def test_rejects_empty_data(self, sample_frequencies) -> None:
        """Test that empty data array raises DataValidationError."""
        empty_data = xr.DataArray(
            np.empty((2, 1, 2, 2, 0)),
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": ["neg", "pos"],
                "freq_range": ["low"],
                "freq_ghz": (["freq_range", "freq_idx"], np.empty((1, 0))),
            },
        )
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        with pytest.raises(DataValidationError, match="empty"):
            fit.fit(empty_data, np.array([]))

    def test_rejects_freq_count_mismatch(self, sample_data) -> None:
        """Test that frequency count mismatch raises DataValidationError."""
        wrong_freqs = np.linspace(2.87, 2.88, 20)
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        with pytest.raises(DataValidationError, match="must match"):
            fit.fit(sample_data, wrong_freqs)

    def test_rejects_too_few_frequencies(self) -> None:
        """Test that fewer than 10 frequency points raises DataValidationError."""
        few_freqs = np.linspace(2.87, 2.88, 5)
        data_5d = np.ones((2, 1, 2, 2, 5))
        da = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": ["neg", "pos"],
                "freq_range": ["low"],
                "freq_ghz": (["freq_range", "freq_idx"], few_freqs.reshape(1, -1)),
            },
        )
        fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())
        with pytest.raises(DataValidationError, match="at least"):
            fit.fit(da, few_freqs)

    def test_rejects_freq_cutoff_unknown_range_key(self) -> None:
        """freq_cutoff rejects unknown top-level range keys."""
        with pytest.raises(DataValidationError, match="unknown range key"):
            FitManager(
                model_name="ESRSINGLE",
                settings=MOCK_SETTINGS,
                freq_cutoff={"middle": {"min": 2.86}},
            )

    def test_rejects_freq_cutoff_invalid_bounds(self) -> None:
        """freq_cutoff rejects min > max for a range."""
        with pytest.raises(DataValidationError, match="must be <="):
            FitManager(
                model_name="ESRSINGLE",
                settings=MOCK_SETTINGS,
                freq_cutoff={"low": {"min": 2.88, "max": 2.87}},
            )

    def test_rejects_freq_cutoff_high_for_single_range(
        self, sample_data, sample_frequencies
    ) -> None:
        """Single-range fits only accept the 'low' cutoff key."""
        fit = FitManager(
            model_name="ESRSINGLE",
            settings=MOCK_SETTINGS,
            backend=FakeFitBackend(),
            freq_cutoff={"high": {"min": 2.875}},
        )
        with pytest.raises(DataValidationError, match="single-range"):
            fit.fit(sample_data, sample_frequencies)

    def test_rejects_freq_cutoff_when_too_few_points_remain(self) -> None:
        """Applying cutoff must keep at least 10 points per frange."""
        freqs = np.linspace(2.87, 2.88, 10)
        data_5d = np.ones((2, 1, 2, 2, 10), dtype=np.float32)
        da = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": ["neg", "pos"],
                "freq_range": ["low"],
                "freq_ghz": (["freq_range", "freq_idx"], freqs.reshape(1, -1)),
            },
        )
        fit = FitManager(
            model_name="ESRSINGLE",
            settings=MOCK_SETTINGS,
            backend=FakeFitBackend(),
            freq_cutoff={"low": {"max": 2.878}},
        )
        with pytest.raises(DataValidationError, match="at least 10"):
            fit.fit(da, freqs)


class _RecordingFitBackend(FakeFitBackend):
    """FakeFitBackend that records the freq_ghz array passed to each fit() call."""

    def __init__(self) -> None:
        self.freq_calls: list[NDArray] = []

    def fit(self, data, freq_ghz, initial_parameters, constraints, constraint_types, model, options):
        self.freq_calls.append(np.asarray(freq_ghz))
        return super().fit(
            data, freq_ghz, initial_parameters, constraints, constraint_types, model, options
        )


def test_fit_applies_freq_cutoff_per_frange() -> None:
    """fit() applies independent low/high frequency cutoffs before fitting."""
    n_pol, n_frange, h, w, n_freq = 2, 2, 2, 2, 20
    data_5d = np.ones((n_pol, n_frange, h, w, n_freq), dtype=np.float32)
    freqs = np.vstack(
        [
            np.linspace(2.82, 2.87, n_freq),
            np.linspace(2.87, 2.93, n_freq),
        ]
    )
    da = xr.DataArray(
        data_5d,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": ["neg", "pos"],
            "freq_range": ["low", "high"],
            "freq_ghz": (["freq_range", "freq_idx"], freqs),
        },
    )

    backend = _RecordingFitBackend()
    fit = FitManager(
        model_name="ESRSINGLE",
        settings=MOCK_SETTINGS,
        backend=backend,
        freq_cutoff={
            "low": {"max": 2.86},
            "high": {"min": 2.89},
        },
    )

    _ = fit.fit(da, freqs)

    assert len(backend.freq_calls) == 2
    low_user_info, high_user_info = backend.freq_calls

    assert np.max(low_user_info) <= 2.86
    assert np.min(high_user_info) >= 2.89
    assert low_user_info.size < n_freq
    assert high_user_info.size < n_freq


def test_fit_auto_model_resolution(sample_data, sample_frequencies) -> None:
    """Test that auto mode resolves the model on first fit() call."""
    fit = FitManager(model_name="auto", settings=MOCK_SETTINGS, backend=FakeFitBackend())
    assert fit.model is None

    resolved_model = ESRSINGLE()

    with patch("qdmpy.fitting.manager.guess_model", return_value=resolved_model):
        result = fit.fit(sample_data, sample_frequencies)
        assert isinstance(result, FitResult)
        assert fit.model is not None
        assert fit.model_name == "ESRSINGLE"


def test_fit_returns_fit_result(sample_data, sample_frequencies) -> None:
    """Test that fit() returns a FitResult with the expected structure."""
    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())

    _, _, n_y, n_x, _ = sample_data.shape

    result = fit.fit(sample_data, sample_frequencies)

    assert isinstance(result, FitResult)
    assert result.model_name == "ESRSINGLE"
    assert result.scan_dimensions == (n_y, n_x)
    assert "center" in result.parameters
    assert "chi2" in result.parameters
    assert "states" in result.parameters


def test_fit_reuse_independent_results(sample_data, sample_frequencies) -> None:
    """Test that the same FitManager returns independent FitResult objects."""
    fit = FitManager(model_name="ESRSINGLE", settings=MOCK_SETTINGS, backend=FakeFitBackend())

    result1 = fit.fit(sample_data, sample_frequencies)
    result2 = fit.fit(sample_data, sample_frequencies)

    assert result1 is not result2
    assert result1.model_name == result2.model_name


def test_param_idx() -> None:
    """Test _param_idx method."""
    fit = FitManager(model_name="ESR14N", settings=MOCK_SETTINGS)

    # ESR14N params: [center, width, contrast_0, contrast_1, contrast_2, offset]
    # center is at index 0
    assert fit._param_idx("center") == [0]

    with pytest.raises(ParameterError):
        fit._param_idx("invalid_param")


def test_param_idx_aliases_are_deprecated() -> None:
    """resonance/mean_contrast are deprecated aliases for center/contrast."""
    fit = FitManager(model_name="ESR14N", settings=MOCK_SETTINGS)

    with pytest.deprecated_call(match="resonance"):
        assert fit._param_idx("resonance") == [0]

    with pytest.deprecated_call(match="mean_contrast"):
        assert fit._param_idx("mean_contrast") == [2, 3, 4]


def test_get_initial_parameter_via_guesser(sample_data, sample_frequencies) -> None:
    """Test ParameterGuesser shape for all models."""
    for model_name in ["ESRSINGLE", "ESR15N", "ESR14N"]:
        model = ModelRegistry.get(model_name)
        guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
        values = sample_data.values
        n_pol, n_frange = values.shape[0], values.shape[1]
        flat_data = values.reshape(n_pol, n_frange, -1, values.shape[-1])
        initial_params = guesser.guess(flat_data)
        expected_shape = (2, 1, 4, model.n_parameters)
        assert initial_params.shape == expected_shape


def test_get_initial_parameter_edge_cases(sample_frequencies) -> None:
    """Test ParameterGuesser with zero data (edge case)."""
    zero_data_4d = np.zeros((2, 1, 4, 10))
    zero_data_xr = _make_xr_data(zero_data_4d)
    model = ModelRegistry.get("ESRSINGLE")
    guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
    values = zero_data_xr.values
    n_pol, n_frange = values.shape[0], values.shape[1]
    flat_data = values.reshape(n_pol, n_frange, -1, values.shape[-1])
    initial_params = guesser.guess(flat_data)
    assert initial_params.shape == (2, 1, 4, model.n_parameters)
    contrast_idx = model.parameter_names.index("contrast")
    assert np.all(initial_params[:, :, :, contrast_idx] == 0)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_fit.py"])
