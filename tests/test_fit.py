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

from QDMpy.exceptions import (
    DataValidationError,
    FitNotPerformedError,
    ModelNotFoundError,
    ParameterError,
)
from QDMpy.fit import CONSTRAINT_TYPES, ConstraintManager, FitManager, ParameterGuesser
from QDMpy.models import ESR14N, ESR15N, ESRSINGLE, Model, ModelRegistry
from QDMpy.settings import (
    FitSettings,
    ModelConstraintsSettings,
    ModelSettings,
    QDMpySettings,
)

# Mock settings for tests (center/width values in GHz, matching default settings convention)
MOCK_SETTINGS = QDMpySettings(
    fit=FitSettings(
        estimator='LSE',
        max_number_iterations=100,
        tolerance=1e-6,
    ),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            center_min=2.8,
            center_max=2.9,
            center_type='FREE',
            width_min=0.001,
            width_max=0.01,
            width_type='FREE',
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type='FREE',
            offset_min=-0.1,
            offset_max=0.1,
            offset_type='FREE',
        )
    ),
)


def _make_xr_data(numpy_4d: np.ndarray) -> xr.DataArray:
    """Convert 4D numpy (n_pol, n_frange, n_pixel, n_freq) to 5D xr.DataArray.

    Assumes pixels can be arranged as a square grid.
    """
    n_pol, n_frange, n_pixel, n_freq = numpy_4d.shape
    side = int(np.sqrt(n_pixel))
    assert side * side == n_pixel, f'n_pixel={n_pixel} is not a perfect square'

    data_5d = numpy_4d.reshape(n_pol, n_frange, side, side, n_freq)
    freq_ghz = np.tile(np.linspace(2.87, 2.88, n_freq), (n_frange, 1))

    return xr.DataArray(
        data_5d,
        dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
        coords={
            'polarity': [f'pol_{i}' for i in range(n_pol)],
            'freq_range': [f'frange_{i}' for i in range(n_frange)],
            'freq_ghz': (['freq_range', 'freq_idx'], freq_ghz),
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

    def test_init_with_default_model(self, sample_data, sample_frequencies) -> None:
        """Test initialization with default 'auto' model."""
        with patch('QDMpy.fit.guess_model') as mock_guess_model:
            mock_model = ESRSINGLE()
            mock_guess_model.return_value = mock_model

            fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)

            mock_guess_model.assert_called_once()
            assert fit.model_name == 'ESRSINGLE'
            assert fit.model == mock_model

    def test_init_with_specific_model(self, sample_data, sample_frequencies) -> None:
        """Test initialization with a specific model."""
        fit = FitManager(
            sample_data, sample_frequencies, model_name='ESR14N', settings=MOCK_SETTINGS
        )
        assert fit.model_name == 'ESR14N'
        assert isinstance(fit.model, ESR14N)

    def test_init_with_invalid_model(self, sample_data, sample_frequencies) -> None:
        """Test initialization with an invalid model name."""
        with pytest.raises(ModelNotFoundError) as excinfo:
            FitManager(
                sample_data, sample_frequencies,
                model_name='INVALID_MODEL', settings=MOCK_SETTINGS,
            )
        assert 'Unknown model' in str(excinfo.value)

    def test_init_with_custom_constraints(self, sample_data, sample_frequencies) -> None:
        """Test initialization with custom constraints."""
        constraints = {'center': {'vmin': 2.87, 'vmax': 2.88, 'constraint_type': 'LOWER_UPPER'}}

        fit = FitManager(
            sample_data, sample_frequencies, constraints=constraints, settings=MOCK_SETTINGS
        )

        assert fit.constraints['center'][0] == 2.87
        assert fit.constraints['center'][1] == 2.88
        assert fit.constraints['center'][2] == 'LOWER_UPPER'


class TestFitProperties:
    """Test property getters and setters of the FitManager class."""

    def test_data_property(self, sample_data, sample_numpy_data, sample_frequencies) -> None:
        """Test data property getter and setter."""
        fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)

        # .data returns 4D flat numpy
        assert fit.data.shape == (2, 1, 4, 10)
        np.testing.assert_array_almost_equal(fit.data, sample_numpy_data)

        # Setting new data
        new_data = np.zeros((2, 1, 4, 10))
        fit.data = new_data
        np.testing.assert_array_equal(fit.data, new_data)

        # Setting identical data should not reset fit
        with patch.object(fit, '_reset_fit') as mock_reset:
            fit.data = new_data
            mock_reset.assert_not_called()

    def test_model_name_property(self, sample_data, sample_frequencies) -> None:
        """Test model_name property getter and setter."""
        fit = FitManager(
            sample_data, sample_frequencies, model_name='ESRSINGLE', settings=MOCK_SETTINGS
        )
        assert fit.model_name == 'ESRSINGLE'

        fit.model_name = 'ESR15N'
        assert fit.model_name == 'ESR15N'
        assert isinstance(fit.model, ESR15N)

        with pytest.raises(ModelNotFoundError):
            fit.model_name = 'INVALID_MODEL'


class TestConstraintsMethods:
    """Test constraint-related methods of the FitManager class."""

    def test_set_constraints(self, sample_data, sample_frequencies) -> None:
        """Test set_constraints method."""
        fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)
        fit.set_constraints('center', vmin=2.85, vmax=2.90, constraint_type='LOWER_UPPER')

        assert fit.constraints['center'][0] == 2.85
        assert fit.constraints['center'][1] == 2.90
        assert fit.constraints['center'][2] == 'LOWER_UPPER'

    def test_set_constraints_with_numeric_type(self, sample_data, sample_frequencies) -> None:
        """Test set_constraints with numeric constraint type."""
        fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)
        fit.set_constraints('width', vmin=1e6, constraint_type=1)

        assert fit.constraints['width'][0] == 1e6
        assert fit.constraints['width'][2] == 'LOWER'

    def test_set_constraints_invalid_type(self, sample_data, sample_frequencies) -> None:
        """Test set_constraints with invalid constraint type."""
        fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)

        with pytest.raises(ParameterError):
            fit.set_constraints('center', constraint_type='INVALID_TYPE')

        with pytest.raises(ParameterError):
            fit.set_constraints('center', constraint_type=10)

    def test_set_free_constraints(self, sample_data, sample_frequencies) -> None:
        """Test set_free_constraints method."""
        fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)
        fit.set_constraints('center', vmin=2.85, vmax=2.90, constraint_type='LOWER_UPPER')
        fit.set_constraints('width', vmin=0.001, constraint_type='LOWER')

        fit.set_free_constraints()

        for param in fit.model_params_unique:
            assert fit.constraints[param][2] == 'FREE'

    def test_get_constraints_array(self, sample_data, sample_frequencies) -> None:
        """Test get_constraints_array method."""
        fit = FitManager(
            sample_data, sample_frequencies, model_name='ESRSINGLE', settings=MOCK_SETTINGS
        )
        fit.set_constraints('center', vmin=2.85, vmax=2.90)
        fit.set_constraints('width', vmin=0.001, vmax=0.01)

        constraints_array = fit.get_constraints_array(2)

        # ESRSINGLE params: center, width, contrast, offset -> 4 params x 2 = 8 columns
        assert constraints_array.shape == (2, 8)

        # to_array converts center from GHz to Hz (* 1e9)
        expected_first_row = [
            fit.constraints['center'][0] * 1e9,
            fit.constraints['center'][1] * 1e9,
            fit.constraints['width'][0],
            fit.constraints['width'][1],
            fit.constraints['contrast'][0],
            fit.constraints['contrast'][1],
            fit.constraints['offset'][0],
            fit.constraints['offset'][1],
        ]
        assert_array_almost_equal(constraints_array[0], expected_first_row)
        assert_array_almost_equal(constraints_array[0], constraints_array[1])

    def test_get_constraint_types(self, sample_data, sample_frequencies) -> None:
        """Test get_constraint_types method."""
        fit = FitManager(
            sample_data, sample_frequencies, model_name='ESRSINGLE', settings=MOCK_SETTINGS
        )
        model_params = fit.model_params_unique

        for i, param in enumerate(model_params):
            constraint_type = CONSTRAINT_TYPES[i % len(CONSTRAINT_TYPES)]
            fit.set_constraints(param, constraint_type=constraint_type)

        constraint_types = fit.get_constraint_types()

        assert len(constraint_types) == len(model_params)
        used_types = set(constraint_types)
        assert len(used_types) > 0
        assert all(t in range(len(CONSTRAINT_TYPES)) for t in used_types)


@pytest.mark.parametrize('model_name', ['ESRSINGLE', 'ESR15N', 'ESR14N'])
def test_get_initial_parameter(sample_data, sample_frequencies, model_name) -> None:
    """Test get_initial_parameter method with different models."""
    fit = FitManager(
        sample_data, sample_frequencies, model_name=model_name, settings=MOCK_SETTINGS
    )
    initial_params = fit.get_initial_parameter()

    model = ModelRegistry.get(model_name)
    expected_shape = (2, 1, 4, model.n_parameters)
    assert initial_params.shape == expected_shape


class TestParamMethods:
    """Test parameter-related methods of the FitManager class."""

    def test_param_idx(self, sample_data, sample_frequencies) -> None:
        """Test _param_idx method."""
        fit = FitManager(
            sample_data, sample_frequencies, model_name='ESR14N', settings=MOCK_SETTINGS
        )

        # ESR14N params: [center, width, contrast_0, contrast_1, contrast_2, offset]
        # center is at index 0 in both model_params and model_params_unique
        assert fit._param_idx('center') == [0]
        assert fit._param_idx('resonance') == [0]

        with pytest.raises(ParameterError):
            fit._param_idx('invalid_param')


def _make_test_model(
    params: list[str],
    param_types: dict[str, str],
    freq_params: list[str],
) -> Model:
    """Create a concrete Model subclass for ConstraintManager tests."""

    class _TestModel(Model):
        def __init__(self) -> None:
            super().__init__('TEST', 1, params)

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
            ['center', 'width_0', 'contrast', 'offset'],
            {'center': 'center', 'width_0': 'width', 'contrast': 'contrast', 'offset': 'offset'},
            ['center'],
        )
        settings = ModelConstraintsSettings(
            center_min=2.8,
            center_max=2.9,
            center_type='FREE',
            width_min=0.001,
            width_max=0.01,
            width_type='LOWER',
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type='UPPER',
            offset_min=-0.1,
            offset_max=0.1,
            offset_type='LOWER_UPPER',
        )

        constraint_manager = ConstraintManager(model, settings)
        constraints = constraint_manager.get_constraints()
        assert len(constraints) == 4

        assert constraints['center'][0] == 2.8
        assert constraints['center'][1] == 2.9
        assert constraints['center'][2] == 'FREE'
        assert constraints['center'][3] == 'GHz'

        assert constraints['width_0'][0] == 0.001
        assert constraints['width_0'][1] == 0.01
        assert constraints['width_0'][2] == 'LOWER'
        assert constraints['width_0'][3] == 'a.u.'

    def test_set_constraint(self) -> None:
        """Test setting constraints."""
        model = _make_test_model(
            ['center', 'width_0', 'contrast', 'offset'],
            {'center': 'center', 'width_0': 'width', 'contrast': 'contrast', 'offset': 'offset'},
            ['center'],
        )
        settings = ModelConstraintsSettings(
            center_min=2.8,
            center_max=2.9,
            center_type='FREE',
            width_min=0.001,
            width_max=0.01,
            width_type='FREE',
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type='FREE',
            offset_min=-0.1,
            offset_max=0.1,
            offset_type='FREE',
        )

        constraint_manager = ConstraintManager(model, settings)
        constraint_manager.set_constraint(
            'center', vmin=2.85, vmax=2.88, constraint_type='LOWER_UPPER'
        )

        constraints = constraint_manager.get_constraints()
        assert constraints['center'][0] == 2.85
        assert constraints['center'][1] == 2.88
        assert constraints['center'][2] == 'LOWER_UPPER'

        constraint_manager.set_constraint('width_0', vmin=0.002)
        assert constraints['width_0'][0] == 0.002
        assert constraints['width_0'][1] == 0.01
        assert constraints['width_0'][2] == 'FREE'

        with pytest.raises(ParameterError):
            constraint_manager.set_constraint('invalid_param', vmin=1.0)

        with pytest.raises(ParameterError):
            constraint_manager.set_constraint('center', constraint_type='INVALID')

    def test_to_array(self) -> None:
        """Test conversion to constraint array."""
        model = _make_test_model(
            ['contrast', 'center', 'width_0', 'offset'],
            {'contrast': 'contrast', 'center': 'center', 'width_0': 'width', 'offset': 'offset'},
            ['center'],
        )
        settings = ModelConstraintsSettings(
            center_min=2.8,
            center_max=2.9,
            center_type='FREE',
            width_min=0.001,
            width_max=0.01,
            width_type='FREE',
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type='FREE',
            offset_min=-0.1,
            offset_max=0.1,
            offset_type='FREE',
        )

        constraint_manager = ConstraintManager(model, settings)
        model_params = model.parameters_unique
        constraints_array = constraint_manager.to_array(2, model_params)

        assert constraints_array.shape == (2, 8)

        expected_first_row = [
            0.0,  # contrast_min
            1.0,  # contrast_max
            2.8e9,  # center_min (GHz * 1e9 → Hz in to_array)
            2.9e9,  # center_max
            0.001,  # width_min (passed through)
            0.01,  # width_max
            -0.1,  # offset_min
            0.1,  # offset_max
        ]
        assert_array_almost_equal(constraints_array[0], expected_first_row)
        assert_array_almost_equal(constraints_array[0], constraints_array[1])

    def test_get_constraint_types(self) -> None:
        """Test getting constraint types as array."""
        model = _make_test_model(
            ['contrast', 'center', 'width_0', 'offset'],
            {'contrast': 'contrast', 'center': 'center', 'width_0': 'width', 'offset': 'offset'},
            ['center'],
        )
        settings = ModelConstraintsSettings(
            center_min=2.8,
            center_max=2.9,
            center_type='LOWER',
            width_min=0.001,
            width_max=0.01,
            width_type='UPPER',
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type='FREE',
            offset_min=-0.1,
            offset_max=0.1,
            offset_type='LOWER_UPPER',
        )

        constraint_manager = ConstraintManager(model, settings)
        model_params = model.parameters_unique
        constraint_types = constraint_manager.get_constraint_types(model_params)

        expected_types = [
            CONSTRAINT_TYPES.index('FREE'),
            CONSTRAINT_TYPES.index('LOWER'),
            CONSTRAINT_TYPES.index('UPPER'),
            CONSTRAINT_TYPES.index('LOWER_UPPER'),
        ]
        assert_array_equal(constraint_types, expected_types)


@pytest.mark.skip(reason='Requires pyGpufit installation')
class TestFitting:
    """Test fitting methods of the FitManager class."""

    def test_fit_odmr(self, sample_data, sample_frequencies) -> None:
        """Test fit_odmr method."""

    def test_reshape_results(self, sample_data, sample_frequencies) -> None:
        """Test reshape_results method."""
        fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)

        n_pixels = 4  # 2x2 spatial
        n_pol = 2
        n_params = fit.n_parameter
        mock_params = np.random.random((n_pol * n_pixels, n_params))
        mock_states = np.zeros(n_pol * n_pixels, dtype=int)
        mock_chi2 = np.random.random(n_pol * n_pixels)
        mock_iters = np.ones(n_pol * n_pixels, dtype=int) * 10
        mock_time = 0.5

        results = [mock_params, mock_states, mock_chi2, mock_iters, mock_time]
        reshaped = fit.reshape_results(results)

        assert reshaped[0].shape == (n_pol, n_pixels, n_params)
        assert reshaped[1].shape == (n_pol, n_pixels)
        assert reshaped[2].shape == (n_pol, n_pixels)
        assert reshaped[3].shape == (n_pol, n_pixels)
        assert reshaped[4] == mock_time


def test_fit_odmr_refit(sample_data, sample_frequencies) -> None:
    """Test fit_odmr with refit=True."""
    fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS, gpu_available=True)

    n_pixels = fit.data.shape[0] * fit.data.shape[2]  # n_pol * n_pixel
    n_params = fit.n_parameter
    mock_results = [
        np.random.random((n_pixels, n_params)),
        np.zeros(n_pixels, dtype=int),
        np.random.random(n_pixels),
        np.ones(n_pixels, dtype=int) * 10,
        0.5,
    ]

    # Set _current_data_shape so reshape_results works when fit_frange is mocked
    flat = fit._flat_data  # (n_pol, n_frange, n_pixel, n_freq)
    fit._current_data_shape = flat[:, 0].shape  # (n_pol, n_pixel, n_freq)

    with patch.object(fit, 'fit_frange', return_value=mock_results) as mock_fit_frange:
        fit.fit_odmr()
        assert fit.fitted is True

        fit.fit_odmr(refit=True)
        assert mock_fit_frange.call_count == 2


def test_set_constraints_missing_param(sample_data, sample_frequencies) -> None:
    """Test set_constraints with a missing parameter."""
    fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)

    with pytest.raises(ParameterError) as excinfo:
        fit.set_constraints('non_existent_param', vmin=0, vmax=1)
    assert 'Unknown parameter' in str(excinfo.value)


def test_get_param_invalid(sample_data, sample_frequencies) -> None:
    """Test get_param with an invalid parameter."""
    fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS)

    # get_param checks fitted status first
    with pytest.raises(FitNotPerformedError, match='No fit has been performed yet'):
        fit.get_param('invalid_param')

    # When fitted, unknown param raises ParameterError
    fit._fitted = True
    fit._fit_results = np.zeros((2, 1, 4, fit.n_parameter))
    with pytest.raises(ParameterError, match='Unknown parameter'):
        fit.get_param('invalid_param')


def test_constraint_manager_missing_settings() -> None:
    """Test ConstraintManager initialization with all settings provided."""
    model = _make_test_model(
        ['center', 'width_0', 'contrast', 'offset'],
        {'center': 'center', 'width_0': 'width', 'contrast': 'contrast', 'offset': 'offset'},
        ['center'],
    )
    settings = ModelConstraintsSettings(
        center_min=2.8,
        center_max=2.9,
        center_type='FREE',
        width_min=0.001,
        width_max=0.01,
        width_type='FREE',
        contrast_min=0.0,
        contrast_max=1.0,
        contrast_type='FREE',
        offset_min=-0.1,
        offset_max=0.1,
        offset_type='FREE',
    )

    cm = ConstraintManager(model, settings)
    assert len(cm.get_constraints()) == 4


def test_to_array_zero_pixels() -> None:
    """Test ConstraintManager.to_array with zero pixels."""
    model = _make_test_model(
        ['center', 'width_0', 'contrast', 'offset'],
        {'center': 'center', 'width_0': 'width', 'contrast': 'contrast', 'offset': 'offset'},
        ['center'],
    )
    settings = ModelConstraintsSettings(
        center_min=2.8,
        center_max=2.9,
        center_type='FREE',
        width_min=0.001,
        width_max=0.01,
        width_type='FREE',
        contrast_min=0.0,
        contrast_max=1.0,
        contrast_type='FREE',
        offset_min=-0.1,
        offset_max=0.1,
        offset_type='FREE',
    )
    constraint_manager = ConstraintManager(model, settings)

    model_params = model.parameters_unique
    constraints_array = constraint_manager.to_array(0, model_params)
    assert constraints_array.shape == (0, len(model_params) * 2)


def test_get_initial_parameter_edge_cases(sample_frequencies) -> None:
    """Test get_initial_parameter with edge cases."""
    zero_data_4d = np.zeros((2, 1, 4, 10))
    zero_data_xr = _make_xr_data(zero_data_4d)
    # Specify model explicitly since auto-detect fails on zero data
    fit = FitManager(
        zero_data_xr, sample_frequencies, model_name='ESRSINGLE', settings=MOCK_SETTINGS
    )

    initial_params = fit.get_initial_parameter()
    # Shape: (n_pol, n_frange, n_pixel, n_params)
    assert initial_params.shape == (2, 1, 4, fit.n_parameter)
    # Contrast should be 0 for zero data (max == 0 → contrast = 0)
    contrast_idx = fit.model_params_unique.index('contrast')
    assert np.all(initial_params[:, :, :, contrast_idx] == 0)


@patch('pygpufit.gpufit.fit_constrained')
def test_fit_frange_mocked(mock_fit_constrained, sample_data, sample_frequencies) -> None:
    """Test fit_frange with mocked pyGpufit."""
    fit = FitManager(sample_data, sample_frequencies, settings=MOCK_SETTINGS, gpu_available=True)

    mock_fit_constrained.return_value = [
        np.random.random((8, fit.n_parameter)),
        np.zeros(8, dtype=int),
        np.random.random(8),
        np.ones(8, dtype=int) * 10,
        0.5,
    ]

    # fit_frange expects 3D: (n_pol, n_pixel, n_freq)
    flat = fit.data  # (2, 1, 4, 10)
    results = fit.fit_frange(flat[:, 0], sample_frequencies, fit.initial_parameter[:, 0])
    assert len(results) == 5
    assert results[0].shape == (8, fit.n_parameter)


def test_set_free_constraints_complex_model(sample_data, sample_frequencies) -> None:
    """Test set_free_constraints with a complex model."""
    fit = FitManager(
        sample_data, sample_frequencies, model_name='ESR14N', settings=MOCK_SETTINGS
    )
    fit.set_constraints('center', vmin=2.85, vmax=2.90, constraint_type='LOWER_UPPER')

    fit.set_free_constraints()

    for param in fit.model_params_unique:
        assert fit.constraints[param][2] == 'FREE'


class TestParameterGuesser:
    """Tests for the ParameterGuesser class."""

    def test_shape_correctness(self, sample_data, sample_frequencies) -> None:
        """Test that guess returns correctly shaped arrays for each model."""
        for model_name in ['ESRSINGLE', 'ESR15N', 'ESR14N']:
            model = ModelRegistry.get(model_name)
            guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
            fit = FitManager(
                sample_data, sample_frequencies,
                model_name=model_name, settings=MOCK_SETTINGS,
            )
            result = guesser.guess(fit._flat_data)
            expected_shape = (2, 1, 4, model.n_parameters)
            assert result.shape == expected_shape, f"Failed for {model_name}"

    def test_caching_behavior(self, sample_data, sample_frequencies) -> None:
        """Test that repeated calls return the same cached object."""
        model = ModelRegistry.get('ESRSINGLE')
        guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
        fit = FitManager(
            sample_data, sample_frequencies,
            model_name='ESRSINGLE', settings=MOCK_SETTINGS,
        )
        first = guesser.guess(fit._flat_data)
        second = guesser.guess(fit._flat_data)
        assert first is second

    def test_reset_clears_cache(self, sample_data, sample_frequencies) -> None:
        """Test that reset clears the cache so next call recomputes."""
        model = ModelRegistry.get('ESRSINGLE')
        guesser = ParameterGuesser(model, np.atleast_2d(sample_frequencies))
        fit = FitManager(
            sample_data, sample_frequencies,
            model_name='ESRSINGLE', settings=MOCK_SETTINGS,
        )
        first = guesser.guess(fit._flat_data)
        guesser.reset()
        assert guesser._cache is None
        second = guesser.guess(fit._flat_data)
        assert first is not second
        assert_array_equal(first, second)


class TestFitManagerValidation:
    """Tests for FitManager input validation."""

    def test_rejects_empty_data(self, sample_frequencies) -> None:
        """Test that empty data array raises DataValidationError."""
        empty_data = xr.DataArray(
            np.empty((2, 1, 2, 2, 0)),
            dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
            coords={
                'polarity': ['pol_0', 'pol_1'],
                'freq_range': ['frange_0'],
                'freq_ghz': (['freq_range', 'freq_idx'], np.empty((1, 0))),
            },
        )
        with pytest.raises(DataValidationError, match='empty'):
            FitManager(empty_data, np.array([]), settings=MOCK_SETTINGS)

    def test_rejects_freq_count_mismatch(self, sample_data) -> None:
        """Test that frequency count mismatch raises DataValidationError."""
        wrong_freqs = np.linspace(2.87, 2.88, 20)
        with pytest.raises(DataValidationError, match='must match'):
            FitManager(sample_data, wrong_freqs, settings=MOCK_SETTINGS)

    def test_rejects_too_few_frequencies(self) -> None:
        """Test that fewer than 10 frequency points raises DataValidationError."""
        few_freqs = np.linspace(2.87, 2.88, 5)
        data_5d = np.ones((2, 1, 2, 2, 5))
        da = xr.DataArray(
            data_5d,
            dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
            coords={
                'polarity': ['pol_0', 'pol_1'],
                'freq_range': ['frange_0'],
                'freq_ghz': (['freq_range', 'freq_idx'], few_freqs.reshape(1, -1)),
            },
        )
        with pytest.raises(DataValidationError, match='at least'):
            FitManager(da, few_freqs, settings=MOCK_SETTINGS)


if __name__ == '__main__':
    pytest.main(['-v', 'tests/test_fit.py'])
