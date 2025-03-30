"""Unit tests for the fit module.

This test suite provides comprehensive testing for the Fit class and related
functionality in the QDMpy.fit module.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from QDMpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from QDMpy.fit import CONSTRAINT_TYPES, ESTIMATOR_ID, UNITS, Fit
from QDMpy.models import ESR14N, ESR15N, ESRSINGLE, ModelRegistry

# Mock settings for tests
MOCK_SETTINGS = {
    'fit': {
        'constraints': {
            'center_min': 2.8e9,
            'center_max': 2.9e9,
            'center_type': 'FREE',
            'width_min': 1e6,
            'width_max': 1e7,
            'width_type': 'FREE',
            'contrast_min': 0.0,
            'contrast_max': 1.0,
            'contrast_type': 'FREE',
            'offset_min': -0.1,
            'offset_max': 0.1,
            'offset_type': 'FREE',
        },
        'estimator': 'LSE',
        'max_number_iterations': 100,
        'tolerance': 1e-6,
    }
}


@pytest.fixture
def sample_data():
    """Create sample data for testing the Fit class."""
    # Create a 4D array with 2 polarities, 1 frequency range, 10 frequencies, and 4 pixels
    data = np.ones((2, 1, 10, 4))
    
    # Add Lorentzian dips for each pixel
    for pol in range(2):
        for pixel in range(4):
            center_idx = 5
            for i in range(10):
                # Create a dip with contrast dependent on pixel
                contrast = 0.1 * (pixel + 1)
                width = 1.0 + 0.5 * pixel
                x = i - center_idx
                data[pol, 0, i, pixel] = 1.0 - contrast * (width**2 / (x**2 + width**2))
                
    return data


@pytest.fixture
def sample_frequencies():
    """Create sample frequencies for testing."""
    return np.linspace(2.87e9, 2.88e9, 10)


class TestFitInitialization:
    """Test initialization of the Fit class."""
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_init_with_default_model(self, sample_data, sample_frequencies):
        """Test initialization with default 'auto' model."""
        with patch('QDMpy.fit.guess_model') as mock_guess_model:
            # Mock the model guessing to return ESRSINGLE
            mock_model = ESRSINGLE()
            mock_guess_model.return_value = mock_model
            
            fit = Fit(sample_data, sample_frequencies)
            
            # Check that model was auto-detected
            mock_guess_model.assert_called_once()
            assert fit.model_name == 'ESRSINGLE'
            assert fit.model == mock_model
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_init_with_specific_model(self, sample_data, sample_frequencies):
        """Test initialization with a specific model."""
        fit = Fit(sample_data, sample_frequencies, model_name='ESR14N')
        assert fit.model_name == 'ESR14N'
        assert isinstance(fit.model, ESR14N)
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_init_with_invalid_model(self, sample_data, sample_frequencies):
        """Test initialization with an invalid model name."""
        with pytest.raises(ValueError) as excinfo:
            Fit(sample_data, sample_frequencies, model_name='INVALID_MODEL')
        assert 'Unknown model' in str(excinfo.value)
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_init_with_custom_constraints(self, sample_data, sample_frequencies):
        """Test initialization with custom constraints."""
        constraints = {
            'center': {
                'vmin': 2.87e9,
                'vmax': 2.88e9,
                'constraint_type': 'LOWER_UPPER'
            }
        }
        
        fit = Fit(sample_data, sample_frequencies, constraints=constraints)
        
        # Check that constraint was applied
        assert fit.constraints['center'][0] == 2.87e9
        assert fit.constraints['center'][1] == 2.88e9
        assert fit.constraints['center'][2] == 'LOWER_UPPER'


class TestFitProperties:
    """Test property getters and setters of the Fit class."""
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_data_property(self, sample_data, sample_frequencies):
        """Test data property getter and setter."""
        fit = Fit(sample_data, sample_frequencies)
        assert np.array_equal(fit.data, sample_data)
        
        # Create new data with same shape
        new_data = np.zeros_like(sample_data)
        fit.data = new_data
        assert np.array_equal(fit.data, new_data)
        
        # Setting identical data should not reset fit
        with patch.object(fit, '_reset_fit') as mock_reset:
            fit.data = new_data
            mock_reset.assert_not_called()
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_model_name_property(self, sample_data, sample_frequencies):
        """Test model_name property getter and setter."""
        fit = Fit(sample_data, sample_frequencies, model_name='ESRSINGLE')
        assert fit.model_name == 'ESRSINGLE'
        
        # Change model
        fit.model_name = 'ESR15N'
        assert fit.model_name == 'ESR15N'
        assert isinstance(fit.model, ESR15N)
        
        # Invalid model name
        with pytest.raises(ValueError):
            fit.model_name = 'INVALID_MODEL'


class TestConstraintsMethods:
    """Test constraint-related methods of the Fit class."""
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_set_constraints(self, sample_data, sample_frequencies):
        """Test set_constraints method."""
        fit = Fit(sample_data, sample_frequencies)
        
        # Set constraint for center parameter
        fit.set_constraints('center', vmin=2.85e9, vmax=2.90e9, constraint_type='LOWER_UPPER')
        
        # Check that constraint was applied
        assert fit.constraints['center'][0] == 2.85e9
        assert fit.constraints['center'][1] == 2.90e9
        assert fit.constraints['center'][2] == 'LOWER_UPPER'
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_set_constraints_with_numeric_type(self, sample_data, sample_frequencies):
        """Test set_constraints with numeric constraint type."""
        fit = Fit(sample_data, sample_frequencies)
        
        # Set constraint with numeric type (1 = 'LOWER')
        fit.set_constraints('width_0', vmin=1e6, constraint_type=1)
        
        # Check that constraint was applied with string type
        assert fit.constraints['width_0'][0] == 1e6
        assert fit.constraints['width_0'][2] == 'LOWER'
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_set_constraints_invalid_type(self, sample_data, sample_frequencies):
        """Test set_constraints with invalid constraint type."""
        fit = Fit(sample_data, sample_frequencies)
        
        # Invalid string type
        with pytest.raises(ValueError):
            fit.set_constraints('center', constraint_type='INVALID_TYPE')
        
        # Invalid numeric type
        with pytest.raises(ValueError):
            fit.set_constraints('center', constraint_type=10)
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_set_free_constraints(self, sample_data, sample_frequencies):
        """Test set_free_constraints method."""
        fit = Fit(sample_data, sample_frequencies)
        
        # First set some constraints
        fit.set_constraints('center', vmin=2.85e9, vmax=2.90e9, constraint_type='LOWER_UPPER')
        fit.set_constraints('width_0', vmin=1e6, constraint_type='LOWER')
        
        # Then set all to FREE
        fit.set_free_constraints()
        
        # Check that all constraints are FREE
        for param in fit.model_params_unique:
            assert fit.constraints[param][2] == 'FREE'
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_get_constraints_array(self, sample_data, sample_frequencies):
        """Test get_constraints_array method."""
        fit = Fit(sample_data, sample_frequencies, model_name='ESRSINGLE')
        
        # Set some constraints
        fit.set_constraints('center', vmin=2.85e9, vmax=2.90e9)
        fit.set_constraints('width_0', vmin=1e6, vmax=1e7)
        
        # Get constraints array for 2 pixels
        constraints_array = fit.get_constraints_array(2)
        
        # Check shape (2 pixels x 8 values - min/max for each of 4 parameters)
        assert constraints_array.shape == (2, 8)
        
        # Check that constraints are correctly ordered and repeated for each pixel
        expected_first_row = [
            fit.constraints['contrast'][0], fit.constraints['contrast'][1],
            fit.constraints['center'][0], fit.constraints['center'][1],
            fit.constraints['width_0'][0], fit.constraints['width_0'][1],
            fit.constraints['offset'][0], fit.constraints['offset'][1]
        ]
        assert_array_almost_equal(constraints_array[0], expected_first_row)
        assert_array_almost_equal(constraints_array[0], constraints_array[1])
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_get_constraint_types(self, sample_data, sample_frequencies):
        """Test get_constraint_types method."""
        fit = Fit(sample_data, sample_frequencies, model_name='ESRSINGLE')
        
        # Set different constraint types - using the exact parameter names from model_params_unique
        model_params = fit.model_params_unique
        
        for i, param in enumerate(model_params):
            # Map parameter to different constraint types
            constraint_type = CONSTRAINT_TYPES[i % len(CONSTRAINT_TYPES)]
            fit.set_constraints(param, constraint_type=constraint_type)
            
        # Get constraint types
        constraint_types = fit.get_constraint_types()
        
        # Check that we have the right number of constraints
        assert len(constraint_types) == len(model_params)
        
        # Check that all expected types are used
        used_types = set(constraint_types)
        assert len(used_types) > 0
        assert all(t in range(len(CONSTRAINT_TYPES)) for t in used_types)


@pytest.mark.parametrize("model_name", ["ESRSINGLE", "ESR15N", "ESR14N"])
@patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
def test_get_initial_parameter(sample_data, sample_frequencies, model_name):
    """Test get_initial_parameter method with different models."""
    fit = Fit(sample_data, sample_frequencies, model_name=model_name)
    
    # Get initial parameters
    initial_params = fit.get_initial_parameter()
    
    # Check shape (2 pols, 1 frange, 4 pixels, n_params)
    model = ModelRegistry.get(model_name)
    expected_shape = (2, 1, 4, model.n_parameters)
    assert initial_params.shape == expected_shape


class TestParamMethods:
    """Test parameter-related methods of the Fit class."""
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_param_idx(self, sample_data, sample_frequencies):
        """Test _param_idx method."""
        fit = Fit(sample_data, sample_frequencies, model_name='ESR14N')
        
        # Test with base parameter
        assert fit._param_idx('center') == [1]
        
        # Test with aliased parameter
        assert fit._param_idx('resonance') == [1]
        
        # Test with invalid parameter
        with pytest.raises(ValueError):
            fit._param_idx('invalid_param')


@pytest.mark.skip(reason="Requires pyGpufit installation")
class TestFitting:
    """Test fitting methods of the Fit class."""
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_fit_odmr(self, sample_data, sample_frequencies):
        """Test fit_odmr method."""
        # This test would require pyGpufit, so implementation depends on environment
        pass
    
    @patch('QDMpy.fit.SETTINGS', MOCK_SETTINGS)
    def test_reshape_results(self, sample_data, sample_frequencies):
        """Test reshape_results method."""
        fit = Fit(sample_data, sample_frequencies)
        
        # Create mock results
        n_pixels = sample_data.shape[3]
        n_pol = sample_data.shape[0]
        n_params = fit.n_parameter
        mock_params = np.random.random((n_pol * n_pixels, n_params))
        mock_states = np.zeros(n_pol * n_pixels, dtype=int)
        mock_chi2 = np.random.random(n_pol * n_pixels)
        mock_iters = np.ones(n_pol * n_pixels, dtype=int) * 10
        mock_time = 0.5
        
        results = [mock_params, mock_states, mock_chi2, mock_iters, mock_time]
        
        # Reshape results
        reshaped = fit.reshape_results(results)
        
        # Check shapes
        assert reshaped[0].shape == (n_pol, n_pixels, n_params)
        assert reshaped[1].shape == (n_pol, n_pixels)
        assert reshaped[2].shape == (n_pol, n_pixels)
        assert reshaped[3].shape == (n_pol, n_pixels)
        assert reshaped[4] == mock_time  # Time should not be reshaped


if __name__ == '__main__':
    pytest.main(['-v', 'tests/test_fit.py'])