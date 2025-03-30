"""ODMR fitting module for Quantum Diamond Microscopy.

This module provides fitting functionality for Optically Detected Magnetic Resonance (ODMR)
spectra from Nitrogen-Vacancy (NV) centers in diamond. It includes a Fit class that manages:

- Model selection: Automatic or manual selection of appropriate spectral models
- Parameter guessing: Estimation of initial fit parameters based on data characteristics
- Constraint management: Defining and enforcing parameter bounds for stability
- GPU-accelerated fitting: Optional acceleration using pyGpufit if available
- Result management: Organizing and accessing fit results by parameter type
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from QDMpy import PYGPUFIT_PRESENT, SETTINGS
from QDMpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from QDMpy.exceptions import ModelGuessNotPossible
from QDMpy.guess import (
    guess_center,
    guess_contrast,
    guess_model,
    guess_width,
)
from QDMpy.models import Model, ModelRegistry

if PYGPUFIT_PRESENT:
    import pygpufit.gpufit as gf

LOG = logging.getLogger(__name__)

# Unit mapping for different parameter types
UNITS = {'center': 'GHz', 'width': 'GHz', 'contrast': 'a.u.', 'offset': 'a.u.'}
# Constraint types recognized by pyGpufit
CONSTRAINT_TYPES = ['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER']
# Estimator IDs for least squares (LSE) or maximum likelihood (MLE)
ESTIMATOR_ID = {'LSE': 0, 'MLE': 1}

class ConstraintManager:
    """Manages parameter constraints for fitting."""

    def __init__(self, model_params: list[str], settings: dict, units: dict[str, str]) -> None:
        """Initialize constraints from configuration settings.

        Args:
            model_params: List of unique model parameters.
            settings: Configuration settings for constraints.
            units: Units for each parameter type.
        """
        self._constraints = {}
        self._units = units
        self._initialize_constraints(model_params, settings)

    def _initialize_constraints(self, model_params: list[str], settings: dict) -> None:
        """Initialize constraints based on model parameters and settings."""
        for param in model_params:
            base_param = param.split('_')[0]
            self._constraints[param] = [
                settings[f'{base_param}_min'],
                settings[f'{base_param}_max'],
                settings[f'{base_param}_type'],
                self._units[base_param],
            ]

    def set_constraint(
        self, param: str, vmin: Optional[float] = None, vmax: Optional[float] = None, constraint_type: Optional[str] = None
    ) -> None:
        """Set or update constraints for a specific parameter."""
        if param not in self._constraints:
            raise ValueError(f"Unknown parameter: {param}")

        current = self._constraints[param]
        if vmin is not None:
            current[0] = vmin
        if vmax is not None:
            current[1] = vmax
        if constraint_type is not None:
            if constraint_type not in CONSTRAINT_TYPES:
                raise ValueError(f"Invalid constraint type: {constraint_type}")
            current[2] = constraint_type

    def get_constraints(self) -> dict[str, list[Any]]:
        """Get the current constraints."""
        return self._constraints

    def to_array(self, n_pixel: int, model_params: list[str]) -> NDArray:
        """Convert constraints to array format for external libraries."""
        constraints_list = []
        for param in model_params:
            constraints_list.extend((self._constraints[param][0], self._constraints[param][1]))
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self, model_params: list[str]) -> NDArray:
        """Get constraint types as integer array."""
        return np.array(
            [CONSTRAINT_TYPES.index(self._constraints[param][2]) for param in model_params], dtype=np.int32
        )

class Fit:
    """Manages fitting operations for ODMR spectral data.

    This class handles all aspects of fitting ODMR spectra, including:
    - Model selection
    - Initial parameter estimation
    - Constraint management
    - Fitting execution
    - Result access and interpretation

    Attributes:
        data (NDArray): 3D array of spectral data (n_polarity, n_frange, n_pixel, n_frequencies)
        f_ghz (NDArray): Frequency values in GHz
        _model (Model): Selected model for fitting (ESR14N, ESR15N, or ESRSINGLE)
        _initial_parameter (NDArray): Initial guess parameters for fitting
        _fit_results (NDArray): Fitted parameters after fitting
        _chi_squares (NDArray): Chi-square values for each fit
        _states (NDArray): Status codes indicating fit success or failure
        _number_iterations (NDArray): Number of iterations for each fit
        _execution_time (NDArray): Execution time for each fit
        _constraints (dict): Parameter constraints for fitting
        estimator_id (int): Estimator ID (0 for LSE, 1 for MLE)
        _fitted (bool): Whether fitting has been performed
    """

    def __init__(
        self: Fit,
        data: NDArray,
        frequencies: NDArray,
        model_name: str = 'auto',
        constraints: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a fitting instance for ODMR data.

        Args:
            data: 3D array of spectral data to fit (n_polarity, n_frange, n_pixel, n_frequencies)
            frequencies: 1D array of frequencies in GHz corresponding to spectral data
            model_name: Name of the model to use ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE')
                If 'auto', the model is automatically determined based on the data.
            constraints: Optional dict of custom constraints for fitting parameters

        Raises:
            ValueError: If model_name is not recognized
            ModelGuessNotPossible: If model_name is 'auto' but model cannot be determined
        """
        self._data = data
        self.f_ghz = frequencies
        LOG.debug(
            f'Initializing Fit instance with data: {self.data.shape} at {frequencies.shape} frequencies.',
        )

        # Determine and set the model
        if model_name == 'auto':
            try:
                self._model = guess_model(data)
            except ModelGuessNotPossible as e:
                LOG.warning(f'Could not auto-detect model: {e}')
                # Default to ESRSINGLE if auto-detection fails
                self._model = ModelRegistry.get('ESRSINGLE')
                LOG.info(f'Defaulting to {self._model.name} model')
        else:
            try:
                self._model = ModelRegistry.get(model_name.upper())
            except KeyError:
                raise ValueError(
                    f'Unknown model: {model_name}. Choose from: {list(ModelRegistry.all().keys())}',
                )

        LOG.info(f'Using model: {self._model.name}')
        
        # Initialize parameters
        self._initial_parameter: Optional[NDArray] = None
        self._reset_fit()
        
        # Set up constraints
        self._constraints = self._set_initial_constraints()
        if constraints:
            for param, constraint in constraints.items():
                self.set_constraints(param, **constraint, reset_fit=False)
        
        # Set estimator from configuration
        self.estimator_id = ESTIMATOR_ID[SETTINGS['fit']['estimator']]

    def __repr__(self: Fit) -> str:
        """Get a string representation of the Fit instance.

        Returns:
            str: A string representation of the Fit object
        """
        return f"Fit(data: {self.data.shape}, f: {self.f_ghz.shape}, model: {self._model.name})"

    @property
    def data(self: Fit) -> NDArray:
        """Get the spectral data.

        Returns:
            NDArray: The spectral data being fitted
        """
        return self._data

    @data.setter
    def data(self: Fit, data: NDArray) -> None:
        """Set new spectral data, resetting fit results.

        Args:
            data: New spectral data array to fit
        """
        LOG.info('Data changed, fits need to be recalculated!')
        if np.all(self._data == data):
            return
        self._data = data
        self._initial_parameter = None
        self._reset_fit()

    def _reset_fit(self: Fit) -> None:
        """Reset all fit results.

        This clears all fit results when the model or data changes.
        """
        self._fitted = False
        self._fit_results = None
        self._states = None
        self._chi_squares = None
        self._number_iterations = None
        self._execution_time = None

    @property
    def model(self: Fit) -> Model:
        """Get the current model.

        Returns:
            Model: The current model instance
        """
        return self._model

    @property
    def model_name(self: Fit) -> str:
        """Get the name of the current model.

        Returns:
            str: The name of the current model
        """
        return self._model.name

    @model_name.setter
    def model_name(self: Fit, model_name: str) -> None:
        """Set a new model by name.

        Args:
            model_name: Name of the model to use

        Raises:
            ValueError: If model_name is not recognized
        """
        try:
            self._model = ModelRegistry.get(model_name.upper())
        except KeyError:
            raise ValueError(
                f'Unknown model: {model_name}. Choose from: {list(ModelRegistry.all().keys())}',
            )

        LOG.debug(
            f'Setting model to {model_name}, resetting all fit results and initial parameters.',
        )
        self._constraints = self._set_initial_constraints()
        self._reset_fit()
        self._initial_parameter = self.get_initial_parameter()

    @property
    def model_params(self: Fit) -> list[str]:
        """Get the list of parameters for the current model.

        Returns:
            list[str]: List of parameter names
        """
        return self._model.parameter

    @property
    def model_params_unique(self: Fit) -> list[str]:
        """Get the list of uniquely identified parameters for the current model.

        Returns:
            list[str]: List of unique parameter names
        """
        return self._model.parameters_unique

    @property
    def n_parameter(self: Fit) -> int:
        """Get the number of parameters in the current model.

        Returns:
            int: Number of parameters
        """
        return self._model.n_parameters

    def _set_initial_constraints(self: Fit) -> dict[str, list[Any]]:
        """Initialize constraints from configuration settings.

        Returns:
            dict[str, list[Any]]: Dictionary of constraints for each parameter
        """
        constraints = SETTINGS['fit']['constraints']
        defaults = {}
        
        for param in self.model_params_unique:
            base_param = param.split('_')[0]
            defaults[param] = [
                constraints[f'{base_param}_min'],
                constraints[f'{base_param}_max'],
                constraints[f'{base_param}_type'],
                UNITS[base_param],
            ]
        return defaults

    def set_constraints(
        self: Fit,
        param: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        constraint_type: Optional[Union[str, int]] = None,
        reset_fit: bool = True,
    ) -> None:
        """Set constraints for a specific parameter.

        Args:
            param: Parameter name to constrain
            vmin: Minimum allowed value (None to keep current)
            vmax: Maximum allowed value (None to keep current)
            constraint_type: Constraint type ('FREE', 'LOWER', 'UPPER', 'LOWER_UPPER')
                            or corresponding index (0-3)
            reset_fit: Whether to reset fit results after changing constraints

        Raises:
            ValueError: If constraint_type is not recognized
        """
        # Handle numeric constraint types
        if isinstance(constraint_type, int):
            if 0 <= constraint_type < len(CONSTRAINT_TYPES):
                constraint_type = CONSTRAINT_TYPES[constraint_type]
            else:
                raise ValueError(
                    f'Invalid constraint type index: {constraint_type}. Must be 0-{len(CONSTRAINT_TYPES)-1}',
                )

        # Validate constraint type
        if constraint_type is not None and constraint_type not in CONSTRAINT_TYPES:
            raise ValueError(
                f'Unknown constraint type: {constraint_type}. Choose from: {CONSTRAINT_TYPES}',
            )

        # Check if this is a base parameter with possible numbered variants
        is_base_param = param == 'contrast' and any('contrast_' in p for p in self.model_params_unique)
        
        if is_base_param:
            # Apply to all numbered variants
            contrast_params = [p for p in self.model_params_unique if p.startswith('contrast_')]
            for contrast_param in contrast_params:
                # Get current constraint values
                current = self._constraints.get(contrast_param, [None, None, None, None])
                
                # Update only provided values
                if vmin is not None:
                    current[0] = vmin
                if vmax is not None:
                    current[1] = vmax
                if constraint_type is not None:
                    current[2] = constraint_type
                
                # Ensure unit is set
                base_param_name = contrast_param.split('_')[0]
                current[3] = UNITS[base_param_name]
                
                LOG.debug(
                    f'Setting constraints for {contrast_param}: ({current[0]}, {current[1]}) with {current[2]}',
                )
                self._constraints[contrast_param] = current
        else:
            # Handle normal parameters
            # Get current constraint values
            current = self._constraints.get(param, [None, None, None, None])
            
            # Update only provided values
            if vmin is not None:
                current[0] = vmin
            if vmax is not None:
                current[1] = vmax
            if constraint_type is not None:
                current[2] = constraint_type
            
            # Ensure unit is set
            base_param = param.split('_')[0]
            current[3] = UNITS[base_param]
            
            LOG.debug(
                f'Setting constraints for {param}: ({current[0]}, {current[1]}) with {current[2]}',
            )
            self._constraints[param] = current

        # Reset fit results if requested
        if reset_fit:
            self._reset_fit()

    def set_free_constraints(self: Fit) -> None:
        """Set all parameters to have unconstrained ('FREE') fitting."""
        # Define a custom constraint dictionary with free constraint type
        self._constraints = self._set_initial_constraints()
        # Override all constraint types to FREE
        for param in self._constraints:
            self._constraints[param][2] = 'FREE'
        self._reset_fit()

    @property
    def constraints(self: Fit) -> dict[str, list[Union[float, str]]]:
        """Get the current constraints dictionary.

        Returns:
            dict[str, list[Union[float, str]]]: Dictionary of parameter constraints
        """
        return self._constraints

    def get_constraints_array(self: Fit, n_pixel: int) -> NDArray:
        """Convert constraints to array format required by pyGpufit.

        Args:
            n_pixel: Number of pixels to generate constraints for

        Returns:
            NDArray: Array of constraints (n_pixel, 2*n_parameters)
        """
        constraints_list: list[float] = []
        for param in self.model_params_unique:
            constraints_list.extend((self._constraints[param][0], self._constraints[param][1]))
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self: Fit) -> NDArray:
        """Get constraint types as integer array required by pyGpufit.

        Returns:
            NDArray: Array of constraint type indices
        """
        fit_bounds = [
            CONSTRAINT_TYPES.index(self._constraints[param][2])
            for param in self.model_params_unique
        ]
        return np.array(fit_bounds, dtype=np.int32)

    @property
    def initial_parameter(self: Fit) -> NDArray:
        """Get the initial parameters for fitting.

        Returns:
            NDArray: Initial parameter values
        """
        if self._initial_parameter is None:
            self._initial_parameter = self.get_initial_parameter()
        return self._initial_parameter

    def get_initial_parameter(self: Fit) -> NDArray:
        """Generate initial parameter guesses based on the data.

        Returns:
            NDArray: Initial parameter values
        """
        n_pol, n_frange, _, n_pixel = self.data.shape
        result = np.zeros((n_pol, n_frange, n_pixel, self.n_parameter), dtype=np.float32)
        
        # Process each parameter in the model's unique parameter list
        for idx, param_name in enumerate(self.model_params_unique):
            param_type = param_name.split('_')[0]
            LOG.debug(f'Guessing {param_type} parameters')
            
            if param_type == 'center':
                param_values = guess_center(self.data, self.f_ghz)
            elif param_type == 'contrast':
                param_values = guess_contrast(self.data)
            elif param_type == 'width':
                param_values = guess_width(self.data, self.f_ghz, DEFAULT_VMIN, DEFAULT_VMAX)
            elif param_type == 'offset':
                param_values = np.zeros((n_pol, n_frange, n_pixel))
            else:
                raise ValueError(f'Unknown parameter type: {param_type}')
            
            # Assign to the appropriate position in the result array
            result[:, :, :, idx] = param_values
        
        return np.ascontiguousarray(result, dtype=np.float32)

    @property
    def parameter(self: Fit) -> NDArray:
        """Get the fitted parameters.

        Returns:
            NDArray: Fitted parameter values

        Raises:
            ValueError: If no fit has been performed yet
        """
        if not self.fitted:
            raise ValueError('No fit has been performed yet. Call fit_odmr() first.')
        return cast(NDArray, self._fit_results)

    def get_param(self: Fit, param: str) -> NDArray:
        """Get a specific parameter from fit results.

        Args:
            param: Parameter name to retrieve ('center', 'width', 'contrast', etc.)
                   or 'chi2'/'chi_squares' for fit quality

        Returns:
            NDArray: Parameter values with dimensions matching data shape

        Raises:
            ValueError: If no fit has been performed or parameter name is not recognized
        """
        if not self.fitted:
            raise ValueError('No fit has been performed yet. Call fit_odmr() first.')
            
        if param in {'chi2', 'chi_squares', 'chi_squared'}:
            return cast(NDArray, self._chi_squares)
            
        idx = self._param_idx(param)
        if param == 'mean_contrast':
            return np.mean(cast(NDArray, self._fit_results)[..., idx], axis=-1)
            
        return cast(NDArray, self._fit_results)[..., idx]

    def _param_idx(self: Fit, parameter: str) -> list[int]:
        """Get the index or indices of a parameter in the fit results.

        Args:
            parameter: Parameter name

        Returns:
            list[int]: List of parameter indices

        Raises:
            ValueError: If parameter name is not recognized
        """
        # Handle parameter aliases
        if parameter == 'resonance':
            parameter = 'center'
        if parameter == 'mean_contrast':
            parameter = 'contrast'
            
        # Try to find parameter in model parameters
        idx = [i for i, p in enumerate(self.model_params) if p == parameter]
        if not idx:
            # Try unique parameters if not found
            idx = [i for i, p in enumerate(self.model_params_unique) if p == parameter]
            
        if not idx:
            raise ValueError(f'Unknown parameter: {parameter}')
            
        return idx

    @property
    def fitted(self: Fit) -> bool:
        """Check if a fit has been performed.

        Returns:
            bool: True if fitting has been performed, False otherwise
        """
        return self._fitted

    def fit_odmr(self: Fit, refit: bool = False) -> None:
        """Fit all ODMR data.

        Args:
            refit: Whether to refit if already fitted

        Raises:
            ImportError: If pygpufit is not available
        """
        if not PYGPUFIT_PRESENT:
            raise ImportError('pyGpufit is required for fitting but not installed')
            
        if self._fitted and not refit:
            LOG.debug('Already fitted')
            return
            
        if self.fitted and refit:
            self._reset_fit()
            LOG.debug('Refitting the ODMR data')
            
        # Fit each frequency range separately
        for irange in range(self.data.shape[1]):
            freq_min = self.f_ghz[irange].min()
            freq_max = self.f_ghz[irange].max()
            LOG.info(f'Fitting frequency range {irange} from {freq_min:.3f}-{freq_max:.3f} GHz')
            
            results = self.fit_frange(
                self.data[:, irange],
                self.f_ghz[irange],
                self.initial_parameter[:, irange],
            )
            results = self.reshape_results(results)
            
            # Store results
            if self._fit_results is None:
                self._fit_results = results[0]
                self._states = results[1]
                self._chi_squares = results[2]
                self._number_iterations = results[3]
                self._execution_time = results[4]
            else:
                self._fit_results = np.stack((self._fit_results, results[0]))
                self._states = np.stack((self._states, results[1]))
                self._chi_squares = np.stack((self._chi_squares, results[2]))
                self._number_iterations = np.stack((self._number_iterations, results[3]))
                self._execution_time = np.stack((self._execution_time, results[4]))
                
            LOG.info(f'Fit finished in {results[4]:.2f} seconds')
            
        # Rearrange results to match input data dimensions
        self._fit_results = np.swapaxes(cast(NDArray, self._fit_results), 0, 1)
        self._fitted = True

    def fit_frange(
        self: Fit, 
        data: NDArray, 
        freq: NDArray, 
        initial_parameters: NDArray,
    ) -> list[NDArray]:
        """Fit a single frequency range.

        Args:
            data: Data for one frequency range (n_polarity, n_pixel, n_freqs)
            freq: Frequency values for this range
            initial_parameters: Initial parameter guesses

        Returns:
            list[NDArray]: List of result arrays:
                0. Fitted parameters
                1. Fit state codes
                2. Chi-square values
                3. Iteration counts
                4. Execution time

        Raises:
            ImportError: If pyGpufit is not available
        """
        if not PYGPUFIT_PRESENT:
            raise ImportError('pyGpufit is required for fitting but not installed')
            
        # Reshape data for pyGpufit
        n_pol, n_pix, n_freqs = data.shape
        data_reshaped = data.reshape((-1, n_freqs))
        initial_parameters_reshaped = initial_parameters.reshape((-1, self.n_parameter))
        n_pixel = data_reshaped.shape[0]
        
        # Prepare constraints
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()
        
        # Execute fit
        results = gf.fit_constrained(
            data=np.ascontiguousarray(data_reshaped, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(initial_parameters_reshaped, dtype=np.float32),
            weights=None,
            model_id=self._model.model_id,
            max_number_iterations=SETTINGS['fit']['max_number_iterations'],
            tolerance=SETTINGS['fit']['tolerance'],
            estimator_id=self.estimator_id,
        )
        
        return list(results)

    def reshape_results(self: Fit, results: list[Any]) -> list[Any]:
        """Reshape fit results to match the original data dimensions.

        Args:
            results: List of result arrays from pyGpufit

        Returns:
            list[Any]: Reshaped result arrays
        """
        for i, result in enumerate(results):
            if not isinstance(result, float):
                results[i] = self.reshape_result(result)
        return results

    def reshape_result(self: Fit, result: NDArray) -> NDArray:
        """Reshape a single result array to match original data dimensions.

        Args:
            result: Flat result array

        Returns:
            NDArray: Reshaped result array
        """
        n_pol, n_pix, _ = self.data.shape[0], self.data.shape[3], None
        result_reshaped = result.reshape((n_pol, n_pix, -1))
        return np.squeeze(result_reshaped)