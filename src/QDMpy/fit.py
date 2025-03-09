"""
Module: QDMpy.fit
================

This module provides the `Fit` class for fitting ODMR (Optically Detected Magnetic Resonance) 
data using various models and constraints. It leverages the pygpufit library for GPU-accelerated 
fitting when available.

Classes:
    - Fit: Class for fitting ODMR data using models and constraints.

Functions:
    - get_constraints_array: Builds a constraints array for use with pygpufit.

Imports:
    - Python standard library: os, sys, logging
    - Third-party: numpy, numba, pygpufit (optional)
    - Local: QDMpy.guess, QDMpy.odmr.data, QDMpy.models
"""

from __future__ import annotations
import os
import sys
import logging
from typing import List, Dict, Any, TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
from numba import njit

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

import QDMpy
from QDMpy.guess import guess_model, guess_initial_fit_parameters
from QDMpy.odmr.data import ODMRData
from QDMpy.models import ModelRegistry, Model

if QDMpy.PYGPUFIT_PRESENT:  # type: ignore[has-type]
    import pygpufit.gpufit as gf

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from QDMpy.odmr.odmr import ODMR


# Constants
UNITS = {"center": "GHz", "width": "GHz", "contrast": "a.u.", "offset": "a.u."}
CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]
ESTIMATOR_ID = {"LSE": 0, "MLE": 1}

LOG = logging.getLogger(__name__)


class Fit:
    """
    Class for fitting ODMR data using models and constraints.
    
    This class interfaces with the ODMR data structure to perform model fitting
    on ODMR spectra. It can automatically detect the appropriate model or use
    a user-specified model. The fitting is performed using pygpufit for
    GPU-accelerated computation when available.
    
    Attributes:
        odmr (ODMR): The ODMR instance containing the data to fit.
        _model (Model): The model used for fitting.
        _constraints (Dict[str, List[Any]]): Fitting constraints for each parameter.
        _initial_parameter (NDArray): Initial parameter values for the fitting.
        estimator_id (int): The ID of the estimator to use (LSE or MLE).
        _fitted (bool): Whether the fit has been performed.
        _fit_results (Optional[Dict[str, NDArray]]): Results from the fitting.
        _chi_squares (Optional[NDArray]): Chi-square values from the fitting.
        _number_iterations (Optional[NDArray]): Number of iterations for each fit.
        _execution_time (Optional[float]): Time taken to perform the fit.
    """

    def __init__(self, odmr: ODMR, model: Optional[Union[str, Model]] = None) -> None:
        """
        Initialize the Fit class.

        Args:
            odmr (ODMR): Instance of ODMR containing processed data, dimensions, and frequencies.
            model (Optional[Union[str, Model]]): The fitting model name or Model instance.
                If None, the model will be automatically determined based on the data.
        
        Raises:
            ValueError: If the ODMR instance does not have processed data available.
        """
        self.odmr = odmr
        self._model = None
        self._constraints = None
        self._initial_parameter = None
        self._fitted = False
        self._fit_results = None
        self._chi_squares = None
        self._number_iterations = None
        self._execution_time = None

        LOG.info("Initializing Fit instance.")

        # Ensure we have processed data to work with
        try:
            processed_data = self.odmr.processed_data
        except ValueError:
            LOG.error("ODMR instance has no processed data available.")
            raise ValueError("ODMR data must be processed before fitting. Call odmr.process_data() first.")

        # Guess Model from processed data if not provided
        if model is None:
            LOG.info("No model specified. Attempting to guess model from data...")
            model = guess_model(self.odmr.processed_data.data)
            LOG.info(f"Guessed model: {model.name}")
        elif isinstance(model, str):
            model_name = model.upper()
            if model_name not in ModelRegistry.all():
                raise ValueError(f"Unknown model: {model}")
            model = ModelRegistry.get(model_name)
            LOG.info(f"Using model: {model.name}")

        self._model = model
        self._constraints = self._set_initial_constraints()
        
        LOG.info("Guessing initial fit parameters...")
        self._initial_parameter = guess_initial_fit_parameters(
            self.odmr.processed_data.data, 
            self.odmr.raw_data.frequencies, 
            self._model
        )
        LOG.info("Initial parameters guessed successfully.")
        
        self.estimator_id = ESTIMATOR_ID[QDMpy.SETTINGS["fit"]["estimator"]]
        LOG.info(f"Using estimator: {QDMpy.SETTINGS['fit']['estimator']}")
        
        LOG.info("Fit instance initialized successfully.")

    @property
    def model_name(self) -> str:
        """
        Get the name of the current fitting model.
        
        Returns:
            str: The name of the fitting model.
        """
        return self._model.name

    @model_name.setter
    def model_name(self, name: str) -> None:
        """
        Set a new fitting model by name.
        
        Args:
            name (str): The name of the model to use.
            
        Raises:
            ValueError: If the model name is not recognized.
        """
        name = name.upper()
        if name not in ModelRegistry.all():
            raise ValueError(f"Unknown model: {name}")
        
        LOG.info(f"Changing model from {self._model.name} to {name}")
        self._model = ModelRegistry.get(name)
        self._reset_fit()
        self._constraints = self._set_initial_constraints()
        
        # Re-guess initial parameters with the new model
        self._initial_parameter = guess_initial_fit_parameters(
            self.odmr.processed_data.data, 
            self.odmr.raw_data.frequencies, 
            self._model
        )
        LOG.info(f"Model changed successfully to {name}")

    def _reset_fit(self) -> None:
        """
        Reset all fit-related attributes.
        
        This method clears any previous fitting results and sets the state
        back to unfitted.
        """
        LOG.debug("Resetting fit state")
        self._fitted = False
        self._fit_results = None
        self._chi_squares = None
        self._number_iterations = None
        self._execution_time = None

    def _set_initial_constraints(self) -> Dict[str, List[Any]]:
        """
        Set default constraints based on model parameters.
        
        Returns:
            Dict[str, List[Any]]: Dictionary of constraints for each parameter.
                Each constraint is a list containing:
                [min_value, max_value, constraint_type, unit]
        """
        constraints = QDMpy.SETTINGS["fit"]["constraints"]
        defaults = {}
        for param in self._model.parameter:
            defaults[param] = [
                constraints[f"{param}_min"],
                constraints[f"{param}_max"],
                constraints[f"{param}_type"],
                UNITS[param],
            ]
        return defaults

    def fit_odmr(self, refit: bool = False) -> Dict[str, Any]:
        """
        Fit the ODMR data using the current model and constraints.
        
        This method processes each frequency range separately and collects
        the results. It uses GPU acceleration via pygpufit when available.
        
        Args:
            refit (bool): Whether to refit the data if it's already been fitted.
                Default is False.
                
        Returns:
            Dict[str, Any]: Dictionary containing the fitting results with keys:
                - 'parameters': The fitted parameters
                - 'chi_squares': Chi-square values for each fit
                - 'n_iterations': Number of iterations for each fit
                - 'execution_time': Time taken to perform the fit
                
        Raises:
            RuntimeError: If pygpufit is not available or the fit fails.
        """
        if self._fitted and not refit:
            LOG.info("Fit already completed; returning existing results.")
            return {
                'parameters': self._fit_results,
                'chi_squares': self._chi_squares,
                'n_iterations': self._number_iterations,
                'execution_time': self._execution_time
            }
            
        if refit:
            LOG.info("Refitting data.")
            self._reset_fit()
        
        if not QDMpy.PYGPUFIT_PRESENT:
            LOG.error("Pygpufit is required for fitting but not available.")
            raise RuntimeError("Pygpufit is required but not available. Please install it.")
        
        LOG.info("Starting ODMR fitting...")
        
        # Initialize result containers
        all_results = []
        all_chi_squares = []
        all_n_iterations = []
        total_execution_time = 0.0
        
        # Process each frequency range separately
        frequency_ranges = self.odmr.processed_data.frequencies
        n_ranges = len(frequency_ranges) if isinstance(frequency_ranges, list) else 1
        
        for irange in range(n_ranges):
            if n_ranges > 1:
                freq = frequency_ranges[irange]
                LOG.info(f"Fitting frequency range {irange+1}/{n_ranges}: {freq.min():.6f}-{freq.max():.6f} GHz")
                data_slice = self.odmr.processed_data.data[:, irange]
                params_slice = self._initial_parameter[:, irange]
            else:
                freq = frequency_ranges
                LOG.info(f"Fitting frequency range: {freq.min():.6f}-{freq.max():.6f} GHz")
                data_slice = self.odmr.processed_data.data
                params_slice = self._initial_parameter
            
            # Fit this range
            try:
                results, chi_squares, n_iterations, exec_time = self._fit_range(
                    data_slice, freq, params_slice
                )
                all_results.append(results)
                all_chi_squares.append(chi_squares)
                all_n_iterations.append(n_iterations)
                total_execution_time += exec_time
                
                LOG.info(f"Range fitted successfully. Average chi-square: {np.mean(chi_squares):.4f}")
            except Exception as e:
                LOG.error(f"Error fitting range {irange}: {str(e)}")
                raise RuntimeError(f"Fitting failed: {str(e)}")
        
        # Store results
        self._fit_results = all_results
        self._chi_squares = all_chi_squares
        self._number_iterations = all_n_iterations
        self._execution_time = total_execution_time
        self._fitted = True
        
        LOG.info(f"ODMR fitting completed in {total_execution_time:.2f} seconds.")
        
        return {
            'parameters': self._fit_results,
            'chi_squares': self._chi_squares,
            'n_iterations': self._number_iterations,
            'execution_time': self._execution_time
        }

    def _fit_range(
        self, data: NDArray, freq: NDArray, initial_params: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray, float]:
        """
        Fit a single frequency range.
        
        This method reshapes the data to match the requirements of pygpufit,
        then performs the GPU-accelerated fitting.
        
        Args:
            data (NDArray): ODMR data for this frequency range with shape:
                (polarities, pixels, frequencies)
            freq (NDArray): Frequency array for this range
            initial_params (NDArray): Initial parameters for the fit
            
        Returns:
            Tuple[NDArray, NDArray, NDArray, float]: Tuple containing:
                - Fitted parameters array
                - Chi-square values array
                - Number of iterations array
                - Execution time in seconds
                
        Raises:
            RuntimeError: If the fitting fails.
        """
        LOG.debug(f"Data shape: {data.shape}, Initial params shape: {initial_params.shape}")
        
        # Get the dimensions
        n_pol, n_pix, n_freqs = data.shape
        
        # Reshape data for fitting - combine polarities and pixels
        reshaped_data = data.reshape((n_pol * n_pix, n_freqs))
        reshaped_params = initial_params.reshape((
            n_pol * n_pix,
            self._model.n_parameters,
        ))
        
        # Get constraints
        constraints = get_constraints_array(
            self._model, self._constraints, reshaped_data.shape[0]
        )
        constraint_types = self.get_constraint_types()
        
        LOG.debug(f"Reshaped data: {reshaped_data.shape}, Reshaped params: {reshaped_params.shape}")
        LOG.debug(f"Constraints shape: {constraints.shape}, Types: {constraint_types}")
        
        # Make arrays contiguous for better GPU performance
        contiguous_data = np.ascontiguousarray(reshaped_data, dtype=np.float32)
        contiguous_freq = np.ascontiguousarray(freq, dtype=np.float32)
        contiguous_params = np.ascontiguousarray(reshaped_params, dtype=np.float32)
        
        # Perform the fit
        try:
            results = gf.fit_constrained(
                data=contiguous_data,
                user_info=contiguous_freq,
                constraints=constraints,
                constraint_types=constraint_types,
                initial_parameters=contiguous_params,
                model_id=self._model.name,
                max_number_iterations=QDMpy.SETTINGS["fit"]["max_number_iterations"],
                tolerance=QDMpy.SETTINGS["fit"]["tolerance"],
                weights=None,
                estimator_id=self.estimator_id
            )
            
            # Extract results
            parameters = results[0]
            states = results[1]
            chi_squares = results[2]
            n_iterations = results[3]
            execution_time = results[4]
            
            # Check for fitting errors
            if np.any(states != 0):
                n_failures = np.sum(states != 0)
                LOG.warning(f"{n_failures} out of {len(states)} fits failed with non-zero state.")
            
            # Reshape parameters back to original structure
            parameters = parameters.reshape((n_pol, n_pix, self._model.n_parameters))
            
            return parameters, chi_squares, n_iterations, execution_time
            
        except Exception as e:
            LOG.error(f"GPU fitting failed with error: {str(e)}")
            raise RuntimeError(f"GPU fitting failed: {str(e)}")

    def get_constraint_types(self) -> NDArray:
        """
        Get constraint types as an array.
        
        Converts the string constraint types to their integer equivalents
        for use with pygpufit.
        
        Returns:
            NDArray: Array of constraint type integers.
        """
        return np.array(
            [CONSTRAINT_TYPES.index(c[2]) for c in self._constraints.values()],
            dtype=np.int32,
        )


def get_constraints_array(model: Model, values: Dict[str, List[Any]], n_pixel: int) -> NDArray:
    """
    Build a constraints array for pygpufit.
    
    This function creates a tiled array of constraints for each parameter and pixel.
    
    Args:
        model (Model): The model being used for fitting.
        values (Dict[str, List[Any]]): Dictionary of constraint values for each parameter.
        n_pixel (int): Number of pixels to create constraints for.
        
    Returns:
        NDArray: A tiled array of constraints with shape (n_pixel, n_parameters * 2),
            where each row contains lower and upper bounds for each parameter.
    """
    constraints_list = model.get_constraint_array(values)
    return np.tile(constraints_list, (n_pixel, 1))


if __name__ == "__main__":
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor
    from QDMpy.odmr.odmr import ODMR

    # User-friendly initialization
    loader = MatlabLoader(data_folder="/home/mike/git/QDMpy/tests/data")
    odmr = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=4))
    odmr.process_data()

    fit = Fit(odmr=odmr)
    print(fit._constraints)
    print(fit._model.parameters_unique)
    print(get_constraints_array(fit._model, fit._constraints, 10))
    # fit.fit_odmr()
