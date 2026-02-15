"""ODMR fitting module for Quantum Diamond Microscopy.

Convention: All frequency values are in GHz. Conversion to Hz occurs only
at the pygpufit boundary in ``fit_frange()`` and ``reshape_results()``.

This module provides fitting functionality for ODMR spectra from NV centers
in diamond, including model selection, parameter estimation, constraint
management, and GPU-accelerated fitting.
"""

from __future__ import annotations

from typing import Any, Optional, Union, cast

import numpy as np
import xarray as xr
from loguru import logger
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
from QDMpy.settings import ModelConstraintsSettings

if PYGPUFIT_PRESENT:
    import pygpufit.gpufit as gf

UNITS = {'center': 'GHz', 'width': 'GHz', 'contrast': 'a.u.', 'offset': 'a.u.'}
CONSTRAINT_TYPES = ['FREE', 'LOWER', 'UPPER', 'LOWER_UPPER']
ESTIMATOR_ID = {'LSE': 0, 'MLE': 1}


class ConstraintManager:
    """Manages parameter constraints for fitting."""

    def __init__(
        self,
        model_params: list[str],
        settings: ModelConstraintsSettings,
        units: dict[str, str],
    ) -> None:
        self._constraints: dict[str, list[Any]] = {}
        self._units = units
        self._initialize_constraints(model_params, settings)

    def _initialize_constraints(
        self,
        model_params: list[str],
        settings: ModelConstraintsSettings,
    ) -> None:
        for param in model_params:
            base_param = param.split('_')[0]
            self._constraints[param] = [
                getattr(settings, f'{base_param}_min'),
                getattr(settings, f'{base_param}_max'),
                getattr(settings, f'{base_param}_type'),
                self._units[base_param],
            ]

    def set_constraint(
        self,
        param: str,
        vmin: float | None = None,
        vmax: float | None = None,
        constraint_type: str | None = None,
    ) -> None:
        if param not in self._constraints:
            raise ValueError(f'Unknown parameter: {param}')
        current = self._constraints[param]
        if vmin is not None:
            current[0] = vmin
        if vmax is not None:
            current[1] = vmax
        if constraint_type is not None:
            if constraint_type not in CONSTRAINT_TYPES:
                raise ValueError(f'Invalid constraint type: {constraint_type}')
            current[2] = constraint_type

    def get_constraints(self) -> dict[str, list[Any]]:
        return self._constraints

    def to_array(self, n_pixel: int, model_params: list[str]) -> NDArray:
        constraints_list: list[float] = []
        for param in model_params:
            param_min, param_max = self._constraints[param][0], self._constraints[param][1]
            if param.startswith('center'):
                param_min *= 1e9
                param_max *= 1e9
            constraints_list.extend((param_min, param_max))
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self, model_params: list[str]) -> NDArray:
        return np.array(
            [CONSTRAINT_TYPES.index(self._constraints[param][2]) for param in model_params],
            dtype=np.int32,
        )


class FitManager:
    """Manages fitting operations for ODMR spectral data.

    Data is stored internally as xr.DataArray with dims
    (polarity, freq_range, y, x, freq_idx). Numpy arrays are extracted
    at boundaries for numba guess functions and pyGpufit.

    Attributes:
        _data_xr: xr.DataArray of spectral data.
        f_ghz: Frequency values in GHz (2D: n_frange x n_freq).
    """

    def __init__(
        self,
        data: xr.DataArray,
        frequencies: NDArray,
        model_name: str = 'auto',
        constraints: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a fitting instance for ODMR data.

        Args:
            data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
            frequencies: Frequency array in GHz, shape (n_frange, n_freq).
            model_name: Model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
            constraints: Optional dict of custom constraints.
        """
        self._data_xr = data
        self.f_ghz = np.atleast_2d(frequencies)
        logger.debug(
            'Initializing FitManager with data shape: %s at %s frequencies.',
            self._data_xr.shape,
            self.f_ghz.shape,
        )

        if model_name == 'auto':
            try:
                self._model = guess_model(self._flat_data)
            except ModelGuessNotPossible as e:
                logger.warning(f'Could not auto-detect model: {e}')
                self._model = ModelRegistry.get('ESRSINGLE')
                logger.info(f'Defaulting to {self._model.name} model')
        else:
            try:
                self._model = ModelRegistry.get(model_name.upper())
            except KeyError:
                raise ValueError(
                    f'Unknown model: {model_name}. Choose from: {list(ModelRegistry.all().keys())}',
                )

        logger.info(f'Using model: {self._model.name}')
        self._initial_parameter: NDArray | None = None
        self._reset_fit()
        self._constraint_manager = ConstraintManager(
            self.model_params_unique, SETTINGS.model.constraints, UNITS
        )
        if constraints:
            for param, constraint in constraints.items():
                self.set_constraints(param, **constraint, reset_fit=False)
        self.estimator_id = ESTIMATOR_ID[SETTINGS.fit.estimator]

    @property
    def _flat_data(self) -> NDArray:
        """4D numpy array (n_pol, n_frange, n_pixel, n_freq) for numba functions."""
        values = self._data_xr.values  # (pol, frange, y, x, freq_idx)
        n_pol, n_frange = values.shape[0], values.shape[1]
        n_freq = values.shape[-1]
        return values.reshape(n_pol, n_frange, -1, n_freq)

    @property
    def data(self) -> NDArray:
        """Get 4D numpy data (n_pol, n_frange, n_pixel, n_freq)."""
        return self._flat_data

    @data.setter
    def data(self, data: NDArray) -> None:
        logger.info('Data changed, fits need to be recalculated!')
        if np.all(self._flat_data == data):
            return
        # Re-wrap into xarray with same coords
        n_pol, n_frange = data.shape[0], data.shape[1]
        n_freq = data.shape[-1]
        n_y = self._data_xr.sizes['y']
        n_x = self._data_xr.sizes['x']
        reshaped = data.reshape(n_pol, n_frange, n_y, n_x, n_freq)
        self._data_xr = xr.DataArray(
            reshaped,
            dims=self._data_xr.dims,
            coords=self._data_xr.coords,
        )
        self._initial_parameter = None
        self._reset_fit()

    def _reset_fit(self) -> None:
        self._fitted = False
        self._fit_results: NDArray | None = None
        self._states: NDArray | None = None
        self._chi_squares: NDArray | None = None
        self._number_iterations: NDArray | None = None
        self._execution_time: NDArray | None = None

    def __repr__(self) -> str:
        return (
            f'FitManager(data: {self._data_xr.shape}, '
            f'f: {self.f_ghz.shape}, model: {self._model.name})'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def model_name(self) -> str:
        return self._model.name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        try:
            self._model = ModelRegistry.get(model_name.upper())
        except KeyError:
            raise ValueError(
                f'Unknown model: {model_name}. Choose from: {list(ModelRegistry.all().keys())}',
            )
        logger.debug('Setting model to %s, resetting fit results.', model_name)
        self._constraint_manager = ConstraintManager(
            self.model_params_unique, SETTINGS.model.constraints, UNITS
        )
        self._reset_fit()
        self._initial_parameter = None

    @property
    def model_params(self) -> list[str]:
        return self._model.parameter

    @property
    def model_params_unique(self) -> list[str]:
        return self._model.parameters_unique

    @property
    def n_parameter(self) -> int:
        return self._model.n_parameters

    def set_constraints(
        self,
        param: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        constraint_type: Optional[Union[str, int]] = None,
        reset_fit: bool = True,
    ) -> None:
        if isinstance(constraint_type, int):
            if 0 <= constraint_type < len(CONSTRAINT_TYPES):
                constraint_type = CONSTRAINT_TYPES[constraint_type]
            else:
                raise ValueError(
                    f'Invalid constraint type index: {constraint_type}. Must be 0-{len(CONSTRAINT_TYPES)-1}',
                )

        is_base_param = param == 'contrast' and any(
            'contrast_' in p for p in self.model_params_unique
        )

        if is_base_param:
            contrast_params = [p for p in self.model_params_unique if p.startswith('contrast_')]
            for contrast_param in contrast_params:
                logger.debug(
                    'Setting constraints for %s: vmin=%s, vmax=%s, type=%s',
                    contrast_param, vmin, vmax, constraint_type,
                )
                self._constraint_manager.set_constraint(contrast_param, vmin, vmax, constraint_type)
        else:
            logger.debug(
                'Setting constraints for %s: vmin=%s, vmax=%s, type=%s',
                param, vmin, vmax, constraint_type,
            )
            self._constraint_manager.set_constraint(param, vmin, vmax, constraint_type)

        if reset_fit:
            self._reset_fit()

    def set_free_constraints(self) -> None:
        for param in self.model_params_unique:
            self._constraint_manager.set_constraint(param, constraint_type='FREE')
        self._reset_fit()

    @property
    def constraints(self) -> dict[str, list[Any]]:
        return self._constraint_manager.get_constraints()

    def get_constraints_array(self, n_pixel: int) -> NDArray:
        return self._constraint_manager.to_array(n_pixel, self.model_params_unique)

    def get_constraint_types(self) -> NDArray:
        return self._constraint_manager.get_constraint_types(self.model_params_unique)

    @property
    def initial_parameter(self) -> NDArray:
        if self._initial_parameter is None:
            self._initial_parameter = self.get_initial_parameter()
        return self._initial_parameter

    def get_initial_parameter(self) -> NDArray:
        """Generate initial parameter guesses.

        Extracts numpy from xarray, flattens spatial dims for numba functions.

        Returns:
            NDArray with shape (n_pol, n_frange, n_pixel, n_params).
        """
        flat = self._flat_data  # (n_pol, n_frange, n_pixel, n_freq)
        n_pol, n_frange, n_pixel, _ = flat.shape
        result = np.zeros((n_pol, n_frange, n_pixel, self.n_parameter), dtype=np.float32)

        for idx, param_name in enumerate(self.model_params_unique):
            param_type = param_name.split('_')[0]
            logger.debug(f'Guessing {param_type} parameters')

            if param_type == 'center':
                param_values = guess_center(flat, self.f_ghz)
            elif param_type == 'contrast':
                param_values = guess_contrast(flat)
            elif param_type == 'width':
                param_values = guess_width(flat, self.f_ghz, DEFAULT_VMIN, DEFAULT_VMAX)
            elif param_type == 'offset':
                param_values = np.zeros((n_pol, n_frange, n_pixel))
            else:
                raise ValueError(f'Unknown parameter type: {param_type}')

            result[:, :, :, idx] = param_values

        return np.ascontiguousarray(result, dtype=np.float32)

    @property
    def parameter(self) -> NDArray:
        if not self.fitted:
            raise ValueError('No fit has been performed yet. Call fit_odmr() first.')
        return cast(NDArray, self._fit_results)

    def get_param(self, param: str) -> NDArray:
        if not self.fitted:
            raise ValueError('No fit has been performed yet. Call fit_odmr() first.')
        if param in {'chi2', 'chi_squares', 'chi_squared'}:
            return cast(NDArray, self._chi_squares)
        idx = self._param_idx(param)
        if param == 'mean_contrast':
            return np.mean(cast(NDArray, self._fit_results)[..., idx], axis=-1)
        return cast(NDArray, self._fit_results)[..., idx]

    def _param_idx(self, parameter: str) -> list[int]:
        if parameter == 'resonance':
            parameter = 'center'
        if parameter == 'mean_contrast':
            parameter = 'contrast'
        idx = [i for i, p in enumerate(self.model_params) if p == parameter]
        if not idx:
            idx = [i for i, p in enumerate(self.model_params_unique) if p == parameter]
        if not idx:
            raise ValueError(f'Unknown parameter: {parameter}')
        return idx

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit_odmr(self, refit: bool = False) -> None:
        if not PYGPUFIT_PRESENT:
            raise ImportError('pyGpufit is required for fitting but not installed')
        if self._fitted and not refit:
            logger.debug('Already fitted')
            return
        if self.fitted and refit:
            self._reset_fit()
            logger.debug('Refitting the ODMR data')

        flat = self._flat_data  # (n_pol, n_frange, n_pixel, n_freq)
        for irange in range(flat.shape[1]):
            freq_min = self.f_ghz[irange].min()
            freq_max = self.f_ghz[irange].max()
            logger.info(
                f'Fitting frequency range {irange} from {freq_min:.3f}-{freq_max:.3f} GHz'
            )

            results = self.fit_frange(
                flat[:, irange],
                self.f_ghz[irange],
                self.initial_parameter[:, irange],
            )
            results = self.reshape_results(results)

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

            logger.info(f'Fit finished in {results[4]:.2f} seconds')

        self._fit_results = np.swapaxes(cast(NDArray, self._fit_results), 0, 1)
        self._fitted = True

    def fit_frange(
        self,
        data: NDArray,
        freq: NDArray,
        initial_parameters: NDArray,
    ) -> list[NDArray]:
        if not PYGPUFIT_PRESENT:
            raise ImportError('pyGpufit is required for fitting but not installed')

        self._current_data_shape = data.shape
        n_pol, n_pix, n_freqs = data.shape
        data_reshaped = data.reshape((-1, n_freqs))
        initial_parameters_reshaped = initial_parameters.reshape((-1, self.n_parameter))

        # --- GHz → Hz boundary for pygpufit ---
        for idx, param_name in enumerate(self.model_params_unique):
            if param_name.startswith('center'):
                initial_parameters_reshaped[:, idx] *= 1e9

        n_pixel = data_reshaped.shape[0]
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()

        results = gf.fit_constrained(
            data=np.ascontiguousarray(data_reshaped, dtype=np.float32),
            user_info=np.ascontiguousarray(freq * 1e9, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(initial_parameters_reshaped, dtype=np.float32),
            weights=None,
            model_id=self._model.model_id,
            max_number_iterations=SETTINGS.fit.max_number_iterations,
            tolerance=SETTINGS.fit.tolerance,
            estimator_id=self.estimator_id,
        )
        return list(results)

    def reshape_results(self, results: list[Any]) -> list[Any]:
        for i, result in enumerate(results):
            if not isinstance(result, float):
                results[i] = self.reshape_result(result)

        if len(results) > 0 and not isinstance(results[0], float):
            fit_parameters = results[0]
            for idx, param_name in enumerate(self.model_params_unique):
                if param_name.startswith('center'):
                    fit_parameters[..., idx] /= 1e9
        return results

    def reshape_result(self, result: NDArray) -> NDArray:
        n_pol, n_pix = self._current_data_shape[0], self._current_data_shape[1]
        result_reshaped = result.reshape((n_pol, n_pix, -1))
        return np.squeeze(result_reshaped)
