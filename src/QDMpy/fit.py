"""ODMR fitting module for Quantum Diamond Microscopy.

Convention: All frequency values are in GHz. Conversion to Hz occurs only
at the pygpufit boundary in ``fit_frange()`` and ``reshape_results()``.

This module provides fitting functionality for ODMR spectra from NV centers
in diamond, including model selection, parameter estimation, constraint
management, and GPU-accelerated fitting.
"""

from __future__ import annotations

from typing import Any, Self, cast

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray

from QDMpy import get_settings, is_pygpufit_available
from QDMpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from QDMpy.exceptions import ModelGuessNotPossibleError
from QDMpy.guess import (
    guess_center,
    guess_contrast,
    guess_model,
    guess_width,
)
from QDMpy.models import Model, ModelRegistry
from QDMpy.settings import ModelConstraintsSettings, QDMpySettings

CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]
ESTIMATOR_ID = {"LSE": 0, "MLE": 1}


class ConstraintManager:
    """Manages parameter constraints for fitting."""

    def __init__(
        self: Self,
        model: Model,
        settings: ModelConstraintsSettings,
    ) -> None:
        """Initialize the constraint manager from a model and settings.

        Args:
            model: Model instance providing parameter metadata.
            settings: ModelConstraintsSettings with constraint bounds and types.
        """
        self._constraints: dict[str, list[Any]] = {}
        self._model = model
        self._initialize_constraints(settings)

    def _initialize_constraints(
        self: Self,
        settings: ModelConstraintsSettings,
    ) -> None:
        units = self._model.units
        for param in self._model.parameters_unique:
            base_param = self._model.parameter_types[param]
            self._constraints[param] = [
                getattr(settings, f'{base_param}_min'),
                getattr(settings, f'{base_param}_max'),
                getattr(settings, f'{base_param}_type'),
                units[param],
            ]

    def set_constraint(
        self: Self,
        param: str,
        vmin: float | None = None,
        vmax: float | None = None,
        constraint_type: str | None = None,
    ) -> None:
        """Set constraint bounds and type for a parameter.

        Args:
            param: Parameter name.
            vmin: Minimum value constraint.
            vmax: Maximum value constraint.
            constraint_type: Type of constraint ('FREE', 'LOWER', 'UPPER', 'LOWER_UPPER').
        """
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

    def get_constraints(self: Self) -> dict[str, list[Any]]:
        """Get all parameter constraints.

        Returns:
            Dictionary mapping parameter names to constraint lists [vmin, vmax, type, unit].
        """
        return self._constraints

    def to_array(
        self: Self, n_pixel: int, model_params: list[str]
    ) -> NDArray:
        """Convert constraints to array format for GPU fitting.

        Args:
            n_pixel: Number of pixels (for array replication).
            model_params: List of parameter names to extract constraints for.

        Returns:
            NDArray of shape (n_pixel, 2*n_params) with min/max bounds for each parameter.
        """
        freq_params = set(self._model.frequency_parameters)
        constraints_list: list[float] = []
        for param in model_params:
            param_min, param_max = self._constraints[param][0], self._constraints[param][1]
            if param in freq_params:
                param_min *= 1e9
                param_max *= 1e9
            constraints_list.extend((param_min, param_max))
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self: Self, model_params: list[str]) -> NDArray:
        """Get constraint type indices for parameters.

        Args:
            model_params: List of parameter names.

        Returns:
            NDArray of constraint type indices (0=FREE, 1=LOWER, 2=UPPER, 3=LOWER_UPPER).
        """
        return np.array(
            [CONSTRAINT_TYPES.index(self._constraints[param][2]) for param in model_params],
            dtype=np.int32,
        )


class ParameterGuesser:
    """Generates initial parameter guesses for ODMR fitting.

    Encapsulates parameter estimation logic with built-in caching.
    The cache is invalidated when reset() is called (e.g. after data
    or model changes).

    Attributes:
        _model: The Model instance providing parameter metadata.
        _f_ghz: Frequency values in GHz (2D: n_frange x n_freq).
        _cache: Cached initial parameter array, or None.
    """

    def __init__(self: Self, model: Model, f_ghz: NDArray) -> None:
        """Initialize the parameter guesser.

        Args:
            model: Model instance providing parameter metadata.
            f_ghz: Frequency values in GHz, shape (n_frange, n_freq).
        """
        self._model = model
        self._f_ghz = f_ghz
        self._cache: NDArray | None = None

    def guess(self: Self, flat_data: NDArray) -> NDArray:
        """Generate initial parameter guesses, using cache if available.

        Args:
            flat_data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).

        Returns:
            NDArray with shape (n_pol, n_frange, n_pixel, n_params).
        """
        if self._cache is not None:
            return self._cache

        n_pol, n_frange, n_pixel, _ = flat_data.shape
        n_params = self._model.n_parameters
        result = np.zeros((n_pol, n_frange, n_pixel, n_params), dtype=np.float32)

        for idx, param_name in enumerate(self._model.parameters_unique):
            param_type = self._model.parameter_types[param_name]
            logger.debug(f"Guessing {param_type} parameters")

            if param_type == 'center':
                param_values = guess_center(flat_data, self._f_ghz)
            elif param_type == 'contrast':
                param_values = guess_contrast(flat_data)
            elif param_type == 'width':
                param_values = guess_width(flat_data, self._f_ghz, DEFAULT_VMIN, DEFAULT_VMAX)
            elif param_type == 'offset':
                param_values = np.zeros((n_pol, n_frange, n_pixel))
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

            result[:, :, :, idx] = param_values

        self._cache = np.ascontiguousarray(result, dtype=np.float32)
        return self._cache

    def reset(self: Self) -> None:
        """Clear the cached initial parameters."""
        self._cache = None


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
        self: Self,
        data: xr.DataArray,
        frequencies: NDArray,
        model_name: str = "auto",
        constraints: dict[str, Any] | None = None,
        *,
        settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ) -> None:
        """Initialize a fitting instance for ODMR data.

        Args:
            data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
            frequencies: Frequency array in GHz, shape (n_frange, n_freq).
            model_name: Model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
            constraints: Optional dict of custom constraints.
            settings: Optional QDMpySettings instance (defaults to global get_settings()).
            gpu_available: Optional GPU availability override (defaults to is_pygpufit_available()).
        """
        self._settings = settings or get_settings()
        self._gpu_available = (
            gpu_available if gpu_available is not None else is_pygpufit_available()
        )
        self._data_xr = data
        self.f_ghz = np.atleast_2d(frequencies)
        logger.debug(
            "Initializing FitManager with data shape: %s at %s frequencies.",
            self._data_xr.shape,
            self.f_ghz.shape,
        )

        if model_name == "auto":
            try:
                self._model = guess_model(self._flat_data)
            except ModelGuessNotPossibleError as e:
                logger.warning(f"Could not auto-detect model: {e}")
                self._model = ModelRegistry.get("ESRSINGLE")
                logger.info(f"Defaulting to {self._model.name} model")
        else:
            try:
                self._model = ModelRegistry.get(model_name.upper())
            except KeyError as e:
                raise ValueError(
                    f"Unknown model: {model_name}. Choose from: {list(ModelRegistry.all().keys())}",
                ) from e

        logger.info(f"Using model: {self._model.name}")
        self._guesser = ParameterGuesser(self._model, self.f_ghz)
        self._reset_fit()
        self._constraint_manager = ConstraintManager(
            self._model, self._settings.model.constraints
        )
        if constraints:
            for param, constraint in constraints.items():
                self.set_constraints(param, **constraint, reset_fit=False)
        self.estimator_id = ESTIMATOR_ID[self._settings.fit.estimator]

    @property
    def _flat_data(self: Self) -> NDArray:
        """4D numpy array (n_pol, n_frange, n_pixel, n_freq) for numba functions."""
        values = self._data_xr.values  # (pol, frange, y, x, freq_idx)
        n_pol, n_frange = values.shape[0], values.shape[1]
        n_freq = values.shape[-1]
        return values.reshape(n_pol, n_frange, -1, n_freq)

    @property
    def data(self: Self) -> NDArray:
        """Get 4D numpy data (n_pol, n_frange, n_pixel, n_freq)."""
        return self._flat_data

    @data.setter
    def data(self: Self, data: NDArray) -> None:
        logger.info("Data changed, fits need to be recalculated!")
        if np.all(self._flat_data == data):
            return
        # Re-wrap into xarray with same coords
        n_pol, n_frange = data.shape[0], data.shape[1]
        n_freq = data.shape[-1]
        n_y = self._data_xr.sizes["y"]
        n_x = self._data_xr.sizes["x"]
        reshaped = data.reshape(n_pol, n_frange, n_y, n_x, n_freq)
        self._data_xr = xr.DataArray(
            reshaped,
            dims=self._data_xr.dims,
            coords=self._data_xr.coords,
        )
        self._guesser.reset()
        self._reset_fit()

    def _reset_fit(self: Self) -> None:
        self._fitted = False
        self._fit_results: NDArray | None = None
        self._states: NDArray | None = None
        self._chi_squares: NDArray | None = None
        self._number_iterations: NDArray | None = None
        self._execution_time: NDArray | None = None

    def __repr__(self: Self) -> str:
        return (
            f"FitManager(data: {self._data_xr.shape}, "
            f"f: {self.f_ghz.shape}, model: {self._model.name})"
        )

    @property
    def model(self: Self) -> Model:
        """Get the current fitting model.

        Returns:
            The Model object currently used for fitting.
        """
        return self._model

    @property
    def model_name(self: Self) -> str:
        """Get the current model name.

        Returns:
            Model name string (e.g., 'ESR14N', 'ESR15N', 'ESRSINGLE').
        """
        return self._model.name

    @model_name.setter
    def model_name(self: Self, model_name: str) -> None:
        try:
            self._model = ModelRegistry.get(model_name.upper())
        except KeyError as e:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from: {list(ModelRegistry.all().keys())}",
            ) from e
        logger.debug("Setting model to %s, resetting fit results.", model_name)
        self._constraint_manager = ConstraintManager(
            self._model, self._settings.model.constraints
        )
        self._guesser = ParameterGuesser(self._model, self.f_ghz)
        self._reset_fit()

    @property
    def model_params(self: Self) -> list[str]:
        """Get all model parameter names including duplicates.

        Returns:
            List of parameter names for the current model.
        """
        return self._model.parameter

    @property
    def model_params_unique(self: Self) -> list[str]:
        """Get unique model parameter names.

        Returns:
            List of unique parameter names (without duplicates) for the current model.
        """
        return self._model.parameters_unique

    @property
    def n_parameter(self: Self) -> int:
        """Get the number of parameters in the model.

        Returns:
            Number of parameters for the current model.
        """
        return self._model.n_parameters

    def set_constraints(
        self: Self,
        param: str,
        vmin: float | None = None,
        vmax: float | None = None,
        constraint_type: str | int | None = None,
        reset_fit: bool = True,
    ) -> None:
        """Set parameter constraints with optional fit reset.

        Args:
            param: Parameter name to constrain.
            vmin: Minimum value constraint.
            vmax: Maximum value constraint.
            constraint_type: Type as string or index (0=FREE, 1=LOWER, 2=UPPER, 3=LOWER_UPPER).
            reset_fit: Whether to reset fit results when constraints change.
        """
        if isinstance(constraint_type, int):
            if 0 <= constraint_type < len(CONSTRAINT_TYPES):
                constraint_type = CONSTRAINT_TYPES[constraint_type]
            else:
                raise ValueError(
                    f"Invalid constraint type index: {constraint_type}. "
                    f"Must be 0-{len(CONSTRAINT_TYPES) - 1}",
                )

        is_base_param = param == "contrast" and any(
            "contrast_" in p for p in self.model_params_unique
        )

        if is_base_param:
            contrast_params = [p for p in self.model_params_unique if p.startswith("contrast_")]
            for contrast_param in contrast_params:
                logger.debug(
                    "Setting constraints for %s: vmin=%s, vmax=%s, type=%s",
                    contrast_param,
                    vmin,
                    vmax,
                    constraint_type,
                )
                self._constraint_manager.set_constraint(contrast_param, vmin, vmax, constraint_type)
        else:
            logger.debug(
                "Setting constraints for %s: vmin=%s, vmax=%s, type=%s",
                param,
                vmin,
                vmax,
                constraint_type,
            )
            self._constraint_manager.set_constraint(param, vmin, vmax, constraint_type)

        if reset_fit:
            self._reset_fit()

    def set_free_constraints(self: Self) -> None:
        """Remove all constraints by setting all parameters to FREE."""
        for param in self.model_params_unique:
            self._constraint_manager.set_constraint(param, constraint_type="FREE")
        self._reset_fit()

    @property
    def constraints(self: Self) -> dict[str, list[Any]]:
        """Get current parameter constraints.

        Returns:
            Dictionary mapping parameter names to constraint lists.
        """
        return self._constraint_manager.get_constraints()

    def get_constraints_array(self: Self, n_pixel: int) -> NDArray:
        """Get constraints as array for GPU fitting.

        Args:
            n_pixel: Number of pixels.

        Returns:
            NDArray of shape (n_pixel, 2*n_params) with constraint bounds.
        """
        return self._constraint_manager.to_array(n_pixel, self.model_params_unique)

    def get_constraint_types(self: Self) -> NDArray:
        """Get constraint type indices for model parameters.

        Returns:
            NDArray of constraint type indices.
        """
        return self._constraint_manager.get_constraint_types(self.model_params_unique)

    @property
    def initial_parameter(self: Self) -> NDArray:
        """Get initial parameter guesses (cached via ParameterGuesser).

        Returns:
            NDArray with shape (n_pol, n_frange, n_pixel, n_params).
        """
        return self._guesser.guess(self._flat_data)

    def get_initial_parameter(self: Self) -> NDArray:
        """Generate initial parameter guesses (always recomputes, no cache).

        Returns:
            NDArray with shape (n_pol, n_frange, n_pixel, n_params).
        """
        self._guesser.reset()
        return self._guesser.guess(self._flat_data)

    @property
    def parameter(self: Self) -> NDArray:
        """Get fitted parameters from most recent fit.

        Returns:
            NDArray of fitted parameter values.

        Raises:
            ValueError: If no fit has been performed yet.
        """
        if not self.fitted:
            raise ValueError("No fit has been performed yet. Call fit_odmr() first.")
        return cast(NDArray, self._fit_results)

    def get_param(self: Self, param: str) -> NDArray:
        """Get specific fitted parameter or fit metric.

        Args:
            param: Parameter name (e.g., 'center', 'width') or metric ('chi_squares', 'chi2').

        Returns:
            NDArray of parameter or metric values.

        Raises:
            ValueError: If no fit has been performed yet.
        """
        if not self.fitted:
            raise ValueError("No fit has been performed yet. Call fit_odmr() first.")
        if param in {"chi2", "chi_squares", "chi_squared"}:
            return cast(NDArray, self._chi_squares)
        idx = self._param_idx(param)
        if param == "mean_contrast":
            return np.mean(cast(NDArray, self._fit_results)[..., idx], axis=-1)
        return cast(NDArray, self._fit_results)[..., idx]

    def _param_idx(self: Self, parameter: str) -> list[int]:
        if parameter == "resonance":
            parameter = "center"
        if parameter == "mean_contrast":
            parameter = "contrast"
        idx = [i for i, p in enumerate(self.model_params) if p == parameter]
        if not idx:
            idx = [i for i, p in enumerate(self.model_params_unique) if p == parameter]
        if not idx:
            raise ValueError(f"Unknown parameter: {parameter}")
        return idx

    @property
    def fitted(self: Self) -> bool:
        """Check if fit has been performed.

        Returns:
            True if fit_odmr() has been called and completed successfully.
        """
        return self._fitted

    def fit_odmr(self: Self, refit: bool = False) -> None:
        """Perform GPU-accelerated ODMR fitting on all frequency ranges.

        Args:
            refit: If True, refit even if already fitted. If False, skip if already fitted.

        Raises:
            ImportError: If pyGpufit is not installed.
        """
        if not self._gpu_available:
            raise ImportError("pyGpufit is required for fitting but not installed")
        if self._fitted and not refit:
            logger.debug("Already fitted")
            return
        if self.fitted and refit:
            self._reset_fit()
            logger.debug("Refitting the ODMR data")

        flat = self._flat_data  # (n_pol, n_frange, n_pixel, n_freq)
        for irange in range(flat.shape[1]):
            freq_min = self.f_ghz[irange].min()
            freq_max = self.f_ghz[irange].max()
            logger.info(f"Fitting frequency range {irange} from {freq_min:.3f}-{freq_max:.3f} GHz")

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
                self._fit_results = np.stack([cast(NDArray, self._fit_results), results[0]])
                self._states = np.stack([cast(NDArray, self._states), results[1]])
                self._chi_squares = np.stack([cast(NDArray, self._chi_squares), results[2]])
                self._number_iterations = np.stack(
                    [cast(NDArray, self._number_iterations), results[3]]
                )
                self._execution_time = np.stack(
                    [cast(NDArray, self._execution_time), results[4]]
                )

            logger.info(f"Fit finished in {results[4]:.2f} seconds")

        self._fit_results = np.swapaxes(cast(NDArray, self._fit_results), 0, 1)
        self._fitted = True

    def fit_frange(
        self: Self,
        data: NDArray,
        freq: NDArray,
        initial_parameters: NDArray,
    ) -> list[NDArray]:
        """Fit a single frequency range using GPU.

        Args:
            data: ODMR data with shape (n_pol, n_pixel, n_freq).
            freq: Frequency values in GHz for this range.
            initial_parameters: Initial parameter guesses.

        Returns:
            List containing [fit_params, states, chi_squares, iterations, exec_time].

        Raises:
            ImportError: If pyGpufit is not installed.
        """
        if not self._gpu_available:
            raise ImportError("pyGpufit is required for fitting but not installed")

        import pygpufit.gpufit as gf

        self._current_data_shape = data.shape
        n_pol, n_pix, n_freqs = data.shape
        data_reshaped = data.reshape((-1, n_freqs))
        initial_parameters_reshaped = initial_parameters.reshape((-1, self.n_parameter))

        # --- GHz → Hz boundary for pygpufit ---
        freq_params = set(self._model.frequency_parameters)
        for idx, param_name in enumerate(self.model_params_unique):
            if param_name in freq_params:
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
            max_number_iterations=self._settings.fit.max_number_iterations,
            tolerance=self._settings.fit.tolerance,
            estimator_id=self.estimator_id,
        )
        return list(results)

    def reshape_results(self: Self, results: list[Any]) -> list[Any]:
        """Reshape fit results and convert center frequencies from Hz to GHz.

        Args:
            results: List of results from pygpufit.

        Returns:
            List of reshaped results with spatial dimensions restored.
        """
        for i, result in enumerate(results):
            if not isinstance(result, float):
                results[i] = self.reshape_result(result)

        if len(results) > 0 and not isinstance(results[0], float):
            freq_params = set(self._model.frequency_parameters)
            fit_parameters = results[0]
            for idx, param_name in enumerate(self.model_params_unique):
                if param_name in freq_params:
                    fit_parameters[..., idx] /= 1e9
        return results

    def reshape_result(self: Self, result: NDArray) -> NDArray:
        """Reshape a single result array to spatial dimensions.

        Args:
            result: Flattened result array.

        Returns:
            NDArray reshaped to (n_pol, n_pixel) or scalar if applicable.
        """
        n_pol, n_pix = self._current_data_shape[0], self._current_data_shape[1]
        result_reshaped = result.reshape((n_pol, n_pix, -1))
        return np.squeeze(result_reshaped)
