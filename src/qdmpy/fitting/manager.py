"""ODMR fitting manager for Quantum Diamond Microscopy.

Convention: All frequency values are in GHz. The pyGpufit ESR kernels have AHYP
hardcoded in GHz, so no Hz conversion is performed at any boundary.

This module provides the FitManager class which orchestrates fitting operations
for ODMR spectra from NV centers in diamond. Parameter constraints and initial
guesses are managed by dedicated modules (constraints.py and guesser.py).
"""

from __future__ import annotations

import datetime
import warnings
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray

from qdmpy.constants import D_ZFS, GAMMA_NV
from qdmpy.exceptions import (
    DataValidationError,
    DependencyError,
    ModelNotFoundError,
    ModelNotResolvedError,
    ParameterError,
)
from qdmpy.fitting.backends import (
    FitBackend,
    FitBackendOptions,
    resolve_backend,
    with_forced_availability,
)
from qdmpy.fitting.constraints import CONSTRAINT_TYPES, Constraint, ConstraintManager
from qdmpy.fitting.guess import guess_model
from qdmpy.fitting.guesser import ParameterGuesser
from qdmpy.fitting.models import Model, ModelRegistry
from qdmpy.fitting.result import FitResult
from qdmpy.odmr._validators import validate_frequencies
from qdmpy.settings import QDMpySettings, get_settings

if TYPE_CHECKING:
    from qdmpy.odmr.folding import FoldedODMR

_MIN_FREQ_POINTS = 10


class FitManager:
    """Manages fitting operations for ODMR spectral data.

    Configuration (model, constraints) is set at construction time.
    Data is provided per-call via fit(), keeping the instance stateless between calls.
    The same FitManager can be reused with different data, returning an independent
    FitResult each time.
    """

    def __init__(
        self: Self,
        model_name: str = "ESR14N",
        constraints: dict[str, Any] | None = None,
        *,
        freq_cutoff: dict[str, dict[str, float | None]] | None = None,
        settings: QDMpySettings | None = None,
        backend: FitBackend | str | None = None,
        gpu_available: bool | None = None,
    ) -> None:
        """Initialize a FitManager with model configuration.

        Args:
            model_name: Model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
                        If 'auto', model is resolved on the first fit() call.
            constraints: Optional dict mapping parameter names to constraint kwargs
                         (vmin, vmax, constraint_type). Applied after model resolution.
            freq_cutoff: Optional per-frange frequency bounds in GHz.
                Schema: {'low': {'min': float|None, 'max': float|None},
                        'high': {'min': float|None, 'max': float|None}}.
            settings: Optional QDMpySettings instance (defaults to global get_settings()).
            backend: Optional FitBackend instance, or a backend name
                ('auto', 'gpufit', 'scipy'). Defaults to ``settings.fit.backend``.
                See :mod:`qdmpy.fitting.backends` (QEP-068).
            gpu_available: Deprecated; use ``backend`` instead. Optional GPU
                availability override.

        Raises:
            ParameterError: If both ``backend`` and ``gpu_available`` are given.
        """
        self._settings = settings or get_settings()
        self._backend = self._resolve_backend(backend, gpu_available)
        self._backend_options = FitBackendOptions(
            estimator=self._settings.fit.estimator,
            max_number_iterations=self._settings.fit.max_number_iterations,
            tolerance=self._settings.fit.tolerance,
        )
        self._freq_cutoff = self._normalize_freq_cutoff(freq_cutoff)

        if model_name == "auto":
            self._model: Model | None = None
            self._constraint_manager: ConstraintManager | None = None
            self._pending_constraints: dict[str, Any] = constraints or {}
            logger.debug("FitManager in auto mode — model resolved on first fit() call")
        else:
            try:
                self._model = ModelRegistry.get(model_name.upper())
            except KeyError as e:
                available = list(ModelRegistry.all().keys())
                msg = f"Unknown model: {model_name}. Choose from: {available}"
                raise ModelNotFoundError(msg) from e
            self._pending_constraints = {}
            self._constraint_manager = ConstraintManager(
                self._model, self._settings.model.constraints
            )
            if constraints:
                for param, constraint in constraints.items():
                    self.set_constraints(param, **constraint)
            logger.info("FitManager initialized with model: {}", self._model.name)

    def _resolve_backend(
        self: Self,
        backend: FitBackend | str | None,
        gpu_available: bool | None,
    ) -> FitBackend:
        """Resolve the constructor's ``backend``/``gpu_available`` arguments.

        Raises:
            ParameterError: If both ``backend`` and ``gpu_available`` are given.
        """
        if gpu_available is not None:
            warnings.warn(
                "FitManager(gpu_available=...) is deprecated and will be removed "
                "in a future release. Pass backend='gpufit'/'scipy' or a "
                "FitBackend instance instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if backend is not None:
                msg = "Pass either 'backend' or the deprecated 'gpu_available', not both"
                raise ParameterError(msg)
            from qdmpy.fitting.backends import GpufitBackend

            return with_forced_availability(GpufitBackend(), available=gpu_available)

        return resolve_backend(backend if backend is not None else self._settings.fit.backend)

    def _require_backend_available(self: Self) -> None:
        """Raise DependencyError if the resolved backend cannot run here."""
        if not self._backend.is_available():
            msg = (
                f"Fit backend '{self._backend.name}' is required for fitting "
                "but is not available"
            )
            raise DependencyError(msg)

    @staticmethod
    def _validate_inputs(data: xr.DataArray, frequencies: NDArray) -> None:
        """Validate data and frequency inputs before fitting.

        Args:
            data: xr.DataArray of ODMR spectral data.
            frequencies: Frequency array in GHz.

        Raises:
            DataValidationError: If inputs fail validation.
        """
        if data.size == 0:
            msg = "Cannot fit empty data array"
            raise DataValidationError(msg)

        freq_2d = np.atleast_2d(frequencies)
        n_freq_data = data.sizes.get("freq_idx", data.shape[-1])
        n_freq_array = freq_2d.shape[-1]
        if n_freq_data != n_freq_array:
            msg = (
                f"Data freq_idx dimension ({n_freq_data}) must match "
                f"frequency count ({n_freq_array})"
            )
            raise DataValidationError(msg)

        if n_freq_array < _MIN_FREQ_POINTS:
            msg = (
                f"Need at least {_MIN_FREQ_POINTS} frequency points for fitting, got {n_freq_array}"
            )
            raise DataValidationError(msg)

        validate_frequencies(freq_2d)

    @staticmethod
    def _coerce_optional_float(value: object, *, field_name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating, np.integer)):
            return float(value)
        msg = f"freq_cutoff field '{field_name}' must be a number or None, got {type(value)!r}"
        raise DataValidationError(msg)

    def _normalize_freq_cutoff(
        self: Self,
        freq_cutoff: dict[str, dict[str, float | None]] | None,
    ) -> dict[str, dict[str, float | None]] | None:
        if freq_cutoff is None:
            return None
        if not isinstance(freq_cutoff, dict):
            msg = "freq_cutoff must be a dictionary"
            raise DataValidationError(msg)

        allowed_ranges = {"low", "high"}
        normalized: dict[str, dict[str, float | None]] = {}

        for range_key, bounds in freq_cutoff.items():
            if range_key not in allowed_ranges:
                msg = (
                    "freq_cutoff has unknown range key "
                    f"'{range_key}'. Allowed keys are: ['low', 'high']"
                )
                raise DataValidationError(msg)
            if not isinstance(bounds, dict):
                msg = f"freq_cutoff['{range_key}'] must be a dictionary"
                raise DataValidationError(msg)

            unknown_bounds = set(bounds).difference({"min", "max"})
            if unknown_bounds:
                unknown = sorted(unknown_bounds)
                msg = (
                    f"freq_cutoff['{range_key}'] has unknown keys {unknown}. "
                    "Allowed keys are: ['min', 'max']"
                )
                raise DataValidationError(msg)

            min_v = self._coerce_optional_float(bounds.get("min"), field_name=f"{range_key}.min")
            max_v = self._coerce_optional_float(bounds.get("max"), field_name=f"{range_key}.max")
            if min_v is not None and max_v is not None and min_v > max_v:
                msg = (
                    f"freq_cutoff['{range_key}'] has invalid bounds: "
                    f"min ({min_v}) must be <= max ({max_v})"
                )
                raise DataValidationError(msg)

            if min_v is not None or max_v is not None:
                normalized[range_key] = {"min": min_v, "max": max_v}

        return normalized or None

    def _validate_freq_cutoff_for_n_ranges(self: Self, n_frange: int) -> None:
        if self._freq_cutoff is None:
            return
        if n_frange == 1:
            if "high" in self._freq_cutoff:
                msg = (
                    "freq_cutoff['high'] is not valid for single-range fits. "
                    "Use 'low' for single-range (including folded) fits."
                )
                raise DataValidationError(msg)
            return
        if n_frange == 2:
            return

        msg = f"freq_cutoff is only supported for 1 or 2 frequency ranges, got {n_frange}"
        raise DataValidationError(msg)

    def _get_range_cutoff(self: Self, irange: int, n_frange: int) -> dict[str, float | None] | None:
        if self._freq_cutoff is None:
            return None
        if n_frange == 1:
            return self._freq_cutoff.get("low")
        range_key = "low" if irange == 0 else "high"
        return self._freq_cutoff.get(range_key)

    def _apply_freq_cutoff_for_range(
        self: Self,
        range_data: NDArray,
        range_freq_ghz: NDArray,
        irange: int,
        n_frange: int,
    ) -> tuple[NDArray, NDArray]:
        cutoff = self._get_range_cutoff(irange, n_frange)
        if cutoff is None:
            return range_data, range_freq_ghz

        fmin = cutoff.get("min")
        fmax = cutoff.get("max")
        mask = np.ones(range_freq_ghz.shape, dtype=bool)
        if fmin is not None:
            mask &= range_freq_ghz >= fmin
        if fmax is not None:
            mask &= range_freq_ghz <= fmax

        n_kept = int(np.sum(mask))
        if n_kept < _MIN_FREQ_POINTS:
            range_label = "low" if n_frange == 1 or irange == 0 else "high"
            msg = (
                f"freq_cutoff for range '{range_label}' keeps {n_kept} frequency points, "
                f"but at least {_MIN_FREQ_POINTS} are required"
            )
            raise DataValidationError(msg)

        if n_kept == range_freq_ghz.size:
            return range_data, range_freq_ghz

        masked_freq = np.ascontiguousarray(range_freq_ghz[mask], dtype=np.float64)
        masked_data = np.ascontiguousarray(range_data[..., mask])
        return masked_data, masked_freq

    def _resolve_auto_model(self: Self, flat_data: NDArray) -> None:
        """Resolve auto model from data and initialize constraint manager.

        Args:
            flat_data: 4D array (n_pol, n_frange, n_pixel, n_freq) used for model detection.
        """
        self._model = guess_model(flat_data)
        logger.info("Auto-resolved model: {}", self._model.name)
        self._constraint_manager = ConstraintManager(self._model, self._settings.model.constraints)
        for param, constraint in self._pending_constraints.items():
            self.set_constraints(param, **constraint)
        self._pending_constraints = {}

    def fit(
        self: Self,
        data: xr.DataArray,
        frequencies: NDArray,
        *,
        pixel_spacing: float = 1.0,
    ) -> FitResult:
        """Fit ODMR data and return a FitResult.

        Args:
            data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
            frequencies: Frequency array in GHz, shape (n_frange, n_freq).
            pixel_spacing: Physical pixel spacing in meters.

        Returns:
            FitResult containing all fitted parameters and analysis methods.

        Raises:
            DependencyError: If pyGpufit is not installed.
            DataValidationError: If data or frequencies fail validation.
        """
        self._require_backend_available()

        self._validate_inputs(data, frequencies)
        f_ghz = np.atleast_2d(frequencies)

        values = data.values  # (n_pol, n_frange, y, x, n_freq)
        n_pol, n_frange = values.shape[0], values.shape[1]
        self._validate_freq_cutoff_for_n_ranges(n_frange)
        n_freq = values.shape[-1]
        flat_data = values.reshape(n_pol, n_frange, -1, n_freq)
        n_pixel = flat_data.shape[2]

        if self._model is None:
            self._resolve_auto_model(flat_data)

        if self._model is None:
            msg = "Model must be set before fitting"
            raise ModelNotResolvedError(msg)
        model: Model = self._model

        all_params = np.empty((n_frange, n_pol, n_pixel, model.n_parameters), dtype=np.float32)
        all_states = np.empty((n_frange, n_pol, n_pixel), dtype=np.int32)
        all_chi2 = np.empty((n_frange, n_pol, n_pixel), dtype=np.float32)
        all_iters = np.empty((n_frange, n_pol, n_pixel), dtype=np.int32)
        exec_times: list[float] = []

        for irange in range(n_frange):
            freq_min = f_ghz[irange].min()
            freq_max = f_ghz[irange].max()
            self._apply_mt_center_window_for_range(f_ghz[irange])
            logger.info(
                "Fitting frequency range {} from {:.3f}-{:.3f} GHz",
                irange,
                freq_min,
                freq_max,
            )

            range_data, range_freq = self._apply_freq_cutoff_for_range(
                flat_data[:, irange],
                f_ghz[irange],
                irange,
                n_frange,
            )
            guesser = ParameterGuesser(model, np.atleast_2d(range_freq))
            initial_params = guesser.guess(range_data[:, np.newaxis, :, :])
            raw = self.fit_frange(range_data, range_freq, initial_params[:, 0])
            shaped = self._reshape_frange_results(raw, data_shape=flat_data[:, irange].shape)
            all_params[irange] = shaped[0]
            all_states[irange] = shaped[1]
            all_chi2[irange] = shaped[2]
            all_iters[irange] = shaped[3]
            exec_times.append(shaped[4])
            logger.info("Fit finished in {:.2f} seconds", shaped[4])

        # Transpose from (n_frange, n_pol, ...) to (n_pol, n_frange, ...)
        params_pf = np.swapaxes(all_params, 0, 1)  # (n_pol, n_frange, n_pixel, n_params)
        states_pf = np.swapaxes(all_states, 0, 1)  # (n_pol, n_frange, n_pixel)
        chi2_pf = np.swapaxes(all_chi2, 0, 1)  # (n_pol, n_frange, n_pixel)

        scan_dimensions = (data.sizes["y"], data.sizes["x"])
        h, w = scan_dimensions
        n_pol, n_frange = params_pf.shape[:2]

        # Reshape flat spatial dimension to 2D (H, W)
        parameters: dict[str, NDArray] = {}
        for idx, param_name in enumerate(model.parameter_names):
            parameters[param_name] = params_pf[:, :, :, idx].reshape(n_pol, n_frange, h, w)
        parameters["chi2"] = chi2_pf.reshape(n_pol, n_frange, h, w)
        parameters["states"] = states_pf.reshape(n_pol, n_frange, h, w)

        quality_metrics = {
            "mean_chi2": float(np.mean(chi2_pf)),
            "median_chi2": float(np.median(chi2_pf)),
            "std_chi2": float(np.std(chi2_pf)),
            "convergence_rate": float(np.mean(states_pf == 0)),
            "n_pixels": int(chi2_pf.size),
            "n_converged": int(np.sum(states_pf == 0)),
            "total_fit_time": sum(exec_times),
        }
        metadata = {
            "fit_timestamp": datetime.datetime.now().isoformat(),
            "quality_metrics": quality_metrics,
        }

        return FitResult(
            parameters=parameters,
            scan_dimensions=scan_dimensions,
            pixel_spacing=pixel_spacing,
            model_name=model.name,
            metadata=metadata,
        )

    def _apply_mt_center_window_for_range(self: Self, freq_ghz: NDArray) -> None:
        """Apply per-range center bounds for mT-mode center windows.

        For `constraint_units='mt'` and `center_type='LOWER_UPPER'`, a non-zero
        `center_min_mt` is interpreted as a true field window `[min, max]` (mT).
        Because non-folded fits run each branch separately, bounds are mapped as:
        - low branch:  D_ZFS-delta_max .. D_ZFS-delta_min
        - high branch: D_ZFS+delta_min .. D_ZFS+delta_max

        Args:
            freq_ghz: Frequency axis (GHz) of the active fit range.
        """
        if self._constraint_manager is None:
            return

        constraints = self._settings.model.constraints
        if constraints.constraint_units != "mt" or constraints.center_type != "LOWER_UPPER":
            return

        delta_min = constraints.center_min_mt * 1e-3 * GAMMA_NV
        if delta_min <= 0.0:
            return
        delta_max = constraints.center_max_mt * 1e-3 * GAMMA_NV

        freq_min = float(np.min(freq_ghz))
        freq_max = float(np.max(freq_ghz))

        if freq_max <= D_ZFS:
            center_min = D_ZFS - delta_max
            center_max = D_ZFS - delta_min
        elif freq_min >= D_ZFS:
            center_min = D_ZFS + delta_min
            center_max = D_ZFS + delta_max
        else:
            # Overlapping-around-D ranges are uncommon; keep existing symmetric
            # bounds from ConstraintManager in this edge case.
            return

        self._constraint_manager.set_constraint(
            "center",
            vmin=center_min,
            vmax=center_max,
            constraint_type="LOWER_UPPER",
        )

    def __repr__(self: Self) -> str:
        """Return a developer-friendly string representation."""
        model_name = self._model.name if self._model is not None else "auto"
        return f"FitManager(model: {model_name})"

    @property
    def model(self: Self) -> Model | None:
        """Get the current fitting model (None if auto mode not yet resolved).

        Returns:
            The Model object or None.
        """
        return self._model

    @property
    def model_name(self: Self) -> str:
        """Get the current model name.

        Returns:
            Model name string (e.g., 'ESR14N', 'ESR15N', 'ESRSINGLE', 'auto').

        Note:
            Model is immutable after construction. To use a different model,
            create a new FitManager instance.
        """
        return self._model.name if self._model is not None else "auto"

    @property
    def parameter_names(self: Self) -> list[str]:
        """Get unique model parameter names.

        Returns:
            List of parameter names for the current model.
        """
        if self._model is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)
        return self._model.parameter_names

    @property
    def n_parameter(self: Self) -> int:
        """Get the number of parameters in the model.

        Returns:
            Number of parameters for the current model.
        """
        if self._model is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)
        return self._model.n_parameters

    def set_constraints(
        self: Self,
        param: str,
        vmin: float | None = None,
        vmax: float | None = None,
        constraint_type: str | int | None = None,
    ) -> None:
        """Set parameter constraints.

        Args:
            param: Parameter name to constrain.
            vmin: Minimum value constraint.
            vmax: Maximum value constraint.
            constraint_type: Type as string or index (0=FREE, 1=LOWER, 2=UPPER, 3=LOWER_UPPER).

        Raises:
            ModelNotResolvedError: If called before model is resolved (auto mode).
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)

        if isinstance(constraint_type, int):
            if 0 <= constraint_type < len(CONSTRAINT_TYPES):
                constraint_type = CONSTRAINT_TYPES[constraint_type]
            else:
                msg = (
                    f"Invalid constraint type index: {constraint_type}. "
                    f"Must be 0-{len(CONSTRAINT_TYPES) - 1}"
                )
                raise ParameterError(msg)

        is_base_param = param == "contrast" and any("contrast_" in p for p in self.parameter_names)

        if is_base_param:
            contrast_params = [p for p in self.parameter_names if p.startswith("contrast_")]
            for contrast_param in contrast_params:
                logger.debug(
                    "Setting constraints for {}: vmin={}, vmax={}, type={}",
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

    def set_free_constraints(self: Self) -> None:
        """Remove all constraints by setting all parameters to FREE."""
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)
        for param in self.parameter_names:
            self._constraint_manager.set_constraint(param, constraint_type="FREE")

    @property
    def constraints(self: Self) -> dict[str, Constraint]:
        """Get current parameter constraints.

        Returns:
            Dictionary mapping parameter names to Constraint objects.
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)
        return self._constraint_manager.get_constraints()

    def get_constraints_array(self: Self, n_pixel: int) -> NDArray:
        """Get constraints as array for GPU fitting.

        Args:
            n_pixel: Number of pixels.

        Returns:
            NDArray of shape (n_pixel, 2*n_params) with constraint bounds.
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)
        return self._constraint_manager.to_array(n_pixel, self.parameter_names)

    def get_constraint_types(self: Self) -> NDArray:
        """Get constraint type indices for model parameters.

        Returns:
            NDArray of constraint type indices.
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise ModelNotResolvedError(msg)
        return self._constraint_manager.get_constraint_types(self.parameter_names)

    def _param_idx(self: Self, parameter: str) -> list[int]:
        if self._model is None:
            msg = "Model not yet resolved"
            raise ModelNotResolvedError(msg)
        if parameter == "resonance":
            parameter = "center"
        if parameter == "mean_contrast":
            parameter = "contrast"
        idx = [
            i
            for i, p in enumerate(self._model.parameter_names)
            if self._model.parameter_types[p] == parameter
        ]
        if not idx:
            idx = [i for i, p in enumerate(self._model.parameter_names) if p == parameter]
        if not idx:
            msg = f"Unknown parameter: {parameter}"
            raise ParameterError(msg)
        return idx

    def fit_frange(
        self: Self,
        data: NDArray,
        freq: NDArray,
        initial_parameters: NDArray,
    ) -> list[Any]:
        """Fit a single frequency range via the resolved backend.

        Thin shape adapter over ``self._backend.fit()`` (QEP-068): builds the
        constraint arrays this FitManager owns, then delegates the actual
        optimization to whichever backend was resolved at construction time
        (gpufit, scipy, or a test fake).

        Args:
            data: ODMR data with shape (n_pol, n_pixel, n_freq).
            freq: Frequency values in GHz for this range.
            initial_parameters: Initial parameter guesses.

        Returns:
            List containing [fit_params, states, chi_squares, iterations, exec_time],
            preserved for compatibility with callers like ``fitting.refit``.

        Raises:
            DependencyError: If the resolved backend is not available, or does
                not support the current model.
            ModelNotResolvedError: If called before the model is resolved.
        """
        self._require_backend_available()

        if self._model is None:
            msg = "Model must be set before fitting"
            raise ModelNotResolvedError(msg)
        model: Model = self._model

        if not self._backend.supports(model):
            msg = (
                f"Backend '{self._backend.name}' does not support model "
                f"'{model.name}' (model_id={model.model_id}). Try "
                "backend='scipy' for custom CPU-only models."
            )
            raise DependencyError(msg)

        n_freqs = data.shape[-1]
        n_pixel = data.reshape((-1, n_freqs)).shape[0]
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()

        output = self._backend.fit(
            data=data,
            freq_ghz=freq,
            initial_parameters=initial_parameters,
            constraints=constraints,
            constraint_types=constraint_types,
            model=model,
            options=self._backend_options,
        )
        return [
            output.parameters,
            output.states,
            output.chi2,
            output.iterations,
            output.execution_time,
        ]

    def _reshape_frange_results(
        self: Self,
        results: list[Any],
        data_shape: tuple[int, ...],
    ) -> list[Any]:
        """Reshape fit results from flat (n_pol*n_pixel, n_params) to (n_pol, n_pixel, n_params).

        Frequency parameters (center, width) remain in GHz — no unit conversion required
        because the pyGpufit kernels use GHz throughout.

        Args:
            results: List of results from pygpufit.
            data_shape: Shape of the per-frange data (n_pol, n_pixel, n_freq).

        Returns:
            List of reshaped results with spatial dimensions restored.
        """
        n_pol, n_pix = data_shape[0], data_shape[1]
        reshaped = []
        for result in results:
            if isinstance(result, float):
                reshaped.append(result)
            else:
                reshaped.append(np.squeeze(result.reshape((n_pol, n_pix, -1))))
        return reshaped

    def fit_folded(
        self: Self,
        folded: FoldedODMR,
        *,
        pixel_spacing: float = 1.0,
        raw_data: NDArray | None = None,
    ) -> FitResult:
        """Fit a folded ODMR spectrum in the absolute-GHz domain.

        The folded spectrum's delta_f frequency axis is shifted to absolute GHz
        (D_ZFS + delta_f) before fitting, so the optimizer works in the same
        domain as non-folded fits. The resulting FitResult uses the standard
        B111 calculation path (center - D_ZFS) with n_frange=1.

        Args:
            folded: FoldedODMR result from SpectralFolder.fold().
            pixel_spacing: Physical pixel spacing in meters.
            raw_data: Optional raw (unfolded) ODMR array with shape
                (n_pol, n_frange, y, x, freq_idx). When provided and model
                is 'auto', peak detection runs on the raw spectrum instead
                of the folded one to avoid spurious peak doubling.

        Returns:
            FitResult with correct B111 maps for folded data.

        Raises:
            DependencyError: If the resolved backend is not available.
        """
        self._require_backend_available()

        # Extract delta_f axis and shift to absolute GHz
        delta_f_ghz: NDArray = folded.folded_spectrum.coords["delta_f_ghz"].values
        abs_freq_ghz = D_ZFS + delta_f_ghz
        n_df = len(abs_freq_ghz)

        spec_vals = folded.folded_spectrum.values  # (n_pol, ny, nx, n_df)
        pol_labels = list(folded.folded_spectrum.coords["polarity"].values)

        # Build xr.DataArray: (n_pol, 1, ny, nx, n_df) matching fit() expected dims
        data_5d = np.expand_dims(spec_vals, axis=1)
        data_xr = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": pol_labels,
                "freq_range": ["folded"],
            },
        )

        # Frequency array in absolute GHz: shape (1, n_df)
        freq_2d = abs_freq_ghz.reshape(1, n_df)

        # Resolve auto model if needed
        if self._model is None:
            if raw_data is not None:
                detection_flat = raw_data.reshape(
                    raw_data.shape[0], raw_data.shape[1], -1, raw_data.shape[-1]
                )
            else:
                n_freq = data_5d.shape[-1]
                detection_flat = data_5d.reshape(data_5d.shape[0], data_5d.shape[1], -1, n_freq)
            self._resolve_auto_model(detection_flat)

        if self._model is None:
            msg = "Model must be set before fitting"
            raise ModelNotResolvedError(msg)
        model: Model = self._model

        # Start from this manager's active constraints so caller-provided overrides
        # (e.g. fit_folded_odmr(..., constraints=...)) are preserved.
        folded_constraints: dict[str, dict[str, float | str]] = {}
        for param_name, constraint in self.constraints.items():
            folded_constraints[param_name] = {
                "vmin": float(constraint.vmin),
                "vmax": float(constraint.vmax),
                "constraint_type": str(constraint.constraint_type),
            }

        # Override contrast/offset bounds for folded spectra (baseline ~1.0 from
        # averaging two branches), while keeping center/width constraints unchanged.
        for param_name, param_type in model.parameter_types.items():
            if param_type == "contrast":
                folded_constraints[param_name] = {
                    "vmin": 0.001,
                    "vmax": 1.0,
                    "constraint_type": "LOWER_UPPER",
                }
            elif param_type == "offset":
                folded_constraints[param_name] = {
                    "vmin": -0.5,
                    "vmax": 3.0,
                    "constraint_type": "LOWER_UPPER",
                }

        folded_mgr = FitManager(
            model.name,
            constraints=folded_constraints,
            freq_cutoff=self._freq_cutoff,
            settings=self._settings,
            backend=self._backend,
        )

        raw = folded_mgr.fit(data_xr, freq_2d, pixel_spacing=pixel_spacing)

        # Return standard FitResult; folded status is carried in metadata.
        params = {k: np.array(v) for k, v in raw.parameters.items()}
        return FitResult(
            parameters=params,
            scan_dimensions=raw.scan_dimensions,
            pixel_spacing=pixel_spacing,
            model_name=model.name,
            metadata={**raw.metadata, "folded_fit": True},
        )
