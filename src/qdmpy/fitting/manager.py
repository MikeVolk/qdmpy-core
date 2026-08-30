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
from dataclasses import dataclass
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
    BackendFitOutput,
    FitBackend,
    FitBackendOptions,
    resolve_backend,
    with_forced_availability,
)
from qdmpy.fitting.constraints import (
    CONSTRAINT_TYPES,
    Constraint,
    ConstraintManager,
    ConstraintOverride,
    constraint_type_indices,
    constraints_to_array,
)
from qdmpy.fitting.freq_cutoff import FreqCutoff
from qdmpy.fitting.guess import guess_model
from qdmpy.fitting.guesser import ParameterGuesser
from qdmpy.fitting.models import Model, ModelRegistry
from qdmpy.fitting.result import FitResult
from qdmpy.odmr._validators import validate_frequencies
from qdmpy.settings import QDMpySettings, get_settings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qdmpy.odmr.folding import FoldedODMR

_MIN_FREQ_POINTS = 10

# Folded spectra average two branches together, shifting contrast/offset
# baselines relative to a single unfolded branch; fit_folded() layers these
# onto the caller's constraints for contrast/offset parameters only. Also
# used by qdmpy.fitting.refit to keep refit consistent with a folded fit.
FOLDED_CONSTRAINT_OVERRIDES: dict[str, ConstraintOverride] = {
    "contrast": ConstraintOverride(vmin=0.001, vmax=1.0, constraint_type="LOWER_UPPER"),
    "offset": ConstraintOverride(vmin=-0.5, vmax=3.0, constraint_type="LOWER_UPPER"),
}

# Undocumented legacy names accepted by _param_idx(); deprecated in favor of
# the parameter-type names used everywhere else (QEP-070).
_PARAM_ALIASES: dict[str, str] = {"resonance": "center", "mean_contrast": "contrast"}


@dataclass(frozen=True)
class _PreparedFitInputs:
    """Flattened, validated fit inputs shared by fit() and fit_folded()."""

    flat_data: NDArray  # (n_pol, n_frange, n_pixel, n_freq)
    freq_ghz: NDArray  # (n_frange, n_freq)
    scan_dimensions: tuple[int, int]

    @property
    def n_pol(self: Self) -> int:
        return self.flat_data.shape[0]

    @property
    def n_frange(self: Self) -> int:
        return self.flat_data.shape[1]

    @property
    def n_pixel(self: Self) -> int:
        return self.flat_data.shape[2]

    @property
    def n_freq(self: Self) -> int:
        return self.flat_data.shape[3]


@dataclass(frozen=True)
class _RangeFitOutputs:
    """Per-frange, per-polarity, per-pixel results from _fit_all_franges()."""

    params: NDArray  # (n_frange, n_pol, n_pixel, n_params)
    states: NDArray  # (n_frange, n_pol, n_pixel)
    chi2: NDArray  # (n_frange, n_pol, n_pixel)
    iterations: NDArray  # (n_frange, n_pol, n_pixel)
    exec_times: tuple[float, ...]


class FitManager:
    """Manages fitting operations for ODMR spectral data.

    Configuration (model, constraints) is set at construction time. Data is
    provided per-call via fit()/fit_folded(), and no call mutates the shared
    constraint state — per-range constraint overrides (mT center windows,
    folded contrast/offset bounds) are computed fresh each call and never
    written back to the manager. The same FitManager can be reused with
    different data, returning an independent FitResult each time.

    The one stateful transition is model resolution: constructing with
    ``model_name='auto'`` leaves the model unset until the first fit() or
    fit_folded() call, after which it is fixed for the manager's lifetime.
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
                ('auto', 'gpufit', 'scipy', 'torch'). Defaults to
                ``settings.fit.backend``. See :mod:`qdmpy.fitting.backends`
                (QEP-068) and :mod:`qdmpy.fitting.torch_backend` (QEP-069).
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
        self._freq_cutoff = FreqCutoff.from_raw(freq_cutoff)

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

            return with_forced_availability(resolve_backend("gpufit"), available=gpu_available)

        return resolve_backend(backend if backend is not None else self._settings.fit.backend)

    def _require_backend_available(self: Self) -> None:
        """Raise DependencyError if the resolved backend cannot run here."""
        if not self._backend.is_available():
            hint = getattr(self._backend, "install_hint", "")
            msg = (
                f"Fit backend '{self._backend.name}' is required for fitting "
                "but is not available." + (f" {hint}" if hint else "")
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

    def _validate_freq_cutoff_for_n_ranges(self: Self, n_frange: int) -> None:
        if self._freq_cutoff is not None:
            self._freq_cutoff.validate_for_n_ranges(n_frange)

    def _apply_freq_cutoff_for_range(
        self: Self,
        range_data: NDArray,
        range_freq_ghz: NDArray,
        irange: int,
        n_frange: int,
    ) -> tuple[NDArray, NDArray]:
        if self._freq_cutoff is None:
            return range_data, range_freq_ghz
        return self._freq_cutoff.apply_to_range(
            range_data, range_freq_ghz, irange, n_frange, min_points=_MIN_FREQ_POINTS
        )

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
        prepared = self._prepare_data(data, frequencies)
        return self._fit_prepared(prepared, pixel_spacing=pixel_spacing)

    @staticmethod
    def _prepare_data(data: xr.DataArray, frequencies: NDArray) -> _PreparedFitInputs:
        """Flatten ODMR data into the pipeline's working shape.

        Args:
            data: xr.DataArray with dims (polarity, freq_range, y, x, freq_idx).
            frequencies: Frequency array in GHz, shape (n_frange, n_freq).

        Returns:
            _PreparedFitInputs with 4D flattened data, 2D frequency array, and
            scan dimensions.
        """
        f_ghz = np.atleast_2d(frequencies)
        values = data.values  # (n_pol, n_frange, y, x, n_freq)
        n_pol, n_frange, n_freq = values.shape[0], values.shape[1], values.shape[-1]
        flat_data = values.reshape(n_pol, n_frange, -1, n_freq)
        scan_dimensions = (data.sizes["y"], data.sizes["x"])
        return _PreparedFitInputs(
            flat_data=flat_data, freq_ghz=f_ghz, scan_dimensions=scan_dimensions
        )

    def _resolve_model(self: Self, detection_data: NDArray) -> Model:
        """Resolve the fitting model, auto-detecting from `detection_data` if needed.

        Args:
            detection_data: 4D array (n_pol, n_frange, n_pixel, n_freq) used for
                model auto-detection when constructed with model_name='auto'.

        Returns:
            The resolved Model.

        Raises:
            ModelNotResolvedError: If the model is still unresolved after detection.
        """
        if self._model is None:
            self._resolve_auto_model(detection_data)
        if self._model is None:
            msg = "Model must be set before fitting"
            raise ModelNotResolvedError(msg)
        return self._model

    def _guess_parameters(
        self: Self, model: Model, range_data: NDArray, range_freq: NDArray
    ) -> NDArray:
        """Generate initial parameter guesses for one (post-cutoff) frequency range.

        Args:
            model: Resolved fitting model.
            range_data: Data for one range, shape (n_pol, n_pixel, n_freq).
            range_freq: Frequency axis (GHz) for this range, shape (n_freq,).

        Returns:
            NDArray of shape (n_pol, n_pixel, n_params).
        """
        guesser = ParameterGuesser(model, np.atleast_2d(range_freq))
        return guesser.guess(range_data[:, np.newaxis, :, :])[:, 0]

    def _fit_all_franges(
        self: Self,
        prepared: _PreparedFitInputs,
        model: Model,
        base_constraints: dict[str, Constraint],
    ) -> _RangeFitOutputs:
        """Fit every frequency range, applying per-range constraints and cutoffs.

        Args:
            prepared: Flattened data/frequencies from _prepare_data().
            model: Resolved fitting model.
            base_constraints: Constraint mapping each range starts from (the
                per-range mT center window, if any, is layered on top without
                mutating shared state).

        Returns:
            _RangeFitOutputs with per-frange, per-pol, per-pixel results.
        """
        n_frange, n_pol, n_pixel = prepared.n_frange, prepared.n_pol, prepared.n_pixel
        all_params = np.empty((n_frange, n_pol, n_pixel, model.n_parameters), dtype=np.float32)
        all_states = np.empty((n_frange, n_pol, n_pixel), dtype=np.int32)
        all_chi2 = np.empty((n_frange, n_pol, n_pixel), dtype=np.float32)
        all_iters = np.empty((n_frange, n_pol, n_pixel), dtype=np.int32)
        exec_times: list[float] = []

        for irange in range(n_frange):
            freq_min = prepared.freq_ghz[irange].min()
            freq_max = prepared.freq_ghz[irange].max()
            effective_constraints = self._effective_constraints_for_range(
                base_constraints, prepared.freq_ghz[irange]
            )
            logger.info(
                "Fitting frequency range {} from {:.3f}-{:.3f} GHz",
                irange,
                freq_min,
                freq_max,
            )

            range_data, range_freq = self._apply_freq_cutoff_for_range(
                prepared.flat_data[:, irange],
                prepared.freq_ghz[irange],
                irange,
                n_frange,
            )
            initial_params = self._guess_parameters(model, range_data, range_freq)
            output = self._run_backend_fit(
                range_data, range_freq, initial_params, model, effective_constraints
            )
            raw = [
                output.parameters,
                output.states,
                output.chi2,
                output.iterations,
                output.execution_time,
            ]
            shaped = self._reshape_frange_results(
                raw, data_shape=prepared.flat_data[:, irange].shape
            )
            all_params[irange] = shaped[0]
            all_states[irange] = shaped[1]
            all_chi2[irange] = shaped[2]
            all_iters[irange] = shaped[3]
            exec_times.append(shaped[4])
            logger.info("Fit finished in {:.2f} seconds", shaped[4])

        return _RangeFitOutputs(
            params=all_params,
            states=all_states,
            chi2=all_chi2,
            iterations=all_iters,
            exec_times=tuple(exec_times),
        )

    @staticmethod
    def _assemble_result(
        raw: _RangeFitOutputs,
        model: Model,
        prepared: _PreparedFitInputs,
        pixel_spacing: float,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> FitResult:
        """Transpose per-frange results into a FitResult with quality metrics.

        Args:
            raw: Per-frange fit outputs from _fit_all_franges().
            model: Resolved fitting model.
            prepared: Flattened data/frequencies (for scan_dimensions).
            pixel_spacing: Physical pixel spacing in meters.
            extra_metadata: Optional extra keys merged into FitResult.metadata.

        Returns:
            FitResult containing all fitted parameters and analysis methods.
        """
        # Transpose from (n_frange, n_pol, ...) to (n_pol, n_frange, ...)
        params_pf = np.swapaxes(raw.params, 0, 1)  # (n_pol, n_frange, n_pixel, n_params)
        states_pf = np.swapaxes(raw.states, 0, 1)  # (n_pol, n_frange, n_pixel)
        chi2_pf = np.swapaxes(raw.chi2, 0, 1)  # (n_pol, n_frange, n_pixel)

        h, w = prepared.scan_dimensions
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
            "total_fit_time": sum(raw.exec_times),
        }
        metadata = {
            "fit_timestamp": datetime.datetime.now().isoformat(),
            "quality_metrics": quality_metrics,
            **(extra_metadata or {}),
        }

        return FitResult(
            parameters=parameters,
            scan_dimensions=prepared.scan_dimensions,
            pixel_spacing=pixel_spacing,
            model_name=model.name,
            metadata=metadata,
        )

    def _base_constraints_with_overrides(
        self: Self,
        model: Model,
        overrides: Mapping[str, ConstraintOverride] | None,
    ) -> dict[str, Constraint]:
        """Return this manager's constraints, with per-parameter-type overrides applied.

        Args:
            model: Resolved fitting model (maps parameter name -> parameter type).
            overrides: Optional constraint override keyed by parameter type
                (e.g. ``{"contrast": ConstraintOverride(...)}``).

        Returns:
            `self.constraints` unchanged when no overrides are given, else a new
            dict with matching parameters replaced via `Constraint.with_updates()`.
        """
        base = self.constraints
        if not overrides:
            return base
        effective = dict(base)
        for param_name, param_type in model.parameter_types.items():
            override = overrides.get(param_type)
            if override is not None:
                effective[param_name] = base[param_name].with_updates(
                    vmin=override.vmin,
                    vmax=override.vmax,
                    constraint_type=override.constraint_type,
                )
        return effective

    def _fit_prepared(
        self: Self,
        prepared: _PreparedFitInputs,
        *,
        pixel_spacing: float,
        detection_data: NDArray | None = None,
        constraint_overrides: Mapping[str, ConstraintOverride] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> FitResult:
        """Shared internal execution path for fit() and fit_folded() (QEP-070).

        Args:
            prepared: Flattened data/frequencies from _prepare_data().
            pixel_spacing: Physical pixel spacing in meters.
            detection_data: 4D array used for auto-model detection; defaults to
                `prepared.flat_data` when not given.
            constraint_overrides: Optional per-parameter-type constraint
                overrides (used by fit_folded()).
            extra_metadata: Optional extra keys merged into FitResult.metadata.

        Returns:
            FitResult containing all fitted parameters and analysis methods.

        Raises:
            DataValidationError: If freq_cutoff is incompatible with the range count.
            ModelNotResolvedError: If the model is still unresolved after detection.
        """
        self._validate_freq_cutoff_for_n_ranges(prepared.n_frange)
        model = self._resolve_model(
            detection_data if detection_data is not None else prepared.flat_data
        )
        base_constraints = self._base_constraints_with_overrides(model, constraint_overrides)
        raw = self._fit_all_franges(prepared, model, base_constraints)
        return self._assemble_result(raw, model, prepared, pixel_spacing, extra_metadata)

    def _mt_center_window_for_range(self: Self, freq_ghz: NDArray) -> tuple[float, float] | None:
        """Compute the mT-mode center window for one frequency range, if any.

        For `constraint_units='mt'` and `center_type='LOWER_UPPER'`, a non-zero
        `center_min_mt` is interpreted as a true field window `[min, max]` (mT).
        Because non-folded fits run each branch separately, bounds are mapped as:
        - low branch:  D_ZFS-delta_max .. D_ZFS-delta_min
        - high branch: D_ZFS+delta_min .. D_ZFS+delta_max

        Args:
            freq_ghz: Frequency axis (GHz) of the active fit range.

        Returns:
            (center_min, center_max) in GHz, or None if no window applies.
        """
        constraints = self._settings.model.constraints
        if constraints.constraint_units != "mt" or constraints.center_type != "LOWER_UPPER":
            return None

        delta_min = constraints.center_min_mt * 1e-3 * GAMMA_NV
        if delta_min <= 0.0:
            return None
        delta_max = constraints.center_max_mt * 1e-3 * GAMMA_NV

        freq_min = float(np.min(freq_ghz))
        freq_max = float(np.max(freq_ghz))

        if freq_max <= D_ZFS:
            return D_ZFS - delta_max, D_ZFS - delta_min
        if freq_min >= D_ZFS:
            return D_ZFS + delta_min, D_ZFS + delta_max
        # Overlapping-around-D ranges are uncommon; keep existing symmetric bounds.
        return None

    def _effective_constraints_for_range(
        self: Self, base: dict[str, Constraint], freq_ghz: NDArray
    ) -> dict[str, Constraint]:
        """Layer the per-range mT center window onto `base` without mutating shared state.

        Args:
            base: Constraint mapping to start from (typically `self.constraints`).
            freq_ghz: Frequency axis (GHz) of the active fit range.

        Returns:
            `base` unchanged, or a new dict with `center` replaced by the window.
        """
        window = self._mt_center_window_for_range(freq_ghz)
        if window is None:
            return base
        center_min, center_max = window
        effective = dict(base)
        effective["center"] = base["center"].with_updates(
            vmin=center_min, vmax=center_max, constraint_type="LOWER_UPPER"
        )
        return effective

    def _run_backend_fit(
        self: Self,
        data: NDArray,
        freq: NDArray,
        initial_parameters: NDArray,
        model: Model,
        constraints_map: dict[str, Constraint],
    ) -> BackendFitOutput:
        """Run the resolved backend on one frequency range's data.

        Args:
            data: ODMR data with shape (n_pol, n_pixel, n_freq).
            freq: Frequency values in GHz for this range.
            initial_parameters: Initial parameter guesses.
            model: Resolved fitting model.
            constraints_map: Constraint mapping to project into backend arrays.

        Returns:
            BackendFitOutput from the resolved FitBackend.

        Raises:
            DependencyError: If the resolved backend does not support `model`.
        """
        if not self._backend.supports(model):
            msg = (
                f"Backend '{self._backend.name}' does not support model "
                f"'{model.name}' (model_id={model.model_id}). Try "
                "backend='scipy' for custom CPU-only models."
            )
            raise DependencyError(msg)

        n_freqs = data.shape[-1]
        n_pixel = data.reshape((-1, n_freqs)).shape[0]
        constraints = constraints_to_array(constraints_map, n_pixel, model.parameter_names)
        constraint_types = constraint_type_indices(constraints_map, model.parameter_names)

        return self._backend.fit(
            data=data,
            freq_ghz=freq,
            initial_parameters=initial_parameters,
            constraints=constraints,
            constraint_types=constraint_types,
            model=model,
            options=self._backend_options,
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
        if parameter in _PARAM_ALIASES:
            canonical = _PARAM_ALIASES[parameter]
            warnings.warn(
                f"Parameter alias '{parameter}' is deprecated; use '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            parameter = canonical
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
        *,
        irange: int,
        n_frange: int,
        constraint_overrides: Mapping[str, ConstraintOverride] | None = None,
    ) -> list[Any]:
        """Fit a single frequency range via the resolved backend.

        Applies the same per-range preprocessing ``_fit_all_franges()`` uses
        (freq_cutoff trimming, mT-center-window / caller-supplied constraint
        overrides) so results stay consistent with however this FitManager
        was configured. Used directly by ``fit_frange``'s own callers and by
        ``qdmpy.fitting.refit``, which must reproduce the original fit's
        cutoff/constraint configuration when refitting outlier pixels.

        Args:
            data: ODMR data with shape (n_pol, n_pixel, n_freq).
            freq: Full (pre-cutoff) frequency axis in GHz for this range.
            initial_parameters: Initial parameter guesses.
            irange: Index of this frequency range among ``n_frange`` ranges.
            n_frange: Total number of frequency ranges being fit.
            constraint_overrides: Optional per-parameter-type constraint
                overrides (e.g. folded-fit contrast/offset bounds), layered
                onto this manager's constraints before the mT-center-window.

        Returns:
            List containing [fit_params, states, chi_squares, iterations, exec_time].

        Raises:
            DependencyError: If the resolved backend is not available, or does
                not support the current model.
            ModelNotResolvedError: If called before the model is resolved.
            DataValidationError: If freq_cutoff is incompatible with n_frange.
        """
        self._require_backend_available()

        if self._model is None:
            msg = "Model must be set before fitting"
            raise ModelNotResolvedError(msg)
        model: Model = self._model

        self._validate_freq_cutoff_for_n_ranges(n_frange)
        base = self._base_constraints_with_overrides(model, constraint_overrides)
        effective = self._effective_constraints_for_range(base, freq)
        range_data, range_freq = self._apply_freq_cutoff_for_range(data, freq, irange, n_frange)

        output = self._run_backend_fit(range_data, range_freq, initial_parameters, model, effective)
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
        because the pyGpufit kernels use GHz throughout. Only the trailing axis is dropped
        when it's a singleton (per-pixel scalars like states/chi2/iterations) — n_pol and
        n_pixel are always kept, even when either is 1, so callers can rely on a fixed rank.

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
                arr = result.reshape((n_pol, n_pix, -1))
                reshaped.append(arr[..., 0] if arr.shape[-1] == 1 else arr)
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

        data_xr, freq_2d = folded.to_fit_inputs()
        self._validate_inputs(data_xr, freq_2d)
        prepared = self._prepare_data(data_xr, freq_2d)

        detection_data = None
        if raw_data is not None:
            detection_data = raw_data.reshape(
                raw_data.shape[0], raw_data.shape[1], -1, raw_data.shape[-1]
            )

        return self._fit_prepared(
            prepared,
            pixel_spacing=pixel_spacing,
            detection_data=detection_data,
            constraint_overrides=FOLDED_CONSTRAINT_OVERRIDES,
            extra_metadata={"folded_fit": True},
        )
