"""ODMR fitting manager for Quantum Diamond Microscopy.

Convention: All frequency values are in GHz. The pyGpufit ESR kernels have AHYP
hardcoded in GHz, so no Hz conversion is performed at any boundary.

This module provides the FitManager class which orchestrates fitting operations
for ODMR spectra from NV centers in diamond. Parameter constraints and initial
guesses are managed by dedicated modules (constraints.py and guesser.py).
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray

from qdmpy_core.exceptions import (
    DataValidationError,
    DependencyError,
    ModelNotFoundError,
    ParameterError,
)
from qdmpy_core.fitting.constraints import CONSTRAINT_TYPES, ConstraintManager
from qdmpy_core.fitting.guess import guess_model
from qdmpy_core.fitting.guesser import ParameterGuesser
from qdmpy_core.fitting.models import Model, ModelRegistry
from qdmpy_core.fitting.result import FitResult, FoldedFitResult
from qdmpy_core.odmr._validators import validate_frequencies
from qdmpy_core.settings import (
    QDMpySettings,
    get_settings,
    is_pygpufit_available,
)

if TYPE_CHECKING:
    from qdmpy_core.odmr.folding import FoldedODMR

ESTIMATOR_ID = {"LSE": 0, "MLE": 1}

# Folded-spectrum fitting domain constraints
_FOLDED_CENTER_MIN = 0.001   # GHz (1 MHz minimum Zeeman shift)
_FOLDED_CENTER_MAX = 0.080   # GHz (80 MHz ~ 2.8 mT)
_FOLDED_WIDTH_MIN = 0.001    # GHz
_FOLDED_WIDTH_MAX = 0.020    # GHz
_FOLDED_CONTRAST_MAX = 1.0   # spectrum baseline is ~1.0 (mean of two normalised ranges)


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
        settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ) -> None:
        """Initialize a FitManager with model configuration.

        Args:
            model_name: Model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
                        If 'auto', model is resolved on the first fit() call.
            constraints: Optional dict mapping parameter names to constraint kwargs
                         (vmin, vmax, constraint_type). Applied after model resolution.
            settings: Optional QDMpySettings instance (defaults to global get_settings()).
            gpu_available: Optional GPU availability override (defaults to is_pygpufit_available()).
        """
        self._settings = settings or get_settings()
        self._gpu_available = (
            gpu_available if gpu_available is not None else is_pygpufit_available()
        )
        self.estimator_id = ESTIMATOR_ID[self._settings.fit.estimator]

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
            logger.info(f"FitManager initialized with model: {self._model.name}")

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

        min_freq_points = 10
        if n_freq_array < min_freq_points:
            msg = (
                f"Need at least {min_freq_points} frequency points for fitting, got {n_freq_array}"
            )
            raise DataValidationError(msg)

        validate_frequencies(freq_2d)

    def _resolve_auto_model(self: Self, flat_data: NDArray) -> None:
        """Resolve auto model from data and initialize constraint manager.

        Args:
            flat_data: 4D array (n_pol, n_frange, n_pixel, n_freq) used for model detection.
        """
        self._model = guess_model(flat_data)
        logger.info(f"Auto-resolved model: {self._model.name}")
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
        if not self._gpu_available:
            msg = "pyGpufit is required for fitting but not installed"
            raise DependencyError(msg)

        self._validate_inputs(data, frequencies)
        f_ghz = np.atleast_2d(frequencies)

        values = data.values  # (n_pol, n_frange, y, x, n_freq)
        n_pol, n_frange = values.shape[0], values.shape[1]
        n_freq = values.shape[-1]
        flat_data = values.reshape(n_pol, n_frange, -1, n_freq)
        n_pixel = flat_data.shape[2]

        if self._model is None:
            self._resolve_auto_model(flat_data)

        model: Model = self._model  # type: ignore[assignment]
        guesser = ParameterGuesser(model, f_ghz)
        initial_params = guesser.guess(flat_data)

        all_params = np.empty((n_frange, n_pol, n_pixel, model.n_parameters), dtype=np.float32)
        all_states = np.empty((n_frange, n_pol, n_pixel), dtype=np.int32)
        all_chi2 = np.empty((n_frange, n_pol, n_pixel), dtype=np.float32)
        all_iters = np.empty((n_frange, n_pol, n_pixel), dtype=np.int32)
        exec_times: list[float] = []

        for irange in range(n_frange):
            freq_min = f_ghz[irange].min()
            freq_max = f_ghz[irange].max()
            logger.info(f"Fitting frequency range {irange} from {freq_min:.3f}-{freq_max:.3f} GHz")
            raw = self.fit_frange(flat_data[:, irange], f_ghz[irange], initial_params[:, irange])
            shaped = self._reshape_frange_results(raw, data_shape=flat_data[:, irange].shape)
            all_params[irange] = shaped[0]
            all_states[irange] = shaped[1]
            all_chi2[irange] = shaped[2]
            all_iters[irange] = shaped[3]
            exec_times.append(shaped[4])
            logger.info(f"Fit finished in {shaped[4]:.2f} seconds")

        # Transpose from (n_frange, n_pol, ...) to (n_pol, n_frange, ...)
        params_pf = np.swapaxes(all_params, 0, 1)  # (n_pol, n_frange, n_pixel, n_params)
        states_pf = np.swapaxes(all_states, 0, 1)  # (n_pol, n_frange, n_pixel)
        chi2_pf = np.swapaxes(all_chi2, 0, 1)  # (n_pol, n_frange, n_pixel)

        parameters: dict[str, NDArray] = {}
        for idx, param_name in enumerate(model.parameter_names):
            parameters[param_name] = params_pf[:, :, :, idx]
        parameters["chi2"] = chi2_pf
        parameters["states"] = states_pf

        scan_dimensions = (data.sizes["y"], data.sizes["x"])
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
            raise RuntimeError(msg)
        return self._model.parameter_names

    @property
    def n_parameter(self: Self) -> int:
        """Get the number of parameters in the model.

        Returns:
            Number of parameters for the current model.
        """
        if self._model is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise RuntimeError(msg)
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
            RuntimeError: If called before model is resolved (auto mode).
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise RuntimeError(msg)

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

    def set_free_constraints(self: Self) -> None:
        """Remove all constraints by setting all parameters to FREE."""
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise RuntimeError(msg)
        for param in self.parameter_names:
            self._constraint_manager.set_constraint(param, constraint_type="FREE")

    @property
    def constraints(self: Self) -> dict[str, list[Any]]:
        """Get current parameter constraints.

        Returns:
            Dictionary mapping parameter names to constraint lists.
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise RuntimeError(msg)
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
            raise RuntimeError(msg)
        return self._constraint_manager.to_array(n_pixel, self.parameter_names)

    def get_constraint_types(self: Self) -> NDArray:
        """Get constraint type indices for model parameters.

        Returns:
            NDArray of constraint type indices.
        """
        if self._constraint_manager is None:
            msg = "Model not yet resolved; call fit() first or specify model_name"
            raise RuntimeError(msg)
        return self._constraint_manager.get_constraint_types(self.parameter_names)

    def _param_idx(self: Self, parameter: str) -> list[int]:
        if self._model is None:
            msg = "Model not yet resolved"
            raise RuntimeError(msg)
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
            msg = "pyGpufit is required for fitting but not installed"
            raise DependencyError(msg)

        import pygpufit.gpufit as gf

        n_freqs = data.shape[-1]
        data_reshaped = data.reshape((-1, n_freqs))
        initial_parameters_reshaped = initial_parameters.reshape((-1, self.n_parameter))

        # All values (freq, center, width, constraints) are kept in GHz.
        # The pyGpufit ESR kernels have AHYP hardcoded in GHz (ahyp=0.0015 for 15N,
        # ahyp=0.002158 for 14N) so any Hz conversion breaks the hyperfine splitting.
        n_pixel = data_reshaped.shape[0]
        constraints = self.get_constraints_array(n_pixel)
        constraint_types = self.get_constraint_types()

        model: Model = self._model  # type: ignore[assignment]
        results = gf.fit_constrained(
            data=np.ascontiguousarray(data_reshaped, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(initial_parameters_reshaped, dtype=np.float32),
            weights=None,
            model_id=model.model_id,
            max_number_iterations=self._settings.fit.max_number_iterations,
            tolerance=self._settings.fit.tolerance,
            estimator_id=self.estimator_id,
        )
        return list(results)

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
    ) -> FoldedFitResult:
        """Fit a folded ODMR spectrum and return a FoldedFitResult.

        The folded spectrum has its frequency axis in Zeeman-offset (delta_f) GHz.
        Fitting uses self._model (e.g. ESR14N, ESR15N, ESRSINGLE) with
        folded-domain constraints. The resulting centre IS the Zeeman shift
        -- no D_ZFS subtraction needed.

        Args:
            folded: FoldedODMR result from SpectralFolder.fold().
            pixel_spacing: Physical pixel spacing in meters.

        Returns:
            FoldedFitResult with correct B111 maps for folded data.

        Raises:
            DependencyError: If pyGpufit is not installed.
        """
        if not self._gpu_available:
            msg = "pyGpufit is required for fitting but not installed"
            raise DependencyError(msg)

        # Extract delta_f axis from the folded spectrum coordinate
        delta_f_ghz: NDArray = folded.folded_spectrum.coords["delta_f_ghz"].values  # (n_df,)
        n_df = len(delta_f_ghz)

        # folded_spectrum dims: (polarity, y, x, freq_idx)
        # The folded spectrum is the mean of S_low and S_high, so its baseline is
        # already ~1.0 (same as a single normalised ODMR spectrum).
        spec_vals = folded.folded_spectrum.values  # (n_pol, ny, nx, n_df)
        pol_labels = list(folded.folded_spectrum.coords["polarity"].values)

        # Build xr.DataArray: (n_pol, 1, ny, nx, n_df) matching fit() expected dims
        data_5d = np.expand_dims(spec_vals, axis=1)  # (n_pol, 1, ny, nx, n_df)
        data_xr = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": pol_labels,
                "freq_range": ["folded"],
            },
        )

        # Frequency array: shape (1, n_df) for a single freq_range
        freq_2d = delta_f_ghz.reshape(1, n_df)

        # Resolve auto model if needed — use the folded 5D data for detection
        if self._model is None:
            n_freq = data_5d.shape[-1]
            flat_data = data_5d.reshape(data_5d.shape[0], data_5d.shape[1], -1, n_freq)
            self._resolve_auto_model(flat_data)

        model: Model = self._model  # type: ignore[assignment]

        # Build folded-domain constraints dynamically from the model's parameter
        # types so that ESR14N (contrast_0/1/2), ESR15N (contrast_0/1), and
        # ESRSINGLE (contrast) all get correct bounds.
        _type_bounds: dict[str, dict[str, float | str]] = {
            "center": {
                "vmin": _FOLDED_CENTER_MIN,
                "vmax": _FOLDED_CENTER_MAX,
                "constraint_type": "LOWER_UPPER",
            },
            "width": {
                "vmin": _FOLDED_WIDTH_MIN,
                "vmax": _FOLDED_WIDTH_MAX,
                "constraint_type": "LOWER_UPPER",
            },
            "contrast": {
                "vmin": 0.001,
                "vmax": _FOLDED_CONTRAST_MAX,
                "constraint_type": "LOWER_UPPER",
            },
            "offset": {
                "vmin": -0.5,
                "vmax": 3.0,
                "constraint_type": "LOWER_UPPER",
            },
        }
        folded_constraints: dict[str, dict[str, float | str]] = {}
        for param_name, param_type in model.parameter_types.items():
            if param_type in _type_bounds:
                folded_constraints[param_name] = _type_bounds[param_type]

        # Build a dedicated FitManager with folded-domain constraints.
        # Use the same model as the outer FitManager (e.g. ESR14N for 14N diamond)
        # so that the correct number of dips is fitted on the folded spectrum.
        folded_mgr = FitManager(
            model.name,
            constraints=folded_constraints,
            settings=self._settings,
            gpu_available=self._gpu_available,
        )

        raw = folded_mgr.fit(data_xr, freq_2d, pixel_spacing=pixel_spacing)

        # Copy parameters (need fresh writable arrays before new object locks them)
        params = {k: np.array(v) for k, v in raw.parameters.items()}
        return FoldedFitResult(
            parameters=params,
            scan_dimensions=raw.scan_dimensions,
            pixel_spacing=pixel_spacing,
            model_name=f"{model.name}+FOLDED",
            metadata={**raw.metadata, "folded_fit": True},
        )
