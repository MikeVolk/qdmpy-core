"""Fit backend seam: swappable optimizers behind a single interface.

QEP-068. This module is the only place ``pygpufit`` may be imported — every
availability check and every call into the GPU library is funneled through
:class:`GpufitBackend`. ``FitManager`` depends on the :class:`FitBackend`
protocol, never on pygpufit directly, which makes the optimizer swappable
(GPU, CPU, or a test fake) without touching fitting logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from qdmpy.exceptions import DependencyError, ParameterError
from qdmpy.fitting.constraints import CONSTRAINT_TYPES
from qdmpy.fitting.models import Model

ESTIMATOR_ID = {"LSE": 0, "MLE": 1}


@dataclass(frozen=True)
class BackendFitOutput:
    """Raw per-fit output; one row per flattened (polarity, pixel) fit.

    Attributes:
        parameters: Fitted parameters, shape (n_fits, n_params).
        states: Convergence state codes, shape (n_fits,). 0 == converged,
            matching the gpufit convention.
        chi2: Chi-squared values, shape (n_fits,).
        iterations: Iteration counts, shape (n_fits,).
        execution_time: Wall-clock fit time in seconds.
    """

    parameters: NDArray
    states: NDArray
    chi2: NDArray
    iterations: NDArray
    execution_time: float


class FitBackendOptions(BaseModel):
    """Optimizer knobs, read from ``QDMpySettings.fit`` at fit time."""

    model_config = ConfigDict(frozen=True)

    estimator: str = "MLE"
    max_number_iterations: int = 1000
    tolerance: float = 1e-10


@runtime_checkable
class FitBackend(Protocol):
    """Everything FitManager needs to know about an optimizer.

    A backend is a swappable seam: FitManager calls only these three
    methods and never imports an optimizer library directly.
    """

    name: str

    def is_available(self) -> bool:
        """Return True if this backend can run on the current machine."""
        ...

    def supports(self, model: Model) -> bool:
        """Return True if this backend can fit the given model."""
        ...

    def fit(
        self,
        data: NDArray,
        freq_ghz: NDArray,
        initial_parameters: NDArray,
        constraints: NDArray,
        constraint_types: NDArray,
        model: Model,
        options: FitBackendOptions,
    ) -> BackendFitOutput:
        """Fit flattened ``data`` (n_fits, n_freq) and return per-fit results."""
        ...


class GpufitBackend:
    """GPU-accelerated fitting via pyGpufit.

    The only class in QDMpy that imports ``pygpufit``. The import is deferred
    to method bodies so that merely importing this module (or constructing
    this backend) never touches the native library — only ``is_available()``
    and ``fit()`` do.
    """

    name = "gpufit"

    def is_available(self: GpufitBackend) -> bool:
        """Return True if pyGpufit can be imported on this machine."""
        from qdmpy.settings import is_pygpufit_available

        return is_pygpufit_available()

    def supports(self: GpufitBackend, model: Model) -> bool:
        """Return True for any model with a real gpufit model_id (>= 0)."""
        return model.model_id >= 0

    def fit(
        self: GpufitBackend,
        data: NDArray,
        freq_ghz: NDArray,
        initial_parameters: NDArray,
        constraints: NDArray,
        constraint_types: NDArray,
        model: Model,
        options: FitBackendOptions,
    ) -> BackendFitOutput:
        """Fit via ``pygpufit.gpufit.fit_constrained``.

        Raises:
            DependencyError: If pyGpufit cannot be imported.
        """
        if not self.is_available():
            msg = "pyGpufit is required for the 'gpufit' backend but is not available"
            raise DependencyError(msg)

        import pygpufit.gpufit as gf

        n_freqs = data.shape[-1]
        data_reshaped = data.reshape((-1, n_freqs))
        n_params = initial_parameters.shape[-1]
        initial_parameters_reshaped = initial_parameters.reshape((-1, n_params))

        # All values (freq, center, width, constraints) are kept in GHz.
        # The pyGpufit ESR kernels have AHYP hardcoded in GHz (ahyp=0.0015 for 15N,
        # ahyp=0.002158 for 14N) so any Hz conversion breaks the hyperfine splitting.
        results = gf.fit_constrained(
            data=np.ascontiguousarray(data_reshaped, dtype=np.float32),
            user_info=np.ascontiguousarray(freq_ghz, dtype=np.float32),
            constraints=np.ascontiguousarray(constraints, dtype=np.float32),
            constraint_types=constraint_types,
            initial_parameters=np.ascontiguousarray(
                initial_parameters_reshaped, dtype=np.float32
            ),
            weights=None,
            model_id=model.model_id,
            max_number_iterations=options.max_number_iterations,
            tolerance=options.tolerance,
            estimator_id=ESTIMATOR_ID[options.estimator],
        )
        params, states, chi2, iterations, exec_time = results
        return BackendFitOutput(
            parameters=np.asarray(params),
            states=np.asarray(states),
            chi2=np.asarray(chi2),
            iterations=np.asarray(iterations),
            execution_time=float(exec_time),
        )


class ScipyBackend:
    """CPU fitting via ``scipy.optimize.least_squares``.

    Fits every pixel independently using ``Model.func`` as the residual
    model, so it supports any registered model — including pure-Python
    custom models with ``model_id = -1`` that ``GpufitBackend`` cannot run.
    Slower than GPU fitting; intended for custom models, small ROIs, and
    GPU-less machines/CI, not production-sized 2k x 2k frames.
    """

    name = "scipy"

    def is_available(self: ScipyBackend) -> bool:
        """Always available — scipy is a core runtime dependency."""
        return True

    def supports(self: ScipyBackend, model: Model) -> bool:  # noqa: ARG002
        """Support any model; least_squares only needs ``model.func``."""
        return True

    def fit(
        self: ScipyBackend,
        data: NDArray,
        freq_ghz: NDArray,
        initial_parameters: NDArray,
        constraints: NDArray,
        constraint_types: NDArray,
        model: Model,
        options: FitBackendOptions,
    ) -> BackendFitOutput:
        """Fit via per-pixel ``scipy.optimize.least_squares``."""
        if options.estimator != "LSE":
            logger.warning(
                "ScipyBackend only supports least-squares (LSE); ignoring "
                "estimator={!r}",
                options.estimator,
            )

        n_freqs = data.shape[-1]
        data_reshaped = np.asarray(data, dtype=np.float64).reshape((-1, n_freqs))
        n_fits = data_reshaped.shape[0]
        n_params = initial_parameters.shape[-1]
        initial_reshaped = np.asarray(initial_parameters, dtype=np.float64).reshape(
            (n_fits, n_params)
        )
        freq_ghz_1d = np.asarray(freq_ghz, dtype=np.float64).reshape(-1)
        lower, upper = self._bounds_from_constraints(constraints, constraint_types, n_params)

        start = time.perf_counter()
        params, states, chi2, iterations = self._fit_all_pixels(
            model, freq_ghz_1d, data_reshaped, initial_reshaped, lower, upper, options
        )
        return BackendFitOutput(
            parameters=params,
            states=states,
            chi2=chi2,
            iterations=iterations,
            execution_time=time.perf_counter() - start,
        )

    @staticmethod
    def _fit_all_pixels(
        model: Model,
        freq_ghz: NDArray,
        data: NDArray,
        initial: NDArray,
        lower: NDArray,
        upper: NDArray,
        options: FitBackendOptions,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        from scipy.optimize import least_squares

        n_fits, n_params = initial.shape
        params_out = np.empty((n_fits, n_params), dtype=np.float32)
        states_out = np.empty(n_fits, dtype=np.int32)
        chi2_out = np.empty(n_fits, dtype=np.float32)
        iterations_out = np.empty(n_fits, dtype=np.int32)

        for i in range(n_fits):
            target = data[i]

            def residuals(p: NDArray, _target: NDArray = target) -> NDArray:
                return model.func(freq_ghz, p[np.newaxis, :])[0] - _target

            # least_squares (unlike gpufit) rejects an x0 outside its bounds;
            # clip rather than raise, matching gpufit's tolerant behaviour.
            x0 = np.clip(initial[i], lower[i], upper[i])
            result = least_squares(
                residuals,
                x0,
                bounds=(lower[i], upper[i]),
                max_nfev=options.max_number_iterations,
                xtol=options.tolerance,
                ftol=options.tolerance,
            )
            params_out[i] = result.x
            states_out[i] = 0 if result.success else 1
            chi2_out[i] = float(np.sum(result.fun**2))
            iterations_out[i] = result.nfev

        return params_out, states_out, chi2_out, iterations_out

    @staticmethod
    def _bounds_from_constraints(
        constraints: NDArray,
        constraint_types: NDArray,
        n_params: int,
    ) -> tuple[NDArray, NDArray]:
        """Map the (n_pixel, 2*n_params) constraint array to scipy bounds.

        ``constraint_types`` (one entry per parameter) selects which bound
        columns are active: LOWER/LOWER_UPPER activates vmin, UPPER/LOWER_UPPER
        activates vmax, FREE leaves both at +/-inf regardless of the array.
        """
        constraints = np.asarray(constraints, dtype=np.float64)
        n_pixel = constraints.shape[0]
        lower = np.full((n_pixel, n_params), -np.inf)
        upper = np.full((n_pixel, n_params), np.inf)
        for j in range(n_params):
            ctype = CONSTRAINT_TYPES[constraint_types[j]]
            if ctype in ("LOWER", "LOWER_UPPER"):
                lower[:, j] = constraints[:, 2 * j]
            if ctype in ("UPPER", "LOWER_UPPER"):
                upper[:, j] = constraints[:, 2 * j + 1]
        return lower, upper


class _ForcedAvailability:
    """Wraps a backend, pinning ``is_available()`` to a fixed value.

    Backs the deprecated ``FitManager(gpu_available=...)`` override: it lets
    callers force the availability decision without touching the real
    dependency check, exactly like the boolean flag it replaces.
    """

    def __init__(self: _ForcedAvailability, backend: FitBackend, available: bool) -> None:
        self._backend = backend
        self._available = available
        self.name = backend.name

    def is_available(self: _ForcedAvailability) -> bool:
        return self._available

    def supports(self: _ForcedAvailability, model: Model) -> bool:
        return self._backend.supports(model)

    def fit(
        self: _ForcedAvailability,
        data: NDArray,
        freq_ghz: NDArray,
        initial_parameters: NDArray,
        constraints: NDArray,
        constraint_types: NDArray,
        model: Model,
        options: FitBackendOptions,
    ) -> BackendFitOutput:
        return self._backend.fit(
            data, freq_ghz, initial_parameters, constraints, constraint_types, model, options
        )


def with_forced_availability(backend: FitBackend, *, available: bool) -> FitBackend:
    """Wrap ``backend`` so ``is_available()`` always returns ``available``.

    Backs the deprecated ``FitManager(gpu_available=...)`` override without
    exposing the wrapper class itself.
    """
    return _ForcedAvailability(backend, available)


_BACKENDS_BY_NAME: dict[str, type] = {}


def _register_backend_name(name: str, factory: type) -> None:
    _BACKENDS_BY_NAME[name] = factory


_register_backend_name("gpufit", GpufitBackend)
_register_backend_name("scipy", ScipyBackend)


def resolve_backend(spec: FitBackend | str | None) -> FitBackend:
    """Resolve a backend spec into a concrete :class:`FitBackend`.

    Resolution never touches the underlying dependency — availability is
    checked lazily, only when a fit is actually attempted (via
    ``backend.is_available()`` / ``FitManager._require_backend_available()``).
    This mirrors QEP-029's configuration/execution split: constructing a
    ``FitManager`` (choosing a backend) must never fail just because a GPU
    happens to be unavailable on this machine; only fitting should.

    Args:
        spec: ``'auto'`` or ``None`` resolves to the gpufit backend.
            ``'gpufit'``/``'scipy'`` request a specific backend by name. Any
            other value is assumed to already be a :class:`FitBackend`
            instance and is returned unchanged.

    Returns:
        A concrete FitBackend, not yet checked for availability. ``'auto'``
        never silently falls back between backends — a missing gpufit
        install surfaces a DependencyError at fit time naming the explicit
        opt-in (``backend='scipy'``) rather than transparently switching to
        a much slower CPU fit on production-sized data.

    Raises:
        ParameterError: If ``spec`` is an unrecognised string.
    """
    if spec is None or spec == "auto":
        return GpufitBackend()

    if isinstance(spec, str):
        if spec not in _BACKENDS_BY_NAME:
            available = sorted(_BACKENDS_BY_NAME)
            msg = f"Unknown fit backend: {spec!r}. Choose from: {available}"
            raise ParameterError(msg)
        return _BACKENDS_BY_NAME[spec]()

    logger.debug("Using caller-supplied FitBackend instance: {}", getattr(spec, "name", spec))
    return spec
