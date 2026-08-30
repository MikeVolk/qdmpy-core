"""Fit backend seam: swappable optimizers behind a single interface.

QEP-068. This module is the only place ``pygpufit`` may be imported — every
availability check and every call into the GPU library is funneled through
:class:`GpufitBackend`. ``FitManager`` depends on the :class:`FitBackend`
protocol, never on pygpufit directly, which makes the optimizer swappable
without touching fitting logic. Adapters: :class:`GpufitBackend` (CUDA),
:class:`~qdmpy.fitting.torch_backend.TorchBackend` (cuda/mps/cpu, QEP-069),
:class:`ScipyBackend` (per-pixel CPU), plus test fakes via the protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from qdmpy.exceptions import DependencyError, ParameterError
from qdmpy.fitting.constraints import CONSTRAINT_TYPES
from qdmpy.fitting.models import Model, resolve_analytic_jacobian_columns

if TYPE_CHECKING:
    from collections.abc import Callable

ESTIMATOR_ID = {"LSE": 0, "MLE": 1}

# Matches TorchBackend._STATE_INVALID (torch_backend.py) -- a pixel whose
# input data contains non-finite values, distinct from gpufit's convergence
# state codes.
_STATE_INVALID = 2


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

    Non-finite input contract: ``fit()`` must never raise because some rows
    of ``data`` contain NaN/Inf (a dead sensor pixel, a masked outlier, ...).
    Only the affected row(s) may fail -- report them with a non-zero
    ``states`` entry (the specific code is backend-defined; e.g.
    :class:`GpufitBackend` uses its native non-convergence codes,
    :class:`~qdmpy.fitting.torch_backend.TorchBackend` and
    :class:`ScipyBackend` use ``2``) and continue fitting every other row
    normally. A backend that aborts the whole batch on one bad pixel is not
    a drop-in replacement for one that degrades gracefully.
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
    install_hint = (
        "Install pyGpufit, or use backend='torch' (`uv sync --extra gpu`) or backend='scipy'."
    )

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
            initial_parameters=np.ascontiguousarray(initial_parameters_reshaped, dtype=np.float32),
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
                "ScipyBackend only supports least-squares (LSE); ignoring estimator={!r}",
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

        # Resolved once, not per pixel: models without an analytic form fall
        # back to scipy's own finite differences.
        jac = _analytic_jacobian_callable(model, freq_ghz, initial) or "2-point"

        for i in range(n_fits):
            target = data[i]

            # least_squares (unlike gpufit) rejects an x0 outside its bounds;
            # clip rather than raise, matching gpufit's tolerant behaviour.
            x0 = np.clip(initial[i], lower[i], upper[i])

            if not np.isfinite(target).all():
                # least_squares raises ValueError on non-finite residuals at
                # the initial point, aborting the whole batch. Mark this
                # pixel invalid instead, matching TorchBackend's convention,
                # so one bad pixel doesn't take down every other fit.
                params_out[i] = x0
                states_out[i] = _STATE_INVALID
                chi2_out[i] = np.nan
                iterations_out[i] = 0
                continue

            def residuals(p: NDArray, _target: NDArray = target) -> NDArray:
                return model.func(freq_ghz, p[np.newaxis, :])[0] - _target

            result = least_squares(
                residuals,
                x0,
                jac=jac,
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
        """Delegate to the shared :func:`bounds_from_constraints`."""
        return bounds_from_constraints(constraints, constraint_types, n_params)


def _analytic_jacobian_callable(
    model: Model,
    freq_ghz: NDArray,
    initial: NDArray,
) -> Callable[[NDArray], NDArray] | None:
    """Adapt ``Model.jacobian`` to scipy's ``jac(p) -> (n_freq, n_params)``, or None.

    ``Model.jacobian`` returns framework-neutral columns shaped (1, n_freq) for
    a single parameter vector (QEP-073); ``least_squares`` wants them stacked
    the other way round. Probed once here so a model without an analytic form
    -- or with a malformed one -- silently keeps scipy's finite differences.
    """
    n_params = initial.shape[-1]
    cols = resolve_analytic_jacobian_columns(
        model,
        probe=lambda: model.jacobian(freq_ghz, initial[:1]),
        n_params=n_params,
        expected_shape=(1, freq_ghz.size),
        shape_of=np.shape,
    )
    if cols is None:
        return None

    def jacobian(p: NDArray) -> NDArray:
        columns = model.jacobian(freq_ghz, p[np.newaxis, :])
        return np.stack([col[0] for col in columns], axis=-1)  # ty: ignore[not-iterable]

    return jacobian


def bounds_from_constraints(
    constraints: NDArray,
    constraint_types: NDArray,
    n_params: int,
) -> tuple[NDArray, NDArray]:
    """Map the (n_pixel, 2*n_params) constraint array to (lower, upper) bounds.

    ``constraint_types`` (one entry per parameter) selects which bound
    columns are active: LOWER/LOWER_UPPER activates vmin, UPPER/LOWER_UPPER
    activates vmax, FREE leaves both at +/-inf regardless of the array.
    Shared by ScipyBackend and TorchBackend (QEP-069).
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


class AutoBackend:
    """Lazy delegate behind ``backend='auto'`` (QEP-069).

    Picks, at first use (never at construction — the config/execution split
    from QEP-029 means choosing a backend must not touch hardware):

    1. :class:`GpufitBackend` if pyGpufit is importable;
    2. else :class:`~qdmpy.fitting.torch_backend.TorchBackend` if torch is
       installed **and** a real GPU device (cuda/mps) is available;
    3. else unavailable — fitting raises a DependencyError listing all
       remedies.

    Torch-CPU and scipy are never auto-selected: a full frame on a CPU
    optimizer silently taking minutes-to-hours is worse than a clear error
    naming the explicit opt-in.
    """

    name = "auto"

    install_hint = (
        "No GPU fit backend found: pyGpufit is not installed and no torch GPU "
        "device (cuda/mps) is available. Install pyGpufit, or install torch via "
        "`uv sync --extra gpu` / `pip install 'qdmpy[gpu]'`, or explicitly opt "
        "into a CPU fit with backend='torch' or backend='scipy'."
    )

    def _delegate(self: AutoBackend) -> FitBackend | None:
        """Resolve the concrete backend.

        Intentionally uncached so test-time availability patching is always
        honoured (the underlying imports are process-cached anyway).
        """
        gpufit = GpufitBackend()
        if gpufit.is_available():
            return gpufit
        from qdmpy.fitting.torch_backend import TorchBackend, torch_gpu_device_available

        if torch_gpu_device_available():
            return TorchBackend()
        return None

    def is_available(self: AutoBackend) -> bool:
        """Return True if either gpufit or a torch GPU device can fit."""
        return self._delegate() is not None

    def supports(self: AutoBackend, model: Model) -> bool:
        """Forward to the resolved delegate; False when none is available."""
        delegate = self._delegate()
        return delegate is not None and delegate.supports(model)

    def fit(
        self: AutoBackend,
        data: NDArray,
        freq_ghz: NDArray,
        initial_parameters: NDArray,
        constraints: NDArray,
        constraint_types: NDArray,
        model: Model,
        options: FitBackendOptions,
    ) -> BackendFitOutput:
        """Fit via the resolved delegate, raising with remedies when none exists."""
        delegate = self._delegate()
        if delegate is None:
            raise DependencyError(self.install_hint)
        logger.debug("Auto backend delegating to '{}'", delegate.name)
        return delegate.fit(
            data, freq_ghz, initial_parameters, constraints, constraint_types, model, options
        )


def _torch_backend_factory() -> FitBackend:
    """Deferred import: torch_backend imports from this module (no cycle)."""
    from qdmpy.fitting.torch_backend import TorchBackend

    return TorchBackend()


_BACKENDS_BY_NAME: dict[str, Any] = {}


def _register_backend_name(name: str, factory: Any) -> None:  # noqa: ANN401
    _BACKENDS_BY_NAME[name] = factory


_register_backend_name("gpufit", GpufitBackend)
_register_backend_name("scipy", ScipyBackend)
_register_backend_name("torch", _torch_backend_factory)


def resolve_backend(spec: FitBackend | str | None) -> FitBackend:
    """Resolve a backend spec into a concrete :class:`FitBackend`.

    Resolution never touches the underlying dependency — availability is
    checked lazily, only when a fit is actually attempted (via
    ``backend.is_available()`` / ``FitManager._require_backend_available()``).
    This mirrors QEP-029's configuration/execution split: constructing a
    ``FitManager`` (choosing a backend) must never fail just because a GPU
    happens to be unavailable on this machine; only fitting should.

    Args:
        spec: ``'auto'`` or ``None`` resolves to :class:`AutoBackend`
            (gpufit if available, else torch on a real GPU device, else
            unavailable). ``'gpufit'``/``'scipy'``/``'torch'`` request a
            specific backend by name. Any other value is assumed to already
            be a :class:`FitBackend` instance and is returned unchanged.

    Returns:
        A concrete FitBackend, not yet checked for availability. ``'auto'``
        never silently falls back to a CPU optimizer — on a machine with
        neither pygpufit nor a torch GPU device, fitting raises a
        DependencyError naming the explicit opt-ins (``backend='torch'`` /
        ``backend='scipy'``) rather than transparently taking minutes on CPU.

    Raises:
        ParameterError: If ``spec`` is an unrecognised string.
    """
    if spec is None or spec == "auto":
        return AutoBackend()

    if isinstance(spec, str):
        if spec not in _BACKENDS_BY_NAME:
            available = sorted(_BACKENDS_BY_NAME)
            msg = f"Unknown fit backend: {spec!r}. Choose from: {available}"
            raise ParameterError(msg)
        return _BACKENDS_BY_NAME[spec]()

    logger.debug("Using caller-supplied FitBackend instance: {}", getattr(spec, "name", spec))
    return spec
