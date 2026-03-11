"""Parameter estimation for ODMR fitting.

This module provides the ParameterGuesser class which generates initial
parameter guesses for GPU-accelerated ODMR fitting operations.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.constants import AHYP_14N, AHYP_15N
from qdmpy.exceptions import ParameterError
from qdmpy.fitting.guess import absorption_centroid, halfpower_width, top3_contrast
from qdmpy.fitting.models import Model

# Floor for individual Lorentzian HWHM after subtracting hyperfine splitting.
# 0.3 MHz in GHz prevents negative or near-zero widths.
_MIN_WIDTH_GHZ = 0.0003

# Mean-squared residual threshold for the initial-guess quality check on the
# median spectrum. Spectra are normalized, so MSR > 0.01 suggests the guess
# is substantially off and warrants investigation.
_RESIDUAL_WARN_THRESHOLD = 0.01


class ParameterGuesser:
    """Generates initial parameter guesses for ODMR fitting.

    Encapsulates parameter estimation logic with built-in caching.
    The cache is invalidated when reset() is called (e.g. after data
    or model changes).

    Each parameter type is estimated by a dedicated ``@njit(parallel=True)``
    function that flattens ``(n_pol, n_frange, n_pixel)`` into a single
    ``prange`` so all pixels are parallelised simultaneously::

        guess(flat_data)
        ├── cumsum_contrast(data)              → (n_pol, n_frange, n_pixel)
        │     prange(n_pol × n_frange × n_pixel)
        │     nanmax, nanmin → abs((mx−mn)/mx)
        │
        ├── absorption_centroid(data, freq)    → (n_pol, n_frange, n_pixel)
        │     prange(n_pol × n_frange × n_pixel)
        │     edge baseline → absorption-weighted centroid
        │
        ├── halfpower_width(data, freq)          → (n_pol, n_frange, n_pixel)
        │     prange(n_pol × n_frange × n_pixel)
        │     half-power point search → envelope HWHM
        │     then subtract AHYP for multi-peak models:
        │       ESR14N  envelope_hwhm - AHYP_14N
        │       ESR15N  envelope_hwhm - AHYP_15N
        │       ESRSINGLE  envelope_hwhm (no correction)
        │
        └── edge-mean baseline - 1.0           → offset (n_pol, n_frange, n_pixel)

        assembled via model.parameter_types
        → (n_pol, n_frange, n_pixel, n_params) float32  [cached]

    Design note — why three separate functions instead of one combined kernel:
    A single ``@njit`` kernel computing all three parameters in one ``prange``
    would call ``normalize_pixel`` only once per pixel (~4× speedup vs old code)
    rather than twice (center + width, ~2.7× speedup). That extra ~50% was
    deliberately traded away to keep the functions independently replaceable:
    a new model may need a different contrast estimate while keeping the same
    center/width logic, or vice versa. When a second implementation exists for
    any one parameter (e.g. ``fft_center``), inject it at the call site in
    ``guess()`` without touching the others. See QEP-024 for the benchmarks.

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

        for idx, param_name in enumerate(self._model.parameter_names):
            param_type = self._model.parameter_types[param_name]
            logger.debug("Guessing {} parameters", param_type)

            if param_type == "center":
                param_values = absorption_centroid(flat_data, self._f_ghz)
            elif param_type == "contrast":
                # cumsum_contrast returns total observed dip depth (max-min)/max.
                # For multi-peak models the hyperfine peaks overlap significantly
                # (AHYP ~ linewidth), so the observed dip is dominated by the
                # central peak with partial contributions from neighbours.
                # Dividing by n_peaks would overcorrect; the total dip is a
                # reasonable starting point for each contrast_i parameter.
                param_values = top3_contrast(flat_data)
            elif param_type == "width":
                # halfpower_width measures the envelope HWHM directly from half-
                # power points (no cumsum artifacts). For multi-peak models the
                # envelope includes hyperfine splitting, so subtract AHYP to
                # recover the individual Lorentzian HWHM.
                envelope_hwhm = halfpower_width(flat_data, self._f_ghz)
                if self._model.n_peaks == 3:  # ESR14N
                    param_values = np.maximum(envelope_hwhm - AHYP_14N, _MIN_WIDTH_GHZ)
                elif self._model.n_peaks == 2:  # ESR15N
                    param_values = np.maximum(envelope_hwhm - AHYP_15N, _MIN_WIDTH_GHZ)
                else:  # ESRSINGLE — envelope = individual
                    param_values = envelope_hwhm
            elif param_type == "offset":
                # Estimate baseline from edge frequencies (mean of first + last 10%).
                # Model formula: f = 1 + offset - dips  =>  offset = baseline - 1.0
                n_freq = flat_data.shape[-1]
                n_edge = max(1, n_freq // 10)
                baseline = (
                    np.mean(flat_data[..., :n_edge], axis=-1)
                    + np.mean(flat_data[..., -n_edge:], axis=-1)
                ) / 2.0
                param_values = (baseline - 1.0).astype(np.float32)
            else:
                msg = f"Unknown parameter type: {param_type}"
                raise ParameterError(msg)

            result[:, :, :, idx] = param_values

        self._log_median_residual(flat_data, result)
        self._cache = np.ascontiguousarray(result, dtype=np.float32)
        return self._cache

    def _log_median_residual(self: Self, flat_data: NDArray, params: NDArray) -> None:
        """Log the model residual on the median spectrum for each (pol, frange).

        Evaluates the model at the guessed parameters on the per-(pol, frange)
        median spectrum. A large residual indicates a poor initial guess and
        may degrade fit convergence.

        Args:
            flat_data: 4D numpy array (n_pol, n_frange, n_pixel, n_freq).
            params: 4D numpy array (n_pol, n_frange, n_pixel, n_params).
        """
        median_data = np.median(flat_data, axis=2)  # (n_pol, n_frange, n_freq)
        median_params = np.median(params, axis=2)  # (n_pol, n_frange, n_params)
        n_pol, n_frange = flat_data.shape[:2]
        for p in range(n_pol):
            for r in range(n_frange):
                freq = self._f_ghz[r]
                predicted = self._model.func(freq, median_params[p, r])
                msr = float(np.mean((median_data[p, r] - predicted) ** 2))
                if msr > _RESIDUAL_WARN_THRESHOLD:
                    logger.warning(
                        "Poor initial guess (pol={}, frange={}): median MSR={:.4f} "
                        "exceeds threshold {:.4f}. Verify model and frequency range.",
                        p,
                        r,
                        msr,
                        _RESIDUAL_WARN_THRESHOLD,
                    )
                else:
                    logger.debug("Median guess residual (pol={}, frange={}): MSR={:.4f}", p, r, msr)

    def reset(self: Self) -> None:
        """Clear the cached initial parameters."""
        self._cache = None
