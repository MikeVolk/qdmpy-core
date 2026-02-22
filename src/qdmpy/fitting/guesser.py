"""Parameter estimation for ODMR fitting.

This module provides the ParameterGuesser class which generates initial
parameter guesses for GPU-accelerated ODMR fitting operations.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.constants import DEFAULT_VMAX, DEFAULT_VMIN
from qdmpy.exceptions import ParameterError
from qdmpy.fitting.guess import cumsum_center, cumsum_contrast, cumsum_width
from qdmpy.fitting.models import Model


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
        ├── cumsum_center(data, freq)          → (n_pol, n_frange, n_pixel)
        │     prange(n_pol × n_frange × n_pixel)
        │     normalize_pixel → freq[argmin|norm−0.5|]
        │
        ├── cumsum_width(data, freq, vmin, vmax) → (n_pol, n_frange, n_pixel)
        │     prange(n_pol × n_frange × n_pixel)
        │     normalize_pixel → |freq[ridx] − freq[lidx]|
        │     vmin/vmax are model-specific:
        │       ESR14N  0.35 / 0.65
        │       ESR15N  0.40 / 0.60
        │       ESRSINGLE (default_vmin) / (default_vmax)
        │
        └── np.zeros(...)                      → offset (n_pol, n_frange, n_pixel)

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
            logger.debug(f"Guessing {param_type} parameters")

            if param_type == "center":
                param_values = cumsum_center(flat_data, self._f_ghz)
            elif param_type == "contrast":
                param_values = cumsum_contrast(flat_data)
            elif param_type == "width":
                # Use model-specific cumsum thresholds; tighter window for multi-peak models.
                # Matches QDMpy_old._core.fit.Fit._cumsum_width() n_peaks-based selection.
                if self._model.n_peaks == 2:  # ESR15N: two close hyperfine lines
                    vmin, vmax = 0.4, 0.6
                elif self._model.n_peaks == 3:  # ESR14N: three hyperfine lines
                    vmin, vmax = 0.35, 0.65
                else:  # ESRSINGLE: single dip
                    vmin, vmax = DEFAULT_VMIN, DEFAULT_VMAX
                param_values = cumsum_width(flat_data, self._f_ghz, vmin, vmax)
            elif param_type == "offset":
                param_values = np.zeros((n_pol, n_frange, n_pixel))
            else:
                msg = f"Unknown parameter type: {param_type}"
                raise ParameterError(msg)

            result[:, :, :, idx] = param_values

        self._cache = np.ascontiguousarray(result, dtype=np.float32)
        return self._cache

    def reset(self: Self) -> None:
        """Clear the cached initial parameters."""
        self._cache = None
