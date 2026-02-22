"""Public testing and tutorial utilities for QDMpy.

These helpers generate synthetic ODMR data and results suitable for tutorials,
demos, unit tests, and CI notebooks. They do not require MATLAB data files,
GPU fitting hardware, or any external resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from qdmpy.constants import D_ZFS, GAMMA_NV

if TYPE_CHECKING:
    from qdmpy.fitting.result import FitResult
    from qdmpy.odmr.data import ODMRData
    from qdmpy.result import QDMResult


def _dipole_field(shape: tuple[int, int], amplitude: float = 50.0) -> NDArray:
    """Create a simple dipole-like 2D B111 pattern (µT)."""
    H, W = shape
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    return amplitude * y * x / (y**2 + x**2 + 0.1)


def make_synthetic_odmr_data(
    shape: tuple[int, int] = (16, 16),
    n_freq: int = 50,
    model_name: str = "ESR14N",
    noise: float = 0.002,
    seed: int = 42,
    pixel_spacing: float = 4e-6,
) -> ODMRData:
    """Generate synthetic ODMR data for tutorials and tests.

    Creates realistic ODMR spectra using the requested model, with a
    dipole-like B111 field pattern embedded in the center frequencies.
    No MATLAB files or GPU fitting required.

    Args:
        shape: Spatial dimensions (height, width) of the scan.
        n_freq: Number of frequency points per range.
        model_name: One of 'ESR14N', 'ESR15N', 'ESRSINGLE'.
        noise: RMS noise added to each spectral point.
        seed: Random seed for reproducibility.
        pixel_spacing: Physical pixel size in metres (stored in metadata).

    Returns:
        ODMRData with dims (polarity=2, freq_range=2, y, x, freq_idx).

    Example:
        >>> from qdmpy.testing import make_synthetic_odmr_data
        >>> data = make_synthetic_odmr_data(shape=(8, 8))
        >>> data.data.dims
        ('polarity', 'freq_range', 'y', 'x', 'freq_idx')
    """
    from qdmpy.fitting.models import ModelRegistry
    from qdmpy.odmr.data import ODMRData

    rng = np.random.default_rng(seed)
    H, W = shape
    n_pixels = H * W

    # Applied bias field splits the two branches away from D_ZFS
    B_BIAS_T = 900e-6  # ~900 µT → low branch ~2.845 GHz, high ~2.895 GHz
    bias_shift = GAMMA_NV * B_BIAS_T  # GHz

    # Frequency axes centred on the two branches
    half_width = 0.013  # GHz — covers the ESR14N triplet + some baseline
    center_low = D_ZFS - bias_shift
    center_high = D_ZFS + bias_shift
    freq_low = np.linspace(center_low - half_width, center_low + half_width, n_freq)
    freq_high = np.linspace(center_high - half_width, center_high + half_width, n_freq)
    freqs = np.stack([freq_low, freq_high])  # (2, n_freq)

    # Synthetic remanent B111 field (µT) → frequency shift per pixel
    b111_µt = _dipole_field(shape, amplitude=50.0).flatten()  # (n_pixels,)
    b111_t = b111_µt * 1e-6  # convert to Tesla

    model = ModelRegistry.get(model_name)
    spectra = np.zeros((2, 2, n_pixels, n_freq))

    for i_pol, pol_sign in enumerate([-1, 1]):  # neg = -1, pos = +1
        # Total field along NV: applied bias ± remanent
        b_total_t = pol_sign * B_BIAS_T + b111_t  # (n_pixels,)

        for i_fr, fr_sign in enumerate([-1, 1]):  # low = -1, high = +1
            center = D_ZFS + fr_sign * GAMMA_NV * np.abs(b_total_t)  # GHz (n_pixels,)
            freq_axis = freqs[i_fr]
            params = _make_params(model_name, center, n_pixels)
            spectra[i_pol, i_fr] = model.func(freq_axis, params)  # (n_pixels, n_freq)

    spectra += rng.normal(0, noise, spectra.shape)

    data_5d = spectra.reshape(2, 2, H, W, n_freq)
    da = xr.DataArray(
        data_5d,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        coords={
            "polarity": ["neg", "pos"],
            "freq_range": ["low", "high"],
            "freq_ghz": (("freq_range", "freq_idx"), freqs),
        },
    )
    return ODMRData(data=da, metadata={"pixel_spacing": pixel_spacing})


def _make_params(model_name: str, center: NDArray, n_pixels: int) -> NDArray:
    """Build a (n_pixels, n_params) parameter array for the given model."""
    width = np.full(n_pixels, 0.0025)
    offset = np.zeros(n_pixels)
    if model_name == "ESR14N":
        contrast = np.full(n_pixels, 0.07)
        return np.column_stack([center, width, contrast, contrast, contrast, offset])
    if model_name == "ESR15N":
        contrast = np.full(n_pixels, 0.08)
        return np.column_stack([center, width, contrast, contrast, offset])
    # ESRSINGLE
    contrast = np.full(n_pixels, 0.10)
    return np.column_stack([center, width, contrast, offset])


def make_synthetic_fit_result(
    shape: tuple[int, int] = (16, 16),
    model_name: str = "ESR14N",
    seed: int = 42,
    pixel_spacing: float = 4e-6,
) -> FitResult:
    """Generate a synthetic FitResult for tutorials and tests.

    Constructs realistic fitted parameters (center, width, contrast, chi2)
    without performing any actual fitting. The center frequencies encode a
    dipole-like B111 field so that ``result.b111_remanent`` returns a
    spatially varying map.

    Args:
        shape: Spatial dimensions (height, width).
        model_name: One of 'ESR14N', 'ESR15N', 'ESRSINGLE'.
        seed: Random seed for reproducibility.
        pixel_spacing: Physical pixel size in metres.

    Returns:
        FitResult with scan_dimensions=shape and all model parameters set.

    Example:
        >>> from qdmpy.testing import make_synthetic_fit_result
        >>> res = make_synthetic_fit_result(shape=(32, 32))
        >>> res.b111_remanent.shape
        (32, 32)
    """
    from qdmpy.fitting.result import FitResult

    rng = np.random.default_rng(seed)
    H, W = shape
    n_pixels = H * W

    B_BIAS_T = 900e-6
    b111_µt = _dipole_field(shape, amplitude=50.0).flatten()  # µT
    b111_t = b111_µt * 1e-6  # T

    # shape: (n_pol=2, n_frange=2, n_pixels)
    center = np.zeros((2, 2, n_pixels))
    for i_pol, pol_sign in enumerate([-1, 1]):
        b_total_t = pol_sign * B_BIAS_T + b111_t
        for i_fr, fr_sign in enumerate([-1, 1]):
            center[i_pol, i_fr] = D_ZFS + fr_sign * GAMMA_NV * np.abs(b_total_t)

    # Width with small spatial noise
    width = 0.0025 + rng.normal(0, 0.0001, (2, 2, n_pixels))
    chi2 = rng.exponential(5e-7, (2, 2, n_pixels))
    states = np.zeros((2, 2, n_pixels), dtype=np.int32)

    if model_name == "ESR14N":
        contrast = 0.07 + rng.normal(0, 0.002, (2, 2, n_pixels))
        parameters = {
            "center": center,
            "width": width,
            "contrast_0": contrast,
            "contrast_1": contrast * 0.95,
            "contrast_2": contrast * 1.05,
            "offset": rng.normal(0, 0.001, (2, 2, n_pixels)),
            "chi2": chi2,
            "states": states,
        }
    elif model_name == "ESR15N":
        contrast = 0.08 + rng.normal(0, 0.002, (2, 2, n_pixels))
        parameters = {
            "center": center,
            "width": width,
            "contrast_0": contrast,
            "contrast_1": contrast * 0.95,
            "offset": rng.normal(0, 0.001, (2, 2, n_pixels)),
            "chi2": chi2,
            "states": states,
        }
    else:  # ESRSINGLE
        contrast = 0.10 + rng.normal(0, 0.003, (2, 2, n_pixels))
        parameters = {
            "center": center,
            "width": width,
            "contrast": contrast,
            "offset": rng.normal(0, 0.001, (2, 2, n_pixels)),
            "chi2": chi2,
            "states": states,
        }

    return FitResult(
        parameters=parameters,
        scan_dimensions=shape,
        pixel_spacing=pixel_spacing,
        model_name=model_name,
        metadata={
            "synthetic": True,
            "quality_metrics": {
                "mean_chi2": float(chi2.mean()),
                "convergence_rate": 1.0,
                "n_pixels": int(n_pixels * 4),
                "n_converged": int(n_pixels * 4),
                "total_fit_time": 0.0,
            },
        },
    )


def make_synthetic_qdm_result(
    shape: tuple[int, int] = (16, 16),
    model_name: str = "ESR14N",
    seed: int = 42,
    pixel_spacing: float = 4e-6,
) -> QDMResult:
    """Generate a synthetic QDMResult for tutorials and tests.

    Wraps :func:`make_synthetic_fit_result` in a ``QDMResult`` container.
    Accessing ``result.magnetic_map`` will trigger the Fourier reconstruction
    from the synthetic B111 map.

    Args:
        shape: Spatial dimensions (height, width).
        model_name: One of 'ESR14N', 'ESR15N', 'ESRSINGLE'.
        seed: Random seed for reproducibility.
        pixel_spacing: Physical pixel size in metres.

    Returns:
        QDMResult ready for ``b111_remanent``, ``show()``, ``magnetic_map``.

    Example:
        >>> from qdmpy.testing import make_synthetic_qdm_result
        >>> result = make_synthetic_qdm_result(shape=(32, 32))
        >>> result.b111_remanent.shape
        (32, 32)
    """
    from qdmpy.result import QDMResult

    fit_result = make_synthetic_fit_result(
        shape=shape, model_name=model_name, seed=seed, pixel_spacing=pixel_spacing
    )
    return QDMResult(fit_result=fit_result)
