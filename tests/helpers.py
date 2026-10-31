"""Shared constructors for ODMR test data.

Every builder here sources its coordinate labels from ``qdmpy.constants`` rather
than hardcoding them. Before QEP-TEST-001 the suite had three independent copies
of the same builder, one of which still used the pre-QEP-025 ``pol_0``/``frange_0``
labels. Production selects on the canonical labels (``fitting/result.py``,
``odmr/analysis.py``, ``odmr/folding.py``), so a fixture carrying the old ones
cannot reach those code paths at all -- it fails with ``KeyError`` or, worse,
passes because the consumer happens to index positionally.

``tests/test_constants.py::test_coordinate_labels_are_canonical`` locks the
literal values in exactly one place, so a rename fails loudly there while every
builder below follows automatically.

This module is deliberately not part of ``qdmpy.testing``: the public helpers
there are documented API, whereas these carry test-only affordances such as the
square-grid assertion in :func:`make_xr_data`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from qdmpy.constants import FRANGE_LABELS, POLARITY_LABELS

ODMR_DIMS = ("polarity", "freq_range", "y", "x", "freq_idx")


def make_xr_data(numpy_4d: np.ndarray) -> xr.DataArray:
    """Convert 4D numpy (n_pol, n_frange, n_pixel, n_freq) to a 5D DataArray.

    Pixels are arranged as a square grid, so ``n_pixel`` must be a perfect square.

    Args:
        numpy_4d: Array of shape (n_pol, n_frange, n_pixel, n_freq).

    Returns:
        5D DataArray with dims (polarity, freq_range, y, x, freq_idx) and
        canonical polarity/freq_range labels.

    Raises:
        AssertionError: If n_pixel is not a perfect square.
    """
    n_pol, n_frange, n_pixel, n_freq = numpy_4d.shape
    side = int(np.sqrt(n_pixel))
    assert side * side == n_pixel, f"n_pixel={n_pixel} is not a perfect square"

    data_5d = numpy_4d.reshape(n_pol, n_frange, side, side, n_freq)
    freq_ghz = np.tile(np.linspace(2.87, 2.88, n_freq), (n_frange, 1))

    return xr.DataArray(
        data_5d,
        dims=ODMR_DIMS,
        coords={
            "polarity": POLARITY_LABELS[:n_pol],
            "freq_range": FRANGE_LABELS[:n_frange],
            "freq_ghz": (("freq_range", "freq_idx"), freq_ghz),
        },
    )


def make_odmr_xr(
    *,
    n_pol: int = 2,
    n_frange: int = 2,
    height: int = 8,
    width: int = 8,
    n_freq: int = 20,
    freq_ghz_range: tuple[float, float] = (2.82, 2.92),
    seed: int = 42,
) -> xr.DataArray:
    """Build a random 5D ODMR DataArray with canonical coordinates.

    ``n_pol`` and ``n_frange`` are genuine parameters rather than fixed at 2.
    Single-polarity data is the case QEP-FIT-001 needs: ``FitResult`` currently
    hardcodes ``n_pol=2`` in ``_normalize_resonance_shape``, so a dataset with one
    polarity yields wrong B111 with no error. Constructing that input is a
    one-argument call here.

    Args:
        n_pol: Number of polarities (1 or 2).
        n_frange: Number of frequency ranges (1 or 2).
        height: Image height in pixels.
        width: Image width in pixels.
        n_freq: Frequencies per range.
        freq_ghz_range: Inclusive (start, stop) of the frequency axis in GHz.
        seed: Seed for the random generator, for reproducibility.

    Returns:
        5D DataArray with dims (polarity, freq_range, y, x, freq_idx).
    """
    rng = np.random.default_rng(seed)
    arr = rng.random((n_pol, n_frange, height, width, n_freq))
    freq_ghz = np.linspace(*freq_ghz_range, n_freq)

    return xr.DataArray(
        arr,
        dims=ODMR_DIMS,
        coords={
            "polarity": POLARITY_LABELS[:n_pol],
            "freq_range": FRANGE_LABELS[:n_frange],
            "freq_ghz": (("freq_range", "freq_idx"), np.tile(freq_ghz, (n_frange, 1))),
        },
    )
