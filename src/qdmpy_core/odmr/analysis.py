"""Physics analysis functions for ODMR data.

This module provides analysis routines that compute derived quantities from
ODMR spectroscopic measurements, such as magnetic field calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from numpy.typing import NDArray

from qdmpy_core.constants import GAMMA_NV
from qdmpy_core.exceptions import DataValidationError

if TYPE_CHECKING:
    pass


def _validate_b111_coords(data: xr.DataArray) -> None:
    """Raise DataValidationError if polarity or freq_range coords are incomplete."""
    for label in ("neg", "pos"):
        if label not in data.coords["polarity"].values:
            found = list(data.coords["polarity"].values)
            msg = f"b111_from_dip_positions requires polarity='{label}'; found {found}"
            raise DataValidationError(msg)
    for label in ("low", "high"):
        if label not in data.coords["freq_range"].values:
            found = list(data.coords["freq_range"].values)
            msg = f"b111_from_dip_positions requires freq_range='{label}'; found {found}"
            raise DataValidationError(msg)


def b111_from_dip_positions(data: xr.DataArray) -> dict[str, NDArray]:
    """Compute B₁₁₁ from argmin dip positions — no spectral fitting required.

    Finds the frequency of minimum intensity at each pixel (argmin over ``freq_idx``)
    for each polarity/range combination and applies the Zeeman-splitting formula::

        δB[pol] = sign[pol] × (f_high − f_low) / 2 / GAMMA_NV   [µT]

    where ``sign = {neg: −1, pos: +1}`` and ``GAMMA_NV = 28.024 GHz/T``.
    Remanent and induced components are then::

        b111_remanent = (δB_neg + δB_pos) / 2
        b111_induced  = (δB_neg − δB_pos) / 2

    Accuracy degrades for noisy data or overlapping dips; use ``FitManager``
    for quantitative analysis.

    Args:
        data: xr.DataArray with dims ``(polarity, freq_range, y, x, freq_idx)``
            and coords ``polarity=['neg', 'pos']``, ``freq_range=['low', 'high']``,
            and ``freq_ghz`` of shape ``(n_frange, n_freq)``.

    Returns:
        Dict with keys ``'remanent'`` and ``'induced'`` — 2D NDArrays in µT,
        shape ``(y, x)``.

    Raises:
        DataValidationError: If required polarity or freq_range labels are absent.
    """
    _validate_b111_coords(data)

    freq_ghz_arr = data.coords["freq_ghz"].values  # (n_frange, n_freq)
    frange_labels = list(data.coords["freq_range"].values)
    i_low = frange_labels.index("low")
    i_high = frange_labels.index("high")

    _sign = {"neg": -1.0, "pos": 1.0}
    delta: dict[str, NDArray] = {}
    for pol in ("neg", "pos"):
        idx_low = data.sel(polarity=pol, freq_range="low").argmin(dim="freq_idx").values
        idx_high = data.sel(polarity=pol, freq_range="high").argmin(dim="freq_idx").values
        dip_low = freq_ghz_arr[i_low][idx_low]
        dip_high = freq_ghz_arr[i_high][idx_high]
        delta[pol] = _sign[pol] * (dip_high - dip_low) / 2.0 / GAMMA_NV * 1e6

    return {
        "remanent": (delta["neg"] + delta["pos"]) / 2.0,
        "induced": (delta["neg"] - delta["pos"]) / 2.0,
    }
