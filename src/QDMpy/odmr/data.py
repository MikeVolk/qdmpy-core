"""Data structures for ODMR spectroscopy in QDM analysis.

Convention: All frequency values are in GHz. Raw Hz input from MATLAB files
is converted at the ``from_numpy()`` input boundary.

This module provides the `ODMRData` class, which serves as the fundamental data
structure for handling Optically Detected Magnetic Resonance (ODMR) spectroscopic
measurements using xarray for named-dimension data representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from QDMpy.constants import GAMMA_NV
from QDMpy.exceptions import DataValidationError

if TYPE_CHECKING:
    from QDMpy.odmr.io import BaseLoader

EXPECTED_DIMS = ("polarity", "freq_range", "y", "x", "freq_idx")
POLARITY_LABELS = ["neg", "pos"]
FRANGE_LABELS = ["low", "high"]

NV_FREQ_MIN_GHZ = 2.0
NV_FREQ_MAX_GHZ = 3.5


def validate_frequencies(frequencies: NDArray) -> None:
    """Validate frequency array for ODMR fitting.

    Checks finiteness and monotonicity (errors). Checks NV diamond range
    (warning only, since non-standard experiments exist).

    Args:
        frequencies: 1D or 2D frequency array in GHz.

    Raises:
        DataValidationError: If frequencies contain non-finite values or
            are not monotonically increasing.
    """
    frequencies = np.asarray(frequencies)

    if frequencies.size == 0:
        msg = "Frequency array must not be empty"
        raise DataValidationError(msg)

    if not np.all(np.isfinite(frequencies)):
        msg = "Frequency array contains non-finite values"
        raise DataValidationError(msg)

    if frequencies.ndim <= 1:
        rows = [frequencies]
    else:
        rows = [frequencies[i] for i in range(frequencies.shape[0])]

    for row in rows:
        if row.size > 1 and not np.all(np.diff(row) > 0):
            msg = "Frequency array must be monotonically increasing"
            raise DataValidationError(msg)

    flat = frequencies.ravel()
    if flat.min() < NV_FREQ_MIN_GHZ or flat.max() > NV_FREQ_MAX_GHZ:
        logger.warning(
            f"Frequencies [{flat.min():.3f}, {flat.max():.3f}] GHz "
            f"are outside expected NV diamond range "
            f"[{NV_FREQ_MIN_GHZ}, {NV_FREQ_MAX_GHZ}] GHz"
        )


class ODMRData(BaseModel):
    """Represents raw and processed ODMR data backed by an xr.DataArray.

    The underlying DataArray has five named dimensions:
    - polarity: measurement field polarity
    - freq_range: low vs high frequency band
    - y: spatial row
    - x: spatial column
    - freq_idx: frequency sweep index within a range

    Attributes:
        data: xr.DataArray with the named dimensions above.
        metadata: Additional metadata associated with the data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: xr.DataArray
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data")
    @classmethod
    def validate_data_array(cls: type[ODMRData], v: xr.DataArray) -> xr.DataArray:
        """Validate ODMR xarray DataArray at construction time."""
        if not isinstance(v, xr.DataArray):
            msg = f"ODMR data must be an xr.DataArray, got {type(v).__name__}"
            raise DataValidationError(msg)
        if v.dims != EXPECTED_DIMS:
            msg = f"ODMR data must have dims {EXPECTED_DIMS}, got {v.dims}"
            raise DataValidationError(msg)
        if not np.issubdtype(v.dtype, np.number):
            msg = f"ODMR data must be numeric, got dtype {v.dtype}"
            raise DataValidationError(msg)
        if "freq_ghz" not in v.coords:
            msg = "ODMR data must have a freq_ghz coordinate"
            raise DataValidationError(msg)
        return v

    @classmethod
    def from_loader(
        cls: type[ODMRData],
        loader: BaseLoader,
        loader_args: dict[str, Any] | None = None,
    ) -> ODMRData:
        """Create an ODMRData instance using a loader.

        Args:
            loader: An instantiated loader to fetch data dynamically.
            loader_args: Arguments to pass to the loader (optional).

        Returns:
            An instance populated with data loaded from the loader.

        Raises:
            DataLoadError: If the loader fails to fetch data.
        """
        logger.info(f"Loading ODMR data using loader: {loader.__class__.__name__}")
        try:
            data = loader.load(**(loader_args or {}))
            return cls(data=data)
        except Exception as e:
            logger.exception(f"Failed to load data using loader {loader.__class__.__name__}: {e}")
            from QDMpy.exceptions import DataLoadError

            msg = f"Data loading failed: {e}"
            raise DataLoadError(msg) from e

    @classmethod
    def from_numpy(
        cls: type[ODMRData],
        data: NDArray,
        scan_dimensions: tuple[int, int],
        frequencies: NDArray,
        metadata: dict[str, Any] | None = None,
    ) -> ODMRData:
        """Create ODMRData from a raw numpy array (convenience constructor).

        Args:
            data: 4D numpy array with shape (n_pol, n_frange, n_pixels, n_freqs).
            scan_dimensions: Spatial dimensions as (rows, cols).
            frequencies: Frequency array in Hz. Shape (n_freqs,) or
                (n_frange, n_freqs).
            metadata: Optional metadata dict.

        Returns:
            ODMRData instance wrapping the data as an xr.DataArray.
        """
        n_pol, n_frange = data.shape[0], data.shape[1]
        n_freqs = data.shape[-1]
        rows, cols = scan_dimensions

        data_5d = data.reshape(n_pol, n_frange, rows, cols, n_freqs)

        frequencies = np.asarray(frequencies)
        if frequencies.ndim == 1:
            freq_ghz = np.tile(frequencies, (n_frange, 1)) / 1e9
        else:
            freq_ghz = frequencies / 1e9

        validate_frequencies(freq_ghz)

        polarity_labels = POLARITY_LABELS[:n_pol]
        frange_labels = FRANGE_LABELS[:n_frange]

        da = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
            coords={
                "polarity": polarity_labels,
                "freq_range": frange_labels,
                "freq_ghz": (["freq_range", "freq_idx"], freq_ghz),
            },
        )
        return cls(data=da, metadata=metadata or {})

    @property
    def scan_dimensions(self: Self) -> tuple[int, int]:
        """Spatial dimensions (rows, cols) derived from the DataArray."""
        return (self.data.sizes["y"], self.data.sizes["x"])

    @property
    def frequencies(self: Self) -> NDArray:
        """Frequency values in GHz as a numpy array.

        Returns shape (n_frange, n_freqs) or (n_freqs,) depending on the data.
        """
        return self.data.coords["freq_ghz"].values

    @property
    def numpy(self: Self) -> NDArray:
        """Raw numpy array for performance-critical code."""
        return self.data.values

    @property
    def shape(self: Self) -> tuple[int, ...]:
        """Shape of the underlying DataArray."""
        return self.data.shape


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
