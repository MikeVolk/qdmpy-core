"""Per-frange frequency cutoff value object for FitManager (QEP-070).

Extracted from ``FitManager``'s inline dict parsing so the schema, validation,
and masking logic have their own test surface independent of the fit pipeline.
"""

from __future__ import annotations

from typing import Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from qdmpy.exceptions import DataValidationError

_MIN_FREQ_POINTS = 10
_ALLOWED_RANGES = {"low", "high"}
_ALLOWED_BOUNDS = {"min", "max"}


def _coerce_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    msg = f"freq_cutoff field '{field_name}' must be a number or None, got {type(value)!r}"
    raise DataValidationError(msg)


def _parse_range_bounds(range_key: str, raw_bounds: object) -> FreqCutoffBounds:
    """Validate and coerce one range's raw ``{'min': ..., 'max': ...}`` dict.

    Raises:
        DataValidationError: If ``raw_bounds`` fails schema/type/range validation.
    """
    if not isinstance(raw_bounds, dict):
        msg = f"freq_cutoff['{range_key}'] must be a dictionary"
        raise DataValidationError(msg)
    bounds = cast("dict[str, object]", raw_bounds)

    unknown_bounds = set(bounds).difference(_ALLOWED_BOUNDS)
    if unknown_bounds:
        unknown = sorted(unknown_bounds)
        msg = (
            f"freq_cutoff['{range_key}'] has unknown keys {unknown}. "
            "Allowed keys are: ['min', 'max']"
        )
        raise DataValidationError(msg)

    min_v = _coerce_optional_float(bounds.get("min"), field_name=f"{range_key}.min")
    max_v = _coerce_optional_float(bounds.get("max"), field_name=f"{range_key}.max")
    if min_v is not None and max_v is not None and min_v > max_v:
        msg = (
            f"freq_cutoff['{range_key}'] has invalid bounds: "
            f"min ({min_v}) must be <= max ({max_v})"
        )
        raise DataValidationError(msg)

    return FreqCutoffBounds(min=min_v, max=max_v)


class FreqCutoffBounds(BaseModel):
    """Optional ``[min, max]`` GHz bounds for one frequency range."""

    model_config = ConfigDict(frozen=True)

    min: float | None = None
    max: float | None = None


class FreqCutoff(BaseModel):
    """Per-frange frequency cutoff bounds in GHz.

    Schema: ``{'low': {'min': float|None, 'max': float|None},
    'high': {'min': float|None, 'max': float|None}}``.
    """

    model_config = ConfigDict(frozen=True)

    low: FreqCutoffBounds | None = None
    high: FreqCutoffBounds | None = None

    @classmethod
    def from_raw(
        cls,
        raw: FreqCutoff | dict[str, dict[str, float | None]] | None,
    ) -> FreqCutoff | None:
        """Normalize and validate a raw ``freq_cutoff`` constructor argument.

        Returns ``None`` for ``None`` input and for a dict whose ranges all
        end up empty (no bounds supplied). Already-normalized ``FreqCutoff``
        instances pass through unchanged.

        Raises:
            DataValidationError: If ``raw`` fails schema, type, or range
                validation (unknown keys, non-numeric bounds, min > max).
        """
        if raw is None:
            return None
        if isinstance(raw, FreqCutoff):
            return raw
        if not isinstance(raw, dict):
            msg = "freq_cutoff must be a dictionary"
            raise DataValidationError(msg)

        parsed: dict[str, FreqCutoffBounds] = {}
        for range_key, bounds in raw.items():
            if range_key not in _ALLOWED_RANGES:
                msg = (
                    "freq_cutoff has unknown range key "
                    f"'{range_key}'. Allowed keys are: ['low', 'high']"
                )
                raise DataValidationError(msg)

            range_bounds = _parse_range_bounds(range_key, bounds)
            if range_bounds.min is not None or range_bounds.max is not None:
                parsed[range_key] = range_bounds

        if not parsed:
            return None
        return cls(**parsed)

    def validate_for_n_ranges(self: Self, n_frange: int) -> None:
        """Raise if this cutoff is incompatible with fitting ``n_frange`` ranges.

        Raises:
            DataValidationError: If ``'high'`` is set for a single-range fit,
                or ``n_frange`` is neither 1 nor 2.
        """
        if n_frange == 1:
            if self.high is not None:
                msg = (
                    "freq_cutoff['high'] is not valid for single-range fits. "
                    "Use 'low' for single-range (including folded) fits."
                )
                raise DataValidationError(msg)
            return
        if n_frange == 2:
            return

        msg = f"freq_cutoff is only supported for 1 or 2 frequency ranges, got {n_frange}"
        raise DataValidationError(msg)

    def bounds_for_range(self: Self, irange: int, n_frange: int) -> FreqCutoffBounds | None:
        """Return the bounds applicable to frequency range index ``irange``."""
        if n_frange == 1:
            return self.low
        return self.low if irange == 0 else self.high

    def apply_to_range(
        self: Self,
        range_data: NDArray,
        range_freq_ghz: NDArray,
        irange: int,
        n_frange: int,
        *,
        min_points: int = _MIN_FREQ_POINTS,
    ) -> tuple[NDArray, NDArray]:
        """Mask ``range_data``/``range_freq_ghz`` to this range's cutoff bounds.

        Returns the inputs unchanged if no bounds apply or nothing is masked.

        Raises:
            DataValidationError: If fewer than ``min_points`` frequencies
                remain after masking.
        """
        bounds = self.bounds_for_range(irange, n_frange)
        if bounds is None:
            return range_data, range_freq_ghz

        mask = np.ones(range_freq_ghz.shape, dtype=bool)
        if bounds.min is not None:
            mask &= range_freq_ghz >= bounds.min
        if bounds.max is not None:
            mask &= range_freq_ghz <= bounds.max

        n_kept = int(np.sum(mask))
        if n_kept < min_points:
            range_label = "low" if n_frange == 1 or irange == 0 else "high"
            msg = (
                f"freq_cutoff for range '{range_label}' keeps {n_kept} frequency points, "
                f"but at least {min_points} are required"
            )
            raise DataValidationError(msg)

        if n_kept == range_freq_ghz.size:
            return range_data, range_freq_ghz

        masked_freq = np.ascontiguousarray(range_freq_ghz[mask], dtype=np.float64)
        masked_data = np.ascontiguousarray(range_data[..., mask])
        return masked_data, masked_freq
