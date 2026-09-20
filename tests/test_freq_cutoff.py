"""Unit tests for the FreqCutoff value object (QEP-070 phase 1).

Message-parity coverage for the parsing/validation logic extracted from
FitManager's inline freq_cutoff dict handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdmpy.exceptions import DataValidationError
from qdmpy.fitting.freq_cutoff import FreqCutoff, FreqCutoffBounds


class TestFromRaw:
    """FreqCutoff.from_raw() schema/type/range validation and normalization."""

    def test_none_returns_none(self) -> None:
        assert FreqCutoff.from_raw(None) is None

    def test_empty_dict_returns_none(self) -> None:
        assert FreqCutoff.from_raw({}) is None

    def test_all_none_bounds_normalize_to_none(self) -> None:
        assert FreqCutoff.from_raw({"low": {"min": None, "max": None}}) is None

    def test_passthrough_of_existing_instance(self) -> None:
        cutoff = FreqCutoff(low=FreqCutoffBounds(min=2.8, max=2.9))
        assert FreqCutoff.from_raw(cutoff) is cutoff

    def test_non_dict_raises(self) -> None:
        with pytest.raises(DataValidationError, match="must be a dictionary"):
            FreqCutoff.from_raw("not a dict")  # type: ignore[arg-type]

    def test_unknown_range_key_raises(self) -> None:
        with pytest.raises(DataValidationError, match="unknown range key"):
            FreqCutoff.from_raw({"middle": {"min": 2.86}})

    def test_range_bounds_non_dict_raises(self) -> None:
        with pytest.raises(DataValidationError, match=r"freq_cutoff\['low'\] must be a dictionary"):
            FreqCutoff.from_raw({"low": "not a dict"})

    def test_unknown_bound_key_raises(self) -> None:
        with pytest.raises(DataValidationError, match="unknown keys"):
            FreqCutoff.from_raw({"low": {"minimum": 2.86}})

    def test_non_numeric_bound_raises(self) -> None:
        with pytest.raises(DataValidationError, match="must be a number or None"):
            FreqCutoff.from_raw({"low": {"min": "2.86"}})

    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(DataValidationError, match="must be <="):
            FreqCutoff.from_raw({"low": {"min": 2.88, "max": 2.87}})

    def test_numpy_scalar_coercion(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"min": np.float64(2.86), "max": np.int32(3)}})
        assert cutoff is not None
        assert cutoff.low is not None
        assert cutoff.low.min == pytest.approx(2.86)
        assert cutoff.low.max == pytest.approx(3.0)

    def test_valid_low_and_high(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"max": 2.86}, "high": {"min": 2.89}})
        assert cutoff is not None
        assert cutoff.low == FreqCutoffBounds(min=None, max=2.86)
        assert cutoff.high == FreqCutoffBounds(min=2.89, max=None)


class TestValidateForNRanges:
    """validate_for_n_ranges() range-count and single-range-'high' guards."""

    def test_n_frange_one_allows_low(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"min": 2.86}})
        assert cutoff is not None
        cutoff.validate_for_n_ranges(1)  # should not raise

    def test_n_frange_one_rejects_high(self) -> None:
        cutoff = FreqCutoff.from_raw({"high": {"min": 2.86}})
        assert cutoff is not None
        with pytest.raises(DataValidationError, match="single-range"):
            cutoff.validate_for_n_ranges(1)

    def test_n_frange_two_allows_both(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"max": 2.86}, "high": {"min": 2.89}})
        assert cutoff is not None
        cutoff.validate_for_n_ranges(2)  # should not raise

    def test_n_frange_three_rejects(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"min": 2.86}})
        assert cutoff is not None
        with pytest.raises(DataValidationError, match="only supported for 1 or 2"):
            cutoff.validate_for_n_ranges(3)


class TestBoundsForRange:
    """bounds_for_range() range-index-to-bounds mapping."""

    def test_single_range_uses_low(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"min": 2.86}})
        assert cutoff is not None
        assert cutoff.bounds_for_range(0, 1) == FreqCutoffBounds(min=2.86, max=None)

    def test_two_range_irange_zero_uses_low(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"max": 2.86}, "high": {"min": 2.89}})
        assert cutoff is not None
        assert cutoff.bounds_for_range(0, 2) == FreqCutoffBounds(min=None, max=2.86)

    def test_two_range_irange_one_uses_high(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"max": 2.86}, "high": {"min": 2.89}})
        assert cutoff is not None
        assert cutoff.bounds_for_range(1, 2) == FreqCutoffBounds(min=2.89, max=None)

    def test_missing_range_returns_none(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"max": 2.86}})
        assert cutoff is not None
        assert cutoff.bounds_for_range(1, 2) is None


class TestApplyToRange:
    """apply_to_range() masking, min-points enforcement, and pass-through."""

    def test_no_bounds_returns_inputs_unchanged(self) -> None:
        cutoff = FreqCutoff.from_raw({"high": {"min": 2.89}})
        assert cutoff is not None
        data = np.ones((2, 20))
        freq = np.linspace(2.80, 2.86, 20)
        out_data, out_freq = cutoff.apply_to_range(data, freq, 0, 2, min_points=10)
        assert out_data is data
        assert out_freq is freq

    def test_masks_below_min_and_above_max(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"min": 2.83, "max": 2.86}})
        assert cutoff is not None
        freq = np.linspace(2.80, 2.90, 20)
        data = np.arange(20.0).reshape(1, 20)
        out_data, out_freq = cutoff.apply_to_range(data, freq, 0, 1, min_points=5)
        assert np.all(out_freq >= 2.83)
        assert np.all(out_freq <= 2.86)
        assert out_data.shape[-1] == out_freq.size
        assert out_freq.size < freq.size

    def test_nothing_masked_returns_inputs_unchanged(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"min": 2.0, "max": 4.0}})
        assert cutoff is not None
        freq = np.linspace(2.80, 2.90, 20)
        data = np.ones((1, 20))
        out_data, out_freq = cutoff.apply_to_range(data, freq, 0, 1, min_points=5)
        assert out_data is data
        assert out_freq is freq

    def test_too_few_points_raises_with_low_label(self) -> None:
        cutoff = FreqCutoff.from_raw({"low": {"max": 2.805}})
        assert cutoff is not None
        freq = np.linspace(2.80, 2.90, 20)
        data = np.ones((1, 20))
        with pytest.raises(DataValidationError, match=r"range 'low'.*at least 10"):
            cutoff.apply_to_range(data, freq, 0, 1, min_points=10)

    def test_too_few_points_raises_with_high_label(self) -> None:
        cutoff = FreqCutoff.from_raw({"high": {"min": 2.895}})
        assert cutoff is not None
        freq = np.linspace(2.80, 2.90, 20)
        data = np.ones((1, 20))
        with pytest.raises(DataValidationError, match=r"range 'high'.*at least 10"):
            cutoff.apply_to_range(data, freq, 1, 2, min_points=10)
