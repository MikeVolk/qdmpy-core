"""Tests for concrete field processor classes (QEP-034 Phase 2).

RED phase — all tests should FAIL until the processors are implemented.

Import paths under test:
    from qdmpy_core.field_processing import (
        HotPixelFilter,
        QuadraticBackgroundSubtractor,
        UpwardContinuation,
        BlankSubtractor,
    )

Test classes:
    TestHotPixelFilter            (27 tests)
    TestQuadraticBackgroundSubtractor  (22 tests)
    TestUpwardContinuation        (21 tests)
    TestBlankSubtractor           (15 tests)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_field_map(
    height: int = 20,
    width: int = 20,
    pixel_spacing: float = 1e-6,
    fill_value: float | None = None,
    rng: np.random.Generator | None = None,
) -> xr.DataArray:
    """Return a synthetic (H, W) field map DataArray in µT."""
    if rng is None:
        rng = np.random.default_rng(0)

    if fill_value is not None:
        values = np.full((height, width), fill_value, dtype=float)
    else:
        values = rng.uniform(-1.0, 1.0, size=(height, width))

    y_coords = np.arange(height) * pixel_spacing
    x_coords = np.arange(width) * pixel_spacing

    return xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': y_coords, 'x': x_coords},
        attrs={'pixel_spacing': pixel_spacing, 'units': 'µT'},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_field_map() -> xr.DataArray:
    """Synthetic 20×20 uniform-random field map."""
    return _make_field_map(height=20, width=20)


@pytest.fixture
def synthetic_hot_pixel_field() -> xr.DataArray:
    """20×20 field with four known hot-pixel spikes injected."""
    rng = np.random.default_rng(1)
    values = rng.uniform(-0.5, 0.5, size=(20, 20))
    # Inject clear outliers: >> 5 sigma from background
    values[5, 5] = 1000.0
    values[10, 10] = -1000.0
    values[3, 15] = 500.0
    values[17, 2] = -500.0
    y = np.arange(20) * 1e-6
    x = np.arange(20) * 1e-6
    return xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': y, 'x': x},
        attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
    )


@pytest.fixture
def synthetic_background_field() -> xr.DataArray:
    """20×20 field with a known quadratic background."""
    rng = np.random.default_rng(2)
    h, w = 20, 20
    ps = 1e-6
    y_raw = np.arange(h) * ps
    x_raw = np.arange(w) * ps
    Xg, Yg = np.meshgrid(x_raw, y_raw)
    # Quadratic surface: a=3, b=2, c=1, d=-4, e=5, f=0.5
    background = 3.0 + 2.0 * Xg / (w * ps) + 1.0 * Yg / (h * ps) + (
        -4.0 * (Xg / (w * ps)) ** 2
        + 5.0 * (Xg / (w * ps)) * (Yg / (h * ps))
        + 0.5 * (Yg / (h * ps)) ** 2
    )
    # Small noise on top
    signal = rng.uniform(-0.01, 0.01, size=(h, w))
    values = background + signal
    return xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': y_raw, 'x': x_raw},
        attrs={'pixel_spacing': ps, 'units': 'µT'},
    )


@pytest.fixture
def synthetic_gaussian_field() -> xr.DataArray:
    """30×30 smooth Gaussian blob for UpwardContinuation testing."""
    h, w = 30, 30
    ps = 1e-6
    y_raw = np.arange(h) * ps
    x_raw = np.arange(w) * ps
    Xg, Yg = np.meshgrid(x_raw, y_raw)
    cy, cx = h // 2 * ps, w // 2 * ps
    sigma = 5 * ps
    values = np.exp(-((Xg - cx) ** 2 + (Yg - cy) ** 2) / (2 * sigma ** 2))
    return xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': y_raw, 'x': x_raw},
        attrs={'pixel_spacing': ps, 'units': 'µT'},
    )


@pytest.fixture
def synthetic_blank() -> xr.DataArray:
    """20×20 pre-computed blank field map."""
    rng = np.random.default_rng(3)
    values = rng.uniform(-0.1, 0.1, size=(20, 20))
    y = np.arange(20) * 1e-6
    x = np.arange(20) * 1e-6
    return xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': y, 'x': x},
        attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
    )


# ---------------------------------------------------------------------------
# TestHotPixelFilter
# ---------------------------------------------------------------------------


class TestHotPixelFilterConfig:
    """HotPixelFilter is a frozen Pydantic model with validated config fields."""

    def test_import(self) -> None:
        """HotPixelFilter can be imported from qdmpy_core.field_processing."""
        from qdmpy_core.field_processing import HotPixelFilter  # noqa: F401

    def test_default_config(self) -> None:
        """Default config: threshold_sigma=5.0, window_size=3, replacement='mean'."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter()
        assert f.threshold_sigma == 5.0
        assert f.window_size == 3
        assert f.replacement == 'mean'
        assert f.absolute_threshold is None

    def test_custom_config(self) -> None:
        """Custom config fields are stored and accessible."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, window_size=5, replacement='nan', absolute_threshold=10.0)
        assert f.threshold_sigma == 3.0
        assert f.window_size == 5
        assert f.replacement == 'nan'
        assert f.absolute_threshold == pytest.approx(10.0)

    def test_frozen_prevents_mutation(self) -> None:
        """HotPixelFilter fields are immutable after construction."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter()
        with pytest.raises((ValidationError, TypeError)):
            f.threshold_sigma = 1.0  # type: ignore[misc]

    def test_invalid_replacement_raises(self) -> None:
        """replacement must be 'mean', 'nan', or 'zero'."""
        from qdmpy_core.field_processing import HotPixelFilter

        with pytest.raises((ValidationError, ValueError)):
            HotPixelFilter(replacement='median')  # type: ignore[arg-type]

    def test_is_base_field_processor_subclass(self) -> None:
        """HotPixelFilter is a subclass of BaseFieldProcessor."""
        from qdmpy_core.field_processing import BaseFieldProcessor, HotPixelFilter

        assert issubclass(HotPixelFilter, BaseFieldProcessor)


class TestHotPixelFilterAlgorithm:
    """HotPixelFilter correctly detects and replaces outlier pixels."""

    def test_single_spike_detected(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """A single extreme spike at known position is detected (its value changes)."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, replacement='zero')
        result = f.process(synthetic_hot_pixel_field)
        # The spike at (5,5) had value 1000; background is ~0. It should change.
        assert abs(result.values[5, 5]) < 1000.0

    def test_multiple_spikes_detected(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """All four injected spikes are replaced."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, replacement='zero')
        result = f.process(synthetic_hot_pixel_field)
        for row, col in [(5, 5), (10, 10), (3, 15), (17, 2)]:
            assert abs(result.values[row, col]) < 100.0, (
                f'Spike at ({row},{col}) not replaced: {result.values[row, col]}'
            )

    def test_no_outliers_leaves_field_unchanged(self) -> None:
        """A uniform constant field has no outliers; output equals input."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=5.0)
        field = _make_field_map(fill_value=1.0)
        result = f.process(field)
        np.testing.assert_allclose(result.values, field.values)

    def test_all_zeros_no_outliers(self) -> None:
        """An all-zero field has no outliers; output is all zeros."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=5.0)
        field = _make_field_map(fill_value=0.0)
        result = f.process(field)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-12)

    def test_output_shape_equals_input_shape(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """Output shape is identical to input shape."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter()
        result = f.process(synthetic_hot_pixel_field)
        assert result.shape == synthetic_hot_pixel_field.shape

    def test_process_does_not_mutate_input(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """process() leaves the input DataArray unchanged."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0)
        original = synthetic_hot_pixel_field.values.copy()
        f.process(synthetic_hot_pixel_field)
        np.testing.assert_array_equal(synthetic_hot_pixel_field.values, original)

    def test_coords_preserved(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """y and x coordinates are preserved in the output."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter()
        result = f.process(synthetic_hot_pixel_field)
        np.testing.assert_array_equal(result.coords['y'].values, synthetic_hot_pixel_field.coords['y'].values)
        np.testing.assert_array_equal(result.coords['x'].values, synthetic_hot_pixel_field.coords['x'].values)

    def test_attrs_preserved(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """pixel_spacing and other attrs are preserved in the output."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter()
        result = f.process(synthetic_hot_pixel_field)
        assert result.attrs.get('pixel_spacing') == pytest.approx(1e-6)
        assert result.attrs.get('units') == 'µT'

    def test_dims_preserved(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """Output dims are ('y', 'x')."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter()
        result = f.process(synthetic_hot_pixel_field)
        assert result.dims == ('y', 'x')


class TestHotPixelFilterReplacementModes:
    """HotPixelFilter supports 'mean', 'nan', and 'zero' replacement modes."""

    def test_replacement_zero(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """'zero' replacement sets detected outliers to 0.0."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, replacement='zero')
        result = f.process(synthetic_hot_pixel_field)
        # Spike positions should become exactly 0.0
        assert result.values[5, 5] == pytest.approx(0.0)
        assert result.values[10, 10] == pytest.approx(0.0)

    def test_replacement_nan(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """'nan' replacement sets detected outliers to NaN."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, replacement='nan')
        result = f.process(synthetic_hot_pixel_field)
        assert np.isnan(result.values[5, 5])
        assert np.isnan(result.values[10, 10])

    def test_replacement_mean(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """'mean' replacement uses nanmean of window neighbours."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, replacement='mean')
        result = f.process(synthetic_hot_pixel_field)
        # Replaced value should be in a sane range (~background level, not 1000)
        assert abs(result.values[5, 5]) < 10.0
        assert not np.isnan(result.values[5, 5])

    def test_replacement_mean_no_nans_in_output(self) -> None:
        """'mean' mode produces no NaNs in a clean field."""
        from qdmpy_core.field_processing import HotPixelFilter

        rng = np.random.default_rng(99)
        values = rng.uniform(-1.0, 1.0, size=(15, 15))
        values[7, 7] = 500.0
        y = np.arange(15) * 1e-6
        x = np.arange(15) * 1e-6
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': 1e-6})
        f = HotPixelFilter(threshold_sigma=3.0, replacement='mean')
        result = f.process(field)
        assert not np.any(np.isnan(result.values))

    def test_replacement_zero_non_outliers_unchanged(self, synthetic_hot_pixel_field: xr.DataArray) -> None:
        """Non-outlier pixels are unmodified when using 'zero' replacement."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=3.0, replacement='zero')
        result = f.process(synthetic_hot_pixel_field)
        # Non-spike pixels should remain the same (checking a known-safe pixel)
        assert result.values[0, 0] == pytest.approx(synthetic_hot_pixel_field.values[0, 0])


class TestHotPixelFilterAbsoluteThreshold:
    """HotPixelFilter supports an absolute threshold pre-filter."""

    def test_absolute_threshold_filters_large_values(self) -> None:
        """absolute_threshold=10.0 causes |field| > 10 to be replaced."""
        from qdmpy_core.field_processing import HotPixelFilter

        values = np.zeros((10, 10))
        values[5, 5] = 50.0  # clearly above threshold
        y = np.arange(10) * 1e-6
        x = np.arange(10) * 1e-6
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': 1e-6})
        f = HotPixelFilter(threshold_sigma=100.0, replacement='zero', absolute_threshold=10.0)
        result = f.process(field)
        assert result.values[5, 5] == pytest.approx(0.0)

    def test_absolute_threshold_none_skipped(self) -> None:
        """When absolute_threshold is None, no absolute pre-filter is applied."""
        from qdmpy_core.field_processing import HotPixelFilter

        values = np.full((10, 10), 5.0)
        values[5, 5] = 8.0  # within sigma range of background
        y = np.arange(10) * 1e-6
        x = np.arange(10) * 1e-6
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': 1e-6})
        # With very high sigma and no absolute threshold, value=8.0 should NOT be filtered
        f = HotPixelFilter(threshold_sigma=100.0, replacement='zero', absolute_threshold=None)
        result = f.process(field)
        # The slightly elevated value should remain
        assert result.values[5, 5] == pytest.approx(8.0)

    def test_absolute_threshold_union_with_sigma(self) -> None:
        """Outlier mask is union: pixels exceeding either criterion are replaced."""
        from qdmpy_core.field_processing import HotPixelFilter

        rng = np.random.default_rng(7)
        values = rng.uniform(-1.0, 1.0, size=(15, 15))
        values[3, 3] = 20.0  # caught by absolute_threshold=5.0
        values[8, 8] = -20.0  # caught by absolute_threshold=5.0
        y = np.arange(15) * 1e-6
        x = np.arange(15) * 1e-6
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': 1e-6})
        f = HotPixelFilter(threshold_sigma=50.0, replacement='zero', absolute_threshold=5.0)
        result = f.process(field)
        assert result.values[3, 3] == pytest.approx(0.0)
        assert result.values[8, 8] == pytest.approx(0.0)


class TestHotPixelFilterPropertyBased:
    """Hypothesis property-based tests for HotPixelFilter invariants."""

    @given(
        height=st.integers(min_value=5, max_value=30),
        width=st.integers(min_value=5, max_value=30),
        threshold=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
    )
    @hyp_settings(max_examples=20)
    def test_output_shape_equals_input_shape(self, height: int, width: int, threshold: float) -> None:
        """For any valid input, output shape equals input shape."""
        from qdmpy_core.field_processing import HotPixelFilter

        f = HotPixelFilter(threshold_sigma=threshold)
        field = _make_field_map(height=height, width=width)
        result = f.process(field)
        assert result.shape == field.shape

    @given(
        height=st.integers(min_value=5, max_value=20),
        width=st.integers(min_value=5, max_value=20),
    )
    @hyp_settings(max_examples=15)
    def test_replacement_mean_produces_no_nans_for_spike_free_field(
        self, height: int, width: int
    ) -> None:
        """'mean' replacement on a field without NaNs should produce no NaNs."""
        from qdmpy_core.field_processing import HotPixelFilter

        rng = np.random.default_rng(42)
        values = rng.uniform(-1.0, 1.0, size=(height, width))
        y = np.arange(height) * 1e-6
        x = np.arange(width) * 1e-6
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': 1e-6})
        f = HotPixelFilter(threshold_sigma=3.0, replacement='mean')
        result = f.process(field)
        assert not np.any(np.isnan(result.values))


# ---------------------------------------------------------------------------
# TestQuadraticBackgroundSubtractor
# ---------------------------------------------------------------------------


class TestQuadraticBackgroundSubtractorConfig:
    """QuadraticBackgroundSubtractor is a frozen Pydantic model."""

    def test_import(self) -> None:
        """QuadraticBackgroundSubtractor can be imported from qdmpy_core.field_processing."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor  # noqa: F401

    def test_default_config(self) -> None:
        """Default: degree=2, mask=None."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor()
        assert p.degree == 2
        assert p.mask is None

    def test_custom_degree(self) -> None:
        """degree field is stored as provided."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=1)
        assert p.degree == 1

    def test_frozen_prevents_mutation(self) -> None:
        """Fields are immutable after construction."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor()
        with pytest.raises((ValidationError, TypeError)):
            p.degree = 3  # type: ignore[misc]

    def test_is_base_field_processor_subclass(self) -> None:
        """QuadraticBackgroundSubtractor subclasses BaseFieldProcessor."""
        from qdmpy_core.field_processing import BaseFieldProcessor, QuadraticBackgroundSubtractor

        assert issubclass(QuadraticBackgroundSubtractor, BaseFieldProcessor)


class TestQuadraticBackgroundSubtractorAlgorithm:
    """QuadraticBackgroundSubtractor fits and removes polynomial backgrounds."""

    def test_output_shape_equals_input_shape(self, synthetic_background_field: xr.DataArray) -> None:
        """Output shape equals input shape."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=2)
        result = p.process(synthetic_background_field)
        assert result.shape == synthetic_background_field.shape

    def test_removes_quadratic_background(self, synthetic_background_field: xr.DataArray) -> None:
        """After subtraction, a field dominated by quadratic background has small residuals."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=2)
        result = p.process(synthetic_background_field)
        # Residuals should be small (close to the injected noise level ~0.01)
        np.testing.assert_allclose(result.values, 0.0, atol=0.1)

    def test_constant_field_becomes_zero(self) -> None:
        """A constant field has a constant background; output should be near zero."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        field = _make_field_map(fill_value=5.0)
        p = QuadraticBackgroundSubtractor(degree=2)
        result = p.process(field)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_plane_fit_removes_linear_background(self) -> None:
        """degree=1 correctly removes a linear background."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        h, w = 20, 20
        ps = 1e-6
        y_raw = np.arange(h) * ps
        x_raw = np.arange(w) * ps
        Xg, Yg = np.meshgrid(x_raw, y_raw)
        # Pure linear field: 2x + 3y + 1
        values = 1.0 + 2.0 * Xg / (w * ps) + 3.0 * Yg / (h * ps)
        field = xr.DataArray(values, dims=('y', 'x'),
                             coords={'y': y_raw, 'x': x_raw},
                             attrs={'pixel_spacing': ps})
        p = QuadraticBackgroundSubtractor(degree=1)
        result = p.process(field)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-8)

    def test_constant_fit_degree_zero(self) -> None:
        """degree=0 fits and removes only a constant offset."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        field = _make_field_map(fill_value=7.5)
        p = QuadraticBackgroundSubtractor(degree=0)
        result = p.process(field)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_process_does_not_mutate_input(self, synthetic_background_field: xr.DataArray) -> None:
        """Input DataArray is not modified."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=2)
        original = synthetic_background_field.values.copy()
        p.process(synthetic_background_field)
        np.testing.assert_array_equal(synthetic_background_field.values, original)

    def test_coords_preserved(self, synthetic_background_field: xr.DataArray) -> None:
        """Coordinates are preserved after background subtraction."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=2)
        result = p.process(synthetic_background_field)
        np.testing.assert_array_equal(
            result.coords['y'].values, synthetic_background_field.coords['y'].values
        )
        np.testing.assert_array_equal(
            result.coords['x'].values, synthetic_background_field.coords['x'].values
        )

    def test_attrs_preserved(self, synthetic_background_field: xr.DataArray) -> None:
        """pixel_spacing and other attrs are preserved in the output."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=2)
        result = p.process(synthetic_background_field)
        assert result.attrs.get('pixel_spacing') == pytest.approx(1e-6)

    def test_dims_preserved(self, synthetic_background_field: xr.DataArray) -> None:
        """Output dims are ('y', 'x')."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p = QuadraticBackgroundSubtractor(degree=2)
        result = p.process(synthetic_background_field)
        assert result.dims == ('y', 'x')


class TestQuadraticBackgroundSubtractorWithMask:
    """QuadraticBackgroundSubtractor supports pixel-index masks to exclude pixels from fit."""

    def test_mask_excludes_pixels_from_fit(self) -> None:
        """Mask indices are excluded from the polynomial fit."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        h, w = 20, 20
        ps = 1e-6
        y_raw = np.arange(h) * ps
        x_raw = np.arange(w) * ps
        Xg, Yg = np.meshgrid(x_raw, y_raw)
        # Linear background with a sample region that has a strong signal
        background = 1.0 + 2.0 * Xg / (w * ps)
        # Add a large signal to a 5×5 region that should be masked
        values = background.copy()
        values[5:10, 5:10] += 100.0

        # Build flat indices for the 5×5 region
        mask_rows = []
        mask_cols = []
        for r in range(5, 10):
            for c in range(5, 10):
                mask_rows.append(r)
                mask_cols.append(c)

        # Encode mask as ((row0, row1, ...), (col0, col1, ...))
        mask = (tuple(mask_rows), tuple(mask_cols))

        field = xr.DataArray(values, dims=('y', 'x'),
                             coords={'y': y_raw, 'x': x_raw},
                             attrs={'pixel_spacing': ps})

        p = QuadraticBackgroundSubtractor(degree=1, mask=mask)
        result = p.process(field)

        # Outside mask: background should be removed, leaving ~0 residuals
        outside_vals = result.values.copy()
        outside_vals[5:10, 5:10] = np.nan
        outside_flat = outside_vals[~np.isnan(outside_vals)]
        np.testing.assert_allclose(outside_flat, 0.0, atol=0.5)

    def test_no_mask_uses_all_pixels(self, synthetic_background_field: xr.DataArray) -> None:
        """When mask is None, all pixels are used in the fit."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        p_no_mask = QuadraticBackgroundSubtractor(degree=2, mask=None)
        result = p_no_mask.process(synthetic_background_field)
        # Full fit of quadratic background should give near-zero residuals
        np.testing.assert_allclose(result.values, 0.0, atol=0.1)

    def test_mask_evaluates_surface_at_all_pixels(self) -> None:
        """The fitted surface is subtracted from ALL pixels, including masked ones."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        h, w = 10, 10
        ps = 1e-6
        y_raw = np.arange(h) * ps
        x_raw = np.arange(w) * ps
        values = np.ones((h, w)) * 3.0  # constant field
        field = xr.DataArray(values, dims=('y', 'x'),
                             coords={'y': y_raw, 'x': x_raw},
                             attrs={'pixel_spacing': ps})
        # Mask first row
        mask = (tuple(range(w)), (0,) * w)
        p = QuadraticBackgroundSubtractor(degree=0, mask=mask)
        result = p.process(field)
        # All pixels (including masked ones) should have background removed
        np.testing.assert_allclose(result.values, 0.0, atol=1e-8)

    def test_mask_stored_as_nested_tuples(self) -> None:
        """mask field stored as nested tuples (Pydantic-serialisable)."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        mask = ((0, 1, 2), (0, 1, 2))
        p = QuadraticBackgroundSubtractor(mask=mask)
        assert isinstance(p.mask, tuple)
        assert isinstance(p.mask[0], tuple)

    def test_output_shape_with_mask(self) -> None:
        """Output shape equals input shape even when a mask is provided."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        field = _make_field_map(height=15, width=15)
        mask = ((0, 1), (0, 1))
        p = QuadraticBackgroundSubtractor(degree=2, mask=mask)
        result = p.process(field)
        assert result.shape == field.shape


class TestQuadraticBackgroundSubtractorPropertyBased:
    """Hypothesis property-based tests for QuadraticBackgroundSubtractor."""

    @given(
        height=st.integers(min_value=6, max_value=25),
        width=st.integers(min_value=6, max_value=25),
        degree=st.integers(min_value=0, max_value=2),
    )
    @hyp_settings(max_examples=15)
    def test_output_shape_equals_input_shape(self, height: int, width: int, degree: int) -> None:
        """For any valid input and degree, output shape equals input shape."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        field = _make_field_map(height=height, width=width)
        p = QuadraticBackgroundSubtractor(degree=degree)
        result = p.process(field)
        assert result.shape == field.shape

    @given(
        a=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        b=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
        c=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    @hyp_settings(max_examples=15)
    def test_removing_plane_from_plane_gives_near_zero(self, a: float, b: float, c: float) -> None:
        """Subtracting a fitted plane from a pure planar field yields near-zero residuals."""
        from qdmpy_core.field_processing import QuadraticBackgroundSubtractor

        h, w = 12, 12
        ps = 1e-6
        y_raw = np.arange(h) * ps
        x_raw = np.arange(w) * ps
        Xg, Yg = np.meshgrid(x_raw, y_raw)
        values = a + b * Xg / (w * ps) + c * Yg / (h * ps)
        field = xr.DataArray(values, dims=('y', 'x'),
                             coords={'y': y_raw, 'x': x_raw},
                             attrs={'pixel_spacing': ps})
        p = QuadraticBackgroundSubtractor(degree=1)
        result = p.process(field)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TestUpwardContinuation
# ---------------------------------------------------------------------------


class TestUpwardContinuationConfig:
    """UpwardContinuation is a frozen Pydantic model."""

    def test_import(self) -> None:
        """UpwardContinuation can be imported from qdmpy_core.field_processing."""
        from qdmpy_core.field_processing import UpwardContinuation  # noqa: F401

    def test_required_dz(self) -> None:
        """dz is required; construction without it raises."""
        from qdmpy_core.field_processing import UpwardContinuation

        with pytest.raises((ValidationError, TypeError)):
            UpwardContinuation()  # type: ignore[call-arg]

    def test_default_config(self) -> None:
        """Default padding_factor=3.0, oversampling=2."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=1e-6)
        assert u.dz == pytest.approx(1e-6)
        assert u.padding_factor == pytest.approx(3.0)
        assert u.oversampling == 2

    def test_custom_config(self) -> None:
        """Custom padding_factor and oversampling are stored."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6, padding_factor=2.0, oversampling=4)
        assert u.padding_factor == pytest.approx(2.0)
        assert u.oversampling == 4

    def test_frozen_prevents_mutation(self) -> None:
        """Fields are immutable after construction."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=1e-6)
        with pytest.raises((ValidationError, TypeError)):
            u.dz = 2e-6  # type: ignore[misc]

    def test_is_base_field_processor_subclass(self) -> None:
        """UpwardContinuation subclasses BaseFieldProcessor."""
        from qdmpy_core.field_processing import BaseFieldProcessor, UpwardContinuation

        assert issubclass(UpwardContinuation, BaseFieldProcessor)


class TestUpwardContinuationAlgorithm:
    """UpwardContinuation correctly transforms fields using FFT-based continuation."""

    def test_output_shape_equals_input_shape(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Output shape equals input shape."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6)
        result = u.process(synthetic_gaussian_field)
        assert result.shape == synthetic_gaussian_field.shape

    def test_upward_attenuates_amplitude(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Upward continuation (dz > 0) attenuates the field amplitude."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=20e-6)
        result = u.process(synthetic_gaussian_field)
        # Max amplitude should decrease
        assert np.max(np.abs(result.values)) < np.max(np.abs(synthetic_gaussian_field.values))

    def test_zero_dz_preserves_field(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """dz=0 returns a field very close to the input."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=0.0)
        result = u.process(synthetic_gaussian_field)
        np.testing.assert_allclose(
            result.values, synthetic_gaussian_field.values, rtol=1e-3, atol=1e-5
        )

    def test_downward_continuation_amplifies(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Downward continuation (dz < 0) amplifies the field amplitude."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=-5e-6)
        result = u.process(synthetic_gaussian_field)
        # Peak amplitude should increase
        assert np.max(np.abs(result.values)) > np.max(np.abs(synthetic_gaussian_field.values))

    def test_downward_continuation_logs_warning(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """dz < 0 logs a warning (downward continuation amplifies noise)."""
        from qdmpy_core.field_processing import UpwardContinuation

        log_calls: list[Any] = []
        with patch('QDMpy.field_processing.logger') as mock_logger:
            mock_logger.warning = lambda *a, **kw: log_calls.append((a, kw))
            u = UpwardContinuation(dz=-1e-6)
            u.process(synthetic_gaussian_field)

        assert len(log_calls) >= 1, 'Expected at least one warning log for downward continuation'

    def test_process_does_not_mutate_input(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Input DataArray is not modified by process()."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6)
        original = synthetic_gaussian_field.values.copy()
        u.process(synthetic_gaussian_field)
        np.testing.assert_array_equal(synthetic_gaussian_field.values, original)

    def test_coords_preserved(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Coordinates are preserved in the output."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6)
        result = u.process(synthetic_gaussian_field)
        np.testing.assert_array_equal(result.coords['y'].values, synthetic_gaussian_field.coords['y'].values)
        np.testing.assert_array_equal(result.coords['x'].values, synthetic_gaussian_field.coords['x'].values)

    def test_attrs_preserved(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """pixel_spacing and units attrs are preserved."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6)
        result = u.process(synthetic_gaussian_field)
        assert result.attrs.get('pixel_spacing') == pytest.approx(1e-6)

    def test_dims_preserved(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Output dims are ('y', 'x')."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6)
        result = u.process(synthetic_gaussian_field)
        assert result.dims == ('y', 'x')

    def test_large_dz_strong_attenuation(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """Very large dz produces strongly attenuated output amplitude.

        Low frequencies (DC and near-DC) survive; high frequencies are exponentially
        attenuated. With dz=1e-3 (1 mm) and ps=1e-6 (1 µm), expect ~99% attenuation.
        """
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=1e-3)  # huge compared to pixel_spacing=1e-6
        result = u.process(synthetic_gaussian_field)
        # Check that amplitude is reduced to ~0.5% of input (strong attenuation)
        # Input Gaussian has peak ~0.01, so output < 0.01 shows strong attenuation
        assert np.max(np.abs(result.values)) < 0.01

    def test_small_image_with_padding(self) -> None:
        """A 5×5 image with default padding_factor works without error."""
        from qdmpy_core.field_processing import UpwardContinuation

        h, w = 5, 5
        ps = 1e-6
        values = np.ones((h, w))
        y = np.arange(h) * ps
        x = np.arange(w) * ps
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': ps})
        u = UpwardContinuation(dz=1e-6, padding_factor=3.0)
        result = u.process(field)
        assert result.shape == (h, w)

    def test_returns_xr_dataarray(self, synthetic_gaussian_field: xr.DataArray) -> None:
        """process() returns an xr.DataArray."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=5e-6)
        result = u.process(synthetic_gaussian_field)
        assert isinstance(result, xr.DataArray)


class TestUpwardContinuationPropertyBased:
    """Hypothesis property-based tests for UpwardContinuation."""

    @given(
        height=st.integers(min_value=5, max_value=25),
        width=st.integers(min_value=5, max_value=25),
        dz=st.floats(min_value=0.0, max_value=50e-6, allow_nan=False),
    )
    @hyp_settings(max_examples=15)
    def test_output_shape_equals_input_shape(self, height: int, width: int, dz: float) -> None:
        """For any valid input, output shape equals input shape."""
        from qdmpy_core.field_processing import UpwardContinuation

        u = UpwardContinuation(dz=dz, padding_factor=2.0, oversampling=1)
        field = _make_field_map(height=height, width=width)
        result = u.process(field)
        assert result.shape == field.shape

    @given(
        dz1=st.floats(min_value=1e-7, max_value=10e-6, allow_nan=False),
        dz2=st.floats(min_value=1e-7, max_value=10e-6, allow_nan=False),
    )
    @hyp_settings(max_examples=10)
    def test_additive_continuation_heights(self, dz1: float, dz2: float) -> None:
        """Applying dz1 then dz2 ≈ applying dz1+dz2 in a single step (for smooth fields)."""
        from qdmpy_core.field_processing import UpwardContinuation

        # Use a smooth Gaussian field
        h, w = 20, 20
        ps = 1e-6
        y_raw = np.arange(h) * ps
        x_raw = np.arange(w) * ps
        Xg, Yg = np.meshgrid(x_raw, y_raw)
        cy, cx = h // 2 * ps, w // 2 * ps
        sigma = 4 * ps
        values = np.exp(-((Xg - cx) ** 2 + (Yg - cy) ** 2) / (2 * sigma ** 2))
        field = xr.DataArray(values, dims=('y', 'x'),
                             coords={'y': y_raw, 'x': x_raw},
                             attrs={'pixel_spacing': ps})

        u1 = UpwardContinuation(dz=dz1, padding_factor=2.0, oversampling=1)
        u2 = UpwardContinuation(dz=dz2, padding_factor=2.0, oversampling=1)
        u_combined = UpwardContinuation(dz=dz1 + dz2, padding_factor=2.0, oversampling=1)

        result_two_step = u2.process(u1.process(field))
        result_one_step = u_combined.process(field)

        # Should be approximately equal (padding/crop introduces small edge errors)
        np.testing.assert_allclose(
            result_two_step.values, result_one_step.values, rtol=0.05, atol=0.05
        )


# ---------------------------------------------------------------------------
# TestBlankSubtractor
# ---------------------------------------------------------------------------


class TestBlankSubtractorConfig:
    """BlankSubtractor is a frozen Pydantic model."""

    def test_import(self) -> None:
        """BlankSubtractor can be imported from qdmpy_core.field_processing."""
        from qdmpy_core.field_processing import BlankSubtractor  # noqa: F401

    def test_blank_required(self) -> None:
        """Construction without 'blank' raises."""
        from qdmpy_core.field_processing import BlankSubtractor

        with pytest.raises((ValidationError, TypeError)):
            BlankSubtractor()  # type: ignore[call-arg]

    def test_blank_stored_as_nested_tuples(self) -> None:
        """blank is stored as nested tuples (Pydantic-serialisable)."""
        from qdmpy_core.field_processing import BlankSubtractor

        blank_data = ((1.0, 2.0), (3.0, 4.0))
        b = BlankSubtractor(blank=blank_data)
        assert isinstance(b.blank, tuple)
        assert isinstance(b.blank[0], tuple)

    def test_frozen_prevents_mutation(self) -> None:
        """blank field is immutable after construction."""
        from qdmpy_core.field_processing import BlankSubtractor

        blank_data = ((1.0, 2.0), (3.0, 4.0))
        b = BlankSubtractor(blank=blank_data)
        with pytest.raises((ValidationError, TypeError)):
            b.blank = ((0.0, 0.0), (0.0, 0.0))  # type: ignore[misc]

    def test_is_base_field_processor_subclass(self) -> None:
        """BlankSubtractor subclasses BaseFieldProcessor."""
        from qdmpy_core.field_processing import BaseFieldProcessor, BlankSubtractor

        assert issubclass(BlankSubtractor, BaseFieldProcessor)


class TestBlankSubtractorAlgorithm:
    """BlankSubtractor subtracts a blank map element-wise."""

    def test_basic_subtraction(self, synthetic_blank: xr.DataArray) -> None:
        """output = input - blank element-wise."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=20, width=20)
        blank_tuple = tuple(tuple(float(v) for v in row) for row in synthetic_blank.values)
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        np.testing.assert_allclose(result.values, field.values - synthetic_blank.values)

    def test_zero_blank_leaves_field_unchanged(self) -> None:
        """A blank of all zeros leaves the field unchanged."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        blank_tuple = tuple(tuple(0.0 for _ in range(10)) for _ in range(10))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        np.testing.assert_allclose(result.values, field.values)

    def test_blank_equal_to_field_gives_zeros(self) -> None:
        """When blank equals the field, output is all zeros."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=8, width=8, fill_value=3.0)
        blank_tuple = tuple(tuple(3.0 for _ in range(8)) for _ in range(8))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-12)

    def test_output_shape_equals_input_shape(self) -> None:
        """Output shape equals input shape."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=12, width=12)
        blank_tuple = tuple(tuple(0.0 for _ in range(12)) for _ in range(12))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        assert result.shape == field.shape

    def test_shape_mismatch_raises_data_shape_error(self) -> None:
        """blank shape != field shape raises DataShapeError."""
        from qdmpy_core.exceptions import DataShapeError
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        # blank is 8×8, field is 10×10 — mismatch
        blank_tuple = tuple(tuple(0.0 for _ in range(8)) for _ in range(8))
        b = BlankSubtractor(blank=blank_tuple)
        with pytest.raises(DataShapeError):
            b.process(field)

    def test_shape_mismatch_error_includes_both_shapes(self) -> None:
        """DataShapeError message mentions both the blank shape and field shape."""
        from qdmpy_core.exceptions import DataShapeError
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        blank_tuple = tuple(tuple(0.0 for _ in range(8)) for _ in range(8))
        b = BlankSubtractor(blank=blank_tuple)
        with pytest.raises(DataShapeError, match=r'.*\(8.*8\)|.*\(10.*10\)'):
            b.process(field)

    def test_process_does_not_mutate_input(self) -> None:
        """Input DataArray is not modified by process()."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        blank_tuple = tuple(tuple(0.0 for _ in range(10)) for _ in range(10))
        b = BlankSubtractor(blank=blank_tuple)
        original = field.values.copy()
        b.process(field)
        np.testing.assert_array_equal(field.values, original)

    def test_coords_preserved(self) -> None:
        """y and x coordinates are preserved after subtraction."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        blank_tuple = tuple(tuple(0.0 for _ in range(10)) for _ in range(10))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        np.testing.assert_array_equal(result.coords['y'].values, field.coords['y'].values)
        np.testing.assert_array_equal(result.coords['x'].values, field.coords['x'].values)

    def test_attrs_preserved(self) -> None:
        """pixel_spacing and other attrs are preserved in the output."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        blank_tuple = tuple(tuple(0.0 for _ in range(10)) for _ in range(10))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        assert result.attrs.get('pixel_spacing') == pytest.approx(1e-6)
        assert result.attrs.get('units') == 'µT'

    def test_dims_preserved(self) -> None:
        """Output dims are ('y', 'x')."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=10, width=10)
        blank_tuple = tuple(tuple(0.0 for _ in range(10)) for _ in range(10))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        assert result.dims == ('y', 'x')

    def test_large_blank_produces_negative_output(self) -> None:
        """A blank much larger than the field produces a negative output."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=8, width=8, fill_value=1.0)
        blank_tuple = tuple(tuple(100.0 for _ in range(8)) for _ in range(8))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        assert np.all(result.values < 0.0)

    def test_blank_converted_to_ndarray_internally(self) -> None:
        """Internal computation converts nested tuples to NDArray (output is numeric)."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=5, width=5, fill_value=2.0)
        blank_tuple = tuple(tuple(1.0 for _ in range(5)) for _ in range(5))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        # Should be 2.0 - 1.0 = 1.0 everywhere
        np.testing.assert_allclose(result.values, 1.0, atol=1e-12)

    def test_returns_xr_dataarray(self) -> None:
        """process() returns an xr.DataArray."""
        from qdmpy_core.field_processing import BlankSubtractor

        field = _make_field_map(height=5, width=5)
        blank_tuple = tuple(tuple(0.0 for _ in range(5)) for _ in range(5))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        assert isinstance(result, xr.DataArray)

    def test_non_square_field_and_blank(self) -> None:
        """BlankSubtractor works for non-square (H != W) fields."""
        from qdmpy_core.field_processing import BlankSubtractor

        h, w = 8, 12
        ps = 1e-6
        y = np.arange(h) * ps
        x = np.arange(w) * ps
        values = np.ones((h, w)) * 5.0
        field = xr.DataArray(values, dims=('y', 'x'), coords={'y': y, 'x': x},
                             attrs={'pixel_spacing': ps})
        blank_tuple = tuple(tuple(2.0 for _ in range(w)) for _ in range(h))
        b = BlankSubtractor(blank=blank_tuple)
        result = b.process(field)
        np.testing.assert_allclose(result.values, 3.0, atol=1e-12)
        assert result.shape == (h, w)
