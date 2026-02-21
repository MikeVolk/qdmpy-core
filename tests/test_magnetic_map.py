"""Tests for MagneticMap and _reconstruct_bxyz (QEP-034 Phase 3).

RED phase — all tests should FAIL until magnetic_map.py is implemented.

Import paths under test:
    from QDMpy.magnetic_map import MagneticMap, _reconstruct_bxyz
    from QDMpy.settings import get_settings, reset_settings

Test classes:
    TestReconstructBxyzSignature           (5 tests)
    TestReconstructBxyzPhysicsKnownCases   (7 tests)
    TestReconstructBxyzNvAxis              (4 tests)
    TestReconstructBxyzEpsilonRegularisation (5 tests)
    TestReconstructBxyzRoundTrip           (4 tests)
    TestReconstructBxyzWavenumberHandling  (3 tests)
    TestReconstructBxyzEdgeCases           (6 tests)
    TestReconstructBxyzPropertyBased       (3 tests)

    TestMagneticMapImmutability            (6 tests)
    TestMagneticMapFromB111Factory         (9 tests)
    TestMagneticMapToDataset               (7 tests)
    TestMagneticMapDisplay                 (6 tests)
    TestMagneticMapSave                    (6 tests)
    TestMagneticMapArrayIndependence       (4 tests)
    TestMagneticMapSettingsIntegration     (5 tests)
    TestMagneticMapEdgeCases               (8 tests)
    TestMagneticMapErrorHandling           (4 tests)
    TestMagneticMapPropertyBased           (3 tests)
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Constants used in tests
# ---------------------------------------------------------------------------

_GAMMA_NV = 28.024  # GHz/T
_DEFAULT_NV_AXIS = (0.0, math.sqrt(2.0 / 3.0), 1.0 / math.sqrt(3.0))
_DEFAULT_EPSILON = 1e-30
_PIXEL_SPACING = 1e-6  # 1 µm in metres


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_b111(
    height: int = 20,
    width: int = 20,
    pixel_spacing: float = _PIXEL_SPACING,
    fill_value: float | None = None,
    rng: np.random.Generator | None = None,
) -> xr.DataArray:
    """Return a synthetic (H, W) B111 DataArray in µT with pixel_spacing in attrs."""
    if rng is None:
        rng = np.random.default_rng(42)

    if fill_value is not None:
        values = np.full((height, width), fill_value, dtype=float)
    else:
        values = rng.uniform(0.0, 10.0, size=(height, width))

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
def synthetic_b111_field() -> xr.DataArray:
    """20×20 B111 DataArray with pixel_spacing=1e-6 m, values 0–10 µT."""
    return _make_b111(height=20, width=20, pixel_spacing=_PIXEL_SPACING)


@pytest.fixture
def synthetic_b111_zeros() -> xr.DataArray:
    """20×20 B111 DataArray filled with zeros."""
    return _make_b111(height=20, width=20, fill_value=0.0)


@pytest.fixture
def synthetic_b111_constant() -> xr.DataArray:
    """20×20 B111 DataArray filled with constant 5.0 µT."""
    return _make_b111(height=20, width=20, fill_value=5.0)


@pytest.fixture
def synthetic_b111_with_spikes() -> xr.DataArray:
    """20×20 B111 DataArray with a known Gaussian-shaped feature for round-trip tests."""
    height, width = 20, 20
    y, x = np.mgrid[0:height, 0:width]
    cy, cx = height // 2, width // 2
    sigma = 3.0
    values = 5.0 * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
    y_coords = np.arange(height) * _PIXEL_SPACING
    x_coords = np.arange(width) * _PIXEL_SPACING
    return xr.DataArray(
        values,
        dims=('y', 'x'),
        coords={'y': y_coords, 'x': x_coords},
        attrs={'pixel_spacing': _PIXEL_SPACING, 'units': 'µT'},
    )


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — signature and return type
# ---------------------------------------------------------------------------


class TestReconstructBxyzSignature:
    """_reconstruct_bxyz has the correct signature and return contract."""

    def test_function_is_importable(self) -> None:
        """_reconstruct_bxyz can be imported from QDMpy.magnetic_map."""
        from QDMpy.magnetic_map import _reconstruct_bxyz  # noqa: F401

    def test_all_parameters_required_no_defaults(self) -> None:
        """_reconstruct_bxyz requires all four positional arguments."""
        from QDMpy.magnetic_map import _reconstruct_bxyz
        import inspect

        sig = inspect.signature(_reconstruct_bxyz)
        params = sig.parameters
        assert len(params) >= 4
        for name in ('b111', 'pixel_spacing', 'nv_axis', 'epsilon'):
            assert name in params, f'Parameter {name!r} missing from signature'

    def test_returns_tuple_of_three(self) -> None:
        """_reconstruct_bxyz returns a tuple/sequence of exactly three arrays."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.zeros((10, 10))
        result = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert len(result) == 3

    def test_each_output_is_ndarray(self) -> None:
        """All three outputs are numpy ndarrays."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.zeros((10, 10))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert isinstance(bx, np.ndarray)
        assert isinstance(by, np.ndarray)
        assert isinstance(bz, np.ndarray)

    def test_all_outputs_have_same_shape_as_input(self) -> None:
        """Output arrays share the shape of the input B111 array."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        for shape in [(10, 10), (5, 15), (20, 30)]:
            b111 = np.zeros(shape)
            bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
            assert bx.shape == shape, f'bx shape mismatch for input {shape}'
            assert by.shape == shape, f'by shape mismatch for input {shape}'
            assert bz.shape == shape, f'bz shape mismatch for input {shape}'


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — physics: known test cases
# ---------------------------------------------------------------------------


class TestReconstructBxyzPhysicsKnownCases:
    """_reconstruct_bxyz produces correct fields for analytically known inputs."""

    def test_zero_b111_gives_zero_bxyz(self) -> None:
        """B111 = all zeros → Bx, By, Bz all zeros."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.zeros((16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        np.testing.assert_allclose(bx, 0.0, atol=1e-10)
        np.testing.assert_allclose(by, 0.0, atol=1e-10)
        np.testing.assert_allclose(bz, 0.0, atol=1e-10)

    def test_constant_b111_only_dc_contribution_in_bz(self) -> None:
        """B111 = constant → only the DC (k=0) component survives; Bx, By are zero.

        A uniform field has no spatial variation so the deconvolution at non-zero k
        contributes nothing; only the mean is preserved in Bz.
        """
        from QDMpy.magnetic_map import _reconstruct_bxyz

        const_val = 5.0
        b111 = np.full((16, 16), const_val)
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        # DC uniform field: Bx and By should be near zero (no k-space variation)
        np.testing.assert_allclose(bx, 0.0, atol=1e-6)
        np.testing.assert_allclose(by, 0.0, atol=1e-6)

    def test_bz_finite_everywhere_for_nonzero_b111(self) -> None:
        """Bz has finite values for a non-trivial B111 input."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(7)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        _, _, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bz)), 'Bz contains non-finite values'

    def test_bx_finite_everywhere_for_nonzero_b111(self) -> None:
        """Bx has finite values for a non-trivial B111 input."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(8)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        bx, _, _ = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx)), 'Bx contains non-finite values'

    def test_by_finite_everywhere_for_nonzero_b111(self) -> None:
        """By has finite values for a non-trivial B111 input."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(9)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        _, by, _ = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(by)), 'By contains non-finite values'

    def test_nv_z_axis_gives_b111_equals_bz(self) -> None:
        """If nv_axis = (0, 0, 1) the NV is purely along z so B111 must equal Bz."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        nv_z = (0.0, 0.0, 1.0)
        rng = np.random.default_rng(11)
        # Use smooth (low-frequency) input to avoid FFT ringing
        b111 = rng.uniform(0.0, 5.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, nv_z, _DEFAULT_EPSILON)
        # Bx and By should be ~0 since ux=uy=0
        np.testing.assert_allclose(bx, 0.0, atol=1e-6)
        np.testing.assert_allclose(by, 0.0, atol=1e-6)
        np.testing.assert_allclose(bz, b111, atol=1e-6)

    def test_negative_and_positive_b111_handled_symmetrically(self) -> None:
        """Negating B111 negates all Bxyz components (linearity)."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(13)
        b111 = rng.uniform(-3.0, 3.0, (16, 16))
        bx_pos, by_pos, bz_pos = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        bx_neg, by_neg, bz_neg = _reconstruct_bxyz(-b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        np.testing.assert_allclose(bx_neg, -bx_pos, atol=1e-10)
        np.testing.assert_allclose(by_neg, -by_pos, atol=1e-10)
        np.testing.assert_allclose(bz_neg, -bz_pos, atol=1e-10)


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — NV axis variations
# ---------------------------------------------------------------------------


class TestReconstructBxyzNvAxis:
    """_reconstruct_bxyz responds correctly to different nv_axis orientations."""

    def test_default_nv_axis_produces_nonzero_bx_by_bz(self) -> None:
        """Standard [111] NV axis yields non-trivial Bx, By, Bz for a structured field."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(20)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        # All components should be non-trivially non-zero for random input
        assert np.any(np.abs(bx) > 1e-10), 'Bx is unexpectedly all near zero'
        assert np.any(np.abs(by) > 1e-10), 'By is unexpectedly all near zero'
        assert np.any(np.abs(bz) > 1e-10), 'Bz is unexpectedly all near zero'

    def test_different_nv_axis_gives_different_bxy_balance(self) -> None:
        """Two distinct NV axes produce different Bx/By distributions."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(21)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        axis_a = (0.0, math.sqrt(2.0 / 3.0), 1.0 / math.sqrt(3.0))
        axis_b = (math.sqrt(2.0 / 3.0), 0.0, 1.0 / math.sqrt(3.0))
        bx_a, by_a, _ = _reconstruct_bxyz(b111, _PIXEL_SPACING, axis_a, _DEFAULT_EPSILON)
        bx_b, by_b, _ = _reconstruct_bxyz(b111, _PIXEL_SPACING, axis_b, _DEFAULT_EPSILON)
        # The Bx and By distributions should differ between axes
        assert not np.allclose(bx_a, bx_b), 'Bx unchanged despite different NV axis'
        assert not np.allclose(by_a, by_b), 'By unchanged despite different NV axis'

    def test_x_only_nv_axis_gives_zero_by_bz(self) -> None:
        """nv_axis = (1, 0, 0) → By and Bz are zero, Bx captures B111."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        nv_x = (1.0, 0.0, 0.0)
        rng = np.random.default_rng(22)
        b111 = rng.uniform(0.0, 5.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, nv_x, _DEFAULT_EPSILON)
        np.testing.assert_allclose(by, 0.0, atol=1e-6)
        np.testing.assert_allclose(bz, 0.0, atol=1e-6)
        np.testing.assert_allclose(bx, b111, atol=1e-6)

    def test_output_shape_invariant_to_nv_axis(self) -> None:
        """Output shape is always (H, W) regardless of NV axis choice."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.ones((12, 15))
        for axis in [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            _DEFAULT_NV_AXIS,
        ]:
            bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, axis, _DEFAULT_EPSILON)
            assert bx.shape == (12, 15)
            assert by.shape == (12, 15)
            assert bz.shape == (12, 15)


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — epsilon regularisation
# ---------------------------------------------------------------------------


class TestReconstructBxyzEpsilonRegularisation:
    """Epsilon parameter controls k=0 singularity handling."""

    def test_default_epsilon_avoids_nan_and_inf(self) -> None:
        """With default epsilon no NaN or Inf appears in the output."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(30)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))

    def test_large_epsilon_smooths_output(self) -> None:
        """Very large epsilon damps high-frequency content, producing a smoother Bz."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(31)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        _, _, bz_default = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        _, _, bz_smooth = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, 1e10)
        # Large epsilon should reduce variance (smoother field)
        assert np.var(bz_smooth) < np.var(bz_default), (
            'Large epsilon did not reduce variance'
        )

    def test_epsilon_zero_may_produce_nan_at_k0(self) -> None:
        """epsilon=0 either produces NaN/Inf at k=0 or raises; must not silently succeed."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.ones((16, 16))
        try:
            bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, 0.0)
            # If it returns without error, k=0 must contain NaN or Inf
            has_nan_or_inf = (
                not np.all(np.isfinite(bx))
                or not np.all(np.isfinite(by))
                or not np.all(np.isfinite(bz))
            )
            assert has_nan_or_inf, (
                'epsilon=0 returned finite values; expected NaN/Inf at k=0'
            )
        except (ZeroDivisionError, FloatingPointError, ValueError):
            pass  # Acceptable — function detected the singularity

    def test_increasing_epsilon_monotonically_reduces_peak_amplitude(self) -> None:
        """Larger epsilon progressively reduces the peak |Bz| value."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(33)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        epsilons = [1e-30, 1e-10, 1e-3, 1.0]
        peak_bz = []
        for eps in epsilons:
            _, _, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, eps)
            peak_bz.append(np.max(np.abs(bz)))
        # Peak amplitudes should be non-increasing as epsilon increases
        for i in range(len(peak_bz) - 1):
            assert peak_bz[i] >= peak_bz[i + 1] - 1e-10, (
                f'Peak |Bz| increased from eps={epsilons[i]:.0e} to eps={epsilons[i+1]:.0e}'
            )

    def test_epsilon_only_affects_k0_component_magnitude(self) -> None:
        """Modifying epsilon should have negligible effect on smooth low-frequency field."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        # A very smooth (low-frequency) field where k>0 terms dominate the answer
        b111 = np.outer(np.sin(np.linspace(0, np.pi, 16)), np.sin(np.linspace(0, np.pi, 16)))
        b111 *= 5.0
        _, _, bz_a = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, 1e-30)
        _, _, bz_b = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, 1e-15)
        # Smooth field: epsilon change should not drastically change Bz
        np.testing.assert_allclose(bz_a, bz_b, rtol=1e-3)


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — round-trip consistency
# ---------------------------------------------------------------------------


class TestReconstructBxyzRoundTrip:
    """Reconstructed Bxyz, projected back onto NV axis, should recover B111."""

    def test_round_trip_z_axis_exact(self) -> None:
        """nv_axis=(0,0,1): B111 = Bz exactly (lossless round-trip)."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(40)
        b111 = rng.uniform(-3.0, 3.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, (0.0, 0.0, 1.0), _DEFAULT_EPSILON)
        b111_recovered = (0.0 * bx + 0.0 * by + 1.0 * bz)
        np.testing.assert_allclose(b111_recovered, b111, atol=1e-6)

    def test_round_trip_x_axis_exact(self) -> None:
        """nv_axis=(1,0,0): B111 = Bx exactly (lossless round-trip)."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(41)
        b111 = rng.uniform(-3.0, 3.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, (1.0, 0.0, 0.0), _DEFAULT_EPSILON)
        b111_recovered = (1.0 * bx + 0.0 * by + 0.0 * bz)
        np.testing.assert_allclose(b111_recovered, b111, atol=1e-6)

    def test_round_trip_default_axis_smooth_field(self) -> None:
        """For a smooth B111 field, nv · Bxyz ≈ B111 within 5% tolerance."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        # Smooth Gaussian-shaped field (few Fourier modes)
        height, width = 32, 32
        y, x = np.mgrid[0:height, 0:width]
        b111 = 5.0 * np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / (2 * 4.0 ** 2))
        ux, uy, uz = _DEFAULT_NV_AXIS
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        b111_recovered = ux * bx + uy * by + uz * bz
        np.testing.assert_allclose(b111_recovered, b111, rtol=0.05, atol=1e-3)

    def test_btotal_always_geq_abs_bz(self) -> None:
        """|B_total| = sqrt(Bx²+By²+Bz²) >= |Bz| everywhere (physical constraint)."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(43)
        b111 = rng.uniform(-5.0, 5.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        btotal = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        assert np.all(btotal >= np.abs(bz) - 1e-10), (
            '|Btotal| < |Bz| at some pixels (unphysical)'
        )


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — wavenumber handling
# ---------------------------------------------------------------------------


class TestReconstructBxyzWavenumberHandling:
    """FFT wavenumber grid is computed correctly from pixel_spacing."""

    def test_high_frequency_b111_preserved_in_bz(self) -> None:
        """High-frequency checkerboard pattern in B111 produces structured Bz."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        size = 16
        y, x = np.mgrid[0:size, 0:size]
        # Nyquist-frequency checkerboard
        b111 = 5.0 * ((-1) ** (y + x)).astype(float)
        _, _, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        # High-frequency content must survive (non-zero Bz)
        assert np.any(np.abs(bz) > 1e-6), 'High-frequency B111 content was lost'

    def test_pixel_spacing_scales_wavenumbers(self) -> None:
        """Doubling pixel_spacing halves all wavenumbers; Bz amplitude must change."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(50)
        b111 = rng.uniform(-3.0, 3.0, (16, 16))
        _, _, bz_fine = _reconstruct_bxyz(b111, 1e-6, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        _, _, bz_coarse = _reconstruct_bxyz(b111, 2e-6, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        # Different pixel_spacing → different wavenumber amplitudes → different Bz
        assert not np.allclose(bz_fine, bz_coarse), (
            'Changing pixel_spacing did not change Bz'
        )

    def test_single_fourier_mode_reconstructed_correctly(self) -> None:
        """A single-frequency sinusoidal B111 produces finite, structured Bz."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        size = 16
        x = np.arange(size)
        b111 = np.outer(np.ones(size), np.sin(2 * np.pi * x / size))
        b111 *= 3.0
        _, _, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bz))
        assert np.any(np.abs(bz) > 1e-10), 'Bz is zero for non-trivial single-mode B111'


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — edge cases
# ---------------------------------------------------------------------------


class TestReconstructBxyzEdgeCases:
    """_reconstruct_bxyz handles boundary and corner inputs gracefully."""

    def test_tiny_2x2_image(self) -> None:
        """2×2 image does not crash and returns (2,2) arrays."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.array([[1.0, 2.0], [3.0, 4.0]])
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert bx.shape == (2, 2)
        assert by.shape == (2, 2)
        assert bz.shape == (2, 2)

    def test_non_square_image_10x20(self) -> None:
        """Non-square (10×20) image works and returns matching shape."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.ones((10, 20))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert bx.shape == (10, 20)
        assert by.shape == (10, 20)
        assert bz.shape == (10, 20)

    def test_very_small_pixel_spacing(self) -> None:
        """pixel_spacing = 1e-9 (1 nm) does not overflow or underflow."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.ones((8, 8))
        bx, by, bz = _reconstruct_bxyz(b111, 1e-9, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))

    def test_very_large_pixel_spacing(self) -> None:
        """pixel_spacing = 1e-3 (1 mm) does not overflow or underflow."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.ones((8, 8))
        bx, by, bz = _reconstruct_bxyz(b111, 1e-3, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))

    def test_negative_b111_values_handled(self) -> None:
        """Negative-valued B111 produces finite outputs."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.full((8, 8), -3.0)
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))

    def test_large_amplitude_b111(self) -> None:
        """B111 with amplitude 1000 µT does not overflow."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(60)
        b111 = rng.uniform(-1000.0, 1000.0, (16, 16))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))


# ---------------------------------------------------------------------------
# _reconstruct_bxyz — property-based (Hypothesis)
# ---------------------------------------------------------------------------


class TestReconstructBxyzPropertyBased:
    """Hypothesis property-based tests for _reconstruct_bxyz."""

    @given(
        height=st.integers(min_value=2, max_value=24),
        width=st.integers(min_value=2, max_value=24),
    )
    @hyp_settings(max_examples=30)
    def test_output_shape_invariant(self, height: int, width: int) -> None:
        """Output shape always equals input (H, W) for any valid dimensions."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.zeros((height, width))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert bx.shape == (height, width)
        assert by.shape == (height, width)
        assert bz.shape == (height, width)

    @given(
        pixel_spacing=st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=20)
    def test_output_finite_for_any_valid_pixel_spacing(self, pixel_spacing: float) -> None:
        """Finite output for any valid pixel_spacing in [1 nm, 1 mm]."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        b111 = np.ones((8, 8))
        bx, by, bz = _reconstruct_bxyz(b111, pixel_spacing, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        assert np.all(np.isfinite(bx))
        assert np.all(np.isfinite(by))
        assert np.all(np.isfinite(bz))

    @given(
        amplitude=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=20)
    def test_btotal_geq_abs_bz_property(self, amplitude: float) -> None:
        """Physical constraint: |Btotal| >= |Bz| for any amplitude."""
        from QDMpy.magnetic_map import _reconstruct_bxyz

        rng = np.random.default_rng(99)
        b111 = amplitude * rng.standard_normal((8, 8))
        bx, by, bz = _reconstruct_bxyz(b111, _PIXEL_SPACING, _DEFAULT_NV_AXIS, _DEFAULT_EPSILON)
        btotal = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        assert np.all(btotal >= np.abs(bz) - 1e-10)


# ---------------------------------------------------------------------------
# MagneticMap — immutability
# ---------------------------------------------------------------------------


class TestMagneticMapImmutability:
    """MagneticMap is a frozen dataclass; all fields are immutable."""

    def test_magnetic_map_is_importable(self) -> None:
        """MagneticMap can be imported from QDMpy.magnetic_map."""
        from QDMpy.magnetic_map import MagneticMap  # noqa: F401

    def test_b111_field_cannot_be_reassigned(self, synthetic_b111_zeros: xr.DataArray) -> None:
        """Assigning to .b111 on a frozen MagneticMap raises AttributeError or TypeError."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_zeros)
        with pytest.raises((AttributeError, TypeError)):
            mm.b111 = synthetic_b111_zeros  # type: ignore[misc]

    def test_bx_field_cannot_be_reassigned(self, synthetic_b111_zeros: xr.DataArray) -> None:
        """Assigning to .bx raises AttributeError or TypeError."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_zeros)
        with pytest.raises((AttributeError, TypeError)):
            mm.bx = mm.bx  # type: ignore[misc]

    def test_bz_field_cannot_be_reassigned(self, synthetic_b111_zeros: xr.DataArray) -> None:
        """Assigning to .bz raises AttributeError or TypeError."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_zeros)
        with pytest.raises((AttributeError, TypeError)):
            mm.bz = mm.bz  # type: ignore[misc]

    def test_nv_axis_field_cannot_be_reassigned(self, synthetic_b111_zeros: xr.DataArray) -> None:
        """Assigning to .nv_axis raises AttributeError or TypeError."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_zeros)
        with pytest.raises((AttributeError, TypeError)):
            mm.nv_axis = (0.0, 0.0, 1.0)  # type: ignore[misc]

    def test_all_array_fields_are_xarray_dataarrays(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """b111, bx, by, bz, btotal are all xr.DataArray instances."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        assert isinstance(mm.b111, xr.DataArray), 'b111 is not a DataArray'
        assert isinstance(mm.bx, xr.DataArray), 'bx is not a DataArray'
        assert isinstance(mm.by, xr.DataArray), 'by is not a DataArray'
        assert isinstance(mm.bz, xr.DataArray), 'bz is not a DataArray'
        assert isinstance(mm.btotal, xr.DataArray), 'btotal is not a DataArray'


# ---------------------------------------------------------------------------
# MagneticMap — from_b111() factory method
# ---------------------------------------------------------------------------


class TestMagneticMapFromB111Factory:
    """MagneticMap.from_b111() creates valid instances from xr.DataArray."""

    def test_from_b111_returns_magnetic_map_instance(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """from_b111() returns a MagneticMap instance."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        assert isinstance(mm, MagneticMap)

    def test_from_b111_missing_pixel_spacing_raises_value_error(self) -> None:
        """from_b111() raises ValueError when pixel_spacing is absent from attrs."""
        from QDMpy.magnetic_map import MagneticMap

        b111_no_attr = xr.DataArray(
            np.zeros((10, 10)),
            dims=('y', 'x'),
            attrs={'units': 'µT'},  # pixel_spacing intentionally omitted
        )
        with pytest.raises(ValueError, match='pixel_spacing'):
            MagneticMap.from_b111(b111_no_attr)

    def test_from_b111_computes_btotal_correctly(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """btotal = sqrt(bx² + by² + bz²)."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        expected_btotal = np.sqrt(mm.bx.values ** 2 + mm.by.values ** 2 + mm.bz.values ** 2)
        np.testing.assert_allclose(mm.btotal.values, expected_btotal, atol=1e-10)

    def test_from_b111_components_have_same_dims_as_input(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """All components share the same dims as the input b111."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        for component_name in ('b111', 'bx', 'by', 'bz', 'btotal'):
            component = getattr(mm, component_name)
            assert set(component.dims) == set(synthetic_b111_field.dims), (
                f'{component_name} dims mismatch'
            )

    def test_from_b111_components_have_same_shape_as_input(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """All components have the same shape as the input b111."""
        from QDMpy.magnetic_map import MagneticMap

        expected_shape = synthetic_b111_field.shape
        mm = MagneticMap.from_b111(synthetic_b111_field)
        for component_name in ('b111', 'bx', 'by', 'bz', 'btotal'):
            component = getattr(mm, component_name)
            assert component.shape == expected_shape, (
                f'{component_name}.shape {component.shape} != {expected_shape}'
            )

    def test_from_b111_accepts_custom_nv_axis(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """from_b111() accepts an explicit nv_axis that overrides the default."""
        from QDMpy.magnetic_map import MagneticMap

        custom_axis = (0.0, 0.0, 1.0)
        mm = MagneticMap.from_b111(synthetic_b111_field, nv_axis=custom_axis)
        assert mm.nv_axis == custom_axis

    def test_from_b111_accepts_custom_epsilon(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """from_b111() accepts an explicit epsilon."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field, epsilon=1e-20)
        assert isinstance(mm, MagneticMap)  # no error

    def test_from_b111_stores_nv_axis_as_tuple(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """nv_axis is stored as a tuple[float, float, float]."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        assert isinstance(mm.nv_axis, tuple)
        assert len(mm.nv_axis) == 3
        assert all(isinstance(v, float) for v in mm.nv_axis)

    def test_from_b111_preserves_b111_values(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """The .b111 attribute matches the input DataArray values."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        np.testing.assert_array_equal(mm.b111.values, synthetic_b111_field.values)


# ---------------------------------------------------------------------------
# MagneticMap — to_dataset()
# ---------------------------------------------------------------------------


class TestMagneticMapToDataset:
    """MagneticMap.to_dataset() returns a well-formed xr.Dataset."""

    def test_to_dataset_returns_xr_dataset(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """to_dataset() returns an xr.Dataset."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        assert isinstance(ds, xr.Dataset)

    def test_to_dataset_contains_all_components(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Dataset contains variables: b111, Bx, By, Bz, Btotal."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        for var in ('b111', 'Bx', 'By', 'Bz', 'Btotal'):
            assert var in ds, f'Variable {var!r} missing from dataset'

    def test_to_dataset_has_units_attr(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Dataset-level attrs include units='µT'."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        assert ds.attrs.get('units') == 'µT'

    def test_to_dataset_stores_nv_axis_in_attrs(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Dataset attrs include nv_axis stored as a list."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        assert 'nv_axis' in ds.attrs, "Dataset attrs missing 'nv_axis'"
        assert isinstance(ds.attrs['nv_axis'], list)
        assert len(ds.attrs['nv_axis']) == 3

    def test_to_dataset_variables_preserve_coords(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Each variable in the dataset preserves the input coords."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        for var in ('b111', 'Bx', 'By', 'Bz', 'Btotal'):
            np.testing.assert_array_equal(
                ds[var].coords['y'].values,
                synthetic_b111_field.coords['y'].values,
            )
            np.testing.assert_array_equal(
                ds[var].coords['x'].values,
                synthetic_b111_field.coords['x'].values,
            )

    def test_to_dataset_bz_values_match_mm_bz(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Dataset['Bz'].values matches mm.bz.values."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        np.testing.assert_array_equal(ds['Bz'].values, mm.bz.values)

    def test_to_dataset_is_independent_copy(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Modifying the returned Dataset does not affect the MagneticMap."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        ds = mm.to_dataset()
        original_bz = mm.bz.values.copy()
        # Attempt to modify the dataset
        try:
            ds['Bz'].values[:] = 0.0
        except (ValueError, TypeError):
            pass  # Immutable arrays are also acceptable
        np.testing.assert_array_equal(mm.bz.values, original_bz)


# ---------------------------------------------------------------------------
# MagneticMap — display()
# ---------------------------------------------------------------------------


class TestMagneticMapDisplay:
    """MagneticMap.display() renders the correct component."""

    def test_display_bz_calls_xarray_plot(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display('Bz') calls the DataArray plot method (mocked)."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with patch.object(mm.bz, 'plot') as mock_plot:
            mm.display(component='Bz')
            mock_plot.assert_called_once()

    def test_display_b111_calls_xarray_plot(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display('b111') calls the b111 DataArray plot method."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with patch.object(mm.b111, 'plot') as mock_plot:
            mm.display(component='b111')
            mock_plot.assert_called_once()

    def test_display_btotal_calls_xarray_plot(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display('Btotal') calls the btotal DataArray plot method."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with patch.object(mm.btotal, 'plot') as mock_plot:
            mm.display(component='Btotal')
            mock_plot.assert_called_once()

    def test_display_invalid_component_raises_value_error(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display() raises ValueError for an unrecognised component name."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with pytest.raises(ValueError, match='component'):
            mm.display(component='nonexistent_component')

    def test_display_passes_kwargs_to_plot(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display() forwards keyword arguments to the underlying plot call."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with patch.object(mm.bz, 'plot') as mock_plot:
            mm.display(component='Bz', cmap='viridis', vmin=0.0, vmax=10.0)
            mock_plot.assert_called_once_with(cmap='viridis', vmin=0.0, vmax=10.0)

    def test_display_valid_all_components_do_not_raise(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Each valid component name does not raise when display() is called."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        valid_components = ['b111', 'Bx', 'By', 'Bz', 'Btotal']
        for comp in valid_components:
            # Patch the specific component's plot to avoid actual rendering
            target_da = getattr(mm, comp.lower() if comp.startswith('b') else comp)
            with patch.object(target_da, 'plot'):
                mm.display(component=comp)  # must not raise


# ---------------------------------------------------------------------------
# MagneticMap — save()
# ---------------------------------------------------------------------------


class TestMagneticMapSave:
    """MagneticMap.save() persists data to NetCDF."""

    def test_save_creates_file_at_given_path(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """save(path) creates a file at the specified Path."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_output.nc'
            mm.save(path)
            assert path.exists(), f'Expected file at {path}'

    def test_save_accepts_string_path(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """save() accepts a str path as well as a Path object."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / 'test_str_path.nc')
            mm.save(path)
            assert Path(path).exists()

    def test_save_netcdf_contains_bz_variable(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """The saved NetCDF file can be re-opened and contains 'Bz'."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_roundtrip.nc'
            mm.save(path)
            ds_loaded = xr.open_dataset(path)
            assert 'Bz' in ds_loaded, "Saved NetCDF does not contain 'Bz'"
            ds_loaded.close()

    def test_save_netcdf_bz_values_preserved(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Bz values survive a save → load round-trip."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_values.nc'
            mm.save(path)
            ds_loaded = xr.open_dataset(path)
            np.testing.assert_allclose(ds_loaded['Bz'].values, mm.bz.values, atol=1e-10)
            ds_loaded.close()

    def test_save_logs_info_message(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """save() emits a loguru INFO-level log message."""
        from loguru import logger

        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_log.nc'
            log_messages: list[str] = []
            with logger.contextualize():
                listener_id = logger.add(lambda msg: log_messages.append(msg), level='INFO')
                try:
                    mm.save(path)
                finally:
                    logger.remove(listener_id)
            assert any(str(path) in msg or 'save' in msg.lower() for msg in log_messages), (
                'No INFO log message was emitted during save()'
            )

    def test_save_netcdf_nv_axis_in_attrs(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """The saved NetCDF file has nv_axis in its global attributes."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_attrs.nc'
            mm.save(path)
            ds_loaded = xr.open_dataset(path)
            assert 'nv_axis' in ds_loaded.attrs, "Saved NetCDF missing 'nv_axis' attr"
            ds_loaded.close()


# ---------------------------------------------------------------------------
# MagneticMap — array independence
# ---------------------------------------------------------------------------


class TestMagneticMapArrayIndependence:
    """Components of MagneticMap are independent; modifying one does not affect others."""

    def test_bz_independent_of_bx(self, synthetic_b111_field: xr.DataArray) -> None:
        """Modifying bz.values (if mutable) does not change bx.values."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        bx_before = mm.bx.values.copy()
        try:
            mm.bz.values[:] = 99.0
        except (ValueError, TypeError):
            return  # Immutable arrays — test passes trivially
        np.testing.assert_array_equal(mm.bx.values, bx_before)

    def test_b111_independent_of_bz(self, synthetic_b111_field: xr.DataArray) -> None:
        """Modifying b111.values (if mutable) does not change bz.values."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        bz_before = mm.bz.values.copy()
        try:
            mm.b111.values[:] = 0.0
        except (ValueError, TypeError):
            return  # Immutable arrays — test passes trivially
        np.testing.assert_array_equal(mm.bz.values, bz_before)

    def test_btotal_consistent_after_construction(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """btotal = sqrt(bx² + by² + bz²) holds after construction."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        expected = np.sqrt(mm.bx.values ** 2 + mm.by.values ** 2 + mm.bz.values ** 2)
        np.testing.assert_allclose(mm.btotal.values, expected, atol=1e-10)

    def test_two_instances_from_same_b111_are_independent(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Two MagneticMap instances built from the same b111 do not share state."""
        from QDMpy.magnetic_map import MagneticMap

        mm_a = MagneticMap.from_b111(synthetic_b111_field)
        mm_b = MagneticMap.from_b111(synthetic_b111_field)
        np.testing.assert_array_equal(mm_a.bz.values, mm_b.bz.values)
        # They should be different objects (not aliased)
        assert mm_a is not mm_b


# ---------------------------------------------------------------------------
# MagneticMap — settings integration
# ---------------------------------------------------------------------------


class TestMagneticMapSettingsIntegration:
    """MagneticMap respects QDMpy settings for defaults."""

    def test_default_nv_axis_from_settings(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """When nv_axis is not provided, it is taken from get_settings().nv.axis."""
        from QDMpy.magnetic_map import MagneticMap
        from QDMpy.settings import get_settings, reset_settings

        reset_settings()
        settings_axis = get_settings().nv.axis
        mm = MagneticMap.from_b111(synthetic_b111_field)
        assert mm.nv_axis == settings_axis

    def test_explicit_nv_axis_overrides_settings(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Explicit nv_axis kwarg overrides whatever is in settings."""
        from QDMpy.magnetic_map import MagneticMap
        from QDMpy.settings import reset_settings

        reset_settings()
        custom_axis = (0.0, 0.0, 1.0)
        mm = MagneticMap.from_b111(synthetic_b111_field, nv_axis=custom_axis)
        assert mm.nv_axis == custom_axis

    def test_explicit_epsilon_overrides_settings(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Explicit epsilon kwarg is used even when settings have a different default."""
        from QDMpy.magnetic_map import MagneticMap
        from QDMpy.settings import reset_settings

        reset_settings()
        # Verify it succeeds with a custom epsilon
        mm = MagneticMap.from_b111(synthetic_b111_field, epsilon=1e-15)
        assert isinstance(mm, MagneticMap)

    def test_reset_settings_does_not_break_existing_instances(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """Existing MagneticMap instances remain valid after reset_settings()."""
        from QDMpy.magnetic_map import MagneticMap
        from QDMpy.settings import reset_settings

        mm = MagneticMap.from_b111(synthetic_b111_field)
        bz_before = mm.bz.values.copy()
        reset_settings()
        # Instance must be unchanged
        np.testing.assert_array_equal(mm.bz.values, bz_before)

    def test_default_epsilon_from_settings(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """When epsilon is not provided, it is taken from get_settings().nv.epsilon."""
        from QDMpy.magnetic_map import MagneticMap
        from QDMpy.settings import get_settings, reset_settings

        reset_settings()
        settings_epsilon = get_settings().nv.epsilon
        # Build two maps: one with explicit default epsilon, one without
        mm_default = MagneticMap.from_b111(synthetic_b111_field)
        mm_explicit = MagneticMap.from_b111(synthetic_b111_field, epsilon=settings_epsilon)
        np.testing.assert_array_equal(mm_default.bz.values, mm_explicit.bz.values)


# ---------------------------------------------------------------------------
# MagneticMap — edge cases
# ---------------------------------------------------------------------------


class TestMagneticMapEdgeCases:
    """MagneticMap handles corner-case inputs gracefully."""

    def test_tiny_2x2_b111(self) -> None:
        """2×2 B111 produces a valid MagneticMap."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=('y', 'x'),
            coords={'y': [0.0, 1e-6], 'x': [0.0, 1e-6]},
            attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
        )
        mm = MagneticMap.from_b111(b111)
        assert mm.bz.shape == (2, 2)

    def test_non_square_b111_10x20(self) -> None:
        """Non-square (10×20) B111 produces a valid MagneticMap."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = _make_b111(height=10, width=20)
        mm = MagneticMap.from_b111(b111)
        assert mm.bz.shape == (10, 20)
        assert mm.bx.shape == (10, 20)

    def test_all_zero_b111_gives_all_zero_bxyz(
        self, synthetic_b111_zeros: xr.DataArray
    ) -> None:
        """B111=0 → Bx=By=Bz=Btotal=0."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_zeros)
        np.testing.assert_allclose(mm.bx.values, 0.0, atol=1e-10)
        np.testing.assert_allclose(mm.by.values, 0.0, atol=1e-10)
        np.testing.assert_allclose(mm.bz.values, 0.0, atol=1e-10)
        np.testing.assert_allclose(mm.btotal.values, 0.0, atol=1e-10)

    def test_large_amplitude_b111_no_overflow(self) -> None:
        """B111 values of ±1000 µT do not produce NaN or Inf."""
        from QDMpy.magnetic_map import MagneticMap

        rng = np.random.default_rng(70)
        values = rng.uniform(-1000.0, 1000.0, (16, 16))
        b111 = xr.DataArray(
            values,
            dims=('y', 'x'),
            attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
        )
        mm = MagneticMap.from_b111(b111)
        assert np.all(np.isfinite(mm.bz.values)), 'Bz has non-finite values'
        assert np.all(np.isfinite(mm.btotal.values)), 'Btotal has non-finite values'

    def test_negative_b111_values_handled(self) -> None:
        """B111 with purely negative values produces finite Bxyz."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = _make_b111(height=16, width=16, fill_value=-5.0)
        mm = MagneticMap.from_b111(b111)
        assert np.all(np.isfinite(mm.bz.values))
        assert np.all(np.isfinite(mm.bx.values))

    def test_custom_nv_axis_not_111(self) -> None:
        """A non-[111] NV axis produces a valid MagneticMap."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = _make_b111(height=10, width=10)
        mm = MagneticMap.from_b111(b111, nv_axis=(0.0, 0.0, 1.0))
        assert isinstance(mm, MagneticMap)
        assert mm.nv_axis == (0.0, 0.0, 1.0)

    def test_b111_with_coords_preserved(self) -> None:
        """Coordinate values from the input b111 are preserved in all components."""
        from QDMpy.magnetic_map import MagneticMap

        y_coords = np.linspace(0, 19e-6, 20)
        x_coords = np.linspace(0, 19e-6, 20)
        b111 = xr.DataArray(
            np.ones((20, 20)),
            dims=('y', 'x'),
            coords={'y': y_coords, 'x': x_coords},
            attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
        )
        mm = MagneticMap.from_b111(b111)
        np.testing.assert_array_equal(mm.bz.coords['y'].values, y_coords)
        np.testing.assert_array_equal(mm.bz.coords['x'].values, x_coords)

    def test_constant_b111_produces_finite_btotal(
        self, synthetic_b111_constant: xr.DataArray
    ) -> None:
        """Constant (non-zero) B111 produces finite Btotal."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_constant)
        assert np.all(np.isfinite(mm.btotal.values))


# ---------------------------------------------------------------------------
# MagneticMap — error handling
# ---------------------------------------------------------------------------


class TestMagneticMapErrorHandling:
    """MagneticMap raises appropriate exceptions for invalid inputs."""

    def test_missing_pixel_spacing_raises_value_error_with_message(self) -> None:
        """ValueError message mentions 'pixel_spacing' when it is absent."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = xr.DataArray(np.zeros((10, 10)), dims=('y', 'x'))
        with pytest.raises(ValueError, match='pixel_spacing'):
            MagneticMap.from_b111(b111)

    def test_display_empty_component_string_raises_value_error(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display('') raises ValueError for empty component string."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with pytest.raises(ValueError):
            mm.display(component='')

    def test_display_case_insensitive_bx_accepted(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """display() handles case-insensitive component lookup (e.g., 'bx' vs 'Bx')."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        # Both 'bx' and 'Bx' should resolve to the same component
        with patch.object(mm.bx, 'plot'):
            mm.display(component='bx')  # lowercase — must not raise

    def test_save_invalid_extension_still_attempts_write(
        self, synthetic_b111_field: xr.DataArray
    ) -> None:
        """save() with an unusual extension does not silently fail."""
        from QDMpy.magnetic_map import MagneticMap

        mm = MagneticMap.from_b111(synthetic_b111_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'output.data'
            mm.save(path)
            # File should still be created (xr.Dataset.to_netcdf handles extensions)
            assert path.exists()


# ---------------------------------------------------------------------------
# MagneticMap — property-based (Hypothesis)
# ---------------------------------------------------------------------------


class TestMagneticMapPropertyBased:
    """Hypothesis property-based tests for MagneticMap."""

    @given(
        height=st.integers(min_value=2, max_value=20),
        width=st.integers(min_value=2, max_value=20),
        fill_value=st.floats(
            min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
        ),
    )
    @hyp_settings(max_examples=20)
    def test_from_b111_always_succeeds_with_valid_pixel_spacing(
        self, height: int, width: int, fill_value: float
    ) -> None:
        """from_b111() succeeds for any valid (H, W) DataArray with pixel_spacing in attrs."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = xr.DataArray(
            np.full((height, width), fill_value),
            dims=('y', 'x'),
            attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
        )
        mm = MagneticMap.from_b111(b111)
        assert isinstance(mm, MagneticMap)
        assert mm.bz.shape == (height, width)

    @given(
        height=st.integers(min_value=2, max_value=16),
        width=st.integers(min_value=2, max_value=16),
    )
    @hyp_settings(max_examples=15)
    def test_result_btotal_geq_abs_bz_always_holds(
        self, height: int, width: int
    ) -> None:
        """Physical constraint: btotal >= |bz| at every pixel."""
        from QDMpy.magnetic_map import MagneticMap

        rng = np.random.default_rng(height * 100 + width)
        b111 = xr.DataArray(
            rng.uniform(-5.0, 5.0, (height, width)),
            dims=('y', 'x'),
            attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
        )
        mm = MagneticMap.from_b111(b111)
        assert np.all(mm.btotal.values >= np.abs(mm.bz.values) - 1e-10)

    @given(
        height=st.integers(min_value=2, max_value=16),
        width=st.integers(min_value=2, max_value=16),
    )
    @hyp_settings(max_examples=15)
    def test_instance_is_always_frozen(self, height: int, width: int) -> None:
        """MagneticMap instances cannot have fields overwritten (frozen contract)."""
        from QDMpy.magnetic_map import MagneticMap

        b111 = xr.DataArray(
            np.zeros((height, width)),
            dims=('y', 'x'),
            attrs={'pixel_spacing': 1e-6, 'units': 'µT'},
        )
        mm = MagneticMap.from_b111(b111)
        with pytest.raises((AttributeError, TypeError)):
            mm.bz = mm.bz  # type: ignore[misc]
