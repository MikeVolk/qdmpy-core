"""Unit tests for qdmpy.source_fitting.

Test strategy
-------------
- Round-trip: generate a single-dipole Bz map via pypole.dipole.dipole_field
  with known parameters, fit it back with fit_source, verify recovery.
- Integration: fit_sources iterates field_sources and returns one result per
  MagneticSource; other FieldSource kinds are skipped.
- Robustness: an all-zero Bz map must not raise, even if the fit fails.
"""

from __future__ import annotations

import numpy as np
import pypole.convert
import pypole.dipole
import pypole.maps

from qdmpy.field_source import FieldSource, MagneticModel, MagneticSource
from qdmpy.source_fitting import FitSourceResult, compute_field, fit_source, fit_sources
from qdmpy.testing import make_synthetic_qdm_result

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
PIXEL_SIZE: float = 1e-6  # 1 µm per pixel
STANDOFF: float = 5e-6  # 5 µm sensor-to-sample distance
N_PIXELS: int = 50  # map side length (pixels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_bz(
    pypole_dec: float,
    inc: float,
    moment: float,
    n: int = N_PIXELS,
    pixel_size: float = PIXEL_SIZE,
    standoff: float = STANDOFF,
) -> np.ndarray:
    """Generate a synthetic Bz map (Tesla) for a single centred dipole.

    Args:
        pypole_dec: Declination in pypole convention (deg, 0 -> -Y).
        inc: Inclination in degrees (0 = horizontal).
        moment: Magnetic moment magnitude in A*m^2.
        n: Map side length in pixels.
        pixel_size: Pixel size in metres.
        standoff: Sensor-to-sample distance in metres.

    Returns:
        Bz map as (n, n) ndarray in Tesla.
    """
    dim = np.array([[pypole_dec, inc, moment]], dtype=np.float64)
    xyz = pypole.convert.dim2xyz(dim)
    mx, my, mz = float(xyz[0, 0]), float(xyz[0, 1]), float(xyz[0, 2])
    x_grid, y_grid = pypole.maps.get_grid(pixels=(n, n), pixel_size=pixel_size)
    return pypole.dipole.dipole_field(x_grid, y_grid, 0.0, 0.0, standoff, mx, my, mz)


def _make_source(
    n: int = N_PIXELS,
    pixel_size: float = PIXEL_SIZE,
    pypole_dec: float = 90.0,
    inc: float = 45.0,
    moment: float = 1e-14,
) -> MagneticSource:
    """Build a MagneticSource whose ROI covers the full n x n map.

    Sets center / half_extent so that roi_pixels == (slice(0, n), slice(0, n)).

    Args:
        n: Map side length in pixels.
        pixel_size: Pixel size in metres.
        pypole_dec: Declination in pypole/MagneticModel convention (dec=0 -> -Y).
        inc: Inclination in degrees for the initial model.
        moment: Magnetic moment magnitude in A*m^2 for the initial model.

    Returns:
        MagneticSource with ROI covering the full n x n grid.
    """
    model = MagneticModel(declination=pypole_dec, inclination=inc, magnetic_moment=moment)
    # center=(n/2 - 0.5, n/2 - 0.5), half=(n/2 - 0.5, n/2 - 0.5) gives
    # round(0) = 0 and round(n-1) + 1 = n, so roi_pixels covers slice(0, n).
    half = (n - 1) / 2.0
    return MagneticSource(
        name="test_source",
        center=(half, half),
        half_extent=(half, half),
        pixel_spacing=pixel_size,
        model=model,
    )


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestFitSourceRoundTrip:
    """Synthetic dipole round-trip: generate Bz map, fit, recover parameters."""

    def test_recovered_moment_within_10_percent(self) -> None:
        """Fitted magnetic moment should be within 10% of the true value."""
        true_moment = 1e-14
        bz_T = _make_synthetic_bz(pypole_dec=90.0, inc=45.0, moment=true_moment)
        source = _make_source(pypole_dec=90.0, inc=45.0, moment=true_moment)

        result = fit_source(bz_T, source, standoff_m=STANDOFF)

        assert isinstance(result, FitSourceResult)
        assert result.raw.success, f"Fit did not converge: {result.raw.message}"
        recovered = result.source.model.magnetic_moment
        rel_error = abs(recovered - true_moment) / true_moment
        assert rel_error < 0.10, f"Moment error {rel_error:.1%} exceeds 10%"

    def test_returns_fit_source_result_type(self) -> None:
        """fit_source always returns a FitSourceResult with a MagneticSource."""
        bz_T = _make_synthetic_bz(pypole_dec=180.0, inc=0.0, moment=5e-15)
        source = _make_source(pypole_dec=180.0, inc=0.0, moment=5e-15)

        result = fit_source(bz_T, source, standoff_m=STANDOFF)

        assert isinstance(result, FitSourceResult)
        assert isinstance(result.source, MagneticSource)

    def test_raw_result_has_scipy_attributes(self) -> None:
        """Raw OptimizeResult exposes success, cost, and nfev."""
        bz_T = _make_synthetic_bz(pypole_dec=0.0, inc=0.0, moment=1e-14)
        source = _make_source(pypole_dec=0.0, inc=0.0, moment=1e-14)

        result = fit_source(bz_T, source, standoff_m=STANDOFF)

        assert hasattr(result.raw, "success")
        assert hasattr(result.raw, "cost")
        assert hasattr(result.raw, "nfev")

    def test_source_is_immutable_copy(self) -> None:
        """Fitted source must be a distinct object from the input source."""
        bz_T = _make_synthetic_bz(pypole_dec=90.0, inc=30.0, moment=1e-14)
        source = _make_source(pypole_dec=90.0, inc=30.0, moment=1e-14)

        result = fit_source(bz_T, source, standoff_m=STANDOFF)

        assert result.source is not source


# ---------------------------------------------------------------------------
# fit_sources integration tests
# ---------------------------------------------------------------------------


class TestFitSources:
    """Integration tests: fit_sources iterates field_sources correctly."""

    def test_single_magnetic_source_returns_one_result(self) -> None:
        """fit_sources returns exactly one FitSourceResult for one MagneticSource."""
        qdm_result = make_synthetic_qdm_result(shape=(N_PIXELS, N_PIXELS), pixel_spacing=PIXEL_SIZE)
        source = _make_source(n=N_PIXELS, pixel_size=PIXEL_SIZE)
        qdm_result = qdm_result.model_copy(update={"field_sources": [source]})

        fits = fit_sources(qdm_result, standoff_m=STANDOFF)

        assert len(fits) == 1
        assert isinstance(fits[0], FitSourceResult)

    def test_skips_generic_field_sources(self) -> None:
        """fit_sources ignores FieldSource (kind='generic') instances."""
        qdm_result = make_synthetic_qdm_result(shape=(N_PIXELS, N_PIXELS), pixel_spacing=PIXEL_SIZE)
        generic = FieldSource(name="generic_source")
        qdm_result = qdm_result.model_copy(update={"field_sources": [generic]})

        fits = fit_sources(qdm_result, standoff_m=STANDOFF)

        assert fits == []

    def test_empty_field_sources_returns_empty_list(self) -> None:
        """fit_sources returns [] when field_sources is empty."""
        qdm_result = make_synthetic_qdm_result(shape=(N_PIXELS, N_PIXELS), pixel_spacing=PIXEL_SIZE)
        # field_sources defaults to []
        fits = fit_sources(qdm_result, standoff_m=STANDOFF)

        assert fits == []

    def test_mixed_sources_only_fits_magnetic(self) -> None:
        """fit_sources fits only MagneticSource entries in a mixed list."""
        qdm_result = make_synthetic_qdm_result(shape=(N_PIXELS, N_PIXELS), pixel_spacing=PIXEL_SIZE)
        magnetic = _make_source(n=N_PIXELS, pixel_size=PIXEL_SIZE)
        generic = FieldSource(name="noise_source")
        qdm_result = qdm_result.model_copy(update={"field_sources": [generic, magnetic]})

        fits = fit_sources(qdm_result, standoff_m=STANDOFF)

        assert len(fits) == 1
        assert isinstance(fits[0], FitSourceResult)


# ---------------------------------------------------------------------------
# Robustness / convergence failure
# ---------------------------------------------------------------------------


class TestConvergenceFailure:
    """Robustness: degenerate inputs must not raise."""

    def test_zero_field_does_not_raise(self) -> None:
        """Fitting an all-zero Bz map must not raise an exception.

        The fit may not succeed (raw.success may be False), but it must
        return a FitSourceResult regardless.
        """
        bz_T = np.zeros((N_PIXELS, N_PIXELS))
        source = _make_source()

        result = fit_source(bz_T, source, standoff_m=STANDOFF)

        assert isinstance(result, FitSourceResult)
        # Confirm the source is still a valid MagneticSource
        assert isinstance(result.source, MagneticSource)
        assert result.source.model.magnetic_moment > 0


# ---------------------------------------------------------------------------
# compute_field tests
# ---------------------------------------------------------------------------


class TestComputeField:
    """compute_field: forward dipole model over the source ROI."""

    def test_returns_correct_shape(self) -> None:
        """Output shape must match roi_pixels dimensions."""
        source = _make_source(n=N_PIXELS)
        field = compute_field(source, standoff_m=STANDOFF)

        roi_row, roi_col = source.roi_pixels
        expected_h = roi_row.stop - roi_row.start
        expected_w = roi_col.stop - roi_col.start
        assert field.shape == (expected_h, expected_w)

    def test_matches_synthetic_generator(self) -> None:
        """compute_field must agree with the pypole synthetic generator used in tests.

        For matching dipole parameters, both should produce the same Bz map.
        """
        pypole_dec, inc, moment = 90.0, 45.0, 1e-14
        model = MagneticModel(declination=pypole_dec, inclination=inc, magnetic_moment=moment)
        source = MagneticSource(
            name="test",
            center=((N_PIXELS - 1) / 2.0, (N_PIXELS - 1) / 2.0),
            half_extent=((N_PIXELS - 1) / 2.0, (N_PIXELS - 1) / 2.0),
            pixel_spacing=PIXEL_SIZE,
            model=model,
        )

        expected = _make_synthetic_bz(pypole_dec=pypole_dec, inc=inc, moment=moment)
        computed = compute_field(source, standoff_m=STANDOFF)

        np.testing.assert_allclose(computed, expected, rtol=1e-10)

    def test_residual_small_after_fit(self) -> None:
        """Residual between measured and forward model is small after fitting.

        Generate a synthetic Bz, fit the source, then evaluate compute_field with
        the fitted parameters. The RMS residual should be < 10% of the RMS signal.
        """
        true_moment = 1e-14
        bz_T = _make_synthetic_bz(pypole_dec=90.0, inc=45.0, moment=true_moment)
        source = _make_source(pypole_dec=90.0, inc=45.0, moment=true_moment)

        fit_result = fit_source(bz_T, source, standoff_m=STANDOFF)
        predicted = compute_field(fit_result.source, standoff_m=STANDOFF)

        residual_rms = float(np.sqrt(np.mean((bz_T - predicted) ** 2)))
        signal_rms = float(np.sqrt(np.mean(bz_T**2)))
        assert residual_rms < 0.10 * signal_rms, (
            f"Residual RMS {residual_rms:.3e} T exceeds 10% of signal RMS {signal_rms:.3e} T"
        )
