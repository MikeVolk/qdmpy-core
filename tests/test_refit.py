"""Tests for fitting/refit.py — bad-fit detection and neighbor-based refitting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from qdmpy.fitting.manager import FOLDED_CONSTRAINT_OVERRIDES, FitManager
from qdmpy.fitting.refit import (
    RefitSettings,
    compute_neighbor_guesses,
    identify_outlier_pixels,
    refit_outliers,
)
from qdmpy.fitting.result import FitResult
from qdmpy.settings import FitSettings, ModelConstraintsSettings, ModelSettings, QDMpySettings
from qdmpy.testing import RecordingFitBackend

# Mock settings for FitManager-backed refit tests (mirrors tests/test_fit.py's MOCK_SETTINGS).
_REFIT_SETTINGS = QDMpySettings(
    fit=FitSettings(estimator="LSE", max_number_iterations=100, tolerance=1e-6),
    model=ModelSettings(
        constraints=ModelConstraintsSettings(
            constraint_units="absolute_ghz",
            center_min=2.8,
            center_max=2.9,
            center_type="FREE",
            width_min=0.001,
            width_max=0.01,
            width_type="FREE",
            contrast_min=0.0,
            contrast_max=1.0,
            contrast_type="FREE",
            offset_min=-0.1,
            offset_max=0.1,
            offset_type="FREE",
        )
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fit_result(
    params: dict,
    h: int = 4,
    w: int = 4,
    n_pol: int = 1,
    n_frange: int = 1,
    model_name: str = "ESRSINGLE",
) -> FitResult:
    """Construct a minimal FitResult with the given parameters."""
    return FitResult(
        parameters=params,
        scan_dimensions=(h, w),
        pixel_spacing=4e-6,
        model_name=model_name,
    )


def _make_data_array(
    n_pol: int = 1, n_frange: int = 1, h: int = 4, w: int = 4, n_freq: int = 20
) -> xr.DataArray:
    """Create an xr.DataArray with the dims expected by refit_outliers."""
    data = np.ones((n_pol, n_frange, h, w, n_freq), dtype=np.float32)
    return xr.DataArray(
        data,
        dims=("polarity", "freq_range", "y", "x", "freq_idx"),
    )


def _make_mock_fm(
    param_names: list[str],
    return_value: float = 2.87,
) -> MagicMock:
    """Create a FitManager mock whose fit_frange returns uniform good results."""
    n_params = len(param_names)
    fm = MagicMock()
    fm.parameter_names = param_names
    fm.n_parameter = n_params

    def _fit_frange(
        data: np.ndarray,
        freq: np.ndarray,
        initial_params: np.ndarray,
        *,
        irange: int,
        n_frange: int,
        constraint_overrides: object = None,
    ) -> list:
        n_data = data.shape[0] * data.shape[1]  # n_pol * n_pixel
        params_out = np.full((n_data, n_params), return_value, dtype=np.float32)
        states_out = np.zeros(n_data, dtype=np.int32)
        chi2_out = np.full(n_data, 0.05, dtype=np.float32)
        iters_out = np.ones(n_data, dtype=np.int32) * 10
        return [params_out, states_out, chi2_out, iters_out, 0.01]

    fm.fit_frange.side_effect = _fit_frange
    return fm


# ---------------------------------------------------------------------------
# RefitSettings
# ---------------------------------------------------------------------------


class TestRefitSettings:
    def test_defaults(self) -> None:
        s = RefitSettings()
        assert s.chi2_percentile == 90.0
        assert s.include_non_converged is True
        assert s.window_size == 5
        assert s.min_good_neighbors == 3
        assert s.max_iterations == 1

    def test_invalid_percentile_above_100(self) -> None:
        with pytest.raises(Exception):
            RefitSettings(chi2_percentile=101.0)

    def test_invalid_percentile_below_0(self) -> None:
        with pytest.raises(Exception):
            RefitSettings(chi2_percentile=-0.1)

    def test_invalid_window_size_even(self) -> None:
        with pytest.raises(Exception):
            RefitSettings(window_size=4)

    def test_invalid_window_size_too_small(self) -> None:
        with pytest.raises(Exception):
            RefitSettings(window_size=1)

    def test_invalid_min_good_neighbors_zero(self) -> None:
        with pytest.raises(Exception):
            RefitSettings(min_good_neighbors=0)

    def test_invalid_max_iterations_zero(self) -> None:
        with pytest.raises(Exception):
            RefitSettings(max_iterations=0)

    def test_frozen_immutable(self) -> None:
        s = RefitSettings()
        with pytest.raises(Exception):
            s.chi2_percentile = 50.0  # type: ignore[misc]

    def test_boundary_percentile_0_and_100(self) -> None:
        # Should not raise
        RefitSettings(chi2_percentile=0.0)
        RefitSettings(chi2_percentile=100.0)


# ---------------------------------------------------------------------------
# identify_outlier_pixels
# ---------------------------------------------------------------------------


class TestIdentifyOutlierPixels:
    def test_high_chi2_pixel_flagged(self) -> None:
        """A single pixel with very high chi2 should be flagged."""
        chi2 = np.ones((1, 1, 4, 4), dtype=np.float32) * 0.1
        chi2[0, 0, 2, 3] = 100.0
        states = np.zeros_like(chi2, dtype=np.int32)
        settings = RefitSettings(chi2_percentile=90.0, include_non_converged=False)

        mask = identify_outlier_pixels(chi2, states, settings)

        assert mask.shape == chi2.shape
        assert mask[0, 0, 2, 3]
        # Lower-chi2 pixels should not all be flagged (only the top 10%)
        assert mask.sum() < chi2.size

    def test_non_converged_flagged_when_enabled(self) -> None:
        """States != 0 should be flagged when include_non_converged=True."""
        chi2 = np.ones((2, 2, 4, 4), dtype=np.float32) * 0.1
        states = np.zeros_like(chi2, dtype=np.int32)
        states[0, 0, 1, 1] = 3  # non-converged

        mask_with = identify_outlier_pixels(
            chi2, states, RefitSettings(chi2_percentile=100.0, include_non_converged=True)
        )
        mask_without = identify_outlier_pixels(
            chi2, states, RefitSettings(chi2_percentile=100.0, include_non_converged=False)
        )

        assert mask_with[0, 0, 1, 1]
        assert not mask_without[0, 0, 1, 1]

    def test_percentile_100_flags_nothing(self) -> None:
        """At percentile=100, no pixel strictly exceeds the maximum."""
        chi2 = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        states = np.zeros_like(chi2, dtype=np.int32)
        settings = RefitSettings(chi2_percentile=100.0, include_non_converged=False)

        mask = identify_outlier_pixels(chi2, states, settings)

        assert not np.any(mask)

    def test_thresholds_are_independent_per_subspace(self) -> None:
        """Each (pol, frange) computes its threshold independently."""
        chi2 = np.ones((2, 2, 4, 4), dtype=np.float32) * 0.5
        chi2[0, 0, 0, 0] = 50.0  # worst in subspace (0, 0)
        chi2[1, 1, 3, 3] = 10.0  # worst in subspace (1, 1), much smaller
        states = np.zeros_like(chi2, dtype=np.int32)
        settings = RefitSettings(chi2_percentile=90.0, include_non_converged=False)

        mask = identify_outlier_pixels(chi2, states, settings)

        assert mask[0, 0, 0, 0]  # outlier in (pol=0, frange=0)
        assert mask[1, 1, 3, 3]  # outlier in (pol=1, frange=1)
        # Cross-subspace: the value 50 in subspace (0,0) is not relevant to (1,1)
        assert not mask[0, 1, 0, 0]  # value is 0.5 in (pol=0, frange=1) — not an outlier

    def test_output_shape_matches_input(self) -> None:
        chi2 = np.random.default_rng(0).random((3, 2, 5, 7)).astype(np.float32)
        states = np.zeros_like(chi2, dtype=np.int32)
        mask = identify_outlier_pixels(chi2, states, RefitSettings())
        assert mask.shape == chi2.shape
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# compute_neighbor_guesses
# ---------------------------------------------------------------------------


class TestComputeNeighborGuesses:
    def test_central_outlier_gets_neighbor_median(self) -> None:
        """An outlier in the center of a uniform grid gets the correct median guess."""
        h, w = 7, 7
        center = np.full((1, 1, h, w), 2.87, dtype=np.float32)
        # Two neighbors with distinct values; rest at 2.87
        center[0, 0, 2, 3] = 2.85
        center[0, 0, 4, 3] = 2.89
        outlier_mask = np.zeros((1, 1, h, w), dtype=bool)
        outlier_mask[0, 0, 3, 3] = True  # center pixel is the outlier

        settings = RefitSettings(window_size=3, min_good_neighbors=2)
        guess_dict, refittable = compute_neighbor_guesses(
            {"center": center}, outlier_mask, settings, ["center"]
        )

        assert refittable[0, 0, 3, 3]
        # Median of the 8 good 3x3 neighbors (mostly 2.87 with 2.85 and 2.89)
        assert abs(guess_dict["center"][0, 0, 3, 3] - 2.87) < 0.02

    def test_too_few_neighbors_not_refittable(self) -> None:
        """A pixel without enough good neighbors is marked not-refittable."""
        h, w = 3, 3
        center = np.ones((1, 1, h, w), dtype=np.float32)
        outlier_mask = np.zeros((1, 1, h, w), dtype=bool)
        # Mark all pixels except (0, 0) as outliers
        for y in range(h):
            for x in range(w):
                if (y, x) != (0, 0):
                    outlier_mask[0, 0, y, x] = True

        # With only one good pixel anywhere, no outlier can satisfy min_good_neighbors=3
        settings = RefitSettings(window_size=3, min_good_neighbors=3)
        _, refittable = compute_neighbor_guesses(
            {"center": center}, outlier_mask, settings, ["center"]
        )

        assert not np.any(refittable)

    def test_chi2_and_states_not_in_guess_dict(self) -> None:
        """chi2 and states keys should not appear in guess_dict."""
        h, w = 5, 5
        center = np.full((1, 1, h, w), 2.87, dtype=np.float32)
        chi2 = np.ones((1, 1, h, w), dtype=np.float32)
        states = np.zeros((1, 1, h, w), dtype=np.int32)
        outlier_mask = np.zeros((1, 1, h, w), dtype=bool)
        outlier_mask[0, 0, 2, 2] = True

        params = {"center": center, "chi2": chi2, "states": states}
        guess_dict, _ = compute_neighbor_guesses(
            params,
            outlier_mask,
            settings=RefitSettings(window_size=3, min_good_neighbors=1),
            model_parameter_names=["center"],
        )

        assert "center" in guess_dict
        assert "chi2" not in guess_dict
        assert "states" not in guess_dict

    def test_good_pixels_retain_original_values(self) -> None:
        """Non-outlier pixels in guess_dict keep their original values."""
        rng = np.random.default_rng(7)
        h, w = 5, 5
        original = (rng.random((1, 1, h, w)) + 2.87).astype(np.float32)
        outlier_mask = np.zeros((1, 1, h, w), dtype=bool)
        outlier_mask[0, 0, 2, 2] = True

        guess_dict, _ = compute_neighbor_guesses(
            {"center": original.copy()},
            outlier_mask,
            settings=RefitSettings(window_size=3, min_good_neighbors=1),
            model_parameter_names=["center"],
        )

        good = ~outlier_mask
        np.testing.assert_array_equal(guess_dict["center"][good], original[good])

    def test_multiple_parameters_computed_independently(self) -> None:
        """All model parameters get independent median guesses."""
        h, w = 5, 5
        center = np.full((1, 1, h, w), 2.87, dtype=np.float32)
        width = np.full((1, 1, h, w), 0.005, dtype=np.float32)
        outlier_mask = np.zeros((1, 1, h, w), dtype=bool)
        outlier_mask[0, 0, 2, 2] = True

        guess_dict, refittable = compute_neighbor_guesses(
            {"center": center, "width": width},
            outlier_mask,
            settings=RefitSettings(window_size=3, min_good_neighbors=1),
            model_parameter_names=["center", "width"],
        )

        assert "center" in guess_dict
        assert "width" in guess_dict
        assert refittable[0, 0, 2, 2]
        assert abs(guess_dict["center"][0, 0, 2, 2] - 2.87) < 1e-4
        assert abs(guess_dict["width"][0, 0, 2, 2] - 0.005) < 1e-5


# ---------------------------------------------------------------------------
# refit_outliers
# ---------------------------------------------------------------------------


class TestRefitOutliers:
    def _make_simple_fit_result(
        self,
        h: int = 6,
        w: int = 6,
        n_pol: int = 1,
        n_frange: int = 1,
        bad_pixel: tuple[int, int] = (3, 3),
    ) -> FitResult:
        """FitResult with one clearly bad pixel and uniform good pixels."""
        shape = (n_pol, n_frange, h, w)
        center = np.full(shape, 2.87, dtype=np.float32)
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[:, :, bad_pixel[0], bad_pixel[1]] = 100.0
        states = np.zeros(shape, dtype=np.int32)
        return FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )

    def test_no_outliers_returns_original_object(self) -> None:
        """When no outliers are found, the exact same FitResult is returned."""
        chi2 = np.ones((1, 1, 4, 4), dtype=np.float32) * 0.1
        fr = _make_fit_result(
            {
                "center": np.full((1, 1, 4, 4), 2.87, dtype=np.float32),
                "chi2": chi2,
                "states": np.zeros((1, 1, 4, 4), dtype=np.int32),
            },
        )
        fm = _make_mock_fm(["center"])
        data = _make_data_array()
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        # percentile=100 → no pixel strictly exceeds the max
        settings = RefitSettings(chi2_percentile=100.0, include_non_converged=False)

        result = refit_outliers(fr, data, freq, fm, settings)

        assert result is fr
        fm.fit_frange.assert_not_called()

    def test_outlier_pixel_value_is_updated(self) -> None:
        """The bad pixel's center value should be replaced with the refit output."""
        h, w = 6, 6
        fr = self._make_simple_fit_result(h=h, w=w, bad_pixel=(3, 3))
        # Mock returns 2.871 for the refitted pixel
        fm = _make_mock_fm(["center"], return_value=2.871)
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        result = refit_outliers(fr, data, freq, fm, settings)

        assert result.parameters["center"][0, 0, 3, 3] == pytest.approx(2.871)

    def test_good_pixels_are_unchanged(self) -> None:
        """Non-outlier pixels must keep their original parameter values."""
        h, w = 6, 6
        fr = self._make_simple_fit_result(h=h, w=w, bad_pixel=(3, 3))
        fm = _make_mock_fm(["center"], return_value=2.871)
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        result = refit_outliers(fr, data, freq, fm, settings)

        # Pixel (0, 0) is a good pixel — should remain at 2.87
        assert result.parameters["center"][0, 0, 0, 0] == pytest.approx(2.87)

    def test_refit_info_metadata_populated(self) -> None:
        """refit_info should be present in the new FitResult's metadata."""
        h, w = 6, 6
        fr = self._make_simple_fit_result(h=h, w=w, bad_pixel=(3, 3))
        fr_with_meta = FitResult(
            parameters={k: np.array(v) for k, v in fr.parameters.items()},
            scan_dimensions=fr.scan_dimensions,
            pixel_spacing=fr.pixel_spacing,
            model_name=fr.model_name,
            metadata={"original_key": "preserved"},
        )
        fm = _make_mock_fm(["center"])
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        result = refit_outliers(fr_with_meta, data, freq, fm, settings)

        assert "refit_info" in result.metadata
        assert "original_key" in result.metadata  # original metadata preserved
        info = result.metadata["refit_info"]
        assert "chi2_percentile" in info
        assert "n_outliers_detected" in info
        assert "n_refitted" in info
        assert "per_frange" in info

    def test_original_fit_result_not_mutated(self) -> None:
        """The input FitResult's parameters must not be modified."""
        h, w = 6, 6
        fr = self._make_simple_fit_result(h=h, w=w, bad_pixel=(3, 3))
        original_center_at_bad = float(fr.parameters["center"][0, 0, 3, 3])
        fm = _make_mock_fm(["center"], return_value=2.999)
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        refit_outliers(fr, data, freq, fm, settings)

        # Original must be unchanged (arrays are read-only in FitResult)
        assert float(fr.parameters["center"][0, 0, 3, 3]) == pytest.approx(original_center_at_bad)

    def _make_cluster_fit_result(
        self,
        h: int = 10,
        w: int = 10,
        cluster_rows: slice = slice(4, 7),
        cluster_cols: slice = slice(4, 7),
    ) -> FitResult:
        """FitResult with a rectangular cluster of bad pixels."""
        shape = (1, 1, h, w)
        center = np.full(shape, 2.87, dtype=np.float32)
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[0, 0, cluster_rows, cluster_cols] = 100.0
        states = np.zeros(shape, dtype=np.int32)
        return FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )

    def test_cluster_border_refitted_in_one_pass(self) -> None:
        """With max_iterations=1, only cluster border pixels are refitted.

        A 3x3 cluster (9 pixels) with window_size=3: the 8 border pixels have
        good neighbors outside the cluster; the center pixel has none.
        """
        h, w = 10, 10
        fr = self._make_cluster_fit_result(
            h=h, w=w, cluster_rows=slice(4, 7), cluster_cols=slice(4, 7)
        )
        fm = _make_mock_fm(["center"], return_value=2.871)
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        settings = RefitSettings(
            chi2_percentile=90.0,
            include_non_converged=False,
            window_size=3,
            min_good_neighbors=1,
            max_iterations=1,
        )

        result = refit_outliers(fr, data, freq, fm, settings)

        # Border pixels get refitted — value changes to mock return (2.871)
        assert result.parameters["center"][0, 0, 4, 4] == pytest.approx(2.871)
        assert result.parameters["center"][0, 0, 4, 5] == pytest.approx(2.871)
        # Center of cluster (5, 5) has no good neighbors in pass 1 — still original value
        assert result.parameters["center"][0, 0, 5, 5] == pytest.approx(2.87)

    def test_cluster_interior_cleared_by_second_pass(self) -> None:
        """With max_iterations=2, the cluster interior is reached after the border is cleared.

        Pass 1: border pixels refitted (mock returns low chi2).
        Pass 2: center pixel now has good neighbors (the refitted border) -> refitted.
        """
        h, w = 10, 10
        fr = self._make_cluster_fit_result(
            h=h, w=w, cluster_rows=slice(4, 7), cluster_cols=slice(4, 7)
        )
        fm = _make_mock_fm(["center"], return_value=2.871)
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        settings = RefitSettings(
            chi2_percentile=90.0,
            include_non_converged=False,
            window_size=3,
            min_good_neighbors=1,
            max_iterations=2,
        )

        result = refit_outliers(fr, data, freq, fm, settings)

        # All cluster pixels (including center) should be refitted across 2 passes
        for r in range(4, 7):
            for c in range(4, 7):
                assert result.parameters["center"][0, 0, r, c] == pytest.approx(2.871), (
                    f"Pixel ({r},{c}) not refitted"
                )

    def test_early_convergence_stops_loop(self) -> None:
        """The iteration loop stops before max_iterations when no more pixels can be refitted."""
        h, w = 6, 6
        fr = self._make_simple_fit_result(h=h, w=w, bad_pixel=(3, 3))
        call_count = []

        def counting_fit_frange(
            data: np.ndarray,
            freq: np.ndarray,
            initial_params: np.ndarray,
            *,
            irange: int,
            n_frange: int,
            constraint_overrides: object = None,
        ) -> list:
            call_count.append(1)
            n_data = data.shape[0] * data.shape[1]
            return [
                np.full((n_data, 1), 2.871, dtype=np.float32),
                np.zeros(n_data, dtype=np.int32),
                np.full(n_data, 0.05, dtype=np.float32),  # good chi2 after refit
                np.ones(n_data, dtype=np.int32),
                0.01,
            ]

        fm = MagicMock()
        fm.parameter_names = ["center"]
        fm.n_parameter = 1
        fm.fit_frange.side_effect = counting_fit_frange

        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.84, 2.90, 20).reshape(1, 20)
        # max_iterations=10, but only 1 bad pixel -> should converge after pass 1
        settings = RefitSettings(
            chi2_percentile=90.0,
            include_non_converged=False,
            min_good_neighbors=1,
            max_iterations=10,
        )

        refit_outliers(fr, data, freq, fm, settings)

        # fit_frange should have been called exactly once (single bad pixel, 1 pass suffices)
        assert len(call_count) == 1

    def test_multi_frange_multi_pol(self) -> None:
        """refit_outliers works correctly with 2 polarities and 2 frequency ranges."""
        n_pol, n_frange, h, w = 2, 2, 6, 6
        shape = (n_pol, n_frange, h, w)
        center = np.full(shape, 2.87, dtype=np.float32)
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[0, 0, 2, 2] = 100.0  # outlier in pol=0, frange=0
        chi2[1, 1, 4, 4] = 100.0  # outlier in pol=1, frange=1
        states = np.zeros(shape, dtype=np.int32)

        fr = FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )
        fm = _make_mock_fm(["center"], return_value=2.88)
        data = _make_data_array(n_pol=n_pol, n_frange=n_frange, h=h, w=w)
        freq = np.stack([np.linspace(2.82, 2.87, 20), np.linspace(2.87, 2.92, 20)])
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        result = refit_outliers(fr, data, freq, fm, settings)

        assert isinstance(result, FitResult)
        assert "refit_info" in result.metadata
        assert fm.fit_frange.call_count >= 1

    def test_folded_fit_result_type_preserved(self) -> None:
        """refit_outliers returns a FitResult when given a folded FitResult."""
        h, w = 6, 6
        n_pol, n_frange = 1, 1
        shape = (n_pol, n_frange, h, w)
        center = np.full(shape, 2.880, dtype=np.float32)  # absolute GHz
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[0, 0, 3, 3] = 100.0
        states = np.zeros(shape, dtype=np.int32)
        fr = FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
            metadata={"folded_fit": True},
        )

        fm = _make_mock_fm(["center"], return_value=2.881)
        data = _make_data_array(n_pol=n_pol, n_frange=n_frange, h=h, w=w)
        freq = np.linspace(2.875, 2.930, 20).reshape(1, 20)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        result = refit_outliers(fr, data, freq, fm, settings)

        assert isinstance(result, FitResult)
        assert "refit_info" in result.metadata

    def _make_esrsingle_fit_result(
        self,
        h: int = 6,
        w: int = 6,
        bad_pixel: tuple[int, int] = (3, 3),
        metadata: dict[str, object] | None = None,
    ) -> FitResult:
        """FitResult with all 4 ESRSINGLE parameters, one clearly bad pixel."""
        shape = (1, 1, h, w)
        params = {
            "center": np.full(shape, 2.87, dtype=np.float32),
            "width": np.full(shape, 0.005, dtype=np.float32),
            "contrast": np.full(shape, 0.1, dtype=np.float32),
            "offset": np.full(shape, 0.0, dtype=np.float32),
            "chi2": np.ones(shape, dtype=np.float32) * 0.1,
            "states": np.zeros(shape, dtype=np.int32),
        }
        params["chi2"][0, 0, bad_pixel[0], bad_pixel[1]] = 100.0
        return FitResult(
            parameters=params,
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
            metadata=metadata or {},
        )

    def test_refit_applies_freq_cutoff(self) -> None:
        """refit_outliers crops the refit frequency axis to the configured freq_cutoff.

        Regression test for review finding F3: fit_frange() used to ignore
        freq_cutoff entirely, so outlier pixels were refit against the full
        frequency axis while the rest of the map was fit against the cutoff
        range.
        """
        h, w, n_freq = 6, 6, 30
        fr = self._make_esrsingle_fit_result(h=h, w=w, bad_pixel=(3, 3))
        backend = RecordingFitBackend()
        fit_manager = FitManager(
            model_name="ESRSINGLE",
            settings=_REFIT_SETTINGS,
            backend=backend,
            freq_cutoff={"low": {"max": 2.87}},
        )
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w, n_freq=n_freq)
        freq = np.linspace(2.82, 2.92, n_freq).reshape(1, n_freq)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        refit_outliers(fr, data, freq, fit_manager, settings)

        assert len(backend.freq_calls) >= 1
        recorded_freq = backend.freq_calls[-1]
        assert float(np.max(recorded_freq)) <= 2.87
        assert recorded_freq.size < n_freq

    def test_refit_applies_folded_constraint_overrides(self) -> None:
        """refit_outliers layers FOLDED_CONSTRAINT_OVERRIDES onto a folded fit's refit.

        Regression test for review finding F3: fit_frange() used to always
        fit with the manager's plain base constraints, so outlier pixels in a
        folded fit lost the folded contrast/offset bounds that the original
        fit_folded() call applied.
        """
        h, w = 6, 6
        fr = self._make_esrsingle_fit_result(
            h=h, w=w, bad_pixel=(3, 3), metadata={"folded_fit": True}
        )
        backend = RecordingFitBackend()
        fit_manager = FitManager(model_name="ESRSINGLE", settings=_REFIT_SETTINGS, backend=backend)
        data = _make_data_array(n_pol=1, n_frange=1, h=h, w=w)
        freq = np.linspace(2.875, 2.930, 20).reshape(1, 20)
        settings = RefitSettings(chi2_percentile=90.0, min_good_neighbors=1)

        refit_outliers(fr, data, freq, fit_manager, settings)

        assert len(backend.constraints_calls) >= 1
        offset_idx = fit_manager.parameter_names.index("offset")
        recorded_offset_bounds = backend.constraints_calls[-1][
            0, 2 * offset_idx : 2 * offset_idx + 2
        ]
        assert recorded_offset_bounds[0] == pytest.approx(
            FOLDED_CONSTRAINT_OVERRIDES["offset"].vmin
        )
        assert recorded_offset_bounds[1] == pytest.approx(
            FOLDED_CONSTRAINT_OVERRIDES["offset"].vmax
        )


# ---------------------------------------------------------------------------
# Measurement.refit_outliers and fit_odmr(refit_outliers=True)
# ---------------------------------------------------------------------------


class TestMeasurementRefit:
    """Integration-style tests for the Measurement-level refit API.

    Uses object.__new__ to bypass __init__ and patches _validate_fit_prerequisites
    so no real ODMR data is needed.
    """

    def _make_mock_processed_data(
        self, n_pol: int = 1, n_frange: int = 1, h: int = 4, w: int = 4, n_freq: int = 20
    ) -> MagicMock:
        data_arr = _make_data_array(n_pol=n_pol, n_frange=n_frange, h=h, w=w, n_freq=n_freq)
        freq = np.linspace(2.84, 2.90, n_freq).reshape(n_frange, n_freq)
        mock_pd = MagicMock()
        mock_pd.data = data_arr
        mock_pd.frequencies = freq
        return mock_pd

    def _make_measurement_stub(self) -> object:
        """Return a Measurement instance with minimal state, bypassing __init__."""
        from qdmpy.measurement import Measurement

        m = object.__new__(Measurement)
        m._fit_model = "ESRSINGLE"
        m.pixel_spacing = 4e-6
        m.light_image = np.zeros((4, 4))
        m.laser_image = np.zeros((4, 4))
        return m

    def _make_qdm_result(self, h: int = 4, w: int = 4) -> object:
        from qdmpy.result import QDMResult

        shape = (1, 1, h, w)
        center = np.full(shape, 2.87, dtype=np.float32)
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[0, 0, 2, 2] = 100.0
        states = np.zeros(shape, dtype=np.int32)
        fr = FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )
        return QDMResult(
            fit_result=fr,
            light_image=np.zeros((h, w)),
            laser_image=np.zeros((h, w)),
        )

    def test_refit_outliers_method_returns_qdm_result(self) -> None:
        """Measurement.refit_outliers() returns a QDMResult with refit_info."""
        from qdmpy.measurement import Measurement
        from qdmpy.result import QDMResult

        m = self._make_measurement_stub()
        mock_pd = self._make_mock_processed_data()
        qdm_result = self._make_qdm_result()

        with (
            patch.object(Measurement, "_validate_fit_prerequisites", return_value=mock_pd),
            patch("qdmpy.fitting.manager.FitManager") as MockFM,
        ):
            fm_instance = _make_mock_fm(["center"])
            MockFM.return_value = fm_instance

            result = m.refit_outliers(
                qdm_result,
                refit_settings=RefitSettings(min_good_neighbors=1),
            )

        assert isinstance(result, QDMResult)

    def test_fit_odmr_refit_outliers_kwarg_triggers_refit(self) -> None:
        """fit_odmr(refit_outliers=True) should invoke self.refit_outliers."""
        from qdmpy.measurement import Measurement
        from qdmpy.result import QDMResult

        h, w = 4, 4
        m = self._make_measurement_stub()
        mock_pd = self._make_mock_processed_data(h=h, w=w)
        shape = (1, 1, h, w)
        center = np.full(shape, 2.87, dtype=np.float32)
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[0, 0, 2, 2] = 100.0
        states = np.zeros(shape, dtype=np.int32)
        initial_fr = FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )
        _ = QDMResult(
            fit_result=initial_fr,
            light_image=np.zeros((h, w)),
            laser_image=np.zeros((h, w)),
        )

        refit_called = []

        def _fake_refit(result: QDMResult, **kwargs: object) -> QDMResult:
            refit_called.append(True)
            return result

        with (
            patch.object(Measurement, "_validate_fit_prerequisites", return_value=mock_pd),
            patch("qdmpy.fitting.manager.FitManager") as MockFM,
            patch.object(Measurement, "refit_outliers", side_effect=_fake_refit),
        ):
            fm_instance = MagicMock()
            fm_instance.fit.return_value = initial_fr
            MockFM.return_value = fm_instance

            m.fit_odmr(refit_outliers=True)

        assert refit_called, "refit_outliers was not called when refit_outliers=True"

    def test_fit_odmr_refit_passes_freq_cutoff(self) -> None:
        """fit_odmr forwards freq_cutoff to refit_outliers when enabled."""
        from qdmpy.measurement import Measurement
        from qdmpy.result import QDMResult

        h, w = 4, 4
        m = self._make_measurement_stub()
        mock_pd = self._make_mock_processed_data(h=h, w=w)
        shape = (1, 1, h, w)
        initial_fr = FitResult(
            parameters={
                "center": np.full(shape, 2.87, dtype=np.float32),
                "chi2": np.ones(shape, dtype=np.float32) * 0.1,
                "states": np.zeros(shape, dtype=np.int32),
            },
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )
        initial_result = QDMResult(
            fit_result=initial_fr,
            light_image=np.zeros((h, w)),
            laser_image=np.zeros((h, w)),
        )
        cutoff = {"low": {"max": 2.86}}
        captured_kwargs: dict[str, object] = {}

        def _fake_refit(result: QDMResult, **kwargs: object) -> QDMResult:
            captured_kwargs.update(kwargs)
            return result

        with (
            patch.object(Measurement, "_validate_fit_prerequisites", return_value=mock_pd),
            patch("qdmpy.fitting.manager.FitManager") as MockFM,
            patch.object(Measurement, "refit_outliers", side_effect=_fake_refit),
        ):
            fm_instance = MagicMock()
            fm_instance.fit.return_value = initial_fr
            MockFM.return_value = fm_instance

            m.fit_odmr(refit_outliers=True, freq_cutoff=cutoff)

        assert captured_kwargs.get("freq_cutoff") == cutoff

    def test_fit_odmr_without_refit_outliers_skips_refit(self) -> None:
        """fit_odmr() without refit_outliers=True must not call self.refit_outliers."""
        from qdmpy.measurement import Measurement

        h, w = 4, 4
        m = self._make_measurement_stub()
        mock_pd = self._make_mock_processed_data(h=h, w=w)
        shape = (1, 1, h, w)
        initial_fr = FitResult(
            parameters={
                "center": np.full(shape, 2.87, dtype=np.float32),
                "chi2": np.ones(shape, dtype=np.float32) * 0.1,
                "states": np.zeros(shape, dtype=np.int32),
            },
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
        )

        refit_called = []

        def _fake_refit(*args: object, **kwargs: object) -> object:
            refit_called.append(True)
            return args[0]

        with (
            patch.object(Measurement, "_validate_fit_prerequisites", return_value=mock_pd),
            patch("qdmpy.fitting.manager.FitManager") as MockFM,
            patch.object(Measurement, "refit_outliers", side_effect=_fake_refit),
        ):
            fm_instance = MagicMock()
            fm_instance.fit.return_value = initial_fr
            MockFM.return_value = fm_instance

            m.fit_odmr()  # refit_outliers defaults to False

        assert not refit_called, "refit_outliers was unexpectedly called"

    def _make_folded_qdm_result(self, h: int = 4, w: int = 4) -> object:
        from qdmpy.result import QDMResult

        shape = (1, 1, h, w)
        center = np.full(shape, 2.880, dtype=np.float32)  # absolute GHz
        chi2 = np.ones(shape, dtype=np.float32) * 0.1
        chi2[0, 0, 2, 2] = 100.0
        states = np.zeros(shape, dtype=np.int32)
        fr = FitResult(
            parameters={"center": center, "chi2": chi2, "states": states},
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
            metadata={"folded_fit": True},
        )
        return QDMResult(
            fit_result=fr,
            light_image=np.zeros((h, w)),
            laser_image=np.zeros((h, w)),
        )

    def _make_mock_folded_odmr(self, h: int = 4, w: int = 4, n_df: int = 20) -> MagicMock:
        """Return a mock FoldedODMR whose folded_spectrum has the expected shape/coords."""
        from qdmpy.constants import D_ZFS

        spec_vals = np.ones((1, h, w, n_df), dtype=np.float32)
        delta_f = np.linspace(-0.05, 0.05, n_df)
        folded_spectrum = xr.DataArray(
            spec_vals,
            dims=("polarity", "y", "x", "freq_idx"),
            coords={"polarity": ["neg"], "delta_f_ghz": ("freq_idx", delta_f)},
        )
        mock_folded = MagicMock()
        mock_folded.folded_spectrum = folded_spectrum

        # Build to_fit_inputs() return value matching FoldedODMR.to_fit_inputs()
        abs_freq_ghz = D_ZFS + delta_f
        data_5d = np.expand_dims(spec_vals, axis=1)
        data_xr = xr.DataArray(
            data_5d,
            dims=("polarity", "freq_range", "y", "x", "freq_idx"),
        )
        frequencies = abs_freq_ghz.reshape(1, -1)
        mock_folded.to_fit_inputs.return_value = (data_xr, frequencies)
        return mock_folded

    def test_refit_outliers_folded_uses_folded_data(self) -> None:
        """Measurement.refit_outliers routes to folded_odmr when metadata['folded_fit']=True."""
        from qdmpy.result import QDMResult

        m = self._make_measurement_stub()
        mock_folded = self._make_mock_folded_odmr()
        qdm_result = self._make_folded_qdm_result()

        # Attach folded_odmr to the stub (bypassing the property)
        m._folded_odmr = mock_folded

        with patch("qdmpy.fitting.manager.FitManager") as MockFM:
            fm_instance = _make_mock_fm(["center"], return_value=2.881)
            MockFM.return_value = fm_instance

            result = m.refit_outliers(
                qdm_result,
                refit_settings=RefitSettings(min_good_neighbors=1),
            )

        assert isinstance(result, QDMResult)
        assert isinstance(result.fit_result, FitResult)
        # _validate_fit_prerequisites must NOT have been called (folded path skips it)
        assert not hasattr(m, "_validate_fit_prerequisites_called")

    def test_fit_folded_odmr_refit_outliers_kwarg(self) -> None:
        """fit_folded_odmr(refit_outliers=True) calls self.refit_outliers."""
        from qdmpy.measurement import Measurement

        h, w = 4, 4
        m = self._make_measurement_stub()
        mock_pd = self._make_mock_processed_data(h=h, w=w)
        mock_folded = self._make_mock_folded_odmr(h=h, w=w)
        m._folded_odmr = mock_folded

        shape = (1, 1, h, w)
        folded_fr = FitResult(
            parameters={
                "center": np.full(shape, 2.880, dtype=np.float32),
                "chi2": np.ones(shape, dtype=np.float32) * 0.1,
                "states": np.zeros(shape, dtype=np.int32),
            },
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
            metadata={"folded_fit": True},
        )

        refit_called = []

        def _fake_refit(result: object, **kwargs: object) -> object:
            refit_called.append(True)
            return result

        with (
            patch.object(Measurement, "_validate_fit_prerequisites", return_value=mock_pd),
            patch("qdmpy.fitting.manager.FitManager") as MockFM,
            patch.object(Measurement, "refit_outliers", side_effect=_fake_refit),
        ):
            fm_instance = MagicMock()
            fm_instance.fit_folded.return_value = folded_fr
            MockFM.return_value = fm_instance

            m.fit_folded_odmr(mock_folded, refit_outliers=True)

        assert refit_called, "refit_outliers was not called when refit_outliers=True"

    def test_fit_folded_odmr_refit_passes_freq_cutoff(self) -> None:
        """fit_folded_odmr forwards freq_cutoff to refit_outliers when enabled."""
        from qdmpy.measurement import Measurement

        h, w = 4, 4
        m = self._make_measurement_stub()
        mock_pd = self._make_mock_processed_data(h=h, w=w)
        mock_folded = self._make_mock_folded_odmr(h=h, w=w)
        m._folded_odmr = mock_folded

        shape = (1, 1, h, w)
        folded_fr = FitResult(
            parameters={
                "center": np.full(shape, 2.880, dtype=np.float32),
                "chi2": np.ones(shape, dtype=np.float32) * 0.1,
                "states": np.zeros(shape, dtype=np.int32),
            },
            scan_dimensions=(h, w),
            pixel_spacing=4e-6,
            model_name="ESRSINGLE",
            metadata={"folded_fit": True},
        )

        cutoff = {"low": {"min": 2.875}}
        captured_kwargs: dict[str, object] = {}

        def _fake_refit(result: object, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return result

        with (
            patch.object(Measurement, "_validate_fit_prerequisites", return_value=mock_pd),
            patch("qdmpy.fitting.manager.FitManager") as MockFM,
            patch.object(Measurement, "refit_outliers", side_effect=_fake_refit),
        ):
            fm_instance = MagicMock()
            fm_instance.fit_folded.return_value = folded_fr
            MockFM.return_value = fm_instance

            m.fit_folded_odmr(mock_folded, refit_outliers=True, freq_cutoff=cutoff)

        assert captured_kwargs.get("freq_cutoff") == cutoff
