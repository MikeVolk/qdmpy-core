"""Tests for the public helpers in qdmpy.testing.

These helpers are documented public API: they are exported from
``qdmpy/__init__.py`` and executed by README, docs/quickstart.md and four
tutorial notebooks. Before QEP-TEST-001 not one of them had a direct test --
their 93% coverage was incidental, earned by other modules using them as
fixtures, and ``make_synthetic_fit_result`` was called by no test at all
despite being the backbone of two notebooks. A regression in them broke
user-facing documentation while the suite stayed green.

The B111 assertions here are deliberately exact rather than structural. They
pin the sign convention and units that QEP-FIT-004 will move when it
deduplicates the physics currently living in both fitting/result.py and
odmr/analysis.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdmpy.fitting.backends import FitBackendOptions
from qdmpy.fitting.models import ModelRegistry
from qdmpy.testing import (
    FakeFitBackend,
    RecordingFitBackend,
    _dipole_field,
    _make_params,
    make_synthetic_fit_result,
    make_synthetic_odmr_data,
    make_synthetic_qdm_result,
)

pytestmark = [pytest.mark.unit, pytest.mark.fitting]

MODEL_NAMES = ["ESR14N", "ESR15N", "ESRSINGLE"]

# The applied bias in make_synthetic_* -- 900 uT, which is what b111_induced
# must recover since the remanent dipole cancels between the two polarities.
BIAS_UT = 900.0

SHAPE = (8, 8)

# Parameter keys each model's synthetic FitResult must expose.
EXPECTED_PARAM_KEYS = {
    "ESR14N": {
        "center",
        "width",
        "contrast_0",
        "contrast_1",
        "contrast_2",
        "offset",
        "chi2",
        "states",
    },
    "ESR15N": {"center", "width", "contrast_0", "contrast_1", "offset", "chi2", "states"},
    "ESRSINGLE": {"center", "width", "contrast", "offset", "chi2", "states"},
}


def _fit_kwargs(model_name: str, n_fits: int, n_params: int) -> dict:
    """Minimal keyword set for a backend fit() call."""
    model = ModelRegistry.get(model_name)
    return {
        "initial_parameters": np.ones((n_fits, n_params), dtype=np.float64),
        "constraints": np.zeros((n_params, 2)),
        "constraint_types": np.zeros(n_params, dtype=np.int32),
        "model": model,
        "options": FitBackendOptions(),
    }


class TestFakeFitBackend:
    """FakeFitBackend echoes its inputs as a converged fit."""

    def test_is_available_and_supports_any_model(self) -> None:
        backend = FakeFitBackend()
        assert backend.is_available() is True
        assert backend.supports(ModelRegistry.get("ESR14N")) is True

    def test_fit_echoes_initial_parameters(self) -> None:
        """Non-square data, to catch a backend that assumes a square grid."""
        n_fits, n_freq, n_params = 6, 10, 4
        data = np.zeros((n_fits, n_freq))
        kwargs = _fit_kwargs("ESRSINGLE", n_fits, n_params)
        kwargs["initial_parameters"] = np.arange(n_fits * n_params, dtype=np.float64).reshape(
            n_fits, n_params
        )

        out = FakeFitBackend().fit(data, np.linspace(2.86, 2.88, n_freq), **kwargs)

        assert out.parameters.shape == (n_fits, n_params)
        np.testing.assert_allclose(out.parameters, kwargs["initial_parameters"], rtol=1e-6)
        assert np.all(out.states == 0), "every fit must report converged"
        assert np.all(out.chi2 == 0)
        assert out.iterations.shape == (n_fits,)

    def test_fit_flattens_multidimensional_data(self) -> None:
        """Data arriving as (n_pol, n_frange, n_pixel, n_freq) is flattened."""
        n_freq, n_params = 10, 4
        data = np.zeros((2, 2, 4, n_freq))
        n_fits = 2 * 2 * 4

        out = FakeFitBackend().fit(
            data, np.linspace(2.86, 2.88, n_freq), **_fit_kwargs("ESRSINGLE", n_fits, n_params)
        )

        assert out.parameters.shape == (n_fits, n_params)


class TestRecordingFitBackend:
    """RecordingFitBackend captures what the manager sent it."""

    def test_records_one_entry_per_call_and_delegates(self) -> None:
        n_fits, n_freq, n_params = 4, 10, 4
        backend = RecordingFitBackend()
        assert backend.freq_calls == []

        freqs = np.linspace(2.86, 2.88, n_freq)
        kwargs = _fit_kwargs("ESRSINGLE", n_fits, n_params)
        out = backend.fit(np.zeros((n_fits, n_freq)), freqs, **kwargs)

        assert len(backend.freq_calls) == 1
        assert len(backend.constraints_calls) == 1
        assert len(backend.constraint_types_calls) == 1
        np.testing.assert_allclose(backend.freq_calls[0], freqs)
        np.testing.assert_allclose(backend.constraints_calls[0], kwargs["constraints"])
        # still behaves as a FakeFitBackend
        assert out.parameters.shape == (n_fits, n_params)

        backend.fit(np.zeros((n_fits, n_freq)), freqs, **kwargs)
        assert len(backend.freq_calls) == 2


class TestDipoleField:
    """_dipole_field is the ground-truth pattern the B111 helpers embed."""

    def test_shape_and_antisymmetry(self) -> None:
        """Antisymmetric in both axes -- it is proportional to x*y."""
        field = _dipole_field(SHAPE, amplitude=50.0)

        assert field.shape == SHAPE
        np.testing.assert_allclose(field, -field[::-1, :], atol=1e-12)
        np.testing.assert_allclose(field, -field[:, ::-1], atol=1e-12)

    def test_amplitude_scales_linearly(self) -> None:
        single = _dipole_field(SHAPE, amplitude=1.0)
        double = _dipole_field(SHAPE, amplitude=2.0)

        np.testing.assert_allclose(double, 2.0 * single, rtol=1e-12)


class TestMakeParams:
    """_make_params must match each model's declared parameter count."""

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_column_count_matches_model(self, model_name: str) -> None:
        n_pixels = 5
        center = np.full(n_pixels, 2.87)

        params = _make_params(model_name, center, n_pixels)

        assert params.shape == (n_pixels, ModelRegistry.get(model_name).n_parameters)


class TestMakeSyntheticOdmrData:
    """make_synthetic_odmr_data builds a canonically-labelled ODMRData."""

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_structure_and_canonical_coords(self, model_name: str) -> None:
        n_freq = 20
        data = make_synthetic_odmr_data(shape=SHAPE, n_freq=n_freq, model_name=model_name)

        da = data.data
        assert da.dims == ("polarity", "freq_range", "y", "x", "freq_idx")
        assert da.shape == (2, 2, *SHAPE, n_freq)
        assert list(da.polarity.values) == ["neg", "pos"]
        assert list(da.freq_range.values) == ["low", "high"]

    def test_frequency_axes_are_ghz_and_ordered(self) -> None:
        """Low branch sits below high, both in GHz, each monotonic."""
        da = make_synthetic_odmr_data(shape=SHAPE, n_freq=20).data
        freq_ghz = da.coords["freq_ghz"].values

        assert freq_ghz.shape == (2, 20)
        assert freq_ghz.min() > 2.5
        assert freq_ghz.max() < 3.2
        assert np.all(np.diff(freq_ghz, axis=1) > 0)
        assert freq_ghz[0].max() < freq_ghz[1].min(), "low and high branches must not overlap"

    def test_noise_free_data_is_reproducible(self) -> None:
        """Same seed, same array -- compared in-process, never to a literal."""
        first = make_synthetic_odmr_data(shape=SHAPE, n_freq=12, noise=0.0, seed=7)
        second = make_synthetic_odmr_data(shape=SHAPE, n_freq=12, noise=0.0, seed=7)

        np.testing.assert_array_equal(first.data.values, second.data.values)

    def test_noise_changes_the_data(self) -> None:
        clean = make_synthetic_odmr_data(shape=SHAPE, n_freq=12, noise=0.0, seed=7)
        noisy = make_synthetic_odmr_data(shape=SHAPE, n_freq=12, noise=0.01, seed=7)

        assert not np.allclose(clean.data.values, noisy.data.values)

    def test_pixel_spacing_round_trips_to_metadata(self) -> None:
        data = make_synthetic_odmr_data(shape=SHAPE, n_freq=12, pixel_spacing=2.5e-6)

        assert data.metadata["pixel_spacing"] == 2.5e-6


class TestMakeSyntheticFitResult:
    """make_synthetic_fit_result is public API with two notebooks depending on it."""

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_parameter_keys_and_shapes(self, model_name: str) -> None:
        """Every parameter array is (n_pol, n_frange, n_pixels).

        Note the leading (2, 2): make_synthetic_fit_result hardcodes two
        polarities and two frequency ranges. That is a second n_pol=2
        assumption, in public API, alongside the one QEP-FIT-001 targets in
        FitResult._normalize_resonance_shape. Parametrising it is QEP-FIT-001
        work, not this QEP's, but it is pinned here so the change is visible.
        """
        result = make_synthetic_fit_result(shape=SHAPE, model_name=model_name)

        assert set(result.parameters) == EXPECTED_PARAM_KEYS[model_name]
        n_pixels = SHAPE[0] * SHAPE[1]
        for name, array in result.parameters.items():
            assert array.shape == (2, 2, n_pixels), f"{name} has unexpected shape"

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_metadata_and_dimensions(self, model_name: str) -> None:
        result = make_synthetic_fit_result(shape=SHAPE, model_name=model_name)

        assert result.scan_dimensions == SHAPE
        assert result.model_name == model_name
        assert result.metadata["synthetic"] is True
        assert result.metadata["quality_metrics"]["convergence_rate"] == 1.0

    def test_b111_remanent_recovers_the_embedded_dipole(self) -> None:
        """The remanent map must be the dipole field that was encoded, in uT.

        This is the tightest available statement of the B111 sign convention:
        remanent = (neg + pos) / 2 cancels the applied bias and leaves the
        per-pixel field. QEP-FIT-004 moves this formula out of
        fitting/result.py; if the sign or the factor of two shifts, this fails.
        """
        result = make_synthetic_fit_result(shape=SHAPE)

        expected = _dipole_field(SHAPE, amplitude=50.0)
        np.testing.assert_allclose(result.b111_remanent, expected, atol=1e-6)

    def test_b111_induced_recovers_the_applied_bias(self) -> None:
        """Induced = (neg - pos) / 2 leaves the bias, constant across pixels."""
        result = make_synthetic_fit_result(shape=SHAPE)

        induced = result.b111_induced
        assert induced.shape == SHAPE
        np.testing.assert_allclose(induced, -BIAS_UT, atol=1e-6)

    def test_same_seed_gives_identical_result(self) -> None:
        first = make_synthetic_fit_result(shape=SHAPE, seed=11)
        second = make_synthetic_fit_result(shape=SHAPE, seed=11)

        for name, array in first.parameters.items():
            np.testing.assert_array_equal(array, second.parameters[name])


class TestMakeSyntheticQdmResult:
    """make_synthetic_qdm_result wraps the FitResult for the tutorial path."""

    def test_b111_matches_the_wrapped_fit_result(self) -> None:
        qdm = make_synthetic_qdm_result(shape=SHAPE)
        fit = make_synthetic_fit_result(shape=SHAPE)

        assert qdm.b111_remanent.shape == SHAPE
        np.testing.assert_allclose(qdm.b111_remanent, fit.b111_remanent)

    def test_magnetic_map_is_computable(self) -> None:
        """The Fourier reconstruction the quickstart advertises must run."""
        qdm = make_synthetic_qdm_result(shape=SHAPE)

        magnetic_map = qdm.magnetic_map

        assert magnetic_map.bz.values.shape == SHAPE
        assert np.all(np.isfinite(magnetic_map.bz.values))
