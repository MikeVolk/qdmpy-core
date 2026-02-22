"""Tests for QEP-045 developer extension points.

Covers:
- Custom Model subclass registered via @ModelRegistry.register
- ModelRegistry.available_models() listing
- Custom Processor satisfying the Processor protocol
- Custom FieldReconstructor wired through MagneticMap.from_b111() and QDMResult
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from qdmpy_core import FieldReconstructor, Model, ModelRegistry, Processor
from qdmpy_core.fitting.result import FitResult
from qdmpy_core.magnetic_map import MagneticMap
from qdmpy_core.odmr.data import ODMRData
from qdmpy_core.result import QDMResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

H, W = 8, 8
N_POL, N_FRANGE = 2, 2


@pytest.fixture
def fit_result() -> FitResult:
    rng = np.random.default_rng(42)
    n_pixels = H * W
    # center needs shape (n_pol, n_frange, n_pixels) for b111_remanent to work
    return FitResult(
        parameters={
            'center': rng.uniform(2.82, 2.92, (N_POL, N_FRANGE, n_pixels)),
            'chi2': rng.random(n_pixels),
        },
        scan_dimensions=(H, W),
        pixel_spacing=4e-6,
        model_name='ESRSINGLE',
    )


@pytest.fixture
def b111_da() -> xr.DataArray:
    rng = np.random.default_rng(7)
    arr = rng.random((H, W)).astype(float)
    return xr.DataArray(arr, dims=('y', 'x'), attrs={'pixel_spacing': 4e-6})


# ---------------------------------------------------------------------------
# Custom Model
# ---------------------------------------------------------------------------


@ModelRegistry.register
class _TestSingleLorentz(Model):
    """Minimal single-Lorentzian model for testing extension point."""

    name: ClassVar[str] = '_TESTLORENTZ'

    def __init__(self) -> None:
        super().__init__(
            '_TESTLORENTZ',
            n_peaks=1,
            parameter_names=['center', 'width', 'contrast', 'offset'],
        )
        self.model_id = -1

    @property
    def parameter_types(self) -> dict[str, str]:
        return {'center': 'center', 'width': 'width', 'contrast': 'contrast', 'offset': 'offset'}

    @property
    def frequency_parameters(self) -> list[str]:
        return ['center']

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        parameters = np.atleast_2d(parameters)
        center = parameters[:, 0:1]
        width_sq = parameters[:, 1:2] ** 2
        contrast = parameters[:, 2:3]
        offset = parameters[:, 3:4]
        dip = contrast * width_sq / ((x - center) ** 2 + width_sq)
        return 1 + offset - dip


class TestCustomModel:
    def test_registered(self) -> None:
        assert '_TESTLORENTZ' in ModelRegistry.available_models()

    def test_get_returns_instance(self) -> None:
        model = ModelRegistry.get('_TESTLORENTZ')
        assert isinstance(model, _TestSingleLorentz)

    def test_n_parameters(self) -> None:
        model = ModelRegistry.get('_TESTLORENTZ')
        assert model.n_parameters == 4

    def test_func_output_shape(self) -> None:
        model = ModelRegistry.get('_TESTLORENTZ')
        x = np.linspace(2.82, 2.92, 50)
        params = np.array([[2.87, 0.002, 0.1, 0.0]])  # shape (1, 4)
        out = model.func(x, params)
        assert out.shape == (1, 50)

    def test_func_scalar_params_broadcast(self) -> None:
        model = ModelRegistry.get('_TESTLORENTZ')
        x = np.linspace(2.82, 2.92, 50)
        params = np.array([2.87, 0.002, 0.1, 0.0])  # 1D — atleast_2d inside func
        out = model.func(x, params)
        assert out.shape == (1, 50)

    def test_available_models_includes_builtins(self) -> None:
        names = ModelRegistry.available_models()
        assert 'ESR14N' in names
        assert 'ESR15N' in names
        assert 'ESRSINGLE' in names

    def test_available_models_is_sorted(self) -> None:
        names = ModelRegistry.available_models()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Processor Protocol
# ---------------------------------------------------------------------------


class _ScalingProcessor:
    """Minimal custom processor: multiplies all values by a constant factor."""

    def __init__(self, scale: float = 1.05) -> None:
        self.scale = scale

    def process(self, data: ODMRData) -> ODMRData:
        return ODMRData(data=data.data * self.scale, metadata=data.metadata.copy())

    def describe(self) -> str:
        return f'_ScalingProcessor(scale={self.scale})'


class TestProcessorProtocol:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(_ScalingProcessor(), Processor)

    def test_process_returns_new_odmrdata(self) -> None:
        rng = np.random.default_rng(0)
        n_freq = 10
        arr = rng.random((2, 2, 4, 4, n_freq))
        freq_ghz = np.tile(np.linspace(2.82, 2.92, n_freq), (2, 1))
        da = xr.DataArray(
            arr,
            dims=('polarity', 'freq_range', 'y', 'x', 'freq_idx'),
            coords={
                'polarity': ['neg', 'pos'],
                'freq_range': ['low', 'high'],
                'freq_ghz': (('freq_range', 'freq_idx'), freq_ghz),
            },
        )
        data = ODMRData(data=da)
        proc = _ScalingProcessor(scale=2.0)
        result = proc.process(data)
        assert result is not data
        np.testing.assert_allclose(result.data.values, data.data.values * 2.0)

    def test_describe_returns_string(self) -> None:
        assert isinstance(_ScalingProcessor().describe(), str)

    def test_baseprocessor_satisfies_protocol(self) -> None:
        from qdmpy_core.odmr.processors import NormalizationProcessor
        assert isinstance(NormalizationProcessor(), Processor)


# ---------------------------------------------------------------------------
# FieldReconstructor Protocol
# ---------------------------------------------------------------------------


class _IdentityReconstructor:
    """Trivial reconstructor: returns b111 as bz and zeros for bx/by."""

    def reconstruct(
        self,
        b111: xr.DataArray,
        nv_axis: tuple[float, float, float],
    ) -> xr.Dataset:
        zeros = xr.zeros_like(b111)
        return xr.Dataset({
            'bx': zeros,
            'by': zeros,
            'bz': b111,
            'btotal': abs(b111),
        })


class TestFieldReconstructorProtocol:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(_IdentityReconstructor(), FieldReconstructor)

    def test_from_b111_uses_custom_reconstructor(self, b111_da: xr.DataArray) -> None:
        rec = _IdentityReconstructor()
        mm = MagneticMap.from_b111(b111_da, reconstructor=rec)
        np.testing.assert_array_equal(mm.bz.values, b111_da.values)
        np.testing.assert_array_equal(mm.bx.values, np.zeros((H, W)))

    def test_from_b111_default_reconstructor_still_works(self, b111_da: xr.DataArray) -> None:
        mm = MagneticMap.from_b111(b111_da)
        assert mm.bz.shape == (H, W)
        assert mm.btotal.shape == (H, W)

    def test_qdm_result_uses_custom_reconstructor(self, fit_result: FitResult) -> None:
        rec = _IdentityReconstructor()
        qdm = QDMResult(fit_result=fit_result, reconstructor=rec)
        mm = qdm.magnetic_map
        # bx / by should be all zeros from identity reconstructor
        np.testing.assert_array_equal(mm.bx.values, np.zeros((H, W)))

    def test_qdm_result_default_reconstructor(self, fit_result: FitResult) -> None:
        qdm = QDMResult(fit_result=fit_result)
        mm = qdm.magnetic_map
        assert mm.bz.shape == (H, W)
