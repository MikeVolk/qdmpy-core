"""Tests for BaseFieldProcessor and FieldProcessingPipeline.

Tests the abstract base class and pipeline orchestrator defined in QEP-034 Phase 1.
Follows TDD RED phase — all tests should fail until the module is implemented.

Import path under test:
    from qdmpy_core.field_processing import BaseFieldProcessor, FieldProcessingPipeline
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_field_map(
    height: int = 10,
    width: int = 10,
    pixel_spacing: float = 1e-6,
    fill_value: float | None = None,
    rng: np.random.Generator | None = None,
) -> xr.DataArray:
    """Return a synthetic (H, W) field map DataArray in µT with pixel_spacing in attrs."""
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
        dims=("y", "x"),
        coords={"y": y_coords, "x": x_coords},
        attrs={"pixel_spacing": pixel_spacing, "units": "µT"},
    )


@pytest.fixture
def simple_field_map() -> xr.DataArray:
    """Synthetic 10×10 field map with pixel_spacing=1e-6 m."""
    return _make_field_map(height=10, width=10, pixel_spacing=1e-6)


@pytest.fixture
def field_map_no_pixel_spacing() -> xr.DataArray:
    """Field map DataArray missing the pixel_spacing attribute."""
    values = np.ones((5, 5))
    return xr.DataArray(values, dims=("y", "x"))


# ---------------------------------------------------------------------------
# Concrete subclasses for testing abstract base
# ---------------------------------------------------------------------------


def _make_identity_processor_class() -> type:
    """Return a concrete BaseFieldProcessor subclass that returns a copy of input."""
    from qdmpy_core.field_processing import BaseFieldProcessor

    class IdentityProcessor(BaseFieldProcessor):
        """Concrete processor that returns an unchanged copy of the field map."""

        def process(self, field_map: xr.DataArray) -> xr.DataArray:
            return field_map.copy(deep=True)

    return IdentityProcessor


def _make_scaling_processor_class(factor: float = 2.0) -> type:
    """Return a concrete BaseFieldProcessor subclass that scales values by factor."""
    from qdmpy_core.field_processing import BaseFieldProcessor

    class ScalingProcessor(BaseFieldProcessor):
        """Concrete processor that multiplies field values by a fixed factor."""

        scale: float = factor

        def process(self, field_map: xr.DataArray) -> xr.DataArray:
            return xr.DataArray(
                field_map.values * self.scale,
                dims=field_map.dims,
                coords=field_map.coords,
                attrs=field_map.attrs,
            )

    return ScalingProcessor


def _make_additive_processor_class(offset: float = 1.0) -> type:
    """Return a concrete BaseFieldProcessor subclass that adds a constant offset."""
    from pydantic import Field

    from qdmpy_core.field_processing import BaseFieldProcessor

    offset_field = Field(default=offset)

    class AdditiveProcessor(BaseFieldProcessor):
        """Concrete processor that adds a fixed offset to field values."""

        offset: float = offset_field

        def process(self, field_map: xr.DataArray) -> xr.DataArray:
            return xr.DataArray(
                field_map.values + self.offset,
                dims=field_map.dims,
                coords=field_map.coords,
                attrs=field_map.attrs,
            )

    return AdditiveProcessor


# ---------------------------------------------------------------------------
# Tests: BaseFieldProcessor (abstract interface)
# ---------------------------------------------------------------------------


class TestBaseFieldProcessorIsAbstract:
    """BaseFieldProcessor cannot be instantiated directly."""

    def test_direct_instantiation_raises_type_error(self) -> None:
        """Instantiating BaseFieldProcessor without implementing process() raises TypeError."""
        from qdmpy_core.field_processing import BaseFieldProcessor

        with pytest.raises(TypeError):
            BaseFieldProcessor()

    def test_subclass_without_process_raises_type_error(self) -> None:
        """A subclass that does not implement process() cannot be instantiated."""
        from qdmpy_core.field_processing import BaseFieldProcessor

        class IncompleteProcessor(BaseFieldProcessor):
            pass  # no process() override

        with pytest.raises(TypeError):
            IncompleteProcessor()

    def test_concrete_subclass_with_process_can_be_instantiated(self) -> None:
        """A subclass that implements process() can be instantiated."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        assert proc is not None


class TestBaseFieldProcessorFrozen:
    """BaseFieldProcessor is a frozen Pydantic model."""

    def test_processor_fields_are_immutable(self) -> None:
        """Reassigning a field on a frozen processor raises ValidationError or TypeError."""
        from pydantic import ValidationError

        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()

        with pytest.raises((ValidationError, TypeError)):
            proc.model_config = {}  # type: ignore[misc]

    def test_scaling_processor_scale_is_immutable(self) -> None:
        """Scale field on ScalingProcessor cannot be reassigned after construction."""
        from pydantic import ValidationError

        ScalingProcessor = _make_scaling_processor_class(factor=3.0)
        proc = ScalingProcessor(scale=3.0)

        with pytest.raises((ValidationError, TypeError)):
            proc.scale = 99.0  # type: ignore[misc]


class TestBaseFieldProcessorPixelSpacing:
    """_pixel_spacing() static method extracts pixel_spacing from attrs."""

    def test_pixel_spacing_extracted_correctly(self, simple_field_map: xr.DataArray) -> None:
        """_pixel_spacing() returns the float stored in field_map.attrs['pixel_spacing']."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        ps = proc._pixel_spacing(simple_field_map)
        assert ps == pytest.approx(1e-6)

    def test_pixel_spacing_returns_float(self, simple_field_map: xr.DataArray) -> None:
        """_pixel_spacing() always returns a Python float."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        ps = proc._pixel_spacing(simple_field_map)
        assert isinstance(ps, float)

    def test_pixel_spacing_raises_value_error_when_missing(
        self, field_map_no_pixel_spacing: xr.DataArray
    ) -> None:
        """_pixel_spacing() raises ValueError when pixel_spacing not in attrs."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()

        with pytest.raises(ValueError, match="pixel_spacing"):
            proc._pixel_spacing(field_map_no_pixel_spacing)

    def test_pixel_spacing_is_static_method(self) -> None:
        """_pixel_spacing is callable on the class without an instance."""
        from qdmpy_core.field_processing import BaseFieldProcessor

        field_map = _make_field_map(pixel_spacing=5e-7)
        ps = BaseFieldProcessor._pixel_spacing(field_map)
        assert ps == pytest.approx(5e-7)

    def test_pixel_spacing_custom_value(self) -> None:
        """_pixel_spacing() correctly reads various pixel_spacing values."""
        from qdmpy_core.field_processing import BaseFieldProcessor

        for expected_ps in [1e-9, 1e-6, 1e-3, 1.0]:
            field_map = _make_field_map(pixel_spacing=expected_ps)
            ps = BaseFieldProcessor._pixel_spacing(field_map)
            assert ps == pytest.approx(expected_ps)


class TestBaseFieldProcessorSubclassContract:
    """Subclasses of BaseFieldProcessor honour the process() contract."""

    def test_process_returns_new_xr_dataarray(self, simple_field_map: xr.DataArray) -> None:
        """process() returns an xr.DataArray, not another type."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        result = proc.process(simple_field_map)
        assert isinstance(result, xr.DataArray)

    def test_process_does_not_mutate_input(self, simple_field_map: xr.DataArray) -> None:
        """process() leaves the input DataArray values unchanged (no in-place mutation)."""
        ScalingProcessor = _make_scaling_processor_class(factor=99.0)
        proc = ScalingProcessor(scale=99.0)
        original_values = simple_field_map.values.copy()
        proc.process(simple_field_map)
        np.testing.assert_array_equal(simple_field_map.values, original_values)

    def test_process_returns_object_with_different_identity(
        self, simple_field_map: xr.DataArray
    ) -> None:
        """process() returns a new object (not the same reference as input)."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        result = proc.process(simple_field_map)
        assert id(result) != id(simple_field_map)

    def test_scaling_processor_doubles_values(self, simple_field_map: xr.DataArray) -> None:
        """ScalingProcessor(factor=2.0) produces values exactly 2× the input."""
        ScalingProcessor = _make_scaling_processor_class(factor=2.0)
        proc = ScalingProcessor(scale=2.0)
        result = proc.process(simple_field_map)
        np.testing.assert_allclose(result.values, simple_field_map.values * 2.0)

    def test_identity_processor_preserves_shape(self, simple_field_map: xr.DataArray) -> None:
        """IdentityProcessor output has the same shape as input."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        result = proc.process(simple_field_map)
        assert result.shape == simple_field_map.shape

    def test_identity_processor_preserves_dims(self, simple_field_map: xr.DataArray) -> None:
        """IdentityProcessor output has the same dim names as input."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        result = proc.process(simple_field_map)
        assert result.dims == simple_field_map.dims

    def test_identity_processor_preserves_coords(self, simple_field_map: xr.DataArray) -> None:
        """IdentityProcessor output has the same coordinates as input."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        result = proc.process(simple_field_map)
        np.testing.assert_array_equal(
            result.coords["y"].values, simple_field_map.coords["y"].values
        )
        np.testing.assert_array_equal(
            result.coords["x"].values, simple_field_map.coords["x"].values
        )

    def test_identity_processor_preserves_attrs(self, simple_field_map: xr.DataArray) -> None:
        """IdentityProcessor output preserves attrs (including pixel_spacing)."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        result = proc.process(simple_field_map)
        assert result.attrs.get("pixel_spacing") == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# Tests: FieldProcessingPipeline
# ---------------------------------------------------------------------------


class TestFieldProcessingPipelineConstruction:
    """FieldProcessingPipeline can be constructed and processors added."""

    def test_pipeline_instantiates_empty(self) -> None:
        """FieldProcessingPipeline() creates an empty pipeline without error."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        pipeline = FieldProcessingPipeline()
        assert pipeline is not None

    def test_add_returns_self_for_method_chaining(self) -> None:
        """add() returns the pipeline instance to allow fluent chaining."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline()
        returned = pipeline.add(IdentityProcessor())
        assert returned is pipeline

    def test_method_chaining_multiple_add_calls(self) -> None:
        """Multiple add() calls can be chained without raising."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        ScalingProcessor = _make_scaling_processor_class(factor=2.0)

        pipeline = (
            FieldProcessingPipeline()
            .add(IdentityProcessor())
            .add(ScalingProcessor(scale=2.0))
            .add(IdentityProcessor())
        )
        assert pipeline is not None

    def test_add_accepts_base_field_processor_subclass(self) -> None:
        """add() accepts any BaseFieldProcessor subclass without error."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline()
        # No exception expected
        pipeline.add(IdentityProcessor())


class TestFieldProcessingPipelineProcess:
    """FieldProcessingPipeline.process() applies processors in order."""

    def test_empty_pipeline_returns_copy_of_input(self, simple_field_map: xr.DataArray) -> None:
        """An empty pipeline returns a DataArray with same values as input."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        pipeline = FieldProcessingPipeline()
        result = pipeline.process(simple_field_map)
        np.testing.assert_array_equal(result.values, simple_field_map.values)

    def test_empty_pipeline_returns_new_object(self, simple_field_map: xr.DataArray) -> None:
        """An empty pipeline returns a new DataArray (not the same reference)."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        pipeline = FieldProcessingPipeline()
        result = pipeline.process(simple_field_map)
        assert id(result) != id(simple_field_map)

    def test_single_identity_processor_preserves_values(
        self, simple_field_map: xr.DataArray
    ) -> None:
        """Single IdentityProcessor pipeline returns identical values."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())
        result = pipeline.process(simple_field_map)
        np.testing.assert_array_equal(result.values, simple_field_map.values)

    def test_single_scaling_processor_doubles_values(self, simple_field_map: xr.DataArray) -> None:
        """Single ScalingProcessor(scale=2.0) produces 2× the original values."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        ScalingProcessor = _make_scaling_processor_class(factor=2.0)
        pipeline = FieldProcessingPipeline().add(ScalingProcessor(scale=2.0))
        result = pipeline.process(simple_field_map)
        np.testing.assert_allclose(result.values, simple_field_map.values * 2.0)

    def test_two_scaling_processors_quadruple_values(self, simple_field_map: xr.DataArray) -> None:
        """Two scale=2 processors applied sequentially multiply values by 4."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        ScalingProcessor = _make_scaling_processor_class(factor=2.0)
        pipeline = (
            FieldProcessingPipeline()
            .add(ScalingProcessor(scale=2.0))
            .add(ScalingProcessor(scale=2.0))
        )
        result = pipeline.process(simple_field_map)
        np.testing.assert_allclose(result.values, simple_field_map.values * 4.0)

    def test_processors_applied_in_order(self, simple_field_map: xr.DataArray) -> None:
        """Processors execute left-to-right; scale-then-offset != offset-then-scale."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        ScalingProcessor = _make_scaling_processor_class(factor=2.0)
        AdditiveProcessor = _make_additive_processor_class(offset=10.0)

        pipeline_scale_first = (
            FieldProcessingPipeline()
            .add(ScalingProcessor(scale=2.0))
            .add(AdditiveProcessor(offset=10.0))
        )
        pipeline_offset_first = (
            FieldProcessingPipeline()
            .add(AdditiveProcessor(offset=10.0))
            .add(ScalingProcessor(scale=2.0))
        )

        result_sf = pipeline_scale_first.process(simple_field_map)
        result_of = pipeline_offset_first.process(simple_field_map)

        # scale-first: (v * 2) + 10
        # offset-first: (v + 10) * 2
        # These differ when v != 10, so results must not be equal for non-uniform data.
        assert not np.allclose(result_sf.values, result_of.values)

    def test_process_does_not_mutate_input(self, simple_field_map: xr.DataArray) -> None:
        """pipeline.process() does not alter the original input DataArray."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        ScalingProcessor = _make_scaling_processor_class(factor=99.0)
        pipeline = FieldProcessingPipeline().add(ScalingProcessor(scale=99.0))
        original_values = simple_field_map.values.copy()
        pipeline.process(simple_field_map)
        np.testing.assert_array_equal(simple_field_map.values, original_values)

    def test_process_preserves_coords(self, simple_field_map: xr.DataArray) -> None:
        """pipeline.process() output retains the input's coordinates."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())
        result = pipeline.process(simple_field_map)
        np.testing.assert_array_equal(
            result.coords["y"].values, simple_field_map.coords["y"].values
        )
        np.testing.assert_array_equal(
            result.coords["x"].values, simple_field_map.coords["x"].values
        )

    def test_process_preserves_attrs(self, simple_field_map: xr.DataArray) -> None:
        """pipeline.process() output retains pixel_spacing in attrs."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())
        result = pipeline.process(simple_field_map)
        assert result.attrs.get("pixel_spacing") == pytest.approx(1e-6)

    def test_process_returns_xr_dataarray(self, simple_field_map: xr.DataArray) -> None:
        """pipeline.process() always returns an xr.DataArray."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())
        result = pipeline.process(simple_field_map)
        assert isinstance(result, xr.DataArray)


class TestFieldProcessingPipelineLogging:
    """FieldProcessingPipeline logs processor name and shape after each step."""

    def test_pipeline_logs_each_processor_name(self, simple_field_map: xr.DataArray) -> None:
        """A debug log entry naming the processor appears for each pipeline step."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        ScalingProcessor = _make_scaling_processor_class(factor=2.0)

        pipeline = (
            FieldProcessingPipeline().add(IdentityProcessor()).add(ScalingProcessor(scale=2.0))
        )

        log_calls: list[tuple[Any, ...]] = []

        with patch("QDMpy.field_processing.logger") as mock_logger:
            mock_logger.debug = MagicMock(side_effect=lambda *a, **kw: log_calls.append((a, kw)))
            pipeline.process(simple_field_map)

        # At least two debug calls (one per processor)
        assert len(log_calls) >= 2

    def test_pipeline_logs_processor_class_name(self, simple_field_map: xr.DataArray) -> None:
        """The processor class name appears somewhere in the debug log call."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())

        log_calls: list[tuple[Any, ...]] = []

        with patch("QDMpy.field_processing.logger") as mock_logger:
            mock_logger.debug = MagicMock(side_effect=lambda *a, **kw: log_calls.append((a, kw)))
            pipeline.process(simple_field_map)

        # Flatten all args/kwargs into strings and check for the class name
        all_text = " ".join(
            str(item)
            for call in log_calls
            for part in call
            for item in (part.values() if isinstance(part, dict) else [part])
        )
        assert "IdentityProcessor" in all_text

    def test_pipeline_logs_shape_after_each_step(self, simple_field_map: xr.DataArray) -> None:
        """Shape information appears in the debug log after each processor step."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())

        log_calls: list[tuple[Any, ...]] = []

        with patch("QDMpy.field_processing.logger") as mock_logger:
            mock_logger.debug = MagicMock(side_effect=lambda *a, **kw: log_calls.append((a, kw)))
            pipeline.process(simple_field_map)

        # Shape (10, 10) should appear somewhere in the logged data
        all_text = " ".join(
            str(item)
            for call in log_calls
            for part in call
            for item in (part.values() if isinstance(part, dict) else [part])
        )
        expected_shape = str(simple_field_map.shape)
        assert expected_shape in all_text or "10" in all_text


class TestFieldProcessingPipelineImmutability:
    """Pipeline processing never mutates the input DataArray."""

    def test_process_does_not_modify_input_values(self, simple_field_map: xr.DataArray) -> None:
        """Values in the input remain unchanged after pipeline.process()."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        ScalingProcessor = _make_scaling_processor_class(factor=3.0)
        pipeline = FieldProcessingPipeline().add(ScalingProcessor(scale=3.0))

        snapshot = simple_field_map.values.copy()
        pipeline.process(simple_field_map)
        np.testing.assert_array_equal(simple_field_map.values, snapshot)

    def test_process_does_not_modify_input_attrs(self, simple_field_map: xr.DataArray) -> None:
        """Attributes dict of the input DataArray is not modified by pipeline.process()."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())

        original_attrs = dict(simple_field_map.attrs)
        pipeline.process(simple_field_map)
        assert simple_field_map.attrs == original_attrs

    def test_result_is_independent_of_input_after_processing(
        self, simple_field_map: xr.DataArray
    ) -> None:
        """Modifying the result after process() does not change the original input."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        IdentityProcessor = _make_identity_processor_class()
        pipeline = FieldProcessingPipeline().add(IdentityProcessor())

        result = pipeline.process(simple_field_map)
        original_snapshot = simple_field_map.values.copy()

        # Mutate the result's underlying array
        result.values[:] = 999.0

        # Original must be unaffected
        np.testing.assert_array_equal(simple_field_map.values, original_snapshot)


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestFieldProcessingPropertyBased:
    """Hypothesis tests for universal invariants on processors and pipelines."""

    @given(
        height=st.integers(min_value=2, max_value=20),
        width=st.integers(min_value=2, max_value=20),
        pixel_spacing=st.floats(min_value=1e-9, max_value=1e-3, allow_nan=False),
    )
    @hyp_settings(max_examples=30)
    def test_identity_processor_output_shape_equals_input_shape(
        self, height: int, width: int, pixel_spacing: float
    ) -> None:
        """For any valid input shape, IdentityProcessor output shape == input shape."""
        IdentityProcessor = _make_identity_processor_class()
        proc = IdentityProcessor()
        field_map = _make_field_map(height=height, width=width, pixel_spacing=pixel_spacing)
        result = proc.process(field_map)
        assert result.shape == field_map.shape

    @given(
        height=st.integers(min_value=2, max_value=20),
        width=st.integers(min_value=2, max_value=20),
        scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=30)
    def test_scaling_processor_output_shape_equals_input_shape(
        self, height: int, width: int, scale: float
    ) -> None:
        """For any valid input, ScalingProcessor output has the same shape as input."""
        ScalingProcessor = _make_scaling_processor_class(factor=scale)
        proc = ScalingProcessor(scale=scale)
        field_map = _make_field_map(height=height, width=width)
        result = proc.process(field_map)
        assert result.shape == field_map.shape

    @given(
        n_steps=st.integers(min_value=0, max_value=5),
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=20)
    def test_pipeline_output_shape_invariant_under_any_n_processors(
        self, n_steps: int, scale: float
    ) -> None:
        """Regardless of how many scaling processors are chained, shape is preserved."""
        from qdmpy_core.field_processing import FieldProcessingPipeline

        ScalingProcessor = _make_scaling_processor_class(factor=scale)
        pipeline = FieldProcessingPipeline()
        for _ in range(n_steps):
            pipeline.add(ScalingProcessor(scale=scale))

        field_map = _make_field_map(height=8, width=8)
        result = pipeline.process(field_map)
        assert result.shape == field_map.shape

    @given(
        fill=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=30)
    def test_scaling_processor_values_match_formula(self, fill: float, scale: float) -> None:
        """ScalingProcessor(scale=s).process(v) == v * s for any constant fill map."""
        ScalingProcessor = _make_scaling_processor_class(factor=scale)
        proc = ScalingProcessor(scale=scale)
        field_map = _make_field_map(fill_value=fill)
        result = proc.process(field_map)
        np.testing.assert_allclose(result.values, fill * scale, rtol=1e-6, atol=1e-12)
