"""Tests for NvSettings in QDMpy settings module.

Tests the NV axis geometry sub-model that is part of QDMpySettings.
Follows TDD RED phase — all tests should fail until NvSettings is implemented.

Import path under test:
    from qdmpy.settings import NvSettings, QDMpySettings, get_settings, reset_settings
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from pydantic import ValidationError

from qdmpy.settings import QDMpySettings

# ---------------------------------------------------------------------------
# Expected constants (from QEP-034 proposal)
# ---------------------------------------------------------------------------
# QDM2 standard [111] orientation with [1-10] along x:
#   û_default = (0, √(2/3), 1/√3) ≈ (0.0, 0.8164966, 0.5773503)
_EXPECTED_AXIS_X = 0.0
_EXPECTED_AXIS_Y = math.sqrt(2.0 / 3.0)  # ≈ 0.8164966
_EXPECTED_AXIS_Z = 1.0 / math.sqrt(3.0)  # ≈ 0.5773503
_EXPECTED_EPSILON = 1e-30

# Tolerance for floating-point comparisons
_ATOL = 1e-6


class TestNvSettingsDefaults:
    """NvSettings default values match QEP-034 specification."""

    def test_default_axis_is_tuple_of_three_floats(self) -> None:
        """Default NV axis is a 3-tuple of floats."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        assert len(nv.axis) == 3
        assert all(isinstance(v, float) for v in nv.axis)

    def test_default_axis_x_component_is_zero(self) -> None:
        """Default NV axis x-component is 0.0."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        assert abs(nv.axis[0] - _EXPECTED_AXIS_X) < _ATOL

    def test_default_axis_y_component_matches_sqrt_two_thirds(self) -> None:
        """Default NV axis y-component equals sqrt(2/3) ≈ 0.8164966."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        assert abs(nv.axis[1] - _EXPECTED_AXIS_Y) < _ATOL

    def test_default_axis_z_component_matches_one_over_sqrt_three(self) -> None:
        """Default NV axis z-component equals 1/sqrt(3) ≈ 0.5773503."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        assert abs(nv.axis[2] - _EXPECTED_AXIS_Z) < _ATOL

    def test_default_axis_is_unit_vector(self) -> None:
        """Default NV axis vector has magnitude 1.0."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        magnitude = math.sqrt(sum(v**2 for v in nv.axis))
        assert abs(magnitude - 1.0) < _ATOL

    def test_default_epsilon_is_1e_minus_30(self) -> None:
        """Default epsilon is 1e-30 (k=0 regularisation term)."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        assert nv.epsilon == _EXPECTED_EPSILON


class TestNvSettingsCustomValues:
    """NvSettings accepts valid custom values."""

    def test_custom_axis_tuple(self) -> None:
        """NvSettings accepts a custom 3-tuple for axis."""
        from qdmpy.settings import NvSettings

        custom = (0.0, 0.0, 1.0)
        nv = NvSettings(axis=custom)
        assert nv.axis == custom

    def test_custom_axis_stores_each_component_correctly(self) -> None:
        """Custom axis stores all three components without truncation."""
        from qdmpy.settings import NvSettings

        custom = (0.1, 0.2, 0.9747958)
        nv = NvSettings(axis=custom)
        assert abs(nv.axis[0] - 0.1) < _ATOL
        assert abs(nv.axis[1] - 0.2) < _ATOL
        assert abs(nv.axis[2] - 0.9747958) < _ATOL

    def test_custom_epsilon(self) -> None:
        """NvSettings accepts a custom epsilon value."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(epsilon=1e-10)
        assert nv.epsilon == pytest.approx(1e-10)

    def test_custom_axis_and_epsilon_together(self) -> None:
        """NvSettings accepts both custom axis and epsilon."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(axis=(0.0, 0.0, 1.0), epsilon=1e-15)
        assert nv.axis == (0.0, 0.0, 1.0)
        assert nv.epsilon == pytest.approx(1e-15)

    def test_axis_as_list_is_accepted(self) -> None:
        """A list of three floats is coerced to tuple for axis."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(axis=[0.0, 0.0, 1.0])
        assert len(nv.axis) == 3


class TestNvSettingsImmutability:
    """NvSettings is a frozen Pydantic model."""

    def test_axis_cannot_be_reassigned(self) -> None:
        """Assigning to axis on a frozen model raises ValidationError or TypeError."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        with pytest.raises((ValidationError, TypeError)):
            nv.axis = (0.0, 0.0, 1.0)

    def test_epsilon_cannot_be_reassigned(self) -> None:
        """Assigning to epsilon on a frozen model raises ValidationError or TypeError."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        with pytest.raises((ValidationError, TypeError)):
            nv.epsilon = 1e-5

    def test_extra_fields_are_ignored(self) -> None:
        """Extra keyword arguments do not raise; they are silently dropped."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(unknown_field="should_be_ignored")
        assert not hasattr(nv, "unknown_field")


class TestNvSettingsTypeValidation:
    """NvSettings validates axis type at construction."""

    def test_axis_wrong_length_raises_validation_error(self) -> None:
        """A tuple with != 3 elements raises a ValidationError."""
        from qdmpy.settings import NvSettings

        with pytest.raises(ValidationError):
            NvSettings(axis=(0.0, 1.0))  # only 2 elements

    def test_axis_with_four_elements_raises_validation_error(self) -> None:
        """A 4-element tuple for axis raises a ValidationError."""
        from qdmpy.settings import NvSettings

        with pytest.raises(ValidationError):
            NvSettings(axis=(0.0, 0.0, 1.0, 0.0))

    def test_axis_with_non_numeric_raises_validation_error(self) -> None:
        """String elements in axis tuple raise a ValidationError."""
        from qdmpy.settings import NvSettings

        with pytest.raises(ValidationError):
            NvSettings(axis=("a", "b", "c"))

    def test_epsilon_zero_is_accepted(self) -> None:
        """Epsilon of 0.0 is technically valid (no Pydantic constraint)."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(epsilon=0.0)
        assert nv.epsilon == 0.0


class TestNvSettingsQDMpySettingsIntegration:
    """NvSettings is accessible via QDMpySettings.nv."""

    def test_qdmpy_settings_has_nv_field(self) -> None:
        """QDMpySettings exposes an 'nv' attribute."""
        settings = QDMpySettings()
        assert hasattr(settings, "nv")

    def test_qdmpy_settings_nv_is_nv_settings_instance(self) -> None:
        """QDMpySettings().nv is an NvSettings instance."""
        from qdmpy.settings import NvSettings, QDMpySettings

        settings = QDMpySettings()
        assert isinstance(settings.nv, NvSettings)

    def test_qdmpy_settings_nv_defaults_are_correct(self) -> None:
        """QDMpySettings().nv uses the correct default axis and epsilon."""
        settings = QDMpySettings()
        assert abs(settings.nv.axis[1] - _EXPECTED_AXIS_Y) < _ATOL
        assert settings.nv.epsilon == _EXPECTED_EPSILON

    def test_get_settings_returns_nv_settings(self) -> None:
        """get_settings().nv returns a valid NvSettings instance."""
        from qdmpy.settings import NvSettings, get_settings, reset_settings

        reset_settings()
        try:
            settings = get_settings()
            assert isinstance(settings.nv, NvSettings)
        finally:
            reset_settings()

    def test_qdmpy_settings_nv_custom_axis_via_constructor(self) -> None:
        """QDMpySettings accepts a custom NvSettings via constructor."""
        from qdmpy.settings import NvSettings, QDMpySettings

        custom_nv = NvSettings(axis=(0.0, 0.0, 1.0))
        settings = QDMpySettings(nv=custom_nv)
        assert settings.nv.axis == (0.0, 0.0, 1.0)


class TestNvSettingsTomlSerialization:
    """NvSettings can be loaded from TOML configuration."""

    def test_nv_settings_can_be_serialized_to_dict(self) -> None:
        """NvSettings.model_dump() produces a plain dict."""
        from qdmpy.settings import NvSettings

        nv = NvSettings()
        data = nv.model_dump()
        assert "axis" in data
        assert "epsilon" in data
        assert len(data["axis"]) == 3

    def test_qdmpy_settings_env_var_nv_axis_override(self) -> None:
        """QDMPY_NV__AXIS env var overrides the NV axis."""
        with patch.dict("os.environ", {"QDMPY_NV__AXIS": "[0.0, 0.0, 1.0]"}):
            settings = QDMpySettings()
            # Env var override: axis should reflect [0, 0, 1]
            assert abs(settings.nv.axis[2] - 1.0) < _ATOL

    def test_qdmpy_settings_env_var_nv_epsilon_override(self) -> None:
        """QDMPY_NV__EPSILON env var overrides epsilon."""
        with patch.dict("os.environ", {"QDMPY_NV__EPSILON": "1e-20"}):
            settings = QDMpySettings()
            assert settings.nv.epsilon == pytest.approx(1e-20)

    def test_nv_settings_round_trip_via_model_dump_and_construct(self) -> None:
        """NvSettings survives a dict round-trip (model_dump → constructor)."""
        from qdmpy.settings import NvSettings

        original = NvSettings(axis=(0.1, 0.8, 0.6), epsilon=1e-25)
        data = original.model_dump()
        reconstructed = NvSettings(**data)
        assert abs(reconstructed.axis[0] - 0.1) < _ATOL
        assert abs(reconstructed.axis[1] - 0.8) < _ATOL
        assert abs(reconstructed.axis[2] - 0.6) < _ATOL
        assert reconstructed.epsilon == pytest.approx(1e-25)


class TestNvSettingsPropertyBased:
    """Hypothesis property-based tests for NvSettings."""

    @given(
        x=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        z=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @hyp_settings(max_examples=50)
    def test_any_valid_three_float_tuple_accepted_as_axis(
        self, x: float, y: float, z: float
    ) -> None:
        """NvSettings accepts any (float, float, float) without raising."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(axis=(x, y, z))
        assert len(nv.axis) == 3

    @given(
        eps=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @hyp_settings(max_examples=50)
    def test_any_non_negative_epsilon_accepted(self, eps: float) -> None:
        """NvSettings accepts any non-negative finite epsilon."""
        from qdmpy.settings import NvSettings

        nv = NvSettings(epsilon=eps)
        assert nv.epsilon == pytest.approx(eps, abs=1e-100)
