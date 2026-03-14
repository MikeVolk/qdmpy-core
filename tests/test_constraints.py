"""Tests for qdmpy.fitting.constraints — Constraint dataclass and ConstraintManager."""

from __future__ import annotations

import numpy as np
import pytest

from qdmpy.exceptions import ParameterError
from qdmpy.fitting.constraints import (
    CONSTRAINT_TYPES,
    Constraint,
    ConstraintManager,
    _mt_to_absolute_ghz,
)
from qdmpy.fitting.models import ModelRegistry
from qdmpy.settings import ModelConstraintsSettings

# ---------------------------------------------------------------------------
# Constraint dataclass
# ---------------------------------------------------------------------------


class TestConstraint:
    """Tests for the frozen Constraint dataclass."""

    def test_basic_construction(self) -> None:
        c = Constraint(vmin=2.8, vmax=2.9, constraint_type="FREE", unit="GHz")
        assert c.vmin == 2.8
        assert c.vmax == 2.9
        assert c.constraint_type == "FREE"
        assert c.unit == "GHz"

    def test_type_index(self) -> None:
        for idx, ctype in enumerate(CONSTRAINT_TYPES):
            c = Constraint(vmin=0.0, vmax=1.0, constraint_type=ctype, unit="a.u.")
            assert c.type_index == idx

    def test_invalid_constraint_type_raises(self) -> None:
        with pytest.raises(ParameterError, match="Invalid constraint type"):
            Constraint(vmin=0.0, vmax=1.0, constraint_type="BOGUS", unit="a.u.")

    def test_frozen_immutability(self) -> None:
        c = Constraint(vmin=2.8, vmax=2.9, constraint_type="FREE", unit="GHz")
        with pytest.raises(AttributeError):
            c.vmin = 3.0  # type: ignore[misc]

    def test_with_updates_selective(self) -> None:
        c = Constraint(vmin=2.8, vmax=2.9, constraint_type="FREE", unit="GHz")
        c2 = c.with_updates(vmin=2.7)
        assert c2.vmin == 2.7
        assert c2.vmax == 2.9
        assert c2.constraint_type == "FREE"
        assert c2.unit == "GHz"

    def test_with_updates_all(self) -> None:
        c = Constraint(vmin=2.8, vmax=2.9, constraint_type="FREE", unit="GHz")
        c2 = c.with_updates(vmin=2.0, vmax=3.0, constraint_type="LOWER_UPPER")
        assert c2.vmin == 2.0
        assert c2.vmax == 3.0
        assert c2.constraint_type == "LOWER_UPPER"

    def test_with_updates_invalid_type_raises(self) -> None:
        c = Constraint(vmin=2.8, vmax=2.9, constraint_type="FREE", unit="GHz")
        with pytest.raises(ParameterError):
            c.with_updates(constraint_type="NOPE")


# ---------------------------------------------------------------------------
# ConstraintManager
# ---------------------------------------------------------------------------


class TestConstraintManager:
    """Tests for ConstraintManager initialisation, mutation, and array generation."""

    @pytest.fixture
    def single_model(self):
        return ModelRegistry.get("ESRSINGLE")

    @pytest.fixture
    def default_settings(self):
        return ModelConstraintsSettings(constraint_units="absolute_ghz")

    def test_initialises_all_params(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        constraints = cm.get_constraints()
        assert set(constraints.keys()) == set(single_model.parameter_names)
        for c in constraints.values():
            assert isinstance(c, Constraint)

    def test_set_constraint_updates_vmin(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        cm.set_constraint("center", vmin=2.85)
        assert cm.get_constraints()["center"].vmin == 2.85

    def test_set_constraint_updates_type(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        cm.set_constraint("center", constraint_type="LOWER")
        assert cm.get_constraints()["center"].constraint_type == "LOWER"

    def test_set_constraint_unknown_param_raises(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        with pytest.raises(ParameterError, match="Unknown parameter"):
            cm.set_constraint("nonexistent", vmin=1.0)

    def test_set_constraint_invalid_type_raises(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        with pytest.raises(ParameterError, match="Invalid constraint type"):
            cm.set_constraint("center", constraint_type="BAD")

    def test_to_array_shape(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        arr = cm.to_array(10, single_model.parameter_names)
        n_params = len(single_model.parameter_names)
        assert arr.shape == (10, 2 * n_params)

    def test_to_array_values(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        cm.set_constraint("center", vmin=2.85, vmax=2.90)
        arr = cm.to_array(1, single_model.parameter_names)
        assert arr[0, 0] == pytest.approx(2.85)
        assert arr[0, 1] == pytest.approx(2.90)

    def test_to_array_rows_identical(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        arr = cm.to_array(5, single_model.parameter_names)
        for i in range(1, 5):
            np.testing.assert_array_equal(arr[0], arr[i])

    def test_get_constraint_types_shape(self, single_model, default_settings) -> None:
        cm = ConstraintManager(single_model, default_settings)
        types = cm.get_constraint_types(single_model.parameter_names)
        assert types.shape == (len(single_model.parameter_names),)
        assert types.dtype == np.int32


# ---------------------------------------------------------------------------
# mT conversion
# ---------------------------------------------------------------------------


class TestMtConversion:
    """Tests for _mt_to_absolute_ghz helper."""

    def test_symmetric_center_bounds(self) -> None:
        from qdmpy.constants import D_ZFS, GAMMA_NV

        settings = ModelConstraintsSettings(
            constraint_units="mt",
            center_max_mt=5.0,
            center_min_mt=0.0,
        )
        converted = _mt_to_absolute_ghz(settings)
        delta = 5.0 * 1e-3 * GAMMA_NV
        assert converted.center_min == pytest.approx(D_ZFS - delta)
        assert converted.center_max == pytest.approx(D_ZFS + delta)
