"""Parameter constraint management for ODMR fitting.

This module provides the ConstraintManager class which handles parameter
constraints for GPU-accelerated ODMR fitting operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
from loguru import logger
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Mapping

from qdmpy.constants import D_ZFS, GAMMA_NV
from qdmpy.exceptions import ParameterError
from qdmpy.fitting.models import Model
from qdmpy.settings import ModelConstraintsSettings

CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]


@dataclass(frozen=True)
class Constraint:
    """A single parameter constraint with typed fields.

    Attributes:
        vmin: Minimum bound for the parameter.
        vmax: Maximum bound for the parameter.
        constraint_type: One of 'FREE', 'LOWER', 'UPPER', 'LOWER_UPPER'.
        unit: Physical unit of the parameter (e.g. 'GHz', 'a.u.').
    """

    vmin: float
    vmax: float
    constraint_type: str
    unit: str

    def __post_init__(self: Self) -> None:
        """Validate constraint_type is a known value."""
        if self.constraint_type not in CONSTRAINT_TYPES:
            msg = f"Invalid constraint type: {self.constraint_type}"
            raise ParameterError(msg)

    @property
    def type_index(self: Self) -> int:
        """Return the integer index for the constraint type."""
        return CONSTRAINT_TYPES.index(self.constraint_type)

    def with_updates(
        self: Self,
        vmin: float | None = None,
        vmax: float | None = None,
        constraint_type: str | None = None,
    ) -> Constraint:
        """Return a new Constraint with selectively updated fields.

        Args:
            vmin: New minimum bound (keeps current if None).
            vmax: New maximum bound (keeps current if None).
            constraint_type: New type (keeps current if None).

        Returns:
            New Constraint instance with updated values.
        """
        return Constraint(
            vmin=vmin if vmin is not None else self.vmin,
            vmax=vmax if vmax is not None else self.vmax,
            constraint_type=(
                constraint_type if constraint_type is not None else self.constraint_type
            ),
            unit=self.unit,
        )


@dataclass(frozen=True)
class ConstraintOverride:
    """A vmin/vmax/constraint_type override applied to one parameter type.

    Used by ``FitManager`` to layer per-call constraint overrides (e.g. the
    folded-fit contrast/offset bounds) onto a base constraint mapping without
    constructing a second manager.
    """

    vmin: float
    vmax: float
    constraint_type: str


def constraints_to_array(
    constraints: Mapping[str, Constraint], n_pixel: int, parameter_names: list[str]
) -> NDArray:
    """Convert a constraint mapping to array format for GPU fitting.

    All frequency values are kept in GHz, matching the GPU kernel convention
    (pyGpufit ESR models have AHYP hardcoded in GHz).

    Args:
        constraints: Mapping of parameter name to Constraint.
        n_pixel: Number of pixels (for array replication).
        parameter_names: List of parameter names to extract constraints for.

    Returns:
        NDArray of shape (n_pixel, 2*n_params) with min/max bounds in GHz.
    """
    constraints_list: list[float] = []
    for param in parameter_names:
        c = constraints[param]
        constraints_list.extend((c.vmin, c.vmax))
    return np.tile(constraints_list, (n_pixel, 1))


def constraint_type_indices(
    constraints: Mapping[str, Constraint], parameter_names: list[str]
) -> NDArray:
    """Get constraint type indices for parameters from a constraint mapping.

    Args:
        constraints: Mapping of parameter name to Constraint.
        parameter_names: List of parameter names.

    Returns:
        NDArray of constraint type indices (0=FREE, 1=LOWER, 2=UPPER, 3=LOWER_UPPER).
    """
    return np.array(
        [constraints[param].type_index for param in parameter_names],
        dtype=np.int32,
    )


def _mt_to_absolute_ghz(settings: ModelConstraintsSettings) -> ModelConstraintsSettings:
    """Convert mT constraint bounds to absolute-GHz bounds.

    The conversion: delta_ghz = mt * 1e-3 * GAMMA_NV, then
    center_min = D_ZFS - delta_max, center_max = D_ZFS + delta_max.

    Args:
        settings: Settings with mT-mode fields populated.

    Returns:
        New ModelConstraintsSettings with absolute-GHz center/width bounds.
    """
    delta_max_ghz = settings.center_max_mt * 1e-3 * GAMMA_NV
    width_min_ghz = settings.width_min_mt * 1e-3 * GAMMA_NV
    width_max_ghz = settings.width_max_mt * 1e-3 * GAMMA_NV

    center_min_ghz = D_ZFS - delta_max_ghz
    center_max_ghz = D_ZFS + delta_max_ghz

    logger.debug(
        "mT -> GHz conversion: center=[{:.4f}, {:.4f}] GHz, width=[{:.6f}, {:.6f}] GHz",
        center_min_ghz,
        center_max_ghz,
        width_min_ghz,
        width_max_ghz,
    )

    return settings.model_copy(
        update={
            "center_min": center_min_ghz,
            "center_max": center_max_ghz,
            "width_min": width_min_ghz,
            "width_max": width_max_ghz,
        }
    )


class ConstraintManager:
    """Manages parameter constraints for fitting."""

    def __init__(
        self: Self,
        model: Model,
        settings: ModelConstraintsSettings,
    ) -> None:
        """Initialize the constraint manager from a model and settings.

        Args:
            model: Model instance providing parameter metadata.
            settings: ModelConstraintsSettings with constraint bounds and types.
        """
        self._constraints: dict[str, Constraint] = {}
        self._model = model
        resolved = self._resolve_settings(settings)
        self._initialize_constraints(resolved)

    @staticmethod
    def _resolve_settings(settings: ModelConstraintsSettings) -> ModelConstraintsSettings:
        """Resolve constraint_units mode to absolute-GHz settings.

        Args:
            settings: Raw settings (may be in mT or absolute_ghz mode).

        Returns:
            Settings with center_min/max and width_min/max in absolute GHz.
        """
        if settings.constraint_units == "mt":
            return _mt_to_absolute_ghz(settings)
        return settings

    def _initialize_constraints(
        self: Self,
        settings: ModelConstraintsSettings,
    ) -> None:
        units = self._model.units
        for param in self._model.parameter_names:
            base_param = self._model.parameter_types[param]
            self._constraints[param] = Constraint(
                vmin=getattr(settings, f"{base_param}_min"),
                vmax=getattr(settings, f"{base_param}_max"),
                constraint_type=getattr(settings, f"{base_param}_type"),
                unit=units[param],
            )

    def set_constraint(
        self: Self,
        param: str,
        vmin: float | None = None,
        vmax: float | None = None,
        constraint_type: str | None = None,
    ) -> None:
        """Set constraint bounds and type for a parameter.

        Args:
            param: Parameter name.
            vmin: Minimum value constraint.
            vmax: Maximum value constraint.
            constraint_type: Type of constraint ('FREE', 'LOWER', 'UPPER', 'LOWER_UPPER').
        """
        if param not in self._constraints:
            msg = f"Unknown parameter: {param}"
            raise ParameterError(msg)
        self._constraints[param] = self._constraints[param].with_updates(
            vmin=vmin,
            vmax=vmax,
            constraint_type=constraint_type,
        )

    def get_constraints(self: Self) -> dict[str, Constraint]:
        """Get a copy of all parameter constraints.

        Returns:
            Dictionary mapping parameter names to Constraint objects. Mutating
            the returned dict does not affect this manager's internal state.
        """
        return dict(self._constraints)

    def to_array(self: Self, n_pixel: int, parameter_names: list[str]) -> NDArray:
        """Convert constraints to array format for GPU fitting.

        Args:
            n_pixel: Number of pixels (for array replication).
            parameter_names: List of parameter names to extract constraints for.

        Returns:
            NDArray of shape (n_pixel, 2*n_params) with min/max bounds in GHz.
        """
        return constraints_to_array(self._constraints, n_pixel, parameter_names)

    def get_constraint_types(self: Self, parameter_names: list[str]) -> NDArray:
        """Get constraint type indices for parameters.

        Args:
            parameter_names: List of parameter names.

        Returns:
            NDArray of constraint type indices (0=FREE, 1=LOWER, 2=UPPER, 3=LOWER_UPPER).
        """
        return constraint_type_indices(self._constraints, parameter_names)
