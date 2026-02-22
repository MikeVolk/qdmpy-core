"""Parameter constraint management for ODMR fitting.

This module provides the ConstraintManager class which handles parameter
constraints for GPU-accelerated ODMR fitting operations.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from qdmpy_core.exceptions import ParameterError
from qdmpy_core.fitting.models import Model
from qdmpy_core.settings import ModelConstraintsSettings

CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]


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
        self._constraints: dict[str, list[Any]] = {}
        self._model = model
        self._initialize_constraints(settings)

    def _initialize_constraints(
        self: Self,
        settings: ModelConstraintsSettings,
    ) -> None:
        units = self._model.units
        for param in self._model.parameter_names:
            base_param = self._model.parameter_types[param]
            self._constraints[param] = [
                getattr(settings, f"{base_param}_min"),
                getattr(settings, f"{base_param}_max"),
                getattr(settings, f"{base_param}_type"),
                units[param],
            ]

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
        current = self._constraints[param]
        if vmin is not None:
            current[0] = vmin
        if vmax is not None:
            current[1] = vmax
        if constraint_type is not None:
            if constraint_type not in CONSTRAINT_TYPES:
                msg = f"Invalid constraint type: {constraint_type}"
                raise ParameterError(msg)
            current[2] = constraint_type

    def get_constraints(self: Self) -> dict[str, list[Any]]:
        """Get all parameter constraints.

        Returns:
            Dictionary mapping parameter names to constraint lists [vmin, vmax, type, unit].
        """
        return self._constraints

    def to_array(self: Self, n_pixel: int, parameter_names: list[str]) -> NDArray:
        """Convert constraints to array format for GPU fitting.

        All frequency values are kept in GHz, matching the GPU kernel convention
        (pyGpufit ESR models have AHYP hardcoded in GHz).

        Args:
            n_pixel: Number of pixels (for array replication).
            parameter_names: List of parameter names to extract constraints for.

        Returns:
            NDArray of shape (n_pixel, 2*n_params) with min/max bounds in GHz.
        """
        constraints_list: list[float] = []
        for param in parameter_names:
            param_min, param_max = self._constraints[param][0], self._constraints[param][1]
            constraints_list.extend((param_min, param_max))
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self: Self, parameter_names: list[str]) -> NDArray:
        """Get constraint type indices for parameters.

        Args:
            parameter_names: List of parameter names.

        Returns:
            NDArray of constraint type indices (0=FREE, 1=LOWER, 2=UPPER, 3=LOWER_UPPER).
        """
        return np.array(
            [CONSTRAINT_TYPES.index(self._constraints[param][2]) for param in parameter_names],
            dtype=np.int32,
        )
