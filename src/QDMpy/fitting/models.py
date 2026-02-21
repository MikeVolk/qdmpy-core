"""Model definitions for fitting ODMR spectra.

Convention: All frequency values are in GHz.

This module provides models for fitting Optically Detected Magnetic Resonance (ODMR)
spectra from Nitrogen-Vacancy (NV) centers in diamond. It includes models for different
nitrogen isotopes (14N and 15N) and different configurations, along with a registry
system for managing and retrieving models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from QDMpy.constants import AHYP_14N, AHYP_15N


def esr14n(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    ahyp: float = AHYP_14N,
) -> NDArray[np.floating]:
    """Evaluate the ESR14N model for 14N nitrogen-vacancy centers.

    This function calculates the ODMR spectrum response for NV centers with 14N nitrogen
    isotope (I=1), which exhibits three resonance dips due to hyperfine interaction.
    The model uses Lorentzian lineshapes for each resonance component.

    The three dips are positioned at:
    - Center frequency - ahyp (mI = -1)
    - Center frequency (mI = 0)
    - Center frequency + ahyp (mI = +1)

    Args:
        x: Array of frequency values in GHz.
        parameter: Parameter array with shape (N, 6) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (GHz)
                - [1] width: Linewidth parameter (GHz)
                - [2] contrast_-1: Contrast of the mI=-1 dip (0-1)
                - [3] contrast_0: Contrast of the mI=0 dip (0-1)
                - [4] contrast_+1: Contrast of the mI=+1 dip (0-1)
                - [5] offset: Baseline offset (0-1)
        ahyp: Hyperfine splitting constant (GHz). Defaults to AHYP_14N.

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - ∑ᵢ (contrastᵢ * width² / ((x - posᵢ)² + width²))
        where posᵢ are the three resonance positions.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87, 2.88, 100)
        >>> params = np.array([2.87, 0.002, 0.1, 0.2, 0.1, 0.0])
        >>> spectrum = esr14n(x, params)
    """
    parameter = np.atleast_2d(parameter)
    center = parameter[:, 0:1]
    width_sq = parameter[:, 1:2] ** 2
    c0, c1, c2 = parameter[:, 2:3], parameter[:, 3:4], parameter[:, 4:5]
    offset = parameter[:, 5:6]

    dip1 = c0 * width_sq / ((x - center + ahyp) ** 2 + width_sq)
    dip2 = c1 * width_sq / ((x - center) ** 2 + width_sq)
    dip3 = c2 * width_sq / ((x - center - ahyp) ** 2 + width_sq)
    return 1 + offset - dip1 - dip2 - dip3


def esr15n(
    x: NDArray[np.floating],
    parameter: NDArray[np.floating],
    ahyp: float = AHYP_15N,
) -> NDArray[np.floating]:
    """Evaluate the ESR15N model for 15N nitrogen-vacancy centers.

    This function calculates the ODMR spectrum response for NV centers with 15N nitrogen
    isotope (I=1/2), which exhibits two resonance dips due to hyperfine interaction.
    The model uses Lorentzian lineshapes for each resonance component.

    The two dips are positioned at:
    - Center frequency - ahyp (mI = -1/2)
    - Center frequency + ahyp (mI = +1/2)

    Args:
        x: Array of frequency values in GHz.
        parameter: Parameter array with shape (N, 5) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (GHz)
                - [1] width: Linewidth parameter (GHz)
                - [2] contrast_-1/2: Contrast of the mI=-1/2 dip (0-1)
                - [3] contrast_+1/2: Contrast of the mI=+1/2 dip (0-1)
                - [4] offset: Baseline offset (0-1)
        ahyp: Hyperfine splitting constant (GHz). Defaults to AHYP_15N.

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - ∑ᵢ (contrastᵢ * width² / ((x - posᵢ)² + width²))
        where posᵢ are the two resonance positions.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87, 2.88, 100)
        >>> params = np.array([2.87, 0.002, 0.15, 0.15, 0.0])
        >>> spectrum = esr15n(x, params)
    """
    parameter = np.atleast_2d(parameter)
    center = parameter[:, 0:1]
    width_sq = parameter[:, 1:2] ** 2
    c0, c1 = parameter[:, 2:3], parameter[:, 3:4]
    offset = parameter[:, 4:5]

    dip1 = c0 * width_sq / ((x - center + ahyp) ** 2 + width_sq)
    dip2 = c1 * width_sq / ((x - center - ahyp) ** 2 + width_sq)
    return 1 + offset - dip1 - dip2


def esrsingle(x: NDArray[np.floating], parameter: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate the ESRSINGLE model for single resonance systems.

    This function calculates the ODMR spectrum response for systems with a single
    resonance dip, without hyperfine splitting. This model is useful for isolated
    spin systems or when hyperfine structure is not resolved.

    Args:
        x: Array of frequency values in GHz.
        parameter: Parameter array with shape (N, 4) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (GHz)
                - [1] width: Linewidth parameter (GHz)
                - [2] contrast: Contrast of the dip (0-1)
                - [3] offset: Baseline offset (0-1)

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - (contrast * width² / ((x - center)² + width²))

        This is a simple Lorentzian absorption line.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87, 2.88, 100)
        >>> params = np.array([2.875, 0.003, 0.2, 0.0])
        >>> spectrum = esrsingle(x, params)
    """
    parameter = np.atleast_2d(parameter)
    center = parameter[:, 0:1]
    width_sq = parameter[:, 1:2] ** 2
    contrast = parameter[:, 2:3]
    offset = parameter[:, 3:4]

    dip = contrast * width_sq / ((x - center) ** 2 + width_sq)
    return 1 + offset - dip


class Model(ABC):
    """Abstract base class for ODMR spectral models.

    This class defines the interface for all models used to fit ODMR spectra.
    Each concrete model implementation must provide a function that evaluates
    the model given a set of parameters. The class also provides utilities for
    parameter management and constraint handling.

    All ODMR models in QDMpy should inherit from this class to ensure consistency
    and compatibility with the fitting infrastructure.

    Attributes:
        name: Unique identifier for the model (e.g., 'ESR14N', 'ESR15N').
        parameter_names: List of parameter names (e.g., ['center', 'width', 'contrast_0', ...]).
        n_peaks: Number of resonance peaks in the model (1 for single, 2 for 15N, 3 for 14N).

    Example:
        >>> from QDMpy.fitting.models import ESR14N
        >>> model = ESR14N()
        >>> print(f"Model: {model.name}, Parameters: {model.n_parameters}")
        >>> Model: ESR14N, Parameters: 6
    """

    model_id: int

    def __init__(self: Model, name: str, n_peaks: int, parameter_names: list[str]) -> None:
        """Initialize a model with basic properties.

        Args:
            name: Unique identifier for the model.
            n_peaks: Number of resonance peaks in the model.
            parameter_names: List of parameter names (e.g., ['center', 'width', 'contrast_0', ...]).
        """
        self.name = name
        self.parameter_names = parameter_names
        self.n_peaks = n_peaks

    @property
    @abstractmethod
    def parameter_types(self: Model) -> dict[str, str]:
        """Map each parameter name to its type category.

        Returns:
            Dict mapping param name -> type ('center', 'width', 'contrast', 'offset').
        """

    @property
    @abstractmethod
    def frequency_parameters(self: Model) -> list[str]:
        """Parameter names stored in GHz units (center, width).

        Used to derive the units dict for display purposes.  The pyGpufit kernels
        use GHz throughout (AHYP constants are in GHz), so no unit conversion is
        performed at the GPU boundary.

        Returns:
            List of parameter names in frequency (GHz) units.
        """

    @property
    def units(self: Model) -> dict[str, str]:
        """Derive units from frequency_parameters."""
        freq = set(self.frequency_parameters)
        return {p: "GHz" if p in freq else "a.u." for p in self.parameter_names}

    @property
    def parameter(self: Model) -> list[str]:
        """Get the type category for each parameter (backwards compatibility).

        Returns:
            List of parameter type strings (e.g., ['center', 'width', 'contrast', ...]).
        """
        return [self.parameter_types[p] for p in self.parameter_names]

    @abstractmethod
    def func(
        self: Model,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the model for given frequency values and parameters.

        This abstract method must be implemented by all concrete model classes.
        It defines the mathematical function that calculates the model response.

        Args:
            x: Array of frequency values in GHz.
            parameters: Array of model parameters with shape appropriate for the
                specific model.

        Returns:
            Model prediction array with shape (N, len(x)) where N is the number of
            parameter sets. Values represent normalized fluorescence intensity.
        """
        raise NotImplementedError

    @property
    def n_parameters(self: Model) -> int:
        """Get the number of parameters in the model.

        Returns:
            Number of parameters.
        """
        return len(self.parameter_names)

    def get_constraint_array(self: Model, constraint: dict[str, Any]) -> NDArray[np.floating]:
        """Create an array of constraints for model parameters.

        Converts a dictionary of parameter constraints into a flattened array
        suitable for optimization algorithms. Each parameter gets two values:
        minimum and maximum bounds.

        Args:
            constraint: Dictionary mapping parameter names to [min, max] constraint
                pairs. If a parameter is not specified, infinite bounds are used.

        Returns:
            Flattened array of constraint values with shape (2 * n_parameters,).
            Organized as [param1_min, param1_max, param2_min, param2_max, ...].

        Example:
            >>> model = ESR14N()
            >>> constraints = {'center': [2.8e9, 2.9e9], 'width': [1e6, 1e7]}
            >>> bounds = model.get_constraint_array(constraints)
        """
        constraint_array = []
        for p in self.parameter_names:
            base_param = self.parameter_types[p]
            if base_param in constraint:
                constraint_array.append(constraint[base_param][0])  # Lower bound
                constraint_array.append(constraint[base_param][1])  # Upper bound
            else:
                # Default constraints if not specified
                constraint_array.append(-np.inf)  # No lower bound
                constraint_array.append(np.inf)  # No upper bound

        return np.array(constraint_array)

    def __repr__(self: Model) -> str:
        """Get a string representation of the model.

        Returns:
            String describing the model's key properties.
        """
        return f"Model({self.name}, n_parameters: {self.n_parameters}, n_peaks: {self.n_peaks})"


class ModelRegistry:
    """Registry for managing ODMR spectral models.

    Models are registered via the ``@ModelRegistry.register`` decorator.
    The registry maps model names to their classes.

    Example:
        >>> model = ModelRegistry.get('ESR14N')
        >>> print(type(model).__name__)
        ESR14N
    """

    _registry: ClassVar[dict[str, type[Model]]] = {}

    @classmethod
    def register(cls: type[ModelRegistry], model_cls: type[Model]) -> type[Model]:
        """Register a model class (usable as a decorator).

        Args:
            model_cls: A Model subclass to register. The model's ``name``
                ClassVar is used as the registry key.

        Returns:
            The model class, unchanged.
        """
        cls._registry[model_cls.name] = model_cls
        logger.info(f"Registered model: {model_cls.name}")
        return model_cls

    @classmethod
    def get(cls: type[ModelRegistry], name: str) -> Model:
        """Get a model instance by name.

        Args:
            name: Name of the model to retrieve.

        Returns:
            Instance of the requested model.

        Raises:
            KeyError: If the model name is not found in the registry.
        """
        if name not in cls._registry:
            error_msg = f"Model '{name}' not found in registry"
            raise KeyError(error_msg)
        logger.debug(f"Instantiating model: {name}")
        return cls._registry[name]()  # type: ignore[call-arg]

    @classmethod
    def all(cls: type[ModelRegistry]) -> dict[str, type[Model]]:
        """Get all registered model classes.

        Returns:
            Dictionary mapping model names to model classes.
        """
        return cls._registry


@ModelRegistry.register
class ESR14N(Model):
    """Model for NV centers with 14N nitrogen isotope (3 hyperfine dips)."""

    name: ClassVar[str] = "ESR14N"

    def __init__(self: ESR14N) -> None:
        """Initialize ESR14N model with 14N-specific parameters."""
        super().__init__(
            "ESR14N",
            3,
            ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"],
        )
        self.ahyp = AHYP_14N
        self.model_id = 13

    @property
    def parameter_types(self: ESR14N) -> dict[str, str]:
        """Map each parameter to its type category."""
        return {
            "center": "center",
            "width": "width",
            "contrast_0": "contrast",
            "contrast_1": "contrast",
            "contrast_2": "contrast",
            "offset": "offset",
        }

    @property
    def frequency_parameters(self: ESR14N) -> list[str]:
        """Parameters in frequency units (GHz)."""
        return ["center"]

    def func(
        self: ESR14N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the 14N triplet Lorentzian model."""
        return esr14n(x, parameters, self.ahyp)


@ModelRegistry.register
class ESR15N(Model):
    """Model for NV centers with 15N nitrogen isotope (2 hyperfine dips)."""

    name: ClassVar[str] = "ESR15N"

    def __init__(self: ESR15N) -> None:
        """Initialize ESR15N model with 15N-specific parameters."""
        super().__init__(
            "ESR15N",
            2,
            ["center", "width", "contrast_0", "contrast_1", "offset"],
        )
        self.ahyp = AHYP_15N
        self.model_id = 14

    @property
    def parameter_types(self: ESR15N) -> dict[str, str]:
        """Map each parameter to its type category."""
        return {
            "center": "center",
            "width": "width",
            "contrast_0": "contrast",
            "contrast_1": "contrast",
            "offset": "offset",
        }

    @property
    def frequency_parameters(self: ESR15N) -> list[str]:
        """Parameters in frequency units (GHz)."""
        return ["center"]

    def func(
        self: ESR15N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the 15N doublet Lorentzian model."""
        return esr15n(x, parameters, self.ahyp)


@ModelRegistry.register
class ESRSINGLE(Model):
    """Model for a single ODMR resonance dip (no hyperfine splitting)."""

    name: ClassVar[str] = "ESRSINGLE"

    def __init__(self: ESRSINGLE) -> None:
        """Initialize ESRSINGLE model with single-dip parameters."""
        super().__init__("ESRSINGLE", 1, ["center", "width", "contrast", "offset"])
        self.model_id = 15

    @property
    def parameter_types(self: ESRSINGLE) -> dict[str, str]:
        """Map each parameter to its type category."""
        return {
            "center": "center",
            "width": "width",
            "contrast": "contrast",
            "offset": "offset",
        }

    @property
    def frequency_parameters(self: ESRSINGLE) -> list[str]:
        """Parameters in frequency units (GHz)."""
        return ["center"]

    def func(
        self: ESRSINGLE,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Evaluate the single Lorentzian model."""
        return esrsingle(x, parameters)


def _main_demo() -> None:
    """Demo function that shows model usage when module is run as script."""
    model = ModelRegistry.get("ESRSINGLE")
    import sys

    sys.stdout.write(f"{model.n_parameters}\n")


if __name__ == "__main__":
    _main_demo()
