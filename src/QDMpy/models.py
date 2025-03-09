"""Model definitions for fitting ODMR spectra.

This module provides models for fitting Optically Detected Magnetic Resonance (ODMR)
spectra from Nitrogen-Vacancy (NV) centers in diamond. It includes models for different
nitrogen isotopes (14N and 15N) and different configurations, along with a registry
system for managing and retrieving models.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

# Handle paths for direct script execution
from QDMpy.utils import setup_package_paths

setup_package_paths()

from QDMpy import SETTINGS  # noqa: E402
from QDMpy.constants import AHYP_14N, AHYP_15N  # noqa: E402

LOG = logging.getLogger(__name__)


def esr14n(x: NDArray, parameter: NDArray, ahyp: float = AHYP_14N) -> NDArray:
    """Evaluate the ESR14N model.

    This function calculates the ESR14N model response for a given set of input
    parameters and x-values. The model is characterized by three resonance dips
    at frequencies shifted by the hyperfine splitting constant (AHYP).

    Args:
        x (NDArray): The independent variable (e.g., frequencies).
        parameter (NDArray): A 2D array of parameters for the model. Each row
            corresponds to one set of parameters with the following order:
                - parameter[0]: Center frequency of the resonance (float).
                - parameter[1]: Width of the resonance peak (float).
                - parameter[2]: Contrast of the first dip (float).
                - parameter[3]: Contrast of the second dip (float).
                - parameter[4]: Contrast of the third dip (float).
                - parameter[5]: Offset added to the model (float).
        ahyp (float): Hyperfine splitting constant.

    Returns:
        NDArray: The calculated model response.
    """
    out = []
    parameter = np.atleast_2d(parameter)
    for p in parameter:
        aux1 = x - p[0] + ahyp
        width_squared = p[1] * p[1]
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        aux2 = x - p[0]
        dip2 = p[3] * width_squared / (aux2 * aux2 + width_squared)

        aux3 = x - p[0] - ahyp
        dip3 = p[4] * width_squared / (aux3 * aux3 + width_squared)

        out.append(1 + p[5] - dip1 - dip2 - dip3)
    return np.array(out)


def esr15n(x: NDArray, parameter: NDArray, ahyp: float = AHYP_15N) -> NDArray:
    """Evaluate the ESR15N model.

    This function calculates the 15N Diamond model response for a given set of input
    parameters and x-values. The model is characterized by two resonance dips
    at frequencies shifted by the hyperfine splitting constant (AHYP).

    Args:
        x (NDArray): The independent variable (e.g., frequencies).
        parameter (NDArray): A 2D array of parameters for the model. Each row
            corresponds to one set of parameters with the following order:
                - parameter[0]: Center frequency of the resonance (float).
                - parameter[1]: Width of the resonance peak (float).
                - parameter[2]: Contrast of the first dip (float).
                - parameter[3]: Contrast of the second dip (float).
                - parameter[4]: Offset added to the model (float).
        ahyp (float): Hyperfine splitting constant.

    Returns:
        NDArray: The calculated model response.
    """
    out = []
    parameter = np.atleast_2d(parameter)
    for p in parameter:
        width_squared = p[1] * p[1]

        aux1 = x - p[0] + ahyp
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        aux2 = x - p[0] - ahyp
        dip2 = p[3] * width_squared / (aux2 * aux2 + width_squared)

        out.append(1 + p[4] - dip1 - dip2)
    return np.array(out)


def esrsingle(x: NDArray, parameter: NDArray) -> NDArray:
    """Evaluate the ESRSINGLE model.

    This function calculates the single resonance model response for a given set of input
    parameters and x-values. The model is characterized by a single resonance dip.

    Args:
        x (NDArray): The independent variable (e.g., frequencies).
        parameter (NDArray): A 2D array of parameters for the model. Each row
            corresponds to one set of parameters with the following order:
                - parameter[0]: Center frequency of the resonance (float).
                - parameter[1]: Width of the resonance peak (float).
                - parameter[2]: Contrast of the dip (float).
                - parameter[3]: Offset added to the model (float).

    Returns:
        NDArray: The calculated model response.
    """
    out = []
    parameter = np.atleast_2d(parameter)
    for p in parameter:
        width_squared = p[1] * p[1]

        aux1 = x - p[0]
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        out.append(1 + p[3] - dip1)
    return np.array(out)


class Model(ABC):
    """Abstract base class for ODMR spectral models.

    This class defines the interface for all models used to fit ODMR spectra.
    Each concrete model implementation must provide a function that evaluates
    the model given a set of parameters.

    Attributes:
        name: Unique identifier for the model.
        parameters_unique: List of parameter names, with unique identifiers for duplicates.
        n_peaks: Number of resonance peaks in the model.
    """

    def __init__(self: Model, name: str, n_peaks: int, parameters_unique: list[str]) -> None:
        """Initialize a model with basic properties.

        Args:
            name: Unique identifier for the model.
            n_peaks: Number of resonance peaks in the model.
            parameters_unique: List of parameter names with unique identifiers.
        """
        self.name = name
        self.parameters_unique = parameters_unique
        self.n_peaks = n_peaks

    @property
    def parameter(self: Model) -> list[str]:
        """Get the base parameter names without unique identifiers.

        Returns:
            List of base parameter names (e.g., 'width' from 'width_0').
        """
        return [i.split('_')[0] for i in self.parameters_unique]

    @abstractmethod
    def func(self: Model, x: NDArray, parameters: NDArray) -> NDArray:
        """Evaluate the model for given frequency values and parameters.

        Args:
            x: Array of frequency values.
            parameters: Array of model parameters.

        Returns:
            Model prediction for the given frequency values and parameters.
        """
        raise NotImplementedError

    @property
    def n_parameters(self: Model) -> int:
        """Get the number of parameters in the model.

        Returns:
            Number of parameters.
        """
        return len(self.parameters_unique)

    def get_constraint_array(self: Model, constraint: dict[str, Any]) -> NDArray:
        """Create an array of constraints for model parameters.

        Args:
            constraint: Dictionary of constraints.

        Returns:
            Array of constraint values.
        """
        constraint_array = []
        for p in self.parameters_unique:
            base_param = p.split('_')[0]
            if base_param in constraint:
                constraint_array.append(constraint[base_param][0])  # Lower bound
                constraint_array.append(constraint[base_param][1])  # Upper bound
            else:
                # Default constraints if not specified
                constraint_array.append(-np.inf)  # No lower bound
                constraint_array.append(np.inf)   # No upper bound

        return np.array(constraint_array)

    def __repr__(self: Model) -> str:
        """Get a string representation of the model.

        Returns:
            String describing the model's key properties.
        """
        return f'Model({self.name}, n_parameters: {self.n_parameters}, n_peaks: {self.n_peaks})'


class ModelRegistry:
    """Registry for managing ODMR spectral models.

    This class provides a central registry for all available models,
    allowing models to be registered, retrieved, and listed.
    """

    _registry: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register(cls: type[ModelRegistry], name: str, model: dict[str, Any]) -> None:
        """Register a model in the registry.

        Args:
            name: Unique name for the model.
            model: Dictionary containing model information, including
                  the model class and hyperfine splitting constant.
        """
        cls._registry[name] = model
        LOG.info('Registered model: %s', name)

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
            # Model not found
            error_msg = f"Model '{name}' not found in registry"
            raise KeyError(error_msg)
        return cls._registry[name]['class']()

    @classmethod
    def all(cls: type[ModelRegistry]) -> dict[str, dict[str, Any]]:
        """Get all registered models.

        Returns:
            Dictionary mapping model names to model information.
        """
        return cls._registry

    @classmethod
    def _initialize_constraints(cls: type[ModelRegistry], model: Model) -> dict[str, list[Any]]:
        """Initialize default constraints for model parameters.

        Uses the constraints defined in the configuration settings.

        Args:
            model: The model for which to initialize constraints.

        Returns:
            Dictionary mapping parameter names to constraint lists.
        """
        settings = SETTINGS['fit']['constraints']
        constraints: dict[str, list[Any]] = {}

        for param in model.parameters_unique:
            base_param = param.split('_')[0]
            constraints[param] = [
                settings[f'{base_param}_min'],
                settings[f'{base_param}_max'],
                settings[f'{base_param}_type'],
            ]
        return constraints

class ESR14N(Model):
    """Model for NV centers with 14N nitrogen isotope.

    This model represents ODMR spectra with three dips due to the hyperfine
    interaction with the 14N nucleus (I=1).
    """
    def __init__(self: ESR14N) -> None:
        """Initialize ESR14N model with default parameters."""
        super().__init__(
            'ESR14N',
            3,
            ['contrast', 'center', 'width_0', 'width_1', 'width_2', 'offset'],
        )
        self.ahyp = AHYP_14N

    def func(self: ESR14N, x: NDArray, parameters: NDArray) -> NDArray:
        """Calculate the model response for the given parameters.

        Args:
            x: The independent variable (e.g., frequencies).
            parameters: The model parameters.

        Returns:
            The calculated model response.
        """
        return esr14n(x, parameters, self.ahyp)


class ESR15N(Model):
    """Model for NV centers with 15N nitrogen isotope.

    This model represents ODMR spectra with two dips due to the hyperfine
    interaction with the 15N nucleus (I=1/2).
    """
    def __init__(self: ESR15N) -> None:
        """Initialize ESR15N model with default parameters."""
        super().__init__(
            'ESR15N', 2, ['contrast', 'center', 'width_0', 'width_1', 'offset'],
        )
        self.ahyp = AHYP_15N

    def func(self: ESR15N, x: NDArray, parameters: NDArray) -> NDArray:
        """Calculate the model response for the given parameters.

        Args:
            x: The independent variable (e.g., frequencies).
            parameters: The model parameters.

        Returns:
            The calculated model response.
        """
        return esr15n(x, parameters, self.ahyp)


class ESRSINGLE(Model):
    """Model for a single ODMR resonance dip.

    This model represents ODMR spectra with a single resonance dip,
    without any hyperfine splitting.
    """
    def __init__(self: ESRSINGLE) -> None:
        """Initialize ESRSINGLE model with default parameters."""
        super().__init__('ESRSINGLE', 1, ['contrast', 'center', 'width_0', 'offset'])

    def func(self: ESRSINGLE, x: NDArray, parameters: NDArray) -> NDArray:
        """Calculate the model response for the given parameters.

        Args:
            x: The independent variable (e.g., frequencies).
            parameters: The model parameters.

        Returns:
            The calculated model response.
        """
        return esrsingle(x, parameters)


# Register models
ModelRegistry.register('ESR14N', {'class': ESR14N, 'hyp': AHYP_14N})
ModelRegistry.register('ESR15N', {'class': ESR15N, 'hyp': AHYP_15N})
ModelRegistry.register('ESRSINGLE', {'class': ESRSINGLE, 'hyp': 0.0})

if __name__ == '__main__':
    model = ModelRegistry.get('ESRSINGLE')
    # Print model parameters
    import sys
    sys.stdout.write(f'{model.n_parameters}\n')
