"""Model definitions for fitting ODMR spectra.

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

# Handle paths for direct script execution
from QDMpy.utils import setup_package_paths

setup_package_paths()

from QDMpy import SETTINGS  # noqa: E402
from QDMpy.constants import AHYP_14N, AHYP_15N  # noqa: E402


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
        x: Array of frequency values in Hz.
        parameter: Parameter array with shape (N, 6) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (Hz)
                - [1] width: Linewidth parameter (Hz)
                - [2] contrast_-1: Contrast of the mI=-1 dip (0-1)
                - [3] contrast_0: Contrast of the mI=0 dip (0-1)
                - [4] contrast_+1: Contrast of the mI=+1 dip (0-1)
                - [5] offset: Baseline offset (0-1)
        ahyp: Hyperfine splitting constant (Hz). Defaults to AHYP_14N.

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - ∑ᵢ (contrastᵢ * width² / ((x - posᵢ)² + width²))
        where posᵢ are the three resonance positions.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87e9, 2.88e9, 100)
        >>> params = np.array([2.87e9, 2e6, 0.1, 0.2, 0.1, 0.0])
        >>> spectrum = esr14n(x, params)
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
        x: Array of frequency values in Hz.
        parameter: Parameter array with shape (N, 5) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (Hz)
                - [1] width: Linewidth parameter (Hz)
                - [2] contrast_-1/2: Contrast of the mI=-1/2 dip (0-1)
                - [3] contrast_+1/2: Contrast of the mI=+1/2 dip (0-1)
                - [4] offset: Baseline offset (0-1)
        ahyp: Hyperfine splitting constant (Hz). Defaults to AHYP_15N.

    Returns:
        Model response array with shape (N, len(x)) where N is the number of
        parameter sets. Values represent normalized fluorescence intensity.

    Note:
        The model implements the equation:
        f(x) = 1 + offset - ∑ᵢ (contrastᵢ * width² / ((x - posᵢ)² + width²))
        where posᵢ are the two resonance positions.

    Example:
        >>> import numpy as np
        >>> x = np.linspace(2.87e9, 2.88e9, 100)
        >>> params = np.array([2.87e9, 2e6, 0.15, 0.15, 0.0])
        >>> spectrum = esr15n(x, params)
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


def esrsingle(x: NDArray[np.floating], parameter: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate the ESRSINGLE model for single resonance systems.

    This function calculates the ODMR spectrum response for systems with a single
    resonance dip, without hyperfine splitting. This model is useful for isolated
    spin systems or when hyperfine structure is not resolved.

    Args:
        x: Array of frequency values in Hz.
        parameter: Parameter array with shape (N, 4) where N is the number of spectra.
            Each row contains parameters in this order:
                - [0] center: Center frequency of the resonance (Hz)
                - [1] width: Linewidth parameter (Hz)
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
        >>> x = np.linspace(2.87e9, 2.88e9, 100)
        >>> params = np.array([2.875e9, 3e6, 0.2, 0.0])
        >>> spectrum = esrsingle(x, params)
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
    the model given a set of parameters. The class also provides utilities for
    parameter management and constraint handling.

    All ODMR models in QDMpy should inherit from this class to ensure consistency
    and compatibility with the fitting infrastructure.

    Attributes:
        name: Unique identifier for the model (e.g., 'ESR14N', 'ESR15N').
        parameters_unique: List of parameter names with unique identifiers for
            duplicates (e.g., ['center', 'width_0', 'width_1', 'contrast', 'offset']).
        n_peaks: Number of resonance peaks in the model (1 for single, 2 for 15N,
            3 for 14N).

    Example:
        >>> from QDMpy.models import ESR14N
        >>> model = ESR14N()
        >>> print(f"Model: {model.name}, Parameters: {model.n_parameters}")
        >>> Model: ESR14N, Parameters: 6
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
        return [i.split("_")[0] for i in self.parameters_unique]

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
            x: Array of frequency values in Hz.
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
        return len(self.parameters_unique)

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
        for p in self.parameters_unique:
            base_param = p.split("_")[0]
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
        return f"Model({self.name}, n_parameters: {self.n_parameters}, " f"n_peaks: {self.n_peaks})"


class ModelRegistry:
    """Registry for managing ODMR spectral models.

    This class provides a central registry for all available models,
    allowing models to be registered, retrieved, and listed. It acts as
    a factory pattern for model instantiation and provides a single point
    of access for all model types.

    The registry is populated at module import time with the standard models
    (ESR14N, ESR15N, ESRSINGLE) and can be extended with custom models.

    Class Attributes:
        _registry: Class-level dictionary storing model information.

    Example:
        >>> # Get available models
        >>> models = ModelRegistry.all()
        >>> print(list(models.keys()))
        ['ESR14N', 'ESR15N', 'ESRSINGLE']

        >>> # Get a specific model instance
        >>> model = ModelRegistry.get('ESR14N')
        >>> print(type(model).__name__)
        ESR14N
    """

    _registry: ClassVar[dict[str, dict[str, Any]]] = {}

    @classmethod
    def register(cls: type[ModelRegistry], name: str, model: dict[str, Any]) -> None:
        """Register a model in the registry.

        Adds a new model to the registry, making it available for retrieval
        via the get() method. The model dictionary should contain at minimum
        a 'class' key with the model class.

        Args:
            name: Unique name for the model (e.g., 'ESR14N', 'CUSTOM_MODEL').
            model: Dictionary containing model information with keys:
                - 'class': The model class (must inherit from Model)
                - 'hyp': Hyperfine splitting constant (optional)
                - Additional metadata as needed

        Example:
            >>> class CustomModel(Model):
            ...     # Implementation details
            ...     pass
            >>> ModelRegistry.register('CUSTOM', {'class': CustomModel, 'hyp': 0.001})
        """
        cls._registry[name] = model
        logger.info(f"Registered model: {name}")

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
        return cls._registry[name]["class"]()

    @classmethod
    def all(cls: type[ModelRegistry]) -> dict[str, dict[str, Any]]:
        """Get all registered models.

        Returns a copy of the internal registry dictionary containing all
        registered models and their associated metadata.

        Returns:
            Dictionary mapping model names to model information dictionaries.
            Each model dictionary contains 'class' and other metadata keys.

        Example:
            >>> registry = ModelRegistry.all()
            >>> for name, info in registry.items():
            ...     print(f"{name}: {info['class'].__name__}")
            ESR14N: ESR14N
            ESR15N: ESR15N
            ESRSINGLE: ESRSINGLE
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
        settings = SETTINGS["fit"]["constraints"]
        constraints: dict[str, list[Any]] = {}

        for param in model.parameters_unique:
            base_param = param.split("_")[0]
            constraints[param] = [
                settings[f"{base_param}_min"],
                settings[f"{base_param}_max"],
                settings[f"{base_param}_type"],
            ]
        return constraints


class ESR14N(Model):
    """Model for NV centers with 14N nitrogen isotope.

    This model represents ODMR spectra with three dips due to the hyperfine
    interaction with the 14N nucleus (I=1). The three transitions correspond
    to the mI = -1, 0, +1 spin states, creating a characteristic triplet pattern.

    The model uses Lorentzian lineshapes and allows independent contrast values
    for each of the three resonance lines, enabling fitting of spectra with
    asymmetric intensities.

    Attributes:
        ahyp: Hyperfine splitting constant for 14N (AHYP_14N).
    """

    def __init__(self: ESR14N) -> None:
        """Initialize ESR14N model with 14N-specific parameters.

        Sets up the model with 6 parameters: center frequency, width, three contrast
        values (one for each hyperfine line), and baseline offset. The hyperfine
        constant is set to the standard 14N value.
        """
        super().__init__(
            "ESR14N",
            3,
            ["center", "width", "contrast_0", "contrast_1", "contrast_2", "offset"],
        )
        self.ahyp = AHYP_14N * 1e9  # Convert GHz to Hz for pygpufit models
        self.model_id = 13

    def func(
        self: ESR14N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Calculate the 14N ODMR spectrum response.

        Evaluates the ESR14N model function for the given frequency array
        and parameter set, using the predefined 14N hyperfine constant.

        Args:
            x: Array of frequency values in Hz.
            parameters: Parameter array with shape (N, 6) containing:
                [center, width, contrast_-1, contrast_0, contrast_+1, offset].

        Returns:
            Model response array with normalized fluorescence intensity values.
        """
        return esr14n(x, parameters, self.ahyp)


class ESR15N(Model):
    """Model for NV centers with 15N nitrogen isotope.

    This model represents ODMR spectra with two dips due to the hyperfine
    interaction with the 15N nucleus (I=1/2). The two transitions correspond
    to the mI = -1/2, +1/2 spin states, creating a characteristic doublet pattern.

    The 15N isotope has a smaller hyperfine interaction compared to 14N, resulting
    in a smaller splitting between the two resonance lines.

    Attributes:
        ahyp: Hyperfine splitting constant for 15N (AHYP_15N).
    """

    def __init__(self: ESR15N) -> None:
        """Initialize ESR15N model with 15N-specific parameters.

        Sets up the model with 5 parameters: center frequency, width, two contrast
        values (one for each hyperfine line), and baseline offset. The hyperfine
        constant is set to the standard 15N value.
        """
        super().__init__(
            "ESR15N",
            2,
            ["center", "width", "contrast_0", "contrast_1", "offset"],
        )
        self.ahyp = AHYP_15N * 1e9  # Convert GHz to Hz for pygpufit models
        self.model_id = 14

    def func(
        self: ESR15N,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Calculate the 15N ODMR spectrum response.

        Evaluates the ESR15N model function for the given frequency array
        and parameter set, using the predefined 15N hyperfine constant.

        Args:
            x: Array of frequency values in Hz.
            parameters: Parameter array with shape (N, 5) containing:
                [center, width, contrast_-1/2, contrast_+1/2, offset].

        Returns:
            Model response array with normalized fluorescence intensity values.
        """
        return esr15n(x, parameters, self.ahyp)


class ESRSINGLE(Model):
    """Model for a single ODMR resonance dip.

    This model represents ODMR spectra with a single resonance dip,
    without any hyperfine splitting. This is useful for systems where:
    - Hyperfine structure is not resolved due to broadening
    - Working with isotopically pure samples without hyperfine interaction
    - Fitting individual components of more complex spectra
    - Initial parameter estimation for more complex models

    The model uses a simple Lorentzian lineshape for the resonance.
    """

    def __init__(self: ESRSINGLE) -> None:
        """Initialize ESRSINGLE model with single resonance parameters.

        Sets up the model with 4 parameters: center frequency, width, contrast,
        and baseline offset. No hyperfine constant is needed for this model.
        """
        super().__init__("ESRSINGLE", 1, ["center", "width", "contrast", "offset"])
        self.model_id = 15

    def func(
        self: ESRSINGLE,
        x: NDArray[np.floating],
        parameters: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Calculate the single resonance ODMR spectrum response.

        Evaluates the ESRSINGLE model function for the given frequency array
        and parameter set, producing a single Lorentzian dip.

        Args:
            x: Array of frequency values in Hz.
            parameters: Parameter array with shape (N, 4) containing:
                [center, width, contrast, offset].

        Returns:
            Model response array with normalized fluorescence intensity values.
        """
        return esrsingle(x, parameters)


# Register models
ModelRegistry.register("ESR14N", {"class": ESR14N, "hyp": AHYP_14N})
ModelRegistry.register("ESR15N", {"class": ESR15N, "hyp": AHYP_15N})
ModelRegistry.register("ESRSINGLE", {"class": ESRSINGLE, "hyp": 0.0})


def _main_demo() -> None:
    """Demo function that shows model usage when module is run as script."""
    model = ModelRegistry.get("ESRSINGLE")
    # Print model parameters
    import sys

    sys.stdout.write(f"{model.n_parameters}\n")


if __name__ == "__main__":
    _main_demo()
