from typing import Dict, List, Any
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import logging
import os
import sys

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

from QDMpy import SETTINGS
from QDMpy.constants import AHYP_14N, AHYP_15N

LOG = logging.getLogger(__name__)


def esr14n(x: NDArray, parameter: NDArray, ahyp: float = AHYP_14N) -> NDArray:
    """
    Evaluate the ESR14N model.

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
        NDArray: The calculated model response, with the same shape as `x`.

    Raises:
        ValueError: If the input arrays are incompatible in size.
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


def esr15n(frequencies: NDArray, parameter: NDArray, ahyp: float = AHYP_15N) -> NDArray:
    """
    Evaluate the ESR15N model.

    This function calculates the 15N Diamond model response for a given set of input
    parameters and x-values. The model is characterized by three resonance dips
    at frequencies shifted by the hyperfine splitting constant (AHYP).

    Args:
        frequencies (NDArray): The independent variable (e.g., frequencies).
        parameter (NDArray): A 2D array of parameters for the model. Each row
            corresponds to one set of parameters with the following order:
                - parameter[0]: Center frequency of the resonance (float).
                - parameter[1]: Width of the resonance peak (float).
                - parameter[2]: Contrast of the first dip (float).
                - parameter[3]: Contrast of the second dip (float).
                - parameter[4]: Offset added to the model (float).
        ahyp (float): Hyperfine splitting constant.

    Returns:
        NDArray: The calculated model response, with the same shape as `x`.

    Raises:
        ValueError: If the input arrays are incompatible in size.
    """
    out = []
    parameter = np.atleast_2d(parameter)
    for p in parameter:
        width_squared = p[1] * p[1]

        aux1 = frequencies - p[0] + ahyp
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        aux2 = frequencies - p[0] - ahyp
        dip2 = p[3] * width_squared / (aux2 * aux2 + width_squared)

        out.append(1 + p[4] - dip1 - dip2)
    return np.array(out)


def esrsingle(x: NDArray, parameter: NDArray) -> NDArray:
    """
    Evaluate the ESRSINGLE model.

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = offset

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
    def __init__(self, name: str, n_peaks: int, parameters_unique: List[str]):
        self.name = name
        self.parameters_unique = parameters_unique
        self.n_peaks = n_peaks

    @property
    def parameter(self):
        return [i.split("_")[0] for i in self.parameters_unique]

    @abstractmethod
    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return len(self.parameters_unique)

    def get_constraint_array(self, constraint:dict, n_pixel:int):
        constraint_array = np.zeros(())
        for p in self.parameters_unique:
            for c in constraint:
                if c in p:
                    constraint_array.append(constraint[c][0])
                    constraint_array.append(constraint[c][1])
        return constraint_array

    def __repr__(self):
        return f"model({self.name}, n_parameters: {self.n_parameters}, n_peaks: {self.n_peaks})"


class ModelRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, model: Dict[str, Any]):
        cls._registry[name] = model
        LOG.info(f"Registered model: {name}")

    @classmethod
    def get(cls, name: str):
        return cls._registry.get(name)["class"]()

    @classmethod
    def all(cls):
        return cls._registry

    def _initialize_constraints(self) -> Dict[str, List[Any]]:
        """Initialize default constraints for model parameters."""
        settings = SETTINGS["fit"]["constraints"]
        constraints = {}
        for param in self.parameters_unique:
            base_param = param.split("_")[0]
            constraints[param] = [
                settings[f"{base_param}_min"],
                settings[f"{base_param}_max"],
                settings[f"{base_param}_type"]
            ]
        return constraints

class ESR14N(Model):
    def __init__(self):
        super().__init__(
            "ESR14N",
            3,
            ["contrast", "center", "width_0", "width_1", "width_2", "offset"],
        )
        self.ahyp = AHYP_14N

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        return esr14n(x, parameters, self.ahyp)


class ESR15N(Model):
    def __init__(self):
        super().__init__(
            "ESR15N", 2, ["contrast", "center", "width_0", "width_1", "offset"]
        )
        self.ahyp = AHYP_15N

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        return esr15n(x, parameters, self.ahyp)


class ESRSINGLE(Model):
    def __init__(self):
        super().__init__("ESRSINGLE", 1, ["contrast", "center", "width_0", "offset"])

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        return esrsingle(x, parameters)


# Register models
ModelRegistry.register("ESR14N", {"class": ESR14N, "hyp": AHYP_14N})
ModelRegistry.register("ESR15N", {"class": ESR15N, "hyp": AHYP_15N})
ModelRegistry.register("ESRSINGLE", {"class": ESRSINGLE, "hyp": 0.0})

if __name__ == "__main__":
    model = ModelRegistry.get("ESRSINGLE")
    print(model.n_parameter)
