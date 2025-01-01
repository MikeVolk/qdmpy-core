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

from QDMpy.constants import AHYP_14N, AHYP_15N

LOG = logging.getLogger(__name__)


def esr14n(x, parameter):
    """ESR14N model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = contrast
            parameter[5] = offset

    Returns:
        np.ndarray: y values
    """
    out = []
    AHYP = 0.002158
    parameter = np.atleast_2d(parameter)
    for i in range(parameter.shape[0]):
        p = parameter[i]
        aux1 = x - p[0] + AHYP
        width_squared = p[1] * p[1]

        dip1 = p[2] * width_squared / (aux1 * aux1 + p[1] * p[1])

        aux2 = x - p[0]
        dip2 = p[3] * width_squared / (aux2 * aux2 + p[1] * p[1])

        aux3 = x - p[0] - AHYP
        dip3 = p[4] * width_squared / (aux3 * aux3 + p[1] * p[1])

        out.append(1 + p[5] - dip1 - dip2 - dip3)
    return np.array(out)


def esr15n(frequencies, parameter):
    """ESR15N model

    Args:
        frequencies (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = contrast
            parameter[4] = offset

    Returns:
        np.ndarray: y values
    """
    out = []
    AHYP = 0.0015
    parameter = np.atleast_2d(parameter)

    for i in range(parameter.shape[0]):
        p = parameter[i]
        width_squared = p[1] * p[1]

        aux1 = frequencies - p[0] + AHYP
        dip1 = p[2] * width_squared / (aux1 * aux1 + width_squared)

        aux2 = frequencies - p[0] - AHYP
        dip2 = p[3] * width_squared / (aux2 * aux2 + width_squared)

        out.append(1 + p[4] - dip1 - dip2)
    return np.array(out)


def esrsingle(x, parameter):
    """ESRSINGLE model

    Args:
        x (np.ndarray): x values
        parameter (np.ndarray): parameters
            parameter[0] = center
            parameter[1] = width
            parameter[2] = contrast
            parameter[3] = offset

    Returns:
        np.ndarray: y values
    """
    out = []

    for i in range(parameter.shape[0]):
        p = parameter[i]
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
        return cls._registry.get(name)

    @classmethod
    def all(cls):
        return cls._registry


class ESR14N(Model):
    def __init__(self):
        super().__init__(
            "ESR14N",
            3,
            ["contrast", "center", "width_0", "width_1", "width_2", "offset"],
        )

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        return esr14n(x, parameters)


class ESR15N(Model):
    def __init__(self):
        super().__init__(
            "ESR15N", 2, ["contrast", "center", "width_0", "width_1", "offset"]
        )

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        return esr15n(x, parameters)


class ESRSINGLE(Model):
    def __init__(self):
        super().__init__("ESRSINGLE", 1, ["contrast", "center", "width_0", "offset"])

    def func(self, x: NDArray, parameters: NDArray) -> NDArray:
        return esrsingle(x, parameters)


# Register models
ModelRegistry.register("ESR14N", {"class": ESR14N, "hyp": AHYP_14N})
ModelRegistry.register("ESR15N", {"class": ESR15N, "hyp": AHYP_15N})
ModelRegistry.register("ESRSINGLE", {"class": ESRSINGLE, "hyp": 0.0})
