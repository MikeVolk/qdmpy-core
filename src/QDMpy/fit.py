from __future__ import annotations
import os
import sys
import logging
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
from numba import njit

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

import QDMpy
from QDMpy.guess import guess_model, guess_initial_fit_parameters
from QDMpy.odmr.data import ODMRData
from QDMpy.models import ModelRegistry

if QDMpy.PYGPUFIT_PRESENT:  # type: ignore[has-type]
    import pygpufit.gpufit as gf

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from QDMpy.odmr.odmr import ODMR


# Constants
UNITS = {"center": "GHz", "width": "GHz", "contrast": "a.u.", "offset": "a.u."}
CONSTRAINT_TYPES = ["FREE", "LOWER", "UPPER", "LOWER_UPPER"]
ESTIMATOR_ID = {"LSE": 0, "MLE": 1}

LOG = logging.getLogger(__name__)


class Fit:
    """Class for fitting ODMR data using models and constraints."""

    def __init__(self, odmr: ODMR, model: str | None = None) -> None:
        """
        Initialize the Fit class.

        Args:
            odmr (ODMRData): Instance of ODMRData containing raw data, dimensions, and frequencies.
            model_name (str): Name of the fitting model. Defaults to 'auto'.
        """
        self.odmr = odmr
        self._model = None
        self._constraints = None
        self._initial_parameter = None

        LOG.info("Initializing Fit instance.")

        # Guess Model from raw data
        if not model:
            model = guess_model(self.odmr.processed_data.data)

        self._model = model

        self._reset_fit()
        self._constraints = self._set_initial_constraints()
        self._initial_parameter = guess_initial_fit_parameters(
            self.odmr.processed_data.data, self.odmr.raw_data.frequencies, self._model
        )
        self.estimator_id = ESTIMATOR_ID[QDMpy.SETTINGS["fit"]["estimator"]]

    @property
    def model_name(self) -> str:
        return self._model.name

    @model_name.setter
    def model_name(self, name: str) -> None:
        name = name.upper()
        if name not in ModelRegistry.all():
            raise ValueError("Unknown model: %s", name)
        self._model = ModelRegistry.get(name)
        self._reset_fit()
        self._constraints = self._set_initial_constraints()

    def _reset_fit(self) -> None:
        """Reset all fit-related attributes."""
        self._fitted = False
        self._fit_results = None
        self._chi_squares = None
        self._number_iterations = None
        self._execution_time = None

    def _set_initial_constraints(self) -> Dict[str, List[Any]]:
        """Set default constraints based on model parameters."""
        constraints = QDMpy.SETTINGS["fit"]["constraints"]
        defaults = {}
        for param in self._model.parameter:
            defaults[param] = [
                constraints[f"{param}_min"],
                constraints[f"{param}_max"],
                constraints[f"{param}_type"],
                UNITS[param],
            ]
        return defaults

    def fit_odmr(self, refit: bool = False) -> None:
        """Fit the ODMR data."""
        if self._fitted and not refit:
            LOG.info("Fit already completed; skipping.")
            return
        if refit:
            LOG.info("Refitting data.")
            self._reset_fit()

        for irange, freq in enumerate(self.odmr.processed_data.frequencies):
            LOG.info(f"Fitting frequency range {irange}: {freq.min()}-{freq.max()} GHz")
            print(self.odmr.processed_data.data.shape, self._initial_parameter.shape)

            results = self._fit_range(
                self.odmr.processed_data.data[:, irange],
                freq,
                self._initial_parameter[:, irange],
            )
        return results

    def _fit_range(
        self, data: NDArray, freq: NDArray, initial_params: NDArray
    ) -> List[NDArray]:
        """Fit a single frequency range."""
        n_pol, n_freqs, n_pix = data.shape

        reshaped_data = data.reshape((n_pol * n_pix, n_freqs))
        reshaped_params = initial_params.reshape((
            n_pol * n_pix,
            self._model.n_parameters,
        ))
        constraints = get_constraints_array(
            self._model, self._constraints, reshaped_data.shape[0]
        )

        print(data.shape, initial_params.shape, constraints.shape)
        print(reshaped_data.shape, reshaped_params.shape)

        results = gf.fit_constrained(
            data=np.ascontiguousarray(reshaped_data, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=constraints,
            constraint_types=self.get_constraint_types(),
            initial_parameters=np.ascontiguousarray(reshaped_params, dtype=np.float32),
            model_id=self._model.name,
            max_number_iterations=QDMpy.SETTINGS["fit"]["max_number_iterations"],
            tolerance=QDMpy.SETTINGS["fit"]["tolerance"],
            weights=None,
        )
        return results

    def get_constraint_types(self) -> NDArray:
        """Get constraint types as array."""
        return np.array(
            [CONSTRAINT_TYPES.index(c[2]) for c in self._constraints.values()],
            dtype=np.int32,
        )


def get_constraints_array(model, values, n_pixel: int) -> NDArray:
    """Build a constraints array."""
    constraints_list = model.get_constraint_array(values)
    return np.tile(constraints_list, (n_pixel, 1))


if __name__ == "__main__":
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor
    from QDMpy.odmr.odmr import ODMR

    # User-friendly initialization
    loader = MatlabLoader(data_folder="/home/mike/git/QDMpy/tests/data")
    odmr = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=4))
    odmr.process_data()

    fit = Fit(odmr=odmr)
    print(fit._constraints)
    print(fit._model.parameters_unique)
    print(get_constraints_array(fit._model, fit._constraints, 10))
    # fit.fit_odmr()
