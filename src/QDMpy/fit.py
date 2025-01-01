from __future__ import annotations
import os
import sys
import logging
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np



# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

import QDMpy
from QDMpy._core.models import guess_model, guess_model_name
from QDMpy.odmr.data import ODMRData
from QDMpy._core import models

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

    def __init__(self, odmr: ODMR, model_name: str|None = None) -> None:
        """
        Initialize the Fit class.

        Args:
            odmr_data (ODMRData): Instance of ODMRData containing raw data, dimensions, and frequencies.
            model_name (str): Name of the fitting model. Defaults to 'auto'.
        """
        self.odmr_data = odmr
        self._model = None
        self._constraints = None
        self._initial_parameter = None

        LOG.info("Initializing Fit instance.")

        self.model_name = model_name or guess_model_name(self.odmr_data.raw_data.data)
        self._reset_fit()
        self._constraints = self._set_initial_constraints()

        self.estimator_id = ESTIMATOR_ID[QDMpy.SETTINGS["fit"]["estimator"]]

    @property
    def model_name(self) -> str:
        return self._model["func_name"]

    @model_name.setter
    def model_name(self, name: str) -> None:
        name = name.upper()
        if name not in models.IMPLEMENTED:
            raise ValueError("Unknown model: %s", name)
        self._model = models.IMPLEMENTED[name]
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
        for param in self._model["params"]:
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

        for irange, freq in enumerate(self.odmr_data.processed_data.frequencies):
            LOG.info(
                f"Fitting frequency range {irange}: {freq.min()}-{freq.max()} GHz"
            )
            results = self._fit_range(
                self.odmr_data.processed_data.data[:, irange],
                freq,
                self.initial_parameter[:, irange],
            )
        return results

    def _fit_range(
        self, data: NDArray, freq: NDArray, initial_params: NDArray
    ) -> List[NDArray]:
        """Fit a single frequency range."""
        reshaped_data = data.reshape((-1, data.shape[-1]))
        reshaped_params = initial_params.reshape((-1, self.n_parameters))

        results = gf.fit_constrained(
            data=np.ascontiguousarray(reshaped_data, dtype=np.float32),
            user_info=np.ascontiguousarray(freq, dtype=np.float32),
            constraints=self.get_constraints_array(reshaped_data.shape[0]),
            constraint_types=self.get_constraint_types(),
            initial_parameters=np.ascontiguousarray(reshaped_params, dtype=np.float32),
            model_id=self._model["model_id"],
            max_number_iterations=QDMpy.SETTINGS["fit"]["max_number_iterations"],
            tolerance=QDMpy.SETTINGS["fit"]["tolerance"],
        )
        return results

    def get_constraints_array(self, n_pixel: int) -> NDArray:
        """Build a constraints array."""
        constraints_list = [
            (c[0], c[1]) for c in self._constraints.values()
        ]  # Min, max only
        return np.tile(constraints_list, (n_pixel, 1))

    def get_constraint_types(self) -> NDArray:
        """Get constraint types as array."""
        return np.array(
            [CONSTRAINT_TYPES.index(c[2]) for c in self._constraints.values()],
            dtype=np.int32,
        )

if __name__ == "__main__":
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor
    from QDMpy.odmr.odmr import ODMR

    # User-friendly initialization
    loader = MatlabLoader(data_folder="/home/mike/git/QDMpy/tests/data")
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.process_data()

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('QtAgg')
    print(odmr.raw_data.data.shape)
    plt.plot(odmr.raw_data.data[0,0,: 100])
    plt.show()
    fit = Fit(odmr)
    # print(fit)