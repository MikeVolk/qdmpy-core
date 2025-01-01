import logging
import numpy as np
import sys
import os

from pathlib import Path
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Add the `src` directory to sys.path for local imports if the script is run directly
if not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

from QDMpy.odmr.odmr import ODMR

LOG = logging.getLogger(__name__)


class Measurement:
    """
    The Measurement class encapsulates all data and processing related to a single QDM
    (Quantum Diamond Microscope) measurement.

    It manages:
        - Raw and processed ODMR data using the ODMR instance.
        - Associated images (light and laser).
        - Fitting operations via external fitting instances (to be integrated later).

    Attributes:
        odmr (ODMR): Instance managing ODMR data and processing.
        light (np.ndarray): Light image array.
        laser (np.ndarray): Laser image array.
        working_directory (Path): Path to the working directory.
        pixel_size (float): Pixel size in meters.
    """

    def __init__(
        self,
        odmr: ODMR,
        light_image: np.ndarray,
        laser_image: np.ndarray,
        output_directory: Union[str, Path],
        pixel_size: float = 4e-6,
        fit_model: str = "auto",
    ) -> None:
        """
        Initialize the Measurement object.

        Args:
            odmr_data (ODMR): An initialized ODMR instance containing ODMR data.
            light_image (np.ndarray): Light image array.
            laser_image (np.ndarray): Laser image array.
            output_directory (Union[str, Path]): Path to the output directory.
            pixel_spacing (float): Spacing between pixels in meters. Default is 4 µm (4e-6).
            fit_model (str): Model name for fitting. Default is "auto".
                            If "auto", the model is chosen based on the mean ODMR data.
        """
        LOG.info("Initializing Measurement object.")
        LOG.info('Output directory: "%s"', output_directory)

        self.output_directory = Path(output_directory)
        self.pixel_spacing = pixel_size

        # Store the ODMR instance
        LOG.debug("Setting ODMR data.")
        self.odmr = odmr

        # Validate ODMR instance data
        LOG.debug("ODMR raw data shape: %s", self.odmr.raw_data.shape)
        LOG.debug("ODMR processed data shape: %s", self.odmr.processed_data.shape)
        LOG.debug("ODMR frequencies: {self.odmr.raw_data.frequencies.shape}")

        # # Initialize outlier mask
        # LOG.debug("Initializing outlier mask.")
        # self._outliers = np.ones(self.odmr.raw_data.shape, dtype=bool)

        # # Store light and laser images
        # LOG.debug("Storing light and laser images.")
        # self.light_image = light_image
        # self.laser_image = laser_image

        # # Initialize B111 field and fit model
        # LOG.debug("Initializing B111 field and fit model.")
        # self._B111 = None
        # self._fit = self.new_fit(fit_model=fit_model)


if __name__ == "__main__":
    from QDMpy.odmr.data import ODMRData
    from QDMpy.odmr.io import MatlabLoader
    from QDMpy.odmr.processors import BinningProcessor

    # User-friendly initialization
    loader = MatlabLoader(data_folder="/home/mike/git/QDMpy/tests/data")
    odmr_data = ODMRData.from_loader(loader=loader)
    odmr = ODMR(odmr_data)
    odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
    odmr.process_data()

    measure = Measurement(
        odmr,
        "/home/git/QDMpy/tests/test_data/light.csv",
        "/home/git/QDMpy/tests/test_data/laser.csv",
        "/home/tst"
    )
    print(measure)
