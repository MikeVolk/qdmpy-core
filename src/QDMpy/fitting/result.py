"""Fit results management for Quantum Diamond Microscopy.

Convention: All frequency values are in GHz.

This module provides the FitResult class which encapsulates the results of ODMR spectral
fitting and provides methods for analysis, visualization, and data export. The FitResult
class separates analysis functionality from data management, allowing for clean separation
of concerns and better code organization.

The FitResult class handles:
- Access to fitted parameters (centers, widths, contrasts, etc.)
- Magnetic field calculations from fitted resonance frequencies
- Visualization of spatial parameter maps and magnetic field maps
- Data export and persistence functionality
- Quality assessment and statistical analysis of fit results
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from QDMpy.constants import D_ZFS, GAMMA_NV
from QDMpy.exceptions import DataLoadError, DataShapeError, DataValidationError, ParameterError


class FitResult(BaseModel):
    """Contains ODMR fitting results and provides analysis methods.

    This class encapsulates the results from ODMR spectral fitting using a
    lightweight, data-only approach. It stores only the essential fitted parameters
    and metadata, without maintaining references to heavy objects like FitManager
    or Measurement.

    This design provides:
    - Clean separation of concerns
    - Easy serialization and persistence
    - Minimal memory footprint
    - No circular dependencies

    Attributes:
        parameters: Dictionary of fitted parameters (centers, widths, contrasts, etc.)
        scan_dimensions: Spatial dimensions as (height, width)
        pixel_spacing: Physical spacing between pixels in meters
        model_name: Name of the model used for fitting
        metadata: Additional fitting metadata (quality metrics, etc.)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parameters: dict[str, NDArray]
    scan_dimensions: tuple[int, int]
    pixel_spacing: float = Field(gt=0)
    model_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    _b_field_cache: NDArray | None = PrivateAttr(default=None)
    _delta_resonance_cache: xr.DataArray | None = PrivateAttr(default=None)
    _b111_cache: xr.Dataset | None = PrivateAttr(default=None)

    @field_validator("scan_dimensions")
    @classmethod
    def validate_scan_dimensions(cls: type[FitResult], v: tuple[int, int]) -> tuple[int, int]:
        """Validate that scan dimensions are positive."""
        if v[0] <= 0 or v[1] <= 0:
            msg = f"scan_dimensions must be positive, got {v}"
            raise DataValidationError(msg)
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls: type[FitResult], v: dict[str, NDArray]) -> dict[str, NDArray]:
        """Validate that parameters dict is not empty."""
        if not v:
            msg = "parameters dict must not be empty"
            raise DataValidationError(msg)
        return v

    def model_post_init(self: Self, __context: object) -> None:
        """Log initialization after Pydantic validation."""
        logger.info(f"FitResult initialized with model: {self.model_name}")
        logger.debug(f"Available parameters: {list(self.parameters.keys())}")

    def __repr__(self: Self) -> str:
        """Return string representation of FitResult."""
        n_pixels = self.scan_dimensions[0] * self.scan_dimensions[1]
        n_params = len(self.parameters)
        return f"FitResult(model='{self.model_name}', n_pixels={n_pixels}, parameters={n_params})"

    @property
    def centers(self: Self) -> NDArray:
        """Get resonance center frequencies in GHz.

        Returns:
            Array of center frequencies with shape matching spatial dimensions
        """
        return self.parameters["center"]

    @property
    def linewidths(self: Self) -> NDArray:
        """Get ODMR linewidths in Hz.

        For models with multiple lines, returns the primary linewidth.

        Returns:
            Array of linewidths with shape matching spatial dimensions
        """
        width_0 = self.parameters.get("width_0")
        if width_0 is not None:
            return width_0
        width = self.parameters.get("width")
        if width is not None:
            return width
        msg = "No linewidth parameter found ('width_0' or 'width')"
        raise ParameterError(msg)

    @property
    def contrasts(self: Self) -> NDArray:
        """Get ODMR contrasts (normalized).

        For models with multiple contrast parameters (ESR14N, ESR15N), returns
        the first contrast (``contrast_0``).  For ESRSINGLE returns ``contrast``.

        Returns:
            Array of primary contrast values with shape ``(n_pol, n_frange, n_pixel)``.

        Raises:
            ParameterError: If no contrast parameter is found.
        """
        contrast = self.parameters.get("contrast")
        if contrast is not None:
            return contrast
        contrast_0 = self.parameters.get("contrast_0")
        if contrast_0 is not None:
            return contrast_0
        msg = "No contrast parameter found ('contrast' or 'contrast_0')"
        raise ParameterError(msg)

    @property
    def offsets(self: Self) -> NDArray:
        """Get baseline offsets.

        Returns:
            Array of offset values with shape matching spatial dimensions
        """
        return self.parameters.get("offset", np.zeros_like(self.centers))

    @property
    def chi2(self: Self) -> NDArray:
        """Get fit quality (chi-squared values).

        Returns:
            Array of chi-squared values with shape matching spatial dimensions
        """
        return self.parameters["chi2"]

    @property
    def fit_states(self: Self) -> NDArray:
        """Get fitting convergence states.

        Returns:
            Array of fit state codes with shape matching spatial dimensions
        """
        return self.parameters.get("states", np.zeros_like(self.centers, dtype=int))

    def get_parameter(self: Self, param_name: str) -> NDArray:
        """Get any fitted parameter by name.

        Args:
            param_name: Name of the parameter to retrieve

        Returns:
            Array of parameter values

        Raises:
            ParameterError: If parameter name is not found
        """
        if param_name not in self.parameters:
            available = list(self.parameters.keys())
            msg = f"Parameter '{param_name}' not found. Available: {available}"
            raise ParameterError(msg)
        return self.parameters[param_name]

    def get_parameter_map(self: Self, param_name: str) -> NDArray:
        """Get parameter reshaped as 2D spatial map.

        Args:
            param_name: Name of the parameter to retrieve

        Returns:
            2D array with shape (height, width) for spatial visualization

        Note:
            For multi-range models (e.g. ESR14N with center shape (n_pol, n_frange, n_pixel)),
            the parameter is flattened to 1D before reshaping to the spatial dimensions.
        """
        param_data = self.get_parameter(param_name)
        # If parameter is multi-dimensional (e.g., 3D for multi-range models),
        # flatten it first to get a 1D array of n_pixels
        if param_data.ndim > 1:
            param_data = param_data.reshape(-1)
        return param_data.reshape(self.scan_dimensions)

    @property
    def delta_resonance(self: Self) -> xr.DataArray:
        """Get the signed frequency difference per polarity.

        Returns:
            xr.DataArray with dims ('polarity', 'y', 'x') and polarity coords
            'neg'/'pos'. Values are in µT with polarity-correct sign applied:
            neg polarity gets sign=-1, pos polarity gets sign=+1.
        """
        if self._delta_resonance_cache is None:
            self._delta_resonance_cache = self._compute_delta_resonance()
        return self._delta_resonance_cache

    def _resolve_spatial_dims(self: Self, n_pixels: int) -> tuple[int, int]:
        """Resolve spatial dimensions from pixel count.

        If n_pixels matches scan_dimensions, returns them directly.
        Otherwise finds the factor pair closest to the original aspect ratio.

        Args:
            n_pixels: Total number of pixels in the data.

        Returns:
            Tuple of (height, width) for spatial reshaping.
        """
        height, width = self.scan_dimensions
        if n_pixels == height * width:
            return height, width

        aspect_ratio = width / height
        factors = []
        for i in range(1, int(np.sqrt(n_pixels)) + 1):
            if n_pixels % i == 0:
                factors.append((i, n_pixels // i))

        if factors:
            best_height, best_width = min(factors, key=lambda f: abs(f[1] / f[0] - aspect_ratio))
        else:
            best_height = int(np.sqrt(n_pixels))
            best_width = n_pixels // best_height
            if best_height * best_width != n_pixels:
                best_height, best_width = n_pixels, 1

        logger.debug(
            f"Pixel count mismatch: data has {n_pixels} pixels, "
            f"scan_dims suggest {height * width}. Using ({best_height}, {best_width})"
        )
        return best_height, best_width

    def _normalize_resonance_shape(self: Self, resonance: NDArray) -> tuple[NDArray, int, int, int]:
        """Normalize resonance array to 3D (n_pol, n_frange, n_pixels).

        Handles 4D (squeeze), 3D (passthrough), and 2D (reshape with n_pol=2).

        Args:
            resonance: Center parameter array of varying dimensionality.

        Returns:
            Tuple of (resonance_3d, n_pol, n_frange, n_pixels).

        Raises:
            DataShapeError: If resonance has an unexpected number of dimensions.
        """
        if resonance.ndim == 4:  # noqa: PLR2004
            n_pol, n_frange, n_pixels, _ = resonance.shape
            resonance = np.squeeze(resonance, axis=-1)
            logger.debug(f"Squeezed 4D resonance to shape {resonance.shape}")
        elif resonance.ndim == 3:  # noqa: PLR2004
            n_pol, n_frange, n_pixels = resonance.shape
        elif resonance.ndim == 2:  # noqa: PLR2004
            n_pol = 2
            n_frange = resonance.shape[0] // n_pol
            n_pixels = resonance.shape[1]
            resonance = resonance.reshape((n_pol, n_frange, n_pixels))
            logger.debug(f"Reshaped 2D resonance to shape {resonance.shape}")
        else:
            msg = f"Unexpected center parameter shape: {resonance.shape}"
            raise DataShapeError(msg)

        return resonance, n_pol, n_frange, n_pixels

    def _calc_delta_from_single_center(
        self: Self,
        resonance: NDArray,
        n_pol: int,
        n_frange: int,
        height: int,
        width: int,
    ) -> NDArray:
        """Calculate delta resonance from a single center parameter.

        For n_frange >= 2, computes frequency difference between ranges.
        For n_frange < 2, computes shift from zero-field splitting (D_ZFS).

        Args:
            resonance: 3D array (n_pol, n_frange, n_pixels).
            n_pol: Number of polarities.
            n_frange: Number of frequency ranges.
            height: Spatial height dimension.
            width: Spatial width dimension.

        Returns:
            Array with shape (n_pol, height, width). Sign is applied per polarity:
            pol_0 (neg) gets sign=-1, pol_1 (pos) gets sign=+1.
        """
        d = np.array([-1, 1])[:n_pol].reshape(n_pol, 1, 1)

        if n_frange >= 2:  # noqa: PLR2004
            freq_diff = (resonance[:, 1] - resonance[:, 0]).reshape(n_pol, height, width)
            return freq_diff / 2 / GAMMA_NV * 1e6 * d

        freq_shift = (resonance[:, 0] - D_ZFS).reshape(n_pol, height, width)
        return freq_shift / GAMMA_NV * 1e6 * d


    def _compute_delta_resonance(self: Self) -> xr.DataArray:
        """Compute signed frequency difference per polarity.

        Returns:
            xr.DataArray with dims ('polarity', 'y', 'x') and polarity coords
            'neg'/'pos'. Values in µT with polarity-correct sign applied.
        """
        from QDMpy.constants import POLARITY_LABELS

        logger.debug("Computing delta resonance for B111 calculations")

        resonance = self.parameters["center"]
        logger.debug(f"Center parameter shape: {resonance.shape}")
        resonance, n_pol, n_frange, n_pixels = self._normalize_resonance_shape(resonance)
        height, width = self._resolve_spatial_dims(n_pixels)
        delta = self._calc_delta_from_single_center(resonance, n_pol, n_frange, height, width)

        polarity_coords = POLARITY_LABELS[:n_pol]
        logger.debug(f"Delta resonance computed with shape: {delta.shape}")
        return xr.DataArray(
            delta,
            dims=("polarity", "y", "x"),
            coords={"polarity": polarity_coords},
            attrs={"units": "µT", "description": "signed dB per polarity"},
        )

    @property
    def b111(self: Self) -> xr.Dataset:
        """Get B111 magnetic field components as a named Dataset.

        Returns:
            xr.Dataset with variables 'remanent' and 'induced', each a
            DataArray with dims ('y', 'x') and units='µT'.
        """
        if self._b111_cache is None:
            self._b111_cache = self._compute_b111()
        return self._b111_cache

    def _compute_b111(self: Self) -> xr.Dataset:
        """Compute B111 magnetic field components.

        Uses the exact algorithm from the old QDM class to calculate remanent
        (permanent) and induced magnetic field components.

        Returns:
            xr.Dataset with 'remanent' and 'induced' DataArrays in µT.

        Raises:
            DataShapeError: If delta_resonance does not contain both 'neg' and 'pos'
                polarity coordinates (requires n_pol == 2).
        """
        logger.info("Computing B111 magnetic field components")

        delta_res = self.delta_resonance  # xr.DataArray (polarity, y, x)
        polarity_vals = list(delta_res.coords["polarity"].values)

        if "neg" not in polarity_vals or "pos" not in polarity_vals:
            msg = f"B111 requires both 'neg' and 'pos' polarities; found: {polarity_vals}"
            raise DataShapeError(msg)

        neg_diff = delta_res.sel(polarity="neg").values  # (height, width)
        pos_diff = delta_res.sel(polarity="pos").values  # (height, width)

        b111_remanent = (neg_diff + pos_diff) / 2
        b111_induced = (neg_diff - pos_diff) / 2

        logger.debug(
            f"B111 remanent: mean={b111_remanent.mean():.2e} µT, std={b111_remanent.std():.2e} µT"
        )
        logger.debug(
            f"B111 induced: mean={b111_induced.mean():.2e} µT, std={b111_induced.std():.2e} µT"
        )

        return xr.Dataset(
            {
                "remanent": xr.DataArray(b111_remanent, dims=("y", "x"), attrs={"units": "µT"}),
                "induced": xr.DataArray(b111_induced, dims=("y", "x"), attrs={"units": "µT"}),
            }
        )

    @property
    def b111_remanent(self: Self) -> NDArray:
        """Remanent B111 field component in µT as a 2D numpy array."""
        return self.b111["remanent"].values

    @property
    def b111_induced(self: Self) -> NDArray:
        """Induced B111 field component in µT as a 2D numpy array."""
        return self.b111["induced"].values

    def calculate_b_field(self: Self, force_recalculate: bool = False) -> NDArray:
        """Calculate magnetic field map from fitted resonance frequencies.

        This method computes the magnetic field strength based on the Zeeman
        splitting of NV center resonances.

        Args:
            force_recalculate: If True, recalculate even if cached result exists

        Returns:
            2D array of magnetic field values in Tesla

        Note:
            The calculation assumes the standard NV center gyromagnetic ratio
            and appropriate model-specific conversion factors.
        """
        if self._b_field_cache is None or force_recalculate:
            logger.info(f"Calculating magnetic field from {self.model_name} fit results")
            self._b_field_cache = self._compute_b_field()

        return self._b_field_cache

    def _compute_b_field(self: Self) -> NDArray:
        """Internal method to compute magnetic field from resonance frequencies.

        Returns:
            2D array of magnetic field values in Tesla

        Raises:
            ParameterError: If the model has multiple frequency ranges (use b111 instead).
        """
        # Check if this is a multi-range model (n_frange > 1)
        center = self.parameters["center"]
        if center.ndim >= 3 and center.shape[1] > 1:  # shape is (n_pol, n_frange, n_pixel)
            msg = (
                f"calculate_b_field() does not support multi-range models "
                f"(center shape: {center.shape} has n_frange={center.shape[1]} > 1). "
                "Use the b111 property instead, which correctly handles "
                "frequency-splitting-based B-field calculations."
            )
            raise ParameterError(msg)

        # Get center frequencies and reshape to spatial map
        centers_map = self.get_parameter_map("center")

        # Calculate magnetic field: |B| = |f_center - D| / gamma
        # Centers are in GHz, GAMMA_NV is GHz/T, D_ZFS is GHz -> result in T
        b_field = np.abs(centers_map - D_ZFS) / GAMMA_NV

        logger.debug(f"B-field calculation: mean={b_field.mean():.2e} T, std={b_field.std():.2e} T")

        return b_field

    def get_fit_quality_metrics(self: Self) -> dict[str, float]:
        """Calculate overall fit quality metrics.

        Returns:
            Dictionary containing various quality metrics
        """
        chi2_values = self.chi2

        # Calculate basic statistics
        metrics = {
            "mean_chi2": float(np.mean(chi2_values)),
            "median_chi2": float(np.median(chi2_values)),
            "std_chi2": float(np.std(chi2_values)),
            "n_pixels": int(chi2_values.size),
        }

        # Add convergence rate if states are available
        if "states" in self.parameters:
            states_values = self.fit_states
            metrics.update(
                {
                    "convergence_rate": float(np.mean(states_values == 0)),
                    "n_converged": int(np.sum(states_values == 0)),
                }
            )

        # Add any pre-computed metrics from metadata
        if "quality_metrics" in self.metadata:
            metrics.update(self.metadata["quality_metrics"])

        logger.info(
            f"Fit quality metrics: mean_chi2={metrics['mean_chi2']:.3f}, "
            f"n_pixels={metrics['n_pixels']}"
        )

        return metrics

    def save_results(self: Self, filepath: str | Path) -> None:
        """Save fit results to file.

        Args:
            filepath: Path where to save the results

        Note:
            This saves the essential fit results in a format that can be
            reloaded later. The full FitManager state may not be preserved.
        """
        filepath = Path(filepath)

        # Prepare data for saving
        save_data = {
            "model_name": self.model_name,
            "scan_dimensions": self.scan_dimensions,
            "pixel_spacing": self.pixel_spacing,
            "metadata": self.metadata,
            "parameters": self.parameters.copy(),  # Copy all parameters
        }

        # Add cached calculations if available
        if self._b_field_cache is not None:
            save_data["b_field"] = self._b_field_cache
        if self._delta_resonance_cache is not None:
            save_data["delta_resonance"] = self._delta_resonance_cache.values
        if self._b111_cache is not None:
            save_data["b111_remanent"] = self._b111_cache["remanent"].values
            save_data["b111_induced"] = self._b111_cache["induced"].values

        # Save to NPZ format for efficiency
        # Extract numpy arrays and convert other types to numpy-compatible formats
        numpy_save_data = {}
        for key, value in save_data.items():
            if isinstance(value, (np.ndarray, np.number, int, float, str, bool)):
                numpy_save_data[key] = value
            elif isinstance(value, dict):
                # Convert dict to array for numpy compatibility
                numpy_save_data[key] = np.array([value], dtype=object)
            else:
                # Convert other types to arrays
                numpy_save_data[key] = np.array(value)

        np.savez_compressed(filepath, **numpy_save_data)
        logger.info(f"Fit results saved to: {filepath}")

    @classmethod
    def load_results(cls: type[FitResult], filepath: str | Path) -> dict[str, Any]:
        """Load saved fit results from file.

        Args:
            filepath: Path to the saved results file

        Returns:
            Dictionary containing loaded results data

        Note:
            This returns the raw data dictionary. Creating a full FitResult
            object would require reconstructing the FitManager and Measurement,
            which may not always be possible or desired.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            msg = f"Results file not found: {filepath}"
            raise DataLoadError(msg)

        data = np.load(filepath, allow_pickle=True)

        # Convert back to regular dict
        result_data = {key: data[key] for key in data.files}

        logger.info(f"Fit results loaded from: {filepath}")
        return result_data
