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
from loguru import logger
from numpy.typing import NDArray

from QDMpy.constants import D_ZFS, GAMMA_NV


class FitResult:
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

    def __init__(
        self: Self,
        parameters: dict[str, NDArray],
        scan_dimensions: tuple[int, int],
        pixel_spacing: float,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FitResult with extracted fit data.

        Args:
            parameters: Dictionary of fitted parameters with keys like 'center',
                       'width_0', 'contrast', 'offset', 'chi2', 'states'
            scan_dimensions: Spatial dimensions as (height, width)
            pixel_spacing: Physical spacing between pixels in meters
            model_name: Name of the model used for fitting
            metadata: Optional additional metadata dictionary
        """
        self.parameters = parameters
        self.scan_dimensions = scan_dimensions
        self.pixel_spacing = pixel_spacing
        self.model_name = model_name
        self.metadata = metadata or {}

        # Cache for expensive calculations
        self._b_field_cache: NDArray | None = None
        self._delta_resonance_cache: NDArray | None = None
        self._b111_cache: tuple[NDArray, NDArray] | None = None

        logger.info(f"FitResult initialized with model: {model_name}")
        logger.debug(f"Available parameters: {list(parameters.keys())}")

    def __repr__(self: Self) -> str:
        """Return string representation of FitResult."""
        n_pixels = self.scan_dimensions[0] * self.scan_dimensions[1]
        n_params = len(self.parameters)
        return (
            f"FitResult(model='{self.model_name}', "
            f"n_pixels={n_pixels}, "
            f"parameters={n_params})"
        )

    @property
    def centers(self: Self) -> NDArray:
        """Get resonance center frequencies in Hz.

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
        raise KeyError("No linewidth parameter found ('width_0' or 'width')")

    @property
    def contrasts(self: Self) -> NDArray:
        """Get ODMR contrasts (normalized).

        Returns:
            Array of contrast values with shape matching spatial dimensions
        """
        return self.parameters["contrast"]

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
            KeyError: If parameter name is not found
        """
        if param_name not in self.parameters:
            available = list(self.parameters.keys())
            raise KeyError(f"Parameter '{param_name}' not found. Available: {available}")
        return self.parameters[param_name]

    def get_parameter_map(self: Self, param_name: str) -> NDArray:
        """Get parameter reshaped as 2D spatial map.

        Args:
            param_name: Name of the parameter to retrieve

        Returns:
            2D array with shape (height, width) for spatial visualization
        """
        param_data = self.get_parameter(param_name)
        return param_data.reshape(self.scan_dimensions)

    @property
    def delta_resonance(self: Self) -> NDArray:
        """Get the frequency difference between resonance peaks.

        This calculates the splitting between high and low frequency resonances,
        which is essential for B111 magnetic field calculations.

        Returns:
            Array with shape (n_pol, 2, height, width) containing frequency differences

        Note:
            For models with 2+ peaks, calculates splitting. For single peak models,
            returns frequency shift from zero-field position.
        """
        if self._delta_resonance_cache is None:
            self._delta_resonance_cache = self._compute_delta_resonance()
        return self._delta_resonance_cache

    def _compute_delta_resonance(self: Self) -> NDArray:  # noqa: C901, PLR0912, PLR0915
        """Compute frequency difference between resonance peaks.

        Implements the exact calculation from the old QDM class:
        delta_resonance = (resonance[:, 1] - resonance[:, 0]) / 2 / GAMMA * d
        where resonance[:, 1] is high freq range, resonance[:, 0] is low freq range
        and d = [-1, 1] for negative and positive field directions

        Returns:
            Array with shape (n_pol, 2, height, width) for spatial maps
        """
        logger.debug("Computing delta resonance for B111 calculations")

        # Look for single 'center' parameter with multiple frequency ranges
        if "center" in self.parameters:
            # Standard case: center parameter with shape (n_pol, n_frange, n_pixels)
            resonance = self.parameters["center"]
            logger.debug(f"Center parameter shape: {resonance.shape}")

            # Handle variable shapes from fit results
            if len(resonance.shape) == 4:
                # Shape is (n_pol, n_frange, n_pixels, 1) - squeeze last dimension
                n_pol, n_frange, n_pixels, _ = resonance.shape
                resonance = np.squeeze(resonance, axis=-1)  # Remove last dimension
                logger.debug(
                    f"Squeezed center parameter from "
                    f"{self.parameters['center'].shape} to {resonance.shape}"
                )
            elif len(resonance.shape) == 3:
                n_pol, n_frange, n_pixels = resonance.shape
            elif len(resonance.shape) == 2:
                # Shape might be (n_pol * n_frange, n_pixels) - reshape to 3D
                n_pol = 2  # assume 2 polarities
                total_pol_frange = resonance.shape[0]
                n_frange = total_pol_frange // n_pol
                n_pixels = resonance.shape[1]
                resonance = resonance.reshape((n_pol, n_frange, n_pixels))
                logger.debug(
                    f"Reshaped center parameter from "
                    f"{self.parameters['center'].shape} to {resonance.shape}"
                )
            else:
                raise ValueError(f"Unexpected center parameter shape: {resonance.shape}")

            # Calculate actual spatial dimensions from pixel count and scan_dimensions ratio
            height, width = self.scan_dimensions
            expected_pixels = height * width

            if n_pixels != expected_pixels:
                # Find the best 2D factorization of n_pixels
                # Try to find factors closest to the original aspect ratio
                aspect_ratio = width / height

                # Find all factor pairs
                factors = []
                for i in range(1, int(np.sqrt(n_pixels)) + 1):
                    if n_pixels % i == 0:
                        factors.append((i, n_pixels // i))

                if factors:
                    # Choose factor pair with aspect ratio closest to original
                    best_factors = min(factors, key=lambda f: abs(f[1] / f[0] - aspect_ratio))
                    adjusted_height, adjusted_width = best_factors
                else:
                    # Fallback: use sqrt approach
                    adjusted_height = int(np.sqrt(n_pixels))
                    adjusted_width = n_pixels // adjusted_height
                    if adjusted_height * adjusted_width != n_pixels:
                        adjusted_height, adjusted_width = n_pixels, 1

                logger.debug(
                    f"Pixel count mismatch: data has {n_pixels} pixels, "
                    f"scan_dims suggest {expected_pixels}. "
                    f"Using ({adjusted_height}, {adjusted_width})"
                )
                height, width = adjusted_height, adjusted_width

            if n_frange >= 2:
                # We have at least 2 frequency ranges (low and high)
                # Calculate difference: high_freq - low_freq
                freq_diff = resonance[:, 1] - resonance[:, 0]  # Shape: (n_pol, n_pixels)

                # Reshape to spatial dimensions
                freq_diff = freq_diff.reshape(
                    (n_pol, height, width)
                )  # Shape: (n_pol, height, width)

                # Convert to magnetic field and apply polarity factor
                # Original formula: (resonance[:, 1] - resonance[:, 0]) / 2 / GAMMA * d
                d = np.array([-1, 1])  # negative and positive field directions

                # Broadcast to create (n_pol, 2, height, width)
                # where the "2" dimension represents [negative_diff, positive_diff]
                delta_resonance = np.zeros((n_pol, 2, height, width))

                for pol in range(n_pol):
                    for direction in range(2):
                        delta_resonance[pol, direction] = (
                            freq_diff[pol] / 2 / GAMMA_NV * 1e6 * d[direction]
                        )

            else:
                # Single frequency range - use frequency shift from zero field
                zero_field_freq = D_ZFS
                freq_shift = resonance[:, 0] - zero_field_freq  # Shape: (n_pol, n_pixels)

                # Reshape to spatial dimensions
                freq_shift = freq_shift.reshape(
                    (n_pol, height, width)
                )  # Shape: (n_pol, height, width)

                d = np.array([-1, 1])
                delta_resonance = np.zeros((n_pol, 2, height, width))

                for pol in range(n_pol):
                    for direction in range(2):
                        delta_resonance[pol, direction] = (
                            freq_shift[pol] / GAMMA_NV * 1e6 * d[direction]
                        )

        else:
            # Multiple center parameters (center_0, center_1, etc.)
            center_params = {k: v for k, v in self.parameters.items() if k.startswith("center")}

            if len(center_params) >= 2:
                # Sort parameters by index to ensure correct order
                sorted_items = sorted(
                    center_params.items(),
                    key=lambda x: int(x[0].split("_")[1]) if "_" in x[0] else 0,
                )

                # Assume first is low freq, second is high freq
                low_freq_centers = sorted_items[0][1]  # Shape: (n_pol, n_frange, n_pixels)
                high_freq_centers = sorted_items[1][1]  # Shape: (n_pol, n_frange, n_pixels)

                # Get actual pixel count
                n_pixels = low_freq_centers.shape[-1]
                height, width = self.scan_dimensions
                expected_pixels = height * width

                if n_pixels != expected_pixels:
                    # Find the best 2D factorization of n_pixels
                    aspect_ratio = width / height

                    # Find all factor pairs
                    factors = []
                    for i in range(1, int(np.sqrt(n_pixels)) + 1):
                        if n_pixels % i == 0:
                            factors.append((i, n_pixels // i))

                    if factors:
                        # Choose factor pair with aspect ratio closest to original
                        best_factors = min(factors, key=lambda f: abs(f[1] / f[0] - aspect_ratio))
                        adjusted_height, adjusted_width = best_factors
                    else:
                        # Fallback: use sqrt approach
                        adjusted_height = int(np.sqrt(n_pixels))
                        adjusted_width = n_pixels // adjusted_height
                        if adjusted_height * adjusted_width != n_pixels:
                            adjusted_height, adjusted_width = n_pixels, 1

                    height, width = adjusted_height, adjusted_width

                # Calculate frequency difference between frequency ranges
                # Take the difference within each frequency range if multiple ranges exist
                if low_freq_centers.shape[1] >= 2:
                    # Use difference between high and low frequency ranges
                    freq_diff_low = (
                        high_freq_centers[:, 1] - low_freq_centers[:, 0]
                    )  # Cross-range difference
                    freq_diff_high = (
                        high_freq_centers[:, 0] - low_freq_centers[:, 1]
                    )  # Cross-range difference
                    freq_diff = (freq_diff_low + freq_diff_high) / 2  # Average
                else:
                    # Simple difference between the two center parameters
                    freq_diff = (
                        high_freq_centers[:, 0] - low_freq_centers[:, 0]
                    )  # Shape: (n_pol, n_pixels)

                # Reshape to spatial dimensions
                freq_diff = freq_diff.reshape(
                    (freq_diff.shape[0], height, width)
                )  # Shape: (n_pol, height, width)

                d = np.array([-1, 1])
                delta_resonance = np.zeros((freq_diff.shape[0], 2, height, width))

                for pol in range(freq_diff.shape[0]):
                    for direction in range(2):
                        delta_resonance[pol, direction] = (
                            freq_diff[pol] / 2 / GAMMA_NV * 1e6 * d[direction]
                        )

            else:
                raise ValueError(
                    f"Insufficient center parameters for delta resonance "
                    f"calculation. Found: {len(center_params)}"
                )

        logger.debug(f"Delta resonance computed with shape: {delta_resonance.shape}")
        return delta_resonance

    @property
    def b111(self: Self) -> tuple[NDArray, NDArray]:
        """Get B111 magnetic field components (remanent and induced).

        This is the core magnetic field calculation for quantum diamond microscopy,
        extracting the magnetic field components along the [111] crystal direction.

        Returns:
            Tuple of (b111_remanent, b111_induced) arrays with spatial dimensions

        Note:
            Implements the exact calculation from the old QDM class:
            b111_remanent = (neg_difference + pos_difference) / 2
            b111_induced = (neg_difference - pos_difference) / 2
        """
        if self._b111_cache is None:
            self._b111_cache = self._compute_b111()
        return self._b111_cache

    def _compute_b111(self: Self) -> tuple[NDArray, NDArray]:
        """Compute B111 magnetic field components.

        Uses the exact algorithm from the old QDM class to calculate remanent
        (permanent) and induced magnetic field components.

        Returns:
            Tuple of (remanent_field, induced_field) arrays
        """
        logger.info("Computing B111 magnetic field components")

        # Get delta resonance
        delta_res = self.delta_resonance
        logger.debug(f"Delta resonance shape for B111 calculation: {delta_res.shape}")

        # Handle the expected shape: (n_pol, 2, height, width)
        # where the "2" dimension represents [negative_diff, positive_diff]
        if delta_res.ndim == 4:  # (n_pol, 2, height, width)
            if delta_res.shape[1] == 2:  # (n_pol, 2, height, width)
                # The "2" dimension is the negative/positive difference
                neg_difference = delta_res[:, 0, :, :]  # (n_pol, height, width)
                pos_difference = delta_res[:, 1, :, :]  # (n_pol, height, width)

                # Average over polarities to get final spatial maps
                neg_diff = np.mean(neg_difference, axis=0)  # (height, width)
                pos_diff = np.mean(pos_difference, axis=0)  # (height, width)

            elif delta_res.shape[0] == 2:  # (2, n_pol, height, width)
                # The first dimension is negative/positive difference
                neg_difference = delta_res[0]  # (n_pol, height, width)
                pos_difference = delta_res[1]  # (n_pol, height, width)

                # Average over polarities
                neg_diff = np.mean(neg_difference, axis=0)  # (height, width)
                pos_diff = np.mean(pos_difference, axis=0)  # (height, width)
            else:
                raise ValueError(f"Cannot interpret delta_resonance shape: {delta_res.shape}")

        elif delta_res.ndim == 3:  # (2, height, width)
            neg_diff = delta_res[0]  # (height, width)
            pos_diff = delta_res[1]  # (height, width)

        else:
            raise ValueError(f"Unexpected delta_resonance shape: {delta_res.shape}")

        # Apply the exact B111 calculation from old QDM class
        b111_remanent = (neg_diff + pos_diff) / 2
        b111_induced = (neg_diff - pos_diff) / 2

        logger.debug(
            f"B111 remanent field: mean={b111_remanent.mean():.2e} μT, "
            f"std={b111_remanent.std():.2e} μT"
        )
        logger.debug(
            f"B111 induced field: mean={b111_induced.mean():.2e} μT, "
            f"std={b111_induced.std():.2e} μT"
        )

        return b111_remanent, b111_induced

    @property
    def b111_remanent(self: Self) -> NDArray:
        """Get the remanent (permanent) B111 magnetic field component.

        The remanent field represents the permanent magnetic field component
        along the [111] crystal direction, typically from magnetized materials.

        Returns:
            2D array of remanent field values in microTesla with spatial dimensions
        """
        return self.b111[0]

    @property
    def b111_induced(self: Self) -> NDArray:
        """Get the induced B111 magnetic field component.

        The induced field represents the field component that varies with
        external magnetic fields or current-carrying conductors.

        Returns:
            2D array of induced field values in microTesla with spatial dimensions
        """
        return self.b111[1]

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
        """
        # Get center frequencies and reshape to spatial map
        centers_map = self.get_parameter_map("center")

        # Calculate magnetic field: |B| = |f_center - D| / γ
        # Centers are in GHz, GAMMA_NV is GHz/T, D_ZFS is GHz → result in T
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
            save_data["delta_resonance"] = self._delta_resonance_cache
        if self._b111_cache is not None:
            save_data["b111_remanent"] = self._b111_cache[0]
            save_data["b111_induced"] = self._b111_cache[1]

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
    def load_results(cls, filepath: str | Path) -> dict[str, Any]:
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
            raise FileNotFoundError(f"Results file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)

        # Convert back to regular dict
        result_data = {key: data[key] for key in data.files}

        logger.info(f"Fit results loaded from: {filepath}")
        return result_data
