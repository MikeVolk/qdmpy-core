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

import json
from pathlib import Path
from typing import Any, Self

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from qdmpy.constants import D_ZFS, GAMMA_NV, POLARITY_LABELS
from qdmpy.exceptions import DataLoadError, DataShapeError, DataValidationError, ParameterError


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
        """Log initialization after Pydantic validation and protect parameter arrays."""
        logger.info("FitResult initialized with model: {}", self.model_name)
        logger.debug("Available parameters: {}", list(self.parameters.keys()))

        # Protect parameter arrays from external mutation to prevent cache invalidation
        for param_array in self.parameters.values():
            if isinstance(param_array, np.ndarray):
                param_array.flags.writeable = False

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
            leading dimensions (polarity, freq_range) are averaged to produce a single
            spatial map.
        """
        param_data = self.get_parameter(param_name)
        height, width = self.scan_dimensions
        if param_data.ndim == 2:
            return param_data
        if param_data.ndim == 4:
            return np.nanmean(param_data.reshape(-1, height, width), axis=0)
        if param_data.ndim == 1:
            return param_data.reshape(height, width)
        return np.nanmean(param_data.reshape(-1, height * width), axis=0).reshape(height, width)

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
            "Pixel count mismatch: data has {} pixels, scan_dims suggest {}. Using ({}, {})",
            n_pixels,
            height * width,
            best_height,
            best_width,
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
        if resonance.ndim == 4:
            n_pol, n_frange, h, w = resonance.shape
            n_pixels = h * w
            resonance = resonance.reshape(n_pol, n_frange, n_pixels)
            logger.debug("Reshaped 4D resonance to shape {}", resonance.shape)
        elif resonance.ndim == 3:
            n_pol, n_frange, n_pixels = resonance.shape
        elif resonance.ndim == 2:
            n_pol = 2
            n_frange = resonance.shape[0] // n_pol
            n_pixels = resonance.shape[1]
            resonance = resonance.reshape((n_pol, n_frange, n_pixels))
            logger.debug("Reshaped 2D resonance to shape {}", resonance.shape)
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

        if n_frange >= 2:
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
        logger.debug("Computing delta resonance for B111 calculations")

        resonance = self.parameters["center"]
        logger.debug("Center parameter shape: {}", resonance.shape)
        resonance, n_pol, n_frange, n_pixels = self._normalize_resonance_shape(resonance)
        height, width = self._resolve_spatial_dims(n_pixels)
        delta = self._calc_delta_from_single_center(resonance, n_pol, n_frange, height, width)

        polarity_coords = POLARITY_LABELS[:n_pol]
        logger.debug("Delta resonance computed with shape: {}", delta.shape)
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
            "B111 remanent: mean={:.2e} uT, std={:.2e} uT",
            b111_remanent.mean(),
            b111_remanent.std(),
        )
        logger.debug(
            "B111 induced: mean={:.2e} uT, std={:.2e} uT",
            b111_induced.mean(),
            b111_induced.std(),
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
            logger.info("Calculating magnetic field from {} fit results", self.model_name)
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
        if center.ndim >= 3 and center.shape[1] > 1:
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

        logger.debug(
            "B-field calculation: mean={:.2e} T, std={:.2e} T",
            b_field.mean(),
            b_field.std(),
        )

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
            "Fit quality metrics: mean_chi2={:.3f}, n_pixels={}",
            metrics["mean_chi2"],
            metrics["n_pixels"],
        )

        return metrics

    def _build_save_dict(self: Self) -> dict[str, NDArray | np.void]:
        """Build a pickle-free save dict for NPZ serialization.

        Returns:
            Dictionary with ``__meta__`` (JSON-encoded as ``np.void``) and
            ``param_{name}`` keys for each fitted parameter, plus optional
            ``cache_*`` keys for cached derived arrays.
        """
        meta = {
            "model_name": self.model_name,
            "scan_dimensions": list(self.scan_dimensions),
            "pixel_spacing": self.pixel_spacing,
            "metadata": self.metadata,
        }
        meta_bytes = json.dumps(meta, default=str).encode()

        save_dict: dict[str, NDArray | np.void] = {
            "__meta__": np.void(meta_bytes),
        }

        for name, arr in self.parameters.items():
            save_dict[f"param_{name}"] = np.asarray(arr)

        if self._b_field_cache is not None:
            save_dict["cache_b_field"] = self._b_field_cache
        if self._delta_resonance_cache is not None:
            save_dict["cache_delta_resonance"] = self._delta_resonance_cache.values
        if self._b111_cache is not None:
            save_dict["cache_b111_remanent"] = self._b111_cache["remanent"].values
            save_dict["cache_b111_induced"] = self._b111_cache["induced"].values

        return save_dict

    def save_results(self: Self, filepath: str | Path) -> None:
        """Save fit results to a pickle-free NPZ file.

        The file uses a flat namespace: ``__meta__`` holds JSON-encoded scalar
        fields and ``param_{name}`` keys hold each fitted parameter array.
        No ``dtype=object`` or pickle is used, so the file can be loaded with
        ``np.load(..., allow_pickle=False)``.

        Args:
            filepath: Path where to save the results.
        """
        filepath = Path(filepath)
        save_dict = self._build_save_dict()
        arrays = {k: np.asarray(v) for k, v in save_dict.items()}
        np.savez_compressed(filepath, allow_pickle=False, **arrays)
        logger.info("Fit results saved to: {}", filepath)

    def plot(
        self: Self,
        param: str = "center",
        *,
        save: bool = False,
        filename: str | None = None,
    ) -> None:
        """Quick-plot a parameter map.

        The special values ``'b111_remanent'`` and ``'b111_induced'`` are
        accepted in addition to raw parameter names; they delegate to
        ``plot_b111_map`` with the appropriate component.

        Args:
            param: Parameter name to visualise (``'center'``, ``'chi2'``,
                ``'contrast'``, ``'b111_remanent'``, ``'b111_induced'``, ...).
            save: If True, save the figure to disk.
            filename: Output filename (uses a default name if None).
        """
        if param in ("b111_remanent", "b111_induced"):
            from qdmpy.plotting import plot_b111_map

            component = "remanent" if param == "b111_remanent" else "induced"
            plot_b111_map(self, component=component, save=save, filename=filename)
        else:
            from qdmpy.plotting import plot_fit_result_parameter_map

            plot_fit_result_parameter_map(self, param, save=save, filename=filename)

    def show(self: Self, *, save: bool = False, filename: str | None = None) -> None:
        """Quick-plot overview of all fitted parameters and B111 maps.

        Args:
            save: If True, save the figure to disk.
            filename: Output filename (uses a default name if None).
        """
        from qdmpy.plotting import plot_fit_result_overview

        plot_fit_result_overview(self, save=save, filename=filename)

    @classmethod
    def _from_npz(cls: type[FitResult], data: Any, *, source: str = "<unknown>") -> FitResult:  # noqa: ANN401
        """Reconstruct a FitResult from an open NpzFile handle.

        Used by both ``load_results`` and ``QDMResult.load`` so the file is
        opened exactly once.

        Args:
            data: NpzFile opened with ``allow_pickle=False``.
            source: File path string for error messages.

        Returns:
            Reconstructed FitResult.

        Raises:
            DataLoadError: If ``__meta__`` is missing, corrupt, or no ``param_*``
                keys are found.
        """
        if "__meta__" not in data.files:
            msg = (
                f"File {source} is missing the __meta__ key. "
                "This file was created with an older format that is no longer supported."
            )
            raise DataLoadError(msg)

        try:
            meta = json.loads(bytes(data["__meta__"]))
        except json.JSONDecodeError as exc:
            msg = f"File {source} has a corrupt __meta__ field: {exc}"
            raise DataLoadError(msg) from exc

        parameters: dict[str, NDArray] = {}
        for key in data.files:
            if key.startswith("param_"):
                param_name = key[len("param_") :]
                parameters[param_name] = data[key]

        if not parameters:
            msg = (
                f"File {source} contains no param_* keys. "
                "The file may be corrupt or from an incompatible format."
            )
            raise DataLoadError(msg)

        return cls(
            parameters=parameters,
            scan_dimensions=tuple(meta["scan_dimensions"]),
            pixel_spacing=float(meta["pixel_spacing"]),
            model_name=str(meta["model_name"]),
            metadata=meta.get("metadata", {}),
        )

    @classmethod
    def load_results(cls: type[FitResult], filepath: str | Path) -> FitResult:
        """Load saved fit results from a pickle-free NPZ file.

        Expects the new format with ``__meta__`` (JSON) and ``param_*`` keys.
        Rejects files that require pickle deserialization.

        Args:
            filepath: Path to the saved results file (NPZ format).

        Returns:
            FitResult instance reconstructed from saved data.

        Raises:
            DataLoadError: If the file does not exist, contains pickle data,
                or is missing the ``__meta__`` key.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            msg = f"Results file not found: {filepath}"
            raise DataLoadError(msg)

        try:
            data = np.load(filepath, allow_pickle=False)
        except ValueError as exc:
            msg = (
                f"File {filepath} contains pickled objects and cannot be loaded safely. "
                "Re-save from the original data using FitResult.save_results() "
                "to convert to the safe format."
            )
            raise DataLoadError(msg) from exc

        result = cls._from_npz(data, source=str(filepath))
        logger.info("Fit results loaded from: {}", filepath)
        return result


class FoldedFitResult(FitResult):
    """FitResult for folded-spectrum fits.

    The folded spectrum has its frequency axis in Zeeman-offset (delta_f) GHz,
    so the fitted centre IS the Zeeman shift directly.  No D_ZFS subtraction.
    """

    def _calc_delta_from_single_center(
        self: Self,
        resonance: NDArray,
        n_pol: int,
        n_frange: int,  # noqa: ARG002
        height: int,
        width: int,
    ) -> NDArray:
        """Calculate delta resonance from folded-domain centre.

        The centre parameter is already delta_f in GHz (Zeeman offset), so we
        divide by GAMMA_NV directly without subtracting D_ZFS.

        Args:
            resonance: 3D array (n_pol, n_frange, n_pixels).
            n_pol: Number of polarities.
            n_frange: Ignored — folded spectra always have a single frequency range.
            height: Spatial height dimension.
            width: Spatial width dimension.

        Returns:
            Array with shape (n_pol, height, width). Sign per polarity:
            pol_0 (neg) gets sign=-1, pol_1 (pos) gets sign=+1.
        """
        d = np.array([-1, 1])[:n_pol].reshape(n_pol, 1, 1)
        freq_shift = resonance[:, 0].reshape(n_pol, height, width)
        return freq_shift / GAMMA_NV * 1e6 * d
