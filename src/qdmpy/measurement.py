"""Comprehensive measurement management for Quantum Diamond Microscopy.

This module provides the central `Measurement` class that serves as the primary interface
for working with Quantum Diamond Microscope (QDM) experiments. Key capabilities include:

- Data integration: Combines ODMR spectral data with optical images
- Spatial analysis: Maps spectral properties across the spatial dimensions
- Image processing: Handles light and laser reference images
- Metadata tracking: Maintains experiment parameters and processing history
- Output management: Organizes results in a structured directory hierarchy
- Statistical analysis: Identifies outliers and performs data quality assessment

The Measurement class integrates data from the ODMR module with optical images and
provides a unified interface for analysis and visualization of QDM experiments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.exceptions import DataNotLoadedError
from qdmpy.io import get_image
from qdmpy.measurement_workflows import (
    fit_folded_measurement_odmr,
    fit_measurement_odmr,
    fold_measurement_odmr,
    load_measurement_folder_data,
    refit_measurement_result,
    validate_processed_odmr,
)
from qdmpy.odmr.folding import FoldedODMR, FoldingSettings, SpectralFolder
from qdmpy.odmr.manager import ODMR
from qdmpy.settings import QDMpySettings

# Sentinel for parameters not explicitly set by the caller, used to distinguish
# "user passed None (meaning: skip)" from "user passed nothing (meaning: use default)".
_UNSET: object = object()

if TYPE_CHECKING:
    from os import PathLike

    from qdmpy.fitting.refit import RefitSettings
    from qdmpy.odmr.data import ODMRData
    from qdmpy.result import QDMResult
    from qdmpy.settings import QDMpySettings


class Measurement:
    """Encapsulate all data and processing for a single QDM measurement.

    The Measurement class encapsulates all data and processing related to a single QDM
    (Quantum Diamond Microscope) measurement.

    It manages:
        - Raw and processed ODMR data using the ODMR instance.
        - Associated images (light and laser).
        - Fitting operations via external fitting instances.

    Attributes:
        odmr (ODMR): Instance managing ODMR data and processing.
        light_image (NDArray): Light image array with shape (height, width).
        laser_image (NDArray): Laser image array with shape (height, width).
        output_directory (Path): Path to the output directory.
        pixel_spacing (float): Spacing between pixels in meters.
        _outliers (Optional[NDArray]): Boolean mask for outlier pixels.
        _fit_model (str): Name of the model used for fitting ODMR spectra.
        metadata (Dict[str, Any]): Additional metadata for the measurement.
    """

    def __init__(
        self: Self,
        odmr: ODMR,
        light_image: NDArray,
        laser_image: NDArray,
        output_directory: str | Path | PathLike,
        pixel_spacing: float = 4e-6,
        fit_model: str = "auto",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Measurement object.

        Args:
            odmr (ODMR): An initialized ODMR instance containing ODMR data.
            light_image (NDArray): Light image array with shape (height, width).
            laser_image (NDArray): Laser image array with shape (height, width).
            output_directory (Union[str, Path, PathLike]): Path to the output directory.
            pixel_spacing (float): Spacing between pixels in meters (pixel size).
                Default is 4 µm (4e-6).
            fit_model (str): Name of the model used for fitting ODMR spectra. Default is "auto".
                            If "auto", the model is chosen based on the mean ODMR data.
            metadata (dict[str, Any], optional): Metadata dictionary. Defaults to empty dict.

        Raises:
            ValueError: If the ODMR instance is not properly initialized or if image shapes
                       don't match the ODMR data.
        """
        logger.info("Initializing Measurement object.")
        logger.info('Output directory: "{}"', output_directory)

        self.output_directory = Path(output_directory)
        self.pixel_spacing = pixel_spacing
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

        # Store the ODMR instance
        logger.debug("Setting ODMR data.")
        self.odmr = odmr

        # Validate ODMR data availability
        try:
            # Use public property instead of accessing protected member
            _ = self.odmr.raw_data
        except (ValueError, DataNotLoadedError) as e:
            msg = "ODMR instance has no raw data"
            raise DataNotLoadedError(msg) from e

        # Validate ODMR instance data
        logger.debug("ODMR raw data shape: {}", self.odmr.raw_data.shape)

        # Check if data has been processed
        try:
            logger.debug("ODMR processed data shape: {}", self.odmr.processed_data.shape)
        except (ValueError, DataNotLoadedError):
            logger.warning(
                "ODMR data has not been processed yet. Some functionality may be limited."
            )

        logger.debug("ODMR frequencies shape: {}", self.odmr.raw_data.frequencies.shape)

        # Initialize outlier mask
        logger.debug("Initializing outlier mask.")
        self._outliers: NDArray | None = np.ones(self.odmr.raw_data.shape, dtype=bool)

        # Store light and laser images
        logger.debug("Storing light and laser images.")
        self.light_image = light_image
        self.laser_image = laser_image

        # Store default fit model preference
        self._fit_model = fit_model

        # Cached folded ODMR result (populated by fold_odmr())
        self._folded_odmr: FoldedODMR | None = None

    @classmethod
    def from_folder(
        cls: type[Measurement],
        path: str | PathLike,
        *,
        bin_factor: int | None = None,
        model: str | None = None,
        pixel_spacing: float | None = None,
        normalize: bool | None = None,
        fluorescence_correction: float | None = _UNSET,  # type: ignore[assignment]
        output_directory: str | PathLike | None = None,
    ) -> Measurement:
        """Load ODMR data from a folder and return a ready-to-fit Measurement.

        Handles data loading, processing pipeline setup, and image loading in
        one call. Missing light/laser images fall back to zero arrays so that
        fitting still works even without reference images.

        metadata.toml is loaded automatically if present. Values from its
        ``[acquisition]`` section act as fallbacks between explicit keyword
        arguments and code-level defaults.  Priority (highest first):

        1. Explicit keyword argument (caller wins).
        2. ``[acquisition]`` value in ``metadata.toml``.
        3. Built-in code default.

        ``[measurement]`` fields (date, sample, subsample, fov, operator,
        notes) are stored verbatim in ``measurement.metadata["measurement"]``.

        Args:
            path: Folder containing MATLAB .mat files from the QDM microscope.
            bin_factor: Spatial binning factor (1 = no binning, 2 = 2x2 bins).
                Defaults to 1 or the value in ``[acquisition]``.
            model: ESR model name ('auto', 'ESR14N', 'ESR15N', 'ESRSINGLE').
                Defaults to 'auto' or the value in ``[acquisition]``.
            pixel_spacing: Physical pixel size in metres.
                Defaults to 4e-6 m or the value in ``[acquisition]``.
            normalize: Apply ODMR normalisation to spectra.
                Defaults to True or the value in ``[acquisition]``.
            fluorescence_correction: Fluorescence correction factor applied via
                FluorescenceCorrectionProcessor. Pass None to skip correction.
                Defaults to 0.2 or the value in ``[acquisition]``.
            output_directory: Directory for saved outputs. Defaults to
                ``path/results``.

        Returns:
            Measurement configured and ready for fit_odmr(). All metadata.toml
            contents are available on ``measurement.metadata``.

        Example:
            >>> m = Measurement.from_folder('/data/FOV18x')
            >>> m.metadata["measurement"]["sample"]
            'MIL2'
            >>> m.pixel_spacing   # from [acquisition] pixel_spacing = 2e-6
            2e-06
        """
        path = Path(path)
        logger.info("Loading measurement from {}", path)

        loaded = load_measurement_folder_data(
            path,
            bin_factor=bin_factor,
            model=model,
            pixel_spacing=pixel_spacing,
            normalize=normalize,
            fluorescence_correction=fluorescence_correction,
            unset_sentinel=_UNSET,
            listdir=os.listdir,
            image_loader=get_image,
        )

        return cls(
            odmr=loaded.odmr,
            light_image=loaded.light_image,
            laser_image=loaded.laser_image,
            pixel_spacing=loaded.pixel_spacing,
            fit_model=loaded.fit_model,
            output_directory=output_directory or path / "results",
            metadata=loaded.metadata,
        )

    def __str__(self: Self) -> str:
        """Return a string representation of the Measurement object.

        Returns:
            str: A human-readable string representation of the Measurement.
        """
        return (
            f"Measurement(odmr={self.odmr}, "
            f"output_directory='{self.output_directory}', "
            f"pixel_spacing={self.pixel_spacing} m)"
        )

    def __repr__(self: Self) -> str:
        """Return a developer string representation of the Measurement object.

        Returns:
            str: A detailed string representation for debugging and development.
        """
        return (
            f"Measurement(odmr={self.odmr!r}, "
            f"light_image.shape={self.light_image.shape}, "
            f"laser_image.shape={self.laser_image.shape}, "
            f"output_directory='{self.output_directory}', "
            f"pixel_spacing={self.pixel_spacing})"
        )

    def _validate_fit_prerequisites(
        self: Self,
        *,
        gpu_available: bool | None = None,
    ) -> ODMRData:
        """Validate that processed data and GPU fitting are available.

        Returns:
            The ProcessedData object.

        Raises:
            DataNotLoadedError: If ODMR data hasn't been processed.
            DependencyError: If pyGpufit is not available.
        """
        return validate_processed_odmr(self.odmr, gpu_available=gpu_available)

    def fit_odmr(
        self: Self,
        model_name: str | None = None,
        *,
        constraints: dict[str, Any] | None = None,
        freq_cutoff: dict[str, dict[str, float | None]] | None = None,
        refit_outliers: bool = False,
        refit_settings: RefitSettings | None = None,
        settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ) -> QDMResult:
        """Fit ODMR spectra and return unified result container.

        Args:
            model_name: Model name or None for auto-detection.
            constraints: Optional parameter constraints for fitting.
            freq_cutoff: Optional per-frange frequency cutoff in GHz.
                Schema: {'low': {'min': float|None, 'max': float|None},
                'high': {'min': float|None, 'max': float|None}}.
            refit_outliers: When True, automatically refit bad pixels after the
                initial fit using neighbor-derived initial guesses.
            refit_settings: Configuration for outlier detection and refitting.
                Only used when refit_outliers=True. Defaults to RefitSettings().
            settings: Optional explicit settings object forwarded to FitManager.
            gpu_available: Optional explicit override for GPU dependency
                availability checks.

        Returns:
            QDMResult containing FitResult and lazy MagneticMap access.

        Raises:
            DataNotLoadedError: If ODMR data hasn't been processed yet.
            DependencyError: If required fitting dependencies are not available.
        """
        model_name = model_name or self._fit_model
        logger.info("Starting ODMR fitting with model: {}", model_name)
        processed_data = self._validate_fit_prerequisites(gpu_available=gpu_available)
        result = fit_measurement_odmr(
            processed_data,
            pixel_spacing=self.pixel_spacing,
            model_name=model_name,
            constraints=constraints,
            freq_cutoff=freq_cutoff,
            light_image=self.light_image,
            laser_image=self.laser_image,
            settings=settings,
            gpu_available=gpu_available,
        )
        logger.info("ODMR fitting completed successfully")

        if refit_outliers:
            result = self.refit_outliers(
                result,
                settings=refit_settings,
                constraints=constraints,
                freq_cutoff=freq_cutoff,
                fit_settings=settings,
                gpu_available=gpu_available,
            )

        return result

    def refit_outliers(
        self: Self,
        result: QDMResult,
        *,
        settings: RefitSettings | None = None,
        constraints: dict[str, Any] | None = None,
        freq_cutoff: dict[str, dict[str, float | None]] | None = None,
        fit_settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ) -> QDMResult:
        """Refit bad pixels in an existing result using neighbor-derived initial guesses.

        Works transparently for both regular (fit_odmr) and folded (fit_folded_odmr)
        results. Identifies pixels with high chi-squared or non-convergence, computes
        initial parameter guesses from spatial neighbors, and refits just those pixels
        via GPU. Returns a new QDMResult; the original is not modified.

        Args:
            result: QDMResult from a previous fit_odmr() or fit_folded_odmr() call.
            settings: Outlier detection and refitting configuration.
                Defaults to RefitSettings().
            constraints: Optional parameter constraints to apply when refitting.
                Defaults to the same constraints used in the original fit.
            freq_cutoff: Optional per-frange frequency cutoff in GHz. Uses the
                same schema as fit_odmr()/fit_folded_odmr().
            fit_settings: Optional explicit settings object forwarded to
                FitManager for the refit.
            gpu_available: Optional explicit override for GPU dependency
                availability checks.

        Returns:
            New QDMResult with outlier pixels replaced by refit values.

        Raises:
            DataNotLoadedError: If required data (ODMR or folded) has not been processed.
            DependencyError: If pyGpufit is not available.
        """
        processed_data = None
        if not result.fit_result.metadata.get("folded_fit", False):
            processed_data = self._validate_fit_prerequisites(gpu_available=gpu_available)

        return refit_measurement_result(
            result,
            processed_data=processed_data,
            folded=getattr(self, "_folded_odmr", None),
            light_image=result.light_image,
            laser_image=result.laser_image,
            settings=settings,
            constraints=constraints,
            freq_cutoff=freq_cutoff,
            fit_settings=fit_settings,
            gpu_available=gpu_available,
        )

    @property
    def folded_odmr(self: Self) -> FoldedODMR:
        """Return the cached FoldedODMR, or raise if fold_odmr() hasn't been called.

        Returns:
            The cached FoldedODMR result.

        Raises:
            DataNotLoadedError: If fold_odmr() has not been called yet.
        """
        if self._folded_odmr is None:
            msg = "No folded ODMR data available. Call fold_odmr() first."
            raise DataNotLoadedError(msg)
        return self._folded_odmr

    def fold_odmr(
        self: Self,
        settings: FoldingSettings | None = None,
    ) -> FoldedODMR:
        """Fold ODMR spectra about the per-pixel D_ZFS and cache the result.

        Creates a SpectralFolder from the processed ODMR data, runs the
        two-scale folding pipeline, and caches the result for use by
        fit_folded_odmr().

        The folded spectrum has sqrt(2) lower noise per frequency point,
        but the D_ZFS estimation step introduces errors that propagate
        into the subsequent fit. For strong B111 signals (std >> 2-3 uT),
        this is negligible; for weak signals the normal (unfolded) fit
        via fit_odmr() may give more accurate B111 maps.

        Args:
            settings: Optional FoldingSettings. Defaults to FoldingSettings().

        Returns:
            The FoldedODMR result (also cached as self.folded_odmr).

        Raises:
            DataNotLoadedError: If ODMR data hasn't been processed yet.
            DataValidationError: If data doesn't have both frequency ranges.
            FoldingOverlapError: If frequency overlap is too narrow.
        """
        resolved_settings = settings if settings is not None else FoldingSettings()
        model_name = self._fit_model if self._fit_model != "auto" else None
        logger.info(
            "Folding ODMR spectra (method={}, model={})",
            resolved_settings.d_zfs_method,
            model_name or "auto",
        )

        self._folded_odmr = fold_measurement_odmr(
            self.odmr,
            settings=resolved_settings,
            model_name=model_name,
            folder_factory=SpectralFolder,
        )

        logger.info("Spectral folding complete")
        return self._folded_odmr

    def plot(self: Self) -> None:
        """Plot the light and laser optical images for this measurement."""
        from qdmpy.plotting import plot_measurement_images

        plot_measurement_images(self)

    def display(self: Self, result: QDMResult) -> None:
        """Comprehensive overview combining parameter maps, images, and pixel spectra.

        Shows B111 remanent/induced maps, chi-squared, mean centre/contrast/
        linewidth maps, the light and laser optical images, and a selection of
        representative pixel ODMR spectra with the fitted model curves overlaid.

        Args:
            result: QDMResult returned by ``fit_odmr()`` or ``fit_folded_odmr()``.
        """
        from qdmpy.plotting import plot_qdm_display

        plot_qdm_display(result, measurement=self)

    def fit_folded_odmr(
        self: Self,
        folded: FoldedODMR | None = None,
        model_name: str | None = None,
        *,
        constraints: dict[str, Any] | None = None,
        freq_cutoff: dict[str, dict[str, float | None]] | None = None,
        refit_outliers: bool = False,
        refit_settings: RefitSettings | None = None,
        settings: QDMpySettings | None = None,
        gpu_available: bool | None = None,
    ) -> QDMResult:
        """Fit a folded ODMR spectrum and return a unified result container.

        Uses the specified model (or the instance default from ``_fit_model``).
        Folded spectra are fitted in the absolute-GHz domain internally, so the
        returned result follows the same center/B111 conventions as non-folded
        fitting.

        Note:
            The folded fit halves the number of GPU fit calls (one spectrum per
            polarity instead of two branches). For strong B111 signals this
            gives comparable accuracy to fit_odmr(); for weak signals
            (B111 std < 2 uT) the D_ZFS estimation error may degrade accuracy.
            Compare results against fit_odmr() when signal strength is uncertain.

        Args:
            folded: FoldedODMR result. If None, uses the cached result from
                fold_odmr() (accessed via self.folded_odmr).
            model_name: Model name or None to use the instance default.
            constraints: Optional additional parameter constraints for fitting.
            freq_cutoff: Optional frequency cutoff for folded fitting in GHz.
                Folded fits have one range, so only the 'low' cutoff key is valid.
            refit_outliers: When True, automatically refit bad pixels after the
                initial fit using neighbor-derived initial guesses.
            refit_settings: Configuration for outlier detection and refitting.
                Only used when refit_outliers=True. Defaults to RefitSettings().
            settings: Optional explicit settings object forwarded to FitManager.
            gpu_available: Optional explicit override for GPU dependency
                availability checks.

        Returns:
            QDMResult containing FitResult and lazy MagneticMap access.

        Raises:
            DataNotLoadedError: If no folded data is available and fold_odmr()
                hasn't been called.
            DependencyError: If pyGpufit is not available.
        """
        resolved_folded = folded if folded is not None else self.folded_odmr
        model_name = model_name or self._fit_model
        self._validate_fit_prerequisites(gpu_available=gpu_available)

        logger.info("Starting folded ODMR fitting")
        result = fit_folded_measurement_odmr(
            resolved_folded,
            pixel_spacing=self.pixel_spacing,
            model_name=model_name,
            constraints=constraints,
            freq_cutoff=freq_cutoff,
            light_image=self.light_image,
            laser_image=self.laser_image,
            fit_settings=settings,
            gpu_available=gpu_available,
        )
        logger.info("Folded ODMR fitting completed successfully")
        if refit_outliers:
            result = self.refit_outliers(
                result,
                settings=refit_settings,
                constraints=constraints,
                freq_cutoff=freq_cutoff,
                fit_settings=settings,
                gpu_available=gpu_available,
            )
        return result
