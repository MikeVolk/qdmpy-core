"""Concrete workflow helpers behind Measurement convenience methods."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from qdmpy.exceptions import DataLoadError, DataNotLoadedError, DependencyError

if TYPE_CHECKING:
    from qdmpy.fitting.backends import FitBackend
    from qdmpy.fitting.manager import FitManager
    from qdmpy.fitting.refit import RefitSettings
    from qdmpy.fitting.result import FitResult
    from qdmpy.odmr.data import ODMRData
    from qdmpy.odmr.folding import FoldedODMR, FoldingSettings
    from qdmpy.odmr.manager import ODMR
    from qdmpy.result import QDMResult
    from qdmpy.settings import QDMpySettings
type FluorescenceCorrectionArg = float | None | object


class FoldedODMRFolder(Protocol):
    """Protocol for folder objects that can produce FoldedODMR results."""

    def fold(self) -> FoldedODMR:
        """Run the folding workflow and return the folded result."""


@dataclass(frozen=True)
class MeasurementFolderData:
    """Loaded measurement data before Measurement construction."""

    odmr: ODMR
    light_image: NDArray
    laser_image: NDArray
    pixel_spacing: float
    fit_model: str
    metadata: dict[str, Any]


def _load_image_or_zeros(
    folder: Path,
    folder_files: list[str],
    kinds: tuple[str, ...],
    scan_dimensions: tuple[int, int],
    image_label: str,
    image_loader: Callable[[Path, list[str]], NDArray],
) -> NDArray:
    matching = [f for f in folder_files if any(kind in f.lower() for kind in kinds)]
    try:
        return image_loader(folder, matching)
    except DataLoadError:
        logger.warning(
            "No {} image found in {}; using zeros array of shape {}",
            image_label,
            folder,
            scan_dimensions,
        )
        return np.zeros(scan_dimensions)


def load_measurement_folder_data(
    path: str | os.PathLike[str],
    *,
    bin_factor: int | None,
    model: str | None,
    pixel_spacing: float | None,
    normalize: bool | None,
    fluorescence_correction: FluorescenceCorrectionArg,
    unset_sentinel: object,
    listdir: Callable[[Path], list[str]],
    image_loader: Callable[[Path, list[str]], NDArray],
) -> MeasurementFolderData:
    """Load ODMR, metadata, and images for Measurement.from_folder()."""
    from qdmpy.io import load_metadata_toml
    from qdmpy.odmr.data import ODMRData
    from qdmpy.odmr.io import MatlabLoader
    from qdmpy.odmr.manager import ODMR
    from qdmpy.odmr.processors import (
        BinningProcessor,
        FluorescenceCorrectionProcessor,
        NormalizationProcessor,
    )

    folder = Path(path)
    metadata = load_metadata_toml(folder)
    acquisition = metadata.get("acquisition", {})

    resolved_bin_factor = (
        bin_factor if bin_factor is not None else int(acquisition.get("bin_factor", 1))
    )
    resolved_model = model if model is not None else str(acquisition.get("model", "auto"))
    resolved_pixel_spacing = (
        pixel_spacing
        if pixel_spacing is not None
        else float(acquisition.get("pixel_spacing", 4e-6))
    )
    resolved_normalize = (
        normalize if normalize is not None else bool(acquisition.get("normalize", True))
    )

    if fluorescence_correction is not unset_sentinel:
        resolved_fluorescence_correction = cast(float | None, fluorescence_correction)
    else:
        fc_value = acquisition.get("fluorescence_correction", 0.2)
        resolved_fluorescence_correction = float(fc_value) if fc_value is not None else None

    odmr = ODMR(ODMRData.from_loader(MatlabLoader(str(folder))))
    if resolved_bin_factor > 1:
        odmr.processor_manager.add_processor(BinningProcessor(bin_factor=resolved_bin_factor))
    if resolved_normalize:
        odmr.processor_manager.add_processor(NormalizationProcessor())
    if resolved_fluorescence_correction is not None:
        odmr.processor_manager.add_processor(
            FluorescenceCorrectionProcessor(correction_factor=resolved_fluorescence_correction)
        )
    odmr.process_data()

    scan_dimensions = odmr.processed_data.scan_dimensions
    folder_files = listdir(folder)
    light_image = _load_image_or_zeros(
        folder,
        folder_files,
        ("light", "led"),
        scan_dimensions,
        image_label="light/led",
        image_loader=image_loader,
    )
    laser_image = _load_image_or_zeros(
        folder,
        folder_files,
        ("laser",),
        scan_dimensions,
        image_label="laser",
        image_loader=image_loader,
    )
    return MeasurementFolderData(
        odmr=odmr,
        light_image=light_image,
        laser_image=laser_image,
        pixel_spacing=resolved_pixel_spacing,
        fit_model=resolved_model,
        metadata=metadata,
    )


def _backend_needs_gpufit_preflight(backend: FitBackend | str | None) -> bool:
    """Whether the workflow-level GPU-availability guard applies to ``backend``.

    Returns False only for an explicitly-requested non-gpufit backend (a
    caller-supplied FitBackend instance, or a backend name other than
    'auto'/'gpufit'), so requesting ``backend='scipy'`` isn't blocked by a
    check that only makes sense for the gpufit path.
    """
    if backend is None:
        return True
    if isinstance(backend, str):
        return backend in ("auto", "gpufit")
    return False


def validate_processed_odmr(
    odmr: ODMR,
    *,
    gpu_available: bool | None = None,
    backend: FitBackend | str | None = None,
) -> ODMRData:
    """Return processed ODMR data after dependency validation."""
    try:
        processed_data = odmr.processed_data
    except (AttributeError, ValueError, DataNotLoadedError) as exc:
        msg = "ODMR data must be processed before fitting. Call odmr.process_data() first."
        raise DataNotLoadedError(msg) from exc

    if gpu_available is None and not _backend_needs_gpufit_preflight(backend):
        return processed_data

    resolved_gpu_available = gpu_available
    if resolved_gpu_available is None:
        from qdmpy.settings import is_pygpufit_available

        resolved_gpu_available = is_pygpufit_available()
    if not resolved_gpu_available:
        msg = (
            "pyGpufit is required for fitting but not available. "
            "Please install pyGpufit to enable fitting functionality."
        )
        raise DependencyError(msg)
    return processed_data


def build_fit_manager(
    *,
    model_name: str,
    constraints: dict[str, Any] | None,
    freq_cutoff: dict[str, dict[str, float | None]] | None,
    settings: QDMpySettings | None = None,
    backend: FitBackend | str | None = None,
    gpu_available: bool | None = None,
) -> FitManager:
    """Build the concrete FitManager used by Measurement workflows."""
    from qdmpy.fitting.manager import FitManager

    return FitManager(
        model_name=model_name,
        constraints=constraints,
        freq_cutoff=freq_cutoff,
        settings=settings,
        backend=backend,
        gpu_available=gpu_available,
    )


def build_qdm_result(
    *, fit_result: FitResult, light_image: NDArray | None, laser_image: NDArray | None
) -> QDMResult:
    """Wrap a FitResult in the public QDMResult container."""
    from qdmpy.result import QDMResult

    return QDMResult(
        fit_result=fit_result,
        light_image=light_image,
        laser_image=laser_image,
    )


def fit_measurement_odmr(
    processed_data: ODMRData,
    *,
    pixel_spacing: float,
    model_name: str,
    constraints: dict[str, Any] | None,
    freq_cutoff: dict[str, dict[str, float | None]] | None,
    light_image: NDArray,
    laser_image: NDArray,
    settings: QDMpySettings | None = None,
    backend: FitBackend | str | None = None,
    gpu_available: bool | None = None,
) -> QDMResult:
    """Fit processed ODMR data and wrap the public result."""
    fit_manager = build_fit_manager(
        model_name=model_name,
        constraints=constraints,
        freq_cutoff=freq_cutoff,
        settings=settings,
        backend=backend,
        gpu_available=gpu_available,
    )
    fit_result = fit_manager.fit(
        processed_data.data,
        processed_data.frequencies,
        pixel_spacing=pixel_spacing,
    )
    return build_qdm_result(
        fit_result=fit_result,
        light_image=light_image,
        laser_image=laser_image,
    )


def refit_measurement_result(
    result: QDMResult,
    *,
    processed_data: ODMRData | None,
    folded: FoldedODMR | None,
    light_image: NDArray | None,
    laser_image: NDArray | None,
    settings: RefitSettings | None,
    constraints: dict[str, Any] | None,
    freq_cutoff: dict[str, dict[str, float | None]] | None,
    fit_settings: QDMpySettings | None = None,
    backend: FitBackend | str | None = None,
    gpu_available: bool | None = None,
) -> QDMResult:
    """Refit outlier pixels for either regular or folded measurement results."""
    from qdmpy.fitting.refit import refit_outliers as _refit_outliers

    model_name = result.fit_result.model_name
    is_folded = result.fit_result.metadata.get("folded_fit", False)
    if is_folded:
        if folded is None:
            msg = "No folded ODMR data available. Call fold_odmr() first."
            raise DataNotLoadedError(msg)
        data_xr, frequencies = folded.to_fit_inputs()
    else:
        if processed_data is None:
            msg = "ODMR data must be processed before fitting. Call odmr.process_data() first."
            raise DataNotLoadedError(msg)
        data_xr = processed_data.data
        frequencies = processed_data.frequencies

    fit_manager = build_fit_manager(
        model_name=model_name,
        constraints=constraints,
        freq_cutoff=freq_cutoff,
        settings=fit_settings,
        backend=backend,
        gpu_available=gpu_available,
    )
    new_fit_result = _refit_outliers(
        result.fit_result,
        data_xr,
        frequencies,
        fit_manager,
        settings,
    )
    return build_qdm_result(
        fit_result=new_fit_result,
        light_image=light_image,
        laser_image=laser_image,
    )


def fold_measurement_odmr(
    odmr: ODMR,
    *,
    settings: FoldingSettings,
    model_name: str | None,
    folder_factory: Callable[..., FoldedODMRFolder],
) -> FoldedODMR:
    """Fold processed ODMR data through the supplied folder factory."""
    try:
        processed_data = odmr.processed_data
    except (AttributeError, ValueError, DataNotLoadedError) as exc:
        msg = "ODMR data must be processed before folding. Call odmr.process_data() first."
        raise DataNotLoadedError(msg) from exc

    folder = folder_factory(processed_data, settings, model_name=model_name)
    return folder.fold()


def fit_folded_measurement_odmr(
    folded: FoldedODMR,
    *,
    pixel_spacing: float,
    model_name: str,
    constraints: dict[str, Any] | None,
    freq_cutoff: dict[str, dict[str, float | None]] | None,
    light_image: NDArray,
    laser_image: NDArray,
    fit_settings: QDMpySettings | None = None,
    backend: FitBackend | str | None = None,
    gpu_available: bool | None = None,
) -> QDMResult:
    """Fit folded ODMR data and wrap the public result."""
    if gpu_available is False:
        msg = (
            "pyGpufit is required for fitting but not available. "
            "Please install pyGpufit to enable fitting functionality."
        )
        raise DependencyError(msg)

    fit_manager = build_fit_manager(
        model_name=model_name,
        constraints=constraints,
        freq_cutoff=freq_cutoff,
        settings=fit_settings,
        backend=backend,
        gpu_available=gpu_available,
    )
    fit_result = fit_manager.fit_folded(folded, pixel_spacing=pixel_spacing)
    return build_qdm_result(
        fit_result=fit_result,
        light_image=light_image,
        laser_image=laser_image,
    )
