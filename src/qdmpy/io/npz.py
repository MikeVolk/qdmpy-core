"""NPZ persistence for QDMResult.

Lightweight checkpoint format. Stores only fitted parameters, scan metadata,
and nv_axis. Does not include images, Bxyz, or field_sources.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from qdmpy.exceptions import DataLoadError
from qdmpy.fitting.result import FitResult

if TYPE_CHECKING:
    from os import PathLike

    from qdmpy.result import QDMResult


def save_npz(result: QDMResult, path: str | PathLike) -> None:
    """Save QDMResult to a pickle-free NPZ file (fit data only).

    Lightweight checkpoint format. Does not include images, Bxyz,
    or field_sources. Logic moved from QDMResult.save().

    Args:
        result: The QDMResult to save.
        path: Destination file path (.npz extension added if absent).
    """
    path = Path(path)
    logger.info("Saving QDMResult (NPZ) to {}", path)

    save_dict = result.fit_result._build_save_dict()
    if result.nv_axis is not None:
        save_dict["nv_axis"] = np.array(result.nv_axis)

    arrays = {k: np.asarray(v) for k, v in save_dict.items()}
    np.savez_compressed(path, allow_pickle=False, **arrays)
    logger.info("QDMResult (NPZ) saved to {}", path)


def load_npz(path: str | PathLike) -> QDMResult:
    """Load QDMResult from a pickle-free NPZ file.

    Logic moved from QDMResult.load().

    Args:
        path: Path to the .npz file created by save_npz() or QDMResult.save().

    Returns:
        Reconstructed QDMResult. MagneticMap will be recomputed on first
        access to .magnetic_map.

    Raises:
        DataLoadError: If the file does not exist or cannot be parsed.
    """
    from qdmpy.result import QDMResult

    path = Path(path)

    if not path.exists():
        msg = f"Results file not found: {path}"
        raise DataLoadError(msg)

    try:
        with np.load(path, allow_pickle=False) as data:
            if "__meta__" in data.files:
                fit_result = FitResult._from_npz(data, source=str(path))

                nv_axis: tuple[float, float, float] | None = None
                if "nv_axis" in data:
                    _nv = [float(v) for v in data["nv_axis"]]
                    nv_axis = (_nv[0], _nv[1], _nv[2])

                logger.info("QDMResult (NPZ) loaded from {}", path)
                return QDMResult(fit_result=fit_result, nv_axis=nv_axis)

            if "parameters" not in data.files:
                msg = (
                    f"File {path} is missing the __meta__ key. "
                    "This file was created with an older format that is no longer supported."
                )
                raise DataLoadError(msg)
    except ValueError as exc:
        msg = f"File {path} contains pickled objects and cannot be loaded safely."
        raise DataLoadError(msg) from exc

    warning_msg = (
        "Loading legacy pickle-format QDMResult NPZ file. "
        "Re-save with QDMResult.save() to migrate. "
        "Pickle support will be removed in v1.0."
    )
    warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
    logger.warning(warning_msg)

    with np.load(path, allow_pickle=True) as data_legacy:
        fit_result = FitResult._from_legacy_npz(data_legacy, source=str(path))

        nv_axis: tuple[float, float, float] | None = None
        if "nv_axis" in data_legacy:
            _nv = [float(v) for v in data_legacy["nv_axis"]]
            nv_axis = (_nv[0], _nv[1], _nv[2])

    logger.info("QDMResult (NPZ) loaded from legacy format: {}", path)
    return QDMResult(fit_result=fit_result, nv_axis=nv_axis)
