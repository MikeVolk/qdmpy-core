"""HDF5 .qdm file format for QDMResult persistence.

The .qdm format is an HDF5 file (extension .qdm) that carries everything
needed after fitting: optical images, fitted parameters, B111 field maps,
optional Bxyz reconstruction, metadata, and field sources.

Layout::

    <name>.qdm   (HDF5 root)
    |
    +-- .attrs
    |   +-- qdm_version    str       "1.0"
    |   +-- model_name     str       e.g. "ESR14N"
    |   +-- pixel_spacing  float     metres
    |   +-- scan_dimensions int[2]   (height, width)
    |   +-- nv_axis        float[3]  (absent if None)
    |   +-- created_at     str       ISO 8601 UTC timestamp
    |   +-- metadata       str       JSON-encoded dict from FitResult.metadata
    |
    +-- images/            (omitted if both images are None)
    |   +-- light          (H, W)  float32
    |   +-- laser          (H, W)  float32
    |
    +-- fit/               (always present)
    |   +-- frequencies    (n_frange, n_freq)  float64
    |   +-- <param_name>   (n_pol, n_frange, H, W) or subset  float32
    |   +-- fit_states     int32
    |
    +-- b_field/           (always present)
    |   +-- b111_remanent  (H, W)  float32
    |   +-- b111_induced   (H, W)  float32
    |   +-- bx             (H, W)  float32  (optional)
    |   +-- by             (H, W)  float32  (optional)
    |   +-- bz             (H, W)  float32  (optional)
    |   +-- btotal         (H, W)  float32  (optional)
    |
    +-- field_sources/     (omitted if list is empty)
        +-- .attrs["count"]  int
        +-- 0/
        |   +-- .attrs["kind"]  str
        |   +-- .attrs["json"]  str   model_dump(mode="json", exclude={"field_map"})
        |   +-- field_map  (H, W)  float32  (optional)
        +-- 1/ ...
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from qdmpy.exceptions import DataLoadError, DataValidationError
from qdmpy.field_source import FieldSource, FieldSourceType
from qdmpy.fitting.result import FitResult

if TYPE_CHECKING:
    from os import PathLike

    import h5py as h5py_t
    from numpy.typing import NDArray

    from qdmpy.result import QDMResult


_QDM_VERSION = "1.0"
_QDM_MAGIC_EXT = ".qdm"


# ---------------------------------------------------------------------------
# Internal save helpers
# ---------------------------------------------------------------------------


def _write_images(
    f: h5py_t.File,
    result: QDMResult,
    compression: str | None,
) -> None:
    """Write images/ group if any image is present."""
    if result.light_image is None and result.laser_image is None:
        return
    img_grp = f.create_group("images")
    if result.light_image is not None:
        ds = img_grp.create_dataset(
            "light", data=result.light_image.astype(np.float32), compression=compression
        )
        ds.attrs["units"] = "a.u."
    if result.laser_image is not None:
        ds = img_grp.create_dataset(
            "laser", data=result.laser_image.astype(np.float32), compression=compression
        )
        ds.attrs["units"] = "a.u."


def _write_fit(
    f: h5py_t.File,
    fit_result: FitResult,
    compression: str | None,
) -> None:
    """Write fit/ group (parameters and frequencies)."""
    fit_grp = f.create_group("fit")
    for param_name, param_array in fit_result.parameters.items():
        if param_name == "frequencies":
            ds = fit_grp.create_dataset(
                "frequencies", data=param_array.astype(np.float64), compression=compression
            )
            ds.attrs["units"] = "GHz"
            continue
        if param_name == "states":
            continue  # stored separately as fit_states
        ds = fit_grp.create_dataset(
            param_name, data=param_array.astype(np.float32), compression=compression
        )
        if param_array.ndim == 4:
            ds.attrs["dims"] = "polarity,freq_range,y,x"
        else:
            ds.attrs["dims"] = ",".join(f"dim{i}" for i in range(param_array.ndim - 1)) + ",pixel"

    if "states" in fit_result.parameters:
        fit_grp.create_dataset(
            "fit_states",
            data=fit_result.parameters["states"].astype(np.int32),
            compression=compression,
        )


def _write_b_field(
    f: h5py_t.File,
    result: QDMResult,
    include_bxyz: bool,
    compression: str | None,
) -> None:
    """Write b_field/ group (B111 and optional Bxyz)."""
    b_grp = f.create_group("b_field")
    ds = b_grp.create_dataset(
        "b111_remanent",
        data=result.fit_result.b111_remanent.astype(np.float32),
        compression=compression,
    )
    ds.attrs["units"] = "uT"
    ds = b_grp.create_dataset(
        "b111_induced",
        data=result.fit_result.b111_induced.astype(np.float32),
        compression=compression,
    )
    ds.attrs["units"] = "uT"

    if include_bxyz:
        mag_map = result.magnetic_map
        for component in ("bx", "by", "bz", "btotal"):
            arr = np.array(getattr(mag_map, component)).astype(np.float32)
            ds = b_grp.create_dataset(component, data=arr, compression=compression)
            ds.attrs["units"] = "uT"


def _write_field_sources(
    f: h5py_t.File,
    field_sources: list[FieldSourceType],
    compression: str | None,
) -> None:
    """Write field_sources/ group."""
    if not field_sources:
        return
    src_grp = f.create_group("field_sources")
    src_grp.attrs["count"] = len(field_sources)
    for i, src in enumerate(field_sources):
        sg = src_grp.create_group(str(i))
        sg.attrs["kind"] = src.kind
        sg.attrs["json"] = src.model_dump_json(exclude={"field_map"})
        if src.field_map is not None:
            ds = sg.create_dataset(
                "field_map",
                data=np.asarray(src.field_map, dtype=np.float32),
                compression=compression,
            )
            ds.attrs["units"] = "uT"


def _validate_images(result: QDMResult, height: int, width: int) -> None:
    """Raise DataValidationError when image shapes do not match scan_dimensions."""
    if result.light_image is not None and result.light_image.shape != (height, width):
        msg = (
            f"light_image shape {result.light_image.shape} does not match "
            f"scan_dimensions {result.fit_result.scan_dimensions}. "
            "Bin images to match or omit them."
        )
        raise DataValidationError(msg)
    if result.laser_image is not None and result.laser_image.shape != (height, width):
        msg = (
            f"laser_image shape {result.laser_image.shape} does not match "
            f"scan_dimensions {result.fit_result.scan_dimensions}. "
            "Bin images to match or omit them."
        )
        raise DataValidationError(msg)


# ---------------------------------------------------------------------------
# Internal load helpers
# ---------------------------------------------------------------------------


def _check_version(f: h5py_t.File, path: Path) -> None:
    """Raise DataLoadError for missing or incompatible major version."""
    if "qdm_version" not in f.attrs:
        msg = f"{path} is not a valid .qdm file (missing qdm_version attribute)."
        raise DataLoadError(msg)
    version_str: str = f.attrs["qdm_version"]
    major = int(version_str.split(".", maxsplit=1)[0])
    code_major = int(_QDM_VERSION.split(".", maxsplit=1)[0])
    if major > code_major:
        msg = (
            f"{path} was written with .qdm format v{version_str}, "
            f"but this qdmpy only understands v{_QDM_VERSION}. "
            "Please upgrade qdmpy."
        )
        raise DataLoadError(msg)


def _read_fit_parameters(f: h5py_t.File) -> dict[str, Any]:
    """Read fit/ group datasets into a parameters dict."""
    parameters: dict[str, Any] = {}
    if "fit" not in f:
        return parameters
    fit_grp = f["fit"]
    for key in fit_grp:
        if key == "fit_states":
            continue
        parameters[key] = np.array(fit_grp[key])
    return parameters


def _read_b111_caches(
    f: h5py_t.File,
) -> tuple[NDArray | None, NDArray | None]:
    """Return (b111_remanent, b111_induced) arrays or (None, None)."""
    if "b_field" not in f:
        return None, None
    b_grp = f["b_field"]
    b111_remanent = (
        np.array(b_grp["b111_remanent"], dtype=np.float64) if "b111_remanent" in b_grp else None
    )
    b111_induced = (
        np.array(b_grp["b111_induced"], dtype=np.float64) if "b111_induced" in b_grp else None
    )
    return b111_remanent, b111_induced


def _read_images(f: h5py_t.File) -> tuple[NDArray | None, NDArray | None]:
    """Return (light_image, laser_image) arrays or (None, None)."""
    if "images" not in f:
        return None, None
    img_grp = f["images"]
    light = np.array(img_grp["light"]) if "light" in img_grp else None
    laser = np.array(img_grp["laser"]) if "laser" in img_grp else None
    return light, laser


def _read_field_sources(f: h5py_t.File) -> list[FieldSourceType]:
    """Reconstruct FieldSource list from field_sources/ group."""
    if "field_sources" not in f:
        return []
    src_grp = f["field_sources"]
    count = int(src_grp.attrs.get("count", 0))
    sources: list[FieldSourceType] = []
    for i in range(count):
        key = str(i)
        if key not in src_grp:
            continue
        sg = src_grp[key]
        src_json: str = sg.attrs.get("json", "{}")
        field_map: NDArray | None = np.array(sg["field_map"]) if "field_map" in sg else None
        src_data = json.loads(src_json)
        src_data["field_map"] = field_map
        sources.append(FieldSource(**src_data))
    return sources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_qdm(
    result: QDMResult,
    path: str | PathLike,
    *,
    include_bxyz: bool = False,
    overwrite: bool = False,
    compress: bool = True,
) -> None:
    """Export a QDMResult to a .qdm (HDF5) file.

    Args:
        result: The QDMResult to export.
        path: Destination file path. Warns if extension is not '.qdm'.
        include_bxyz: If True, compute MagneticMap (if not cached) and
            store Bx/By/Bz/Btotal. Default False.
        overwrite: If True, overwrite existing file. If False (default),
            raise FileExistsError when the file already exists.
        compress: Apply GZIP compression to each dataset. Default True.

    Raises:
        FileExistsError: If the file already exists and overwrite=False.
        DataValidationError: If image shapes do not match scan_dimensions.
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        msg = "h5py is required for .qdm export. Install it with: uv add h5py"
        raise ImportError(msg) from exc

    path = Path(path)
    if path.suffix.lower() != _QDM_MAGIC_EXT:
        logger.warning("save_qdm: path '{}' does not have .qdm extension", path)

    if path.exists() and not overwrite:
        msg = f"File already exists: {path}. Pass overwrite=True to replace it."
        raise FileExistsError(msg)

    fit_result = result.fit_result
    height, width = fit_result.scan_dimensions
    _validate_images(result, height, width)

    compression: str | None = "gzip" if compress else None
    logger.info("Saving QDMResult (.qdm) to {}", path)

    with h5py.File(path, "w") as f:
        f.attrs["qdm_version"] = _QDM_VERSION
        f.attrs["model_name"] = fit_result.model_name
        f.attrs["pixel_spacing"] = fit_result.pixel_spacing
        f.attrs["scan_dimensions"] = np.array(list(fit_result.scan_dimensions), dtype=np.int32)
        f.attrs["created_at"] = datetime.now(UTC).isoformat()
        f.attrs["metadata"] = json.dumps(fit_result.metadata)
        if result.nv_axis is not None:
            f.attrs["nv_axis"] = np.array(result.nv_axis, dtype=np.float64)

        _write_images(f, result, compression)
        _write_fit(f, fit_result, compression)
        _write_b_field(f, result, include_bxyz, compression)
        _write_field_sources(f, result.field_sources, compression)

    logger.info("QDMResult (.qdm) saved to {}", path)


def load_qdm(path: str | PathLike) -> QDMResult:
    """Load a QDMResult from a .qdm (HDF5) file.

    Reconstructs FitResult, optional images, field_sources, and nv_axis.
    B111 fields stored in the file are loaded into FitResult caches so
    they are available immediately without recomputation. MagneticMap is
    NOT reconstructed on load -- access .magnetic_map to trigger it.

    Args:
        path: Path to a .qdm file created by save_qdm().

    Returns:
        QDMResult with all fields present in the file populated.

    Raises:
        DataLoadError: File not found, not a valid .qdm file, or
            incompatible major version.
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        msg = "h5py is required for .qdm loading. Install it with: uv add h5py"
        raise ImportError(msg) from exc

    import xarray as xr

    from qdmpy.result import QDMResult

    path = Path(path)
    if not path.exists():
        msg = f".qdm file not found: {path}"
        raise DataLoadError(msg)

    logger.info("Loading QDMResult (.qdm) from {}", path)

    try:
        with h5py.File(path, "r") as f:
            _check_version(f, path)

            model_name: str = f.attrs["model_name"]
            pixel_spacing: float = float(f.attrs["pixel_spacing"])
            scan_dimensions: tuple[int, int] = tuple(  # type: ignore[assignment]
                int(x) for x in f.attrs["scan_dimensions"]
            )
            metadata: dict[str, Any] = json.loads(f.attrs.get("metadata", "{}"))

            nv_axis: tuple[float, float, float] | None = None
            if "nv_axis" in f.attrs:
                nv_axis = tuple(float(v) for v in f.attrs["nv_axis"])  # type: ignore[assignment]

            parameters = _read_fit_parameters(f)
            b111_remanent, b111_induced = _read_b111_caches(f)
            light_image, laser_image = _read_images(f)
            field_sources = _read_field_sources(f)

    except OSError as exc:
        msg = f"Could not open {path} as an HDF5 file: {exc}"
        raise DataLoadError(msg) from exc

    fit_result = FitResult(
        parameters=parameters,
        scan_dimensions=scan_dimensions,
        pixel_spacing=pixel_spacing,
        model_name=model_name,
        metadata=metadata,
    )

    # Pre-populate B111 caches so no recomputation is needed
    if b111_remanent is not None and b111_induced is not None:
        fit_result._b111_cache = xr.Dataset(
            {
                "remanent": xr.DataArray(b111_remanent, dims=("y", "x")),
                "induced": xr.DataArray(b111_induced, dims=("y", "x")),
            }
        )

    logger.info("QDMResult (.qdm) loaded from {}", path)
    return QDMResult(
        fit_result=fit_result,
        nv_axis=nv_axis,
        light_image=light_image,
        laser_image=laser_image,
        field_sources=field_sources,
    )
