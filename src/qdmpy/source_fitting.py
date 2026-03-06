"""Magnetic dipole source fitting using pypole.

Fits each MagneticSource in a QDMResult.field_sources list to a single
magnetic dipole using the vertical field component (Bz) from the Fourier
reconstruction.

Coordinate convention
---------------------
MagneticModel.declination uses the pypole convention:
    dec=0 -> -Y, dec=90 -> +X (East), measured counterclockwise from -Y.
No conversion is needed between MagneticModel and pypole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pypole.convert
import pypole.fit
import pypole.maps
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from qdmpy.field_source import MagneticModel, MagneticSource

if TYPE_CHECKING:
    from qdmpy.result import QDMResult


@dataclass(frozen=True)
class FitSourceResult:
    """Result of fitting a single MagneticSource to a Bz field map.

    Attributes:
        source: Updated MagneticSource with fitted parameters. The center
            has been shifted by the in-plane position offset and the model
            carries the recovered (declination, inclination, magnetic_moment).
        raw: Raw scipy OptimizeResult. Inspect .success, .cost, and .nfev
            to assess convergence quality.
    """

    source: MagneticSource
    raw: OptimizeResult


def _build_p0(
    source: MagneticSource,
    standoff_m: float,
) -> tuple[float, float, float, float, float, float]:
    """Build the initial-guess vector for pypole.fit.fit_dipole.

    Converts the qdmpy declination convention (+Y = 0 degrees) to the
    pypole convention (-Y = 0 degrees) and returns the six-element
    parameter tuple (x_source, y_source, z_source, mx, my, mz).

    x_source and y_source are both set to 0.0, placing the dipole at
    the ROI centre at the start of the optimisation.

    Args:
        source: MagneticSource supplying the initial magnetic model.
        standoff_m: Initial sensor-to-sample distance guess in metres.

    Returns:
        Six-element tuple suitable as p0 for pypole.fit.fit_dipole.
    """
    dim = np.array(
        [[source.model.declination, source.model.inclination, source.model.magnetic_moment]],
        dtype=np.float64,
    )
    xyz = pypole.convert.dim2xyz(dim)  # shape (1, 3)
    mx = float(xyz[0, 0])
    my = float(xyz[0, 1])
    mz = float(xyz[0, 2])
    return (0.0, 0.0, standoff_m, mx, my, mz)


def _parse_result(
    fit_x: NDArray,
    source: MagneticSource,
    raw: OptimizeResult,
) -> FitSourceResult:
    """Convert a pypole fit result into an updated MagneticSource.

    Applies the fitted in-plane position offset to the source centre and
    converts the fitted moment (mx, my, mz) back to qdmpy convention.

    If the fitted moment is near-zero (e.g. degenerate field) or produces
    invalid model parameters, the original model is retained.

    Args:
        fit_x: Fitted parameter array [x_offset_m, y_offset_m, z_m, mx, my, mz].
        source: Original MagneticSource (used for pixel_spacing and fallback model).
        raw: Raw scipy OptimizeResult (passed through unchanged).

    Returns:
        FitSourceResult wrapping the updated MagneticSource and raw result.
    """
    x_offset_m, y_offset_m, _z, mx, my, mz = fit_x

    xyz_arr = np.array([[mx, my, mz]], dtype=np.float64)
    dim = pypole.convert.xyz2dim(xyz_arr)  # shape (1, 3): [dec, inc, mag]
    dec = float(dim[0, 0])
    inc = float(dim[0, 1])
    mag = float(dim[0, 2])

    try:
        new_model = MagneticModel(
            declination=dec,
            inclination=inc,
            magnetic_moment=mag,
        )
    except ValueError:
        logger.warning(
            "Model validation failed after fit "
            "(dec={:.1f}, inc={:.1f}, mag={:.3e}); retaining original model.",
            dec,
            inc,
            mag,
        )
        new_model = source.model

    new_center = (
        source.center[0] + x_offset_m / source.pixel_spacing,
        source.center[1] + y_offset_m / source.pixel_spacing,
    )
    updated = source.model_copy(update={"center": new_center, "model": new_model})
    return FitSourceResult(source=updated, raw=raw)


def fit_source(
    bz_map_T: NDArray,  # noqa: N803 -- T suffix denotes Tesla (physics unit)
    source: MagneticSource,
    standoff_m: float,
) -> FitSourceResult:
    """Fit a single magnetic dipole to the Bz map within the source ROI.

    Uses pypole.fit.fit_dipole (scipy least_squares with Huber loss, TRF
    method) to optimise the six-parameter dipole model: in-plane position
    offset (x, y), sensor distance (z), and moment components (mx, my, mz).

    Args:
        bz_map_T: Full Bz field map in Tesla, shape (H, W).
        source: MagneticSource defining the ROI via roi_pixels and supplying
            the initial magnetic model for the guess.
        standoff_m: Initial sensor-to-sample distance guess in metres
            (e.g. 5e-6 for typical NV-diamond experiments).

    Returns:
        FitSourceResult with updated MagneticSource (fitted parameters)
        and raw scipy OptimizeResult for convergence diagnostics.
    """
    roi_T: NDArray = bz_map_T[source.roi_pixels]  # noqa: N806 -- T suffix = Tesla
    logger.info(
        "Fitting dipole for source '{}': ROI shape={}, standoff={:.2e} m",
        source.name,
        roi_T.shape,
        standoff_m,
    )
    p0 = _build_p0(source, standoff_m)
    raw = pypole.fit.fit_dipole(roi_T, p0, pixel_size=source.pixel_spacing)
    logger.info(
        "Source '{}' fit complete: success={}, cost={:.4e}",
        source.name,
        raw.success,
        raw.cost,
    )
    return _parse_result(raw.x, source, raw)


def fit_sources(
    result: QDMResult,
    standoff_m: float,
) -> list[FitSourceResult]:
    """Fit all MagneticSource objects in a QDMResult to magnetic dipoles.

    Extracts the Bz map (vertical field component) from the Fourier
    reconstruction and calls fit_source for each MagneticSource in
    result.field_sources. Other FieldSource subclasses are skipped.

    Accessing result.magnetic_map triggers Fourier reconstruction from
    b111_remanent if it has not been computed yet.

    Args:
        result: QDMResult containing fitted B111 data and the field_sources
            list populated with MagneticSource objects.
        standoff_m: Initial sensor-to-sample distance guess in metres
            (e.g. 5e-6 for 5 micrometres).

    Returns:
        List of FitSourceResult, one per MagneticSource in field_sources,
        in the same order. Sources of other kinds are silently skipped.
    """
    bz_uT: NDArray = result.magnetic_map.bz.values  # noqa: N806 -- uT = microTesla
    bz_T = bz_uT * 1e-6  # noqa: N806 -- T = Tesla; convert µT -> T

    magnetic_sources = [s for s in result.field_sources if isinstance(s, MagneticSource)]
    logger.info(
        "fit_sources: Bz shape={}, fitting {} magnetic source(s)",
        bz_T.shape,
        len(magnetic_sources),
    )
    return [fit_source(bz_T, s, standoff_m) for s in magnetic_sources]


def compute_field(
    source: MagneticSource,
    standoff_m: float,
) -> NDArray:
    """Compute the predicted Bz field over the source ROI from the dipole model.

    Evaluates the analytical magnetic dipole formula over a pixel grid matching
    the source ROI, with the dipole placed at the ROI centre (offset 0, 0).

    The result can be compared with the measured Bz ROI:

        measured = bz_map_T[source.roi_pixels]
        residual = measured - compute_field(source, standoff_m)

    Args:
        source: MagneticSource whose model and ROI define the dipole and grid.
        standoff_m: Sensor-to-sample distance in metres (e.g. 5e-6).

    Returns:
        Predicted Bz field in Tesla, shape (roi_H, roi_W).
    """
    roi_row, roi_col = source.roi_pixels
    roi_h = roi_row.stop - roi_row.start
    roi_w = roi_col.stop - roi_col.start

    # get_grid takes (n_cols, n_rows) and returns arrays of shape (n_rows, n_cols)
    x_grid, y_grid = pypole.maps.get_grid(pixels=(roi_w, roi_h), pixel_size=source.pixel_spacing)

    dim = np.array(
        [[source.model.declination, source.model.inclination, source.model.magnetic_moment]],
        dtype=np.float64,
    )
    xyz = pypole.convert.dim2xyz(dim)  # shape (1, 3)
    mx, my, mz = float(xyz[0, 0]), float(xyz[0, 1]), float(xyz[0, 2])

    logger.debug(
        "compute_field: source='{}', ROI=({}, {}), standoff={:.2e} m",
        source.name,
        roi_h,
        roi_w,
        standoff_m,
    )
    return pypole.fit.dipole_field(x_grid, y_grid, 0.0, 0.0, standoff_m, mx, my, mz)
