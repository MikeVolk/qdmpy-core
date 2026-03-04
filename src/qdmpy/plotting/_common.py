"""Shared plotting helpers used across the plotting subpackage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.image
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes

# Set white background for all QDMpy figures
mpl.rcParams["figure.facecolor"] = "white"


def _add_colorbar(
    im: mpl.image.AxesImage,
    ax: MplAxes,
    label: str | None = None,
) -> mpl.colorbar.Colorbar:
    """Add a colorbar whose height matches the axes it belongs to.

    Uses ``make_axes_locatable`` so the colorbar is always the same height as
    the plot, regardless of figure layout.

    Args:
        im: The mappable (return value of ``imshow``).
        ax: The axes the image lives on.
        label: Optional text label for the colorbar.

    Returns:
        The created ``Colorbar`` instance.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    if label is not None:
        cbar.set_label(label)
    return cbar


def resolve_pixel_indices(
    n_y: int,
    n_x: int,
    x: list[int] | int | None = None,
    y: list[int] | int | None = None,
) -> list[tuple[int, int]]:
    """Resolve pixel coordinate arguments into a list of (y, x) index pairs.

    Expansion rules:
    - Both None: one random pixel is chosen.
    - Both scalar: single pixel ``[(y, x)]``.
    - x is list, y is scalar: ``[(y, x0), (y, x1), ...]``.
    - x is scalar, y is list: ``[(y0, x), (y1, x), ...]``.
    - Both lists (same length): ``zip(y, x)``.

    Args:
        n_y: Height of the scan grid (number of rows).
        n_x: Width of the scan grid (number of columns).
        x: Column index or indices. None means random.
        y: Row index or indices. None means random.

    Returns:
        List of (row, col) index pairs.

    Raises:
        ValueError: If both x and y are lists with mismatched lengths.
    """
    if x is None and y is None:
        rand_y = int(np.random.randint(0, n_y))
        rand_x = int(np.random.randint(0, n_x))
        return [(rand_y, rand_x)]

    x_list: list[int] = [x] if isinstance(x, int) else (list(x) if x is not None else [])
    y_list: list[int] = [y] if isinstance(y, int) else (list(y) if y is not None else [])

    # Scalar x with list y, or vice versa
    if isinstance(y, list) and isinstance(x, int):
        return [(yi, x) for yi in y]
    if isinstance(x, list) and isinstance(y, int):
        return [(y, xi) for xi in x]
    if isinstance(x, list) and isinstance(y, list):
        if len(x_list) != len(y_list):
            msg = f"x and y lists must have the same length, got {len(x_list)} vs {len(y_list)}"
            raise ValueError(msg)
        return list(zip(y_list, x_list, strict=True))

    # Both scalars or one-element case
    yi = y_list[0] if y_list else int(np.random.randint(0, n_y))
    xi = x_list[0] if x_list else int(np.random.randint(0, n_x))
    return [(yi, xi)]


def _label_spatial_axes(ax: MplAxes) -> None:
    """Add standard x/y um labels to a spatial-map axes."""
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")


def _avg_param_map(arr: NDArray, h: int, w: int) -> NDArray:
    """Reshape a parameter array to (h, w) by averaging over leading dims.

    Args:
        arr: Parameter array with shape (n_pol, n_frange, H, W), (H, W),
            (..., n_pixel), or (n_pixel,).
        h: Spatial height.
        w: Spatial width.

    Returns:
        2-D array with shape (h, w).
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4:
        return np.nanmean(arr.reshape(-1, h, w), axis=0)
    if arr.ndim == 1:
        return arr.reshape(h, w)
    return np.nanmean(arr.reshape(-1, h * w), axis=0).reshape(h, w)
