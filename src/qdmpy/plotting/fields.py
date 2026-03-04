"""Magnetic field map visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from qdmpy.plotting._common import _add_colorbar

if TYPE_CHECKING:
    from qdmpy.magnetic_map import MagneticMap


def plot_magnetic_component(
    mag_map: MagneticMap,
    component: str = "Bz",
) -> None:
    """Plot one MagneticMap component as a spatially-resolved map.

    Uses a symmetric RdBu_r colormap clipped to the 99th percentile of |B|.
    Aspect ratio is preserved (one pixel = one pixel).

    Args:
        mag_map: MagneticMap instance.
        component: Which component to display: ``'b111'``, ``'Bx'``, ``'By'``,
            ``'Bz'``, or ``'Btotal'``.

    Raises:
        ValueError: If component is not recognized.
    """
    logger.debug("Plotting magnetic component: {}", component)
    component_lower = component.lower()
    valid_components = {"b111", "bx", "by", "bz", "btotal"}

    if component_lower not in valid_components:
        msg = f"Component {component!r} not in {valid_components}"
        raise ValueError(msg)

    da = getattr(mag_map, component_lower)
    arr = da.values  # (H, W), uT
    ps_um = float(da.attrs.get("pixel_spacing", 1.0)) * 1e6
    h, w = arr.shape
    extent = (0, w * ps_um, h * ps_um, 0)  # origin="upper"

    vmax = float(np.nanpercentile(np.abs(arr), 99))
    cmap = "RdBu_r" if component_lower != "btotal" else "viridis"
    vmin = -vmax if component_lower != "btotal" else 0.0

    _fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        arr, extent=extent, origin="upper", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax
    )
    _add_colorbar(im, ax, label=f"{component} (µT)")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title(component)
    plt.tight_layout()
    plt.show()
