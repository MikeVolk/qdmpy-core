"""Magnetic field reconstruction from B111 maps.

Provides the MagneticMap result object and core Bxyz reconstruction physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from loguru import logger


@runtime_checkable
class FieldReconstructor(Protocol):
    """Protocol for B111 → 3D magnetic field reconstruction.

    Implement this protocol to replace the default Fourier inversion used
    by ``MagneticMap.from_b111()``.

    **Custom reconstructor contract:**

    .. code-block:: python

        import xarray as xr
        from qdmpy import FieldReconstructor, QDMResult

        class MyReconstructor:
            def reconstruct(
                self,
                b111: xr.DataArray,
                nv_axis: tuple[float, float, float],
            ) -> xr.Dataset:
                # b111: dims (y, x), values in µT, attrs['pixel_spacing'] in metres
                # Must return Dataset with variables: 'bx', 'by', 'bz', 'btotal'
                # Units: µT, dims (y, x) on each variable
                bz = b111  # trivial placeholder
                return xr.Dataset({'bx': bz * 0, 'by': bz * 0,
                                   'bz': bz, 'btotal': abs(bz)})

        result = QDMResult(fit_result=fit_result, reconstructor=MyReconstructor())
        result.magnetic_map   # uses MyReconstructor

    Note:
        The returned Dataset must contain exactly the variables ``bx``, ``by``,
        ``bz``, and ``btotal`` with dims ``(y, x)`` and values in µT.
    """

    def reconstruct(
        self,
        b111: xr.DataArray,
        nv_axis: tuple[float, float, float],
    ) -> xr.Dataset:
        """Reconstruct (bx, by, bz, btotal) from a B111 map.

        Args:
            b111: DataArray with dims (y, x), values in µT, and
                ``pixel_spacing`` (metres) in ``.attrs``.
            nv_axis: NV unit vector (ux, uy, uz) in the lab frame.

        Returns:
            Dataset with variables 'bx', 'by', 'bz', 'btotal', each a
            DataArray with dims (y, x) and values in µT.
        """
        ...


def _reconstruct_bxyz(
    b111: np.ndarray,
    pixel_spacing: float,
    nv_axis: tuple[float, float, float],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct (Bx, By, Bz) from B111 in the Fourier domain.

    Uses free-space Maxwell equations ∇×B=0, ∇·B=0 above source to invert
    the NV projection and recover the full 3D field vector.

    Args:
        b111: 2D numpy array of B111 values (units: µT).
        pixel_spacing: Pixel size in metres.
        nv_axis: NV unit vector (ux, uy, uz) in lab frame.
        epsilon: Regularisation term for k=0 singularity (typically 1e-30).

    Returns:
        Tuple (bx, by, bz), each (H, W) ndarray in µT.

    References:
        QDMBzFromBu.m (Eduardo A. Lima, 2017)
        MITBxByFromBz.m (Eduardo A. Lima, 2007)
    """
    ux, uy, uz = nv_axis
    ny, nx = b111.shape

    # Wavenumber grid
    fy = np.fft.fftfreq(ny, d=pixel_spacing)
    fx = np.fft.fftfreq(nx, d=pixel_spacing)
    Fx, Fy = np.meshgrid(fx, fy)
    kx = 2 * np.pi * Fx
    ky = 2 * np.pi * Fy
    k = np.sqrt(kx**2 + ky**2)

    # FFT of B111
    F_b111 = np.fft.fft2(b111)

    # Step 1: B111 → Bz (invert NV projection)
    # Regularize denominator to avoid k=0 singularity
    denom = uz * k - uy * 1j * ky - ux * 1j * kx + epsilon
    H_bz = k / denom
    F_bz = F_b111 * H_bz

    # Step 2: Bz → Bx, By (free-space Maxwell)
    # Regularize k to avoid division by zero at k=0
    k_safe = k + epsilon
    bz = np.real(np.fft.ifft2(F_bz))
    bx = np.real(np.fft.ifft2(F_bz * (-1j * kx / k_safe)))
    by = np.real(np.fft.ifft2(F_bz * (-1j * ky / k_safe)))

    return bx, by, bz


@dataclass(frozen=True)
class MagneticMap:
    """Full 3D magnetic field reconstructed from a B111 map.

    All field components are xr.DataArray with dims (y, x) and units µT.
    This is an immutable result object; no modifications are allowed post-construction.
    """

    b111: xr.DataArray
    bx: xr.DataArray
    by: xr.DataArray
    bz: xr.DataArray
    btotal: xr.DataArray
    nv_axis: tuple[float, float, float]

    @classmethod
    def from_b111(
        cls,
        b111: xr.DataArray,
        nv_axis: tuple[float, float, float] | None = None,
        epsilon: float | None = None,
        reconstructor: FieldReconstructor | None = None,
    ) -> MagneticMap:
        """Reconstruct Bxyz from a preprocessed B111 map.

        Args:
            b111: DataArray with dims (y, x), values in µT, and
                  ``pixel_spacing`` (metres) in ``.attrs``.
            nv_axis: NV unit vector (ux, uy, uz). Defaults to
                     ``get_settings().nv.axis``.
            epsilon: k=0 regularisation. Defaults to
                     ``get_settings().nv.epsilon``. Ignored when
                     ``reconstructor`` is provided.
            reconstructor: Optional custom :class:`FieldReconstructor`. When
                provided, the default Fourier inversion is bypassed and
                ``reconstructor.reconstruct(b111, nv_axis)`` is called instead.

        Returns:
            MagneticMap with b111, bx, by, bz, btotal.

        Raises:
            ValueError: If pixel_spacing not in b111.attrs.
        """
        from qdmpy.settings import get_settings

        if "pixel_spacing" not in b111.attrs:
            raise ValueError("b111.attrs must contain 'pixel_spacing' (metres)")

        logger.info("Reconstructing 3D magnetic field from B111 map")
        settings = get_settings()
        nv = nv_axis or settings.nv.axis

        def _da(arr: np.ndarray, name: str) -> xr.DataArray:
            return xr.DataArray(
                arr,
                dims=b111.dims,
                coords=b111.coords,
                attrs={**b111.attrs, "component": name},
            )

        if reconstructor is not None:
            logger.info("Using custom FieldReconstructor for Bxyz reconstruction")
            ds = reconstructor.reconstruct(b111, nv)
            return cls(
                b111=b111,
                bx=ds["bx"],
                by=ds["by"],
                bz=ds["bz"],
                btotal=ds["btotal"],
                nv_axis=nv,
            )

        eps = epsilon if epsilon is not None else settings.nv.epsilon
        ps = float(b111.attrs["pixel_spacing"])

        bx_arr, by_arr, bz_arr = _reconstruct_bxyz(b111.values, ps, nv, eps)
        btotal_arr = np.sqrt(bx_arr**2 + by_arr**2 + bz_arr**2)

        return cls(
            b111=b111,
            bx=_da(bx_arr, "Bx"),
            by=_da(by_arr, "By"),
            bz=_da(bz_arr, "Bz"),
            btotal=_da(btotal_arr, "Btotal"),
            nv_axis=nv,
        )

    def to_dataset(self) -> xr.Dataset:
        """Return all components as a single xr.Dataset.

        Returns:
            Dataset with variables {b111, Bx, By, Bz, Btotal} and metadata.
        """
        return xr.Dataset(
            {
                "b111": self.b111,
                "Bx": self.bx,
                "By": self.by,
                "Bz": self.bz,
                "Btotal": self.btotal,
            },
            attrs={"units": "µT", "nv_axis": list(self.nv_axis)},
        )

    def display(
        self,
        component: Literal["b111", "Bx", "By", "Bz", "Btotal"] = "Bz",
        **imshow_kwargs: object,
    ) -> None:
        """Quick matplotlib display of one component.

        Args:
            component: Which component to display (case-insensitive for Bx/By/Bz).
            **imshow_kwargs: Passed to xarray ``.plot(**imshow_kwargs)``.

        Raises:
            ValueError: If component is not recognized.
        """
        from qdmpy.plotting import plot_magnetic_component

        plot_magnetic_component(self, component, **imshow_kwargs)

    def save(self, path: str | Path) -> None:
        """Save all components to NetCDF.

        Args:
            path: File path for NetCDF output.
        """
        path_obj = Path(path) if isinstance(path, str) else path
        self.to_dataset().to_netcdf(path_obj)
        logger.info("MagneticMap saved to {}", path_obj)
