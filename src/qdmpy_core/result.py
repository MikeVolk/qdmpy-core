"""Top-level result container combining FitResult and MagneticMap.

QDMResult is returned by Measurement.fit_odmr() and serves as the single
object a user interacts with after fitting. It delegates all FitResult
properties directly and provides lazy access to MagneticMap (3D field
reconstruction) without requiring the user to bridge the two manually.

Layering note: QDMResult lives above both fitting/ and magnetic_map.py and
is the only module that imports from both layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr

from qdmpy_core.fitting.result import FitResult

if TYPE_CHECKING:
    from os import PathLike

    from qdmpy_core.magnetic_map import MagneticMap


class QDMResult(BaseModel):
    """Unified result container from a single QDM measurement.

    Wraps a FitResult and provides lazy access to MagneticMap (Fourier-domain
    3D field reconstruction). All FitResult properties are delegated directly
    so existing code that accesses b111_remanent, centers, chi2, etc. works
    without modification once the return type of fit_odmr() is updated.

    Attributes:
        fit_result: The underlying fitted parameters and B111 analysis.
        nv_axis: NV axis unit vector (ux, uy, uz). When None, the value is
            read from qdmpy_core settings at the time magnetic_map is first accessed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fit_result: FitResult
    nv_axis: tuple[float, float, float] | None = None
    reconstructor: Any | None = None  # FieldReconstructor | None

    _magnetic_map_cache: Any | None = PrivateAttr(default=None)  # MagneticMap | None

    def model_post_init(self: Self, __context: object) -> None:
        """Log initialization."""
        logger.info(
            f"QDMResult initialized: model={self.fit_result.model_name}, "
            f"scan={self.fit_result.scan_dimensions}"
        )

    def __repr__(self: Self) -> str:
        """Return a concise string representation of this QDMResult."""
        return (
            f"QDMResult(model={self.fit_result.model_name!r}, "
            f"scan_dimensions={self.fit_result.scan_dimensions})"
        )

    # ------------------------------------------------------------------
    # Delegated FitResult properties
    # ------------------------------------------------------------------

    @property
    def scan_dimensions(self: Self) -> tuple[int, int]:
        """Spatial dimensions (height, width)."""
        return self.fit_result.scan_dimensions

    @property
    def pixel_spacing(self: Self) -> float:
        """Physical pixel size in metres."""
        return self.fit_result.pixel_spacing

    @property
    def model_name(self: Self) -> str:
        """Name of the ESR model used for fitting."""
        return self.fit_result.model_name

    @property
    def metadata(self: Self) -> dict[str, Any]:
        """Fitting metadata from FitResult."""
        return self.fit_result.metadata

    @property
    def centers(self: Self) -> NDArray:
        """Resonance center frequencies in GHz."""
        return self.fit_result.centers

    @property
    def linewidths(self: Self) -> NDArray:
        """ODMR linewidths."""
        return self.fit_result.linewidths

    @property
    def contrasts(self: Self) -> NDArray:
        """ODMR contrasts."""
        return self.fit_result.contrasts

    @property
    def offsets(self: Self) -> NDArray:
        """Baseline offsets."""
        return self.fit_result.offsets

    @property
    def chi2(self: Self) -> NDArray:
        """Fit quality (chi-squared) values."""
        return self.fit_result.chi2

    @property
    def fit_states(self: Self) -> NDArray:
        """Fitting convergence state codes."""
        return self.fit_result.fit_states

    @property
    def b111(self: Self) -> xr.Dataset:
        """B111 magnetic field as xr.Dataset with 'remanent' and 'induced' (µT)."""
        return self.fit_result.b111

    @property
    def b111_remanent(self: Self) -> NDArray:
        """Remanent B111 field in µT, shape (height, width)."""
        return self.fit_result.b111_remanent

    @property
    def b111_induced(self: Self) -> NDArray:
        """Induced B111 field in µT, shape (height, width)."""
        return self.fit_result.b111_induced

    def get_parameter(self: Self, param_name: str) -> NDArray:
        """Get any fitted parameter by name."""
        return self.fit_result.get_parameter(param_name)

    def get_parameter_map(self: Self, param_name: str) -> NDArray:
        """Get a fitted parameter reshaped as a 2D spatial map (height, width)."""
        return self.fit_result.get_parameter_map(param_name)

    def get_fit_quality_metrics(self: Self) -> dict[str, float]:
        """Return fit quality statistics (chi2 stats, convergence rate)."""
        return self.fit_result.get_fit_quality_metrics()

    def plot(
        self: Self,
        param: str = "center",
        *,
        save: bool = False,
        filename: str | None = None,
    ) -> None:
        """Quick-plot a fitted parameter map. Delegates to FitResult.plot()."""
        self.fit_result.plot(param, save=save, filename=filename)

    def show(self: Self, *, save: bool = False, filename: str | None = None) -> None:
        """Quick-plot overview of all fitted parameters. Delegates to FitResult.show()."""
        self.fit_result.show(save=save, filename=filename)

    # ------------------------------------------------------------------
    # Magnetic map (lazy)
    # ------------------------------------------------------------------

    @property
    def magnetic_map(self: Self) -> MagneticMap:
        """3D magnetic field reconstruction (Bx, By, Bz) via Fourier inversion.

        Computed lazily on first access. Subsequent accesses return the cached
        result. The MagneticMap is built from b111_remanent with pixel_spacing
        embedded in attrs, and nv_axis from settings when not provided.

        Returns:
            MagneticMap with bx, by, bz, btotal DataArrays (µT).
        """
        if self._magnetic_map_cache is None:
            self._magnetic_map_cache = self._build_magnetic_map()
        return self._magnetic_map_cache

    def _build_magnetic_map(self: Self) -> MagneticMap:
        """Construct MagneticMap from b111_remanent and pixel_spacing."""
        from qdmpy_core.magnetic_map import MagneticMap

        logger.info("Building MagneticMap from B111 remanent field")
        b111_da = xr.DataArray(
            self.fit_result.b111_remanent,
            dims=("y", "x"),
            attrs={"pixel_spacing": self.fit_result.pixel_spacing},
        )
        return MagneticMap.from_b111(
            b111_da, nv_axis=self.nv_axis, reconstructor=self.reconstructor
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self: Self, path: str | PathLike) -> None:
        """Save QDMResult to a pickle-free NPZ file.

        Delegates to ``FitResult._build_save_dict()`` for the fitting data and
        appends ``nv_axis`` if present. MagneticMap is not serialised — it is
        recomputed lazily after loading.

        Args:
            path: Destination file path (.npz extension added if absent).
        """
        path = Path(path)
        logger.info(f"Saving QDMResult to {path}")

        save_dict = self.fit_result._build_save_dict()
        if self.nv_axis is not None:
            save_dict["nv_axis"] = np.array(self.nv_axis)

        arrays = {k: np.asarray(v) for k, v in save_dict.items()}
        np.savez_compressed(path, allow_pickle=False, **arrays)
        logger.info(f"QDMResult saved to {path}")

    @classmethod
    def load(cls: type[QDMResult], path: str | PathLike) -> QDMResult:
        """Load a QDMResult from a pickle-free NPZ file.

        Opens the file exactly once and reconstructs both FitResult and
        nv_axis from the same data handle.

        Args:
            path: Path to the .npz file created by QDMResult.save().

        Returns:
            Reconstructed QDMResult. MagneticMap will be recomputed on first
            access to .magnetic_map.

        Raises:
            DataLoadError: If the file does not exist or is not in the safe format.
        """
        from qdmpy_core.exceptions import DataLoadError

        path = Path(path)

        if not path.exists():
            msg = f"Results file not found: {path}"
            raise DataLoadError(msg)

        try:
            data = np.load(path, allow_pickle=False)
        except ValueError as exc:
            msg = f"File {path} contains pickled objects and cannot be loaded safely."
            raise DataLoadError(msg) from exc

        fit_result = FitResult._from_npz(data, source=str(path))

        nv_axis: tuple[float, float, float] | None = None
        if "nv_axis" in data:
            nv_axis = tuple(float(v) for v in data["nv_axis"])  # type: ignore[assignment]

        logger.info(f"QDMResult loaded from {path}")
        return cls(fit_result=fit_result, nv_axis=nv_axis)
