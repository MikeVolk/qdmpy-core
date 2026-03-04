"""Top-level result container combining FitResult and MagneticMap.

QDMResult is returned by Measurement.fit_odmr() and serves as the single
object a user interacts with after fitting. It delegates all FitResult
properties directly and provides lazy access to MagneticMap (3D field
reconstruction) without requiring the user to bridge the two manually.

QDMResult is a **pure data container**. All I/O is in ``qdmpy.io``.
All plotting is in ``qdmpy.plotting``.

Layering note: QDMResult lives above both fitting/ and magnetic_map.py and
is the only module that imports from both layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PrivateAttr

from qdmpy.field_source import FieldSourceType
from qdmpy.fitting.result import FitResult

if TYPE_CHECKING:
    from qdmpy.magnetic_map import MagneticMap


class QDMResult(BaseModel):
    """Unified result container from a single QDM measurement.

    Pure data container. All I/O is handled by ``qdmpy.io``.
    All plotting is handled by ``qdmpy.plotting``.

    Wraps a FitResult and provides lazy access to MagneticMap (Fourier-domain
    3D field reconstruction). All FitResult properties are delegated directly
    so existing code that accesses b111_remanent, centers, chi2, etc. works
    without modification.

    Attributes:
        fit_result: The underlying fitted parameters and B111 analysis.
        nv_axis: NV axis unit vector (ux, uy, uz). When None, the value is
            read from qdmpy settings at the time magnetic_map is first accessed.
        reconstructor: Optional FieldReconstructor override. When None, the
            default reconstructor is used.
        light_image: Optional LED reflectance image (H, W). Attached by
            Measurement.fit_odmr() and preserved through save/load.
        laser_image: Optional NV fluorescence image (H, W). Also called the
            "diamond image". Attached by Measurement.fit_odmr().
        field_sources: Physical B-field sources contributing to the measured
            field. Extended by QEP-050 with concrete subclass types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fit_result: FitResult
    nv_axis: tuple[float, float, float] | None = None
    reconstructor: Any | None = None  # FieldReconstructor | None
    light_image: NDArray | None = None
    laser_image: NDArray | None = None
    field_sources: list[FieldSourceType] = []

    _magnetic_map_cache: Any | None = PrivateAttr(default=None)  # MagneticMap | None

    def model_post_init(self: Self, __context: object) -> None:
        """Log initialization."""
        logger.info(
            "QDMResult initialized: model={}, scan={}",
            self.fit_result.model_name,
            self.fit_result.scan_dimensions,
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
    def parameters(self: Self) -> dict[str, NDArray]:
        """Fitted parameter arrays keyed by name."""
        return self.fit_result.parameters

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
        """B111 magnetic field as xr.Dataset with 'remanent' and 'induced' (uT)."""
        return self.fit_result.b111

    @property
    def b111_remanent(self: Self) -> NDArray:
        """Remanent B111 field in uT, shape (height, width)."""
        return self.fit_result.b111_remanent

    @property
    def b111_induced(self: Self) -> NDArray:
        """Induced B111 field in uT, shape (height, width)."""
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

    def calculate_b_field(self: Self, force_recalculate: bool = False) -> NDArray:
        """Calculate magnetic field map from fitted resonance frequencies."""
        return self.fit_result.calculate_b_field(force_recalculate=force_recalculate)

    # ------------------------------------------------------------------
    # Magnetic map (lazy)
    # ------------------------------------------------------------------

    @property
    def has_cached_magnetic_map(self: Self) -> bool:
        """Whether the MagneticMap has already been computed."""
        return self._magnetic_map_cache is not None

    @property
    def magnetic_map(self: Self) -> MagneticMap:
        """3D magnetic field reconstruction (Bx, By, Bz) via Fourier inversion.

        Computed lazily on first access. Subsequent accesses return the cached
        result. The MagneticMap is built from b111_remanent with pixel_spacing
        embedded in attrs, and nv_axis from settings when not provided.

        Returns:
            MagneticMap with bx, by, bz, btotal DataArrays (uT).
        """
        if self._magnetic_map_cache is None:
            self._magnetic_map_cache = self._build_magnetic_map()
        return self._magnetic_map_cache

    # ------------------------------------------------------------------
    # Convenience I/O wrappers (thin delegation to qdmpy.io)
    # ------------------------------------------------------------------

    def save(
        self: Self,
        path: str | Path,
        *,
        include_bxyz: bool = False,
        overwrite: bool = False,
        compress: bool = True,
    ) -> None:
        """Save this result to disk.

        Dispatches to :func:`qdmpy.io.save_qdm` for ``.qdm`` files and
        :func:`qdmpy.io.save_npz` for everything else.

        Args:
            path: Destination path. Use ``.qdm`` extension for the full HDF5
                archive (images, B111, field sources). Use ``.npz`` for the
                lightweight fit-data-only checkpoint.
            include_bxyz: Include Bxyz reconstruction in the ``.qdm`` file.
                Ignored for NPZ format. Default False.
            overwrite: Overwrite existing ``.qdm`` file. Default False.
            compress: Apply GZIP compression in ``.qdm`` file. Default True.
        """
        from qdmpy.io import save_npz, save_qdm

        if Path(path).suffix.lower() == ".qdm":
            save_qdm(self, path, include_bxyz=include_bxyz, overwrite=overwrite, compress=compress)
        else:
            save_npz(self, path)

    @classmethod
    def load(cls: type[QDMResult], path: str | Path) -> QDMResult:
        """Load a QDMResult from disk.

        Dispatches to :func:`qdmpy.io.load_qdm` for ``.qdm`` files and
        :func:`qdmpy.io.load_npz` for everything else.

        Args:
            path: Path to a ``.qdm`` or ``.npz`` file.

        Returns:
            Reconstructed QDMResult.

        Raises:
            DataLoadError: If the file does not exist or cannot be parsed.
        """
        from qdmpy.io import load_npz, load_qdm

        if Path(path).suffix.lower() == ".qdm":
            return load_qdm(path)
        return load_npz(path)

    def _build_magnetic_map(self: Self) -> MagneticMap:
        """Construct MagneticMap from b111_remanent and pixel_spacing."""
        from qdmpy.magnetic_map import MagneticMap

        logger.info("Building MagneticMap from B111 remanent field")
        b111_da = xr.DataArray(
            self.fit_result.b111_remanent,
            dims=("y", "x"),
            attrs={"pixel_spacing": self.fit_result.pixel_spacing},
        )
        return MagneticMap.from_b111(
            b111_da, nv_axis=self.nv_axis, reconstructor=self.reconstructor
        )
