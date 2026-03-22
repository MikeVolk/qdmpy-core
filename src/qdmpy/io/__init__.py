"""QDMpy I/O package.

Provides all file-based persistence utilities:

- :mod:`qdmpy.io.images` -- optical image loading (CSV/JPG) and metadata TOML.
- :mod:`qdmpy.io.npz`    -- lightweight NPZ checkpoint (fit data only).
- :mod:`qdmpy.io.qdm`    -- full-fidelity HDF5 .qdm archive format.

All public symbols are re-exported from this package so that existing code
using ``from qdmpy.io import get_image`` continues to work unchanged.
"""

from __future__ import annotations

from qdmpy.io.images import get_image, get_image_file, has_csv, load_metadata_toml
from qdmpy.io.magnetic_map import save_magnetic_map
from qdmpy.io.npz import load_npz, save_npz
from qdmpy.io.qdm import load_qdm, save_qdm

__all__ = [
    # images
    "get_image",
    "get_image_file",
    "has_csv",
    "load_metadata_toml",
    "save_magnetic_map",
    # npz
    "load_npz",
    "save_npz",
    # qdm
    "load_qdm",
    "save_qdm",
]
