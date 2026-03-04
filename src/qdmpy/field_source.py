"""Physical field source descriptions for QDM measurements.

A FieldSource describes a physical contributor to the measured B-field (e.g.
a ferromagnetic layer, a current-carrying wire, an applied bias coil). Each
source carries optional pre-computed spatial field maps.

The full subclass taxonomy (MagneticSource, UpwardContinuedSource, etc.) is
defined in QEP-050. This module provides only the base class needed by
QDMResult and the .qdm serialisation format.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FieldSource(BaseModel):
    """Base class for physical sources contributing to a measured B field.

    Subclasses add the parameters specific to the source geometry and
    material (current loops, ferromagnetic layers, uniform bias fields,
    etc.). The full subclass taxonomy is defined in QEP-050.

    The ``kind`` field is the Pydantic discriminator. All subclasses must
    declare it as a Literal. The base class defaults to "generic" so that
    bare FieldSource instances can participate in the discriminated union.

    Attributes:
        kind: Discriminator literal identifying the source type.
        name: Human-readable label for this source.
        field_map: Optional pre-computed spatial field map (H, W) in uT.
            Excluded from JSON serialisation (stored as HDF5 dataset).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "generic"
    name: str
    field_map: Any | None = None  # NDArray | None -- Any avoids import at runtime


# Type alias for discriminated union -- extended by QEP-050 to include
# concrete subclasses (MagneticSource, UpwardContinuedSource, etc.).
FieldSourceType = FieldSource
