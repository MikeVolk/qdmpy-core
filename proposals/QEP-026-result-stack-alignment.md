# QEP-026 — ResultStack and Image Alignment

**Status:** Draft
**Created:** 2026-02-18
**Depends on:** QEP-025 (semantic coordinates)

---

## Motivation

QDMlab's most powerful workflows operate on *series* of measurements — repeated scans of
the same sample at different applied fields, different demagnetization states, or different
time points. These measurements are spatially misregistered because the sample shifts
slightly between remounting. QDMpy has no equivalent of:

- `get_tform_multi` — batch alignment of N folders to one reference
- `get_transformed_maps` — align + filter + upward-continue + stack pipeline
- `align_images` — interactive alignment with fallback to manual control points

Without these, multi-run paleomagnetic workflows (coercivity, viscosity, demagnetization
series) are impossible in QDMpy.

---

## Goals

1. `AlignmentTransform` — serialisable affine transform between two images.
2. `AlignmentEstimator` — computes transforms from optical images.
3. `StackEntry` — one measurement: `FitResult` + optical images + alignment + mask.
4. `ResultStack` — ordered collection of `StackEntry` objects with alignment, stacking,
   and iteration API.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### 3.1  `AlignmentTransform`

Wraps a 3×3 affine matrix in homogeneous coordinates. All transforms are computed in the
coordinate system of the *source image* (LED resolution). A separate method scales the
matrix to binned (B111 map) resolution.

```python
@dataclass
class AlignmentTransform:
    matrix: NDArray          # shape (3, 3) — maps [output] ← [input] in homogeneous coords
    source_shape: tuple[int, int]   # (H, W) of the image it was computed on
    method: str              # 'phase_correlation' | 'feature_based' | 'manual' | 'identity'
```

**Core methods:**

```python
def apply(self, image: NDArray, output_shape: tuple[int, int] | None = None) -> NDArray:
    """Apply this transform to an image using bilinear interpolation.

    Uses scipy.ndimage.affine_transform with the inverse matrix so that
    each output pixel is filled by sampling the input at the mapped location.
    """
    T_inv = np.linalg.inv(self.matrix)
    return affine_transform(
        image,
        T_inv[:2, :2],
        offset=T_inv[:2, 2],
        output_shape=output_shape or self.source_shape,
        order=1,           # bilinear — appropriate for field maps
        mode='constant',
        cval=np.nan,       # out-of-bounds → NaN so callers can build valid masks
    )

def scale_to(self, target_shape: tuple[int, int]) -> AlignmentTransform:
    """Return a new transform rescaled to a different image resolution.

    Compose with scale matrices to convert from source resolution to target resolution.
    If T_source maps output ← input in source pixel coords, the rescaled transform
    in target coords is: T_target = S @ T_source @ S⁻¹
    where S = diag(sy, sx, 1) with sy = target_h / source_h, sx = target_w / source_w.

    This correctly handles translation, rotation, and scale components.

    **Verification note:** Before implementation, verify this composition with a unit
    test using a known translation (e.g., 4 LED pixels, bin_factor=4 → 1 B111 pixel)
    and a known rotation to ensure the matrix direction is correct.
    """
    sy = target_shape[0] / self.source_shape[0]
    sx = target_shape[1] / self.source_shape[1]
    S     = np.diag([sy,    sx,    1.0])
    S_inv = np.diag([1/sy,  1/sx,  1.0])
    # NOTE: Verify this composition before implementation — ensure translation
    # direction and rotation sense match QDMlab's behavior on real LED/B111 pairs.
    return AlignmentTransform(S @ self.matrix @ S_inv, target_shape, self.method)

def valid_mask(self, shape: tuple[int, int] | None = None) -> NDArray:
    """Boolean mask of pixels that are within the source image after transformation."""
    # Apply transform to a ones-array; NaN positions are out-of-bounds
    ones = np.ones(shape or self.source_shape)
    warped = self.apply(ones, shape)
    return ~np.isnan(warped)

def to_dict(self) -> dict: ...
@classmethod
def from_dict(cls, d: dict) -> AlignmentTransform: ...

@classmethod
def identity(cls, shape: tuple[int, int]) -> AlignmentTransform:
    return cls(np.eye(3), shape, 'identity')
```

---

### 3.2  `AlignmentEstimator`

Stateless factory class — computes `AlignmentTransform` from a pair of images.

```python
class AlignmentEstimator:
    method: Literal['phase_correlation', 'feature_based'] = 'phase_correlation'
    upsample_factor: int = 10      # sub-pixel precision for phase correlation
```

**Phase correlation** (default — handles translation only, very robust for QDM):

```python
from skimage.registration import phase_cross_correlation

shift, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=self.upsample_factor)
# shift = [dy, dx] — how much `moving` is offset relative to `fixed`
# Build affine: to transform `moving` into `fixed` frame, shift by -shift
matrix = np.array([
    [1, 0, -shift[1]],
    [0, 1, -shift[0]],
    [0, 0,  1       ],
])
```

**Feature-based** (handles rotation and scale — falls back on phase failure):

```python
from skimage.feature import ORB
from skimage.measure import ransac
from skimage.transform import AffineTransform

# Detect ORB keypoints in both images
# Match descriptors → filter outliers with RANSAC
# Estimate AffineTransform from inliers
# Convert skimage AffineTransform to 3×3 homogeneous matrix
```

**Sequence vs reference mode** (matching QDMlab's `sequence` flag):

- **Reference mode** (default): every entry is aligned to the single fixed reference.
  Best for absolute field values, small drifts.
- **Sequence mode**: each entry is aligned to the *previous* entry. Composes transforms
  back to the reference. Best for large drifts where frame-to-frame shifts are small.

---

### 3.3  `StackEntry`

```python
@dataclass
class StackEntry:
    result: FitResult
    label: str                        # e.g. 'NRM', '10mT', '20mT'
    led_image: NDArray                # (H_led, W_led) — used for alignment
    laser_image: NDArray              # (H_led, W_led) — alternative alignment source
    metadata: dict[str, Any]          # arbitrary: applied_field, timestamp, folder, ...
    alignment: AlignmentTransform | None = None   # None = is the reference, or uncomputed
    valid_mask: NDArray | None = None             # (H_b111, W_b111) bool, post-alignment
```

The `alignment` on the reference entry is always `None` (or identity). All other entries
hold a transform that brings them into the reference frame.

---

### 3.4  `ResultStack`

```python
class ResultStack:
    def __init__(self, reference_idx: int = 0) -> None: ...

    # Building the stack
    def add(
        self,
        result: FitResult,
        led_image: NDArray,
        laser_image: NDArray,
        label: str,
        metadata: dict | None = None,
    ) -> None: ...

    # Alignment
    def compute_alignments(
        self,
        method: Literal['phase_correlation', 'feature_based'] = 'phase_correlation',
        source: Literal['led', 'laser'] = 'led',
        sequence: bool = False,
    ) -> None:
        """Compute AlignmentTransform for every non-reference entry.

        Transforms are computed in LED image space, then scaled to B111 map
        resolution via AlignmentTransform.scale_to(result.scan_dimensions).
        Sets entry.alignment and entry.valid_mask on each non-reference entry.
        """

    # Transform persistence
    def save_transforms(self, path: Path) -> None:
        """Save computed transforms to a JSON file for reuse."""

    def load_transforms(self, path: Path) -> None:
        """Load previously computed transforms, skipping re-computation."""

    # Validity
    @property
    def common_mask(self) -> NDArray:
        """(H, W) bool — pixels that are valid in every aligned entry.

        Raises:
            RuntimeError: if compute_alignments() has not been called yet
            (all valid_mask values are None, indicating alignment not computed).
        """
        masks = [e.valid_mask for e in self.entries if e.valid_mask is not None]
        if not masks:
            raise RuntimeError(
                'compute_alignments() must be called before accessing common_mask. '
                'All entries have uncomputed alignments (valid_mask is None).'
            )
        return np.logical_and.reduce(masks)

    @property
    def scan_dimensions(self) -> tuple[int, int]:
        return self.entries[0].result.scan_dimensions

    # Stacking
    def get_b111(self, idx: int, component: str = 'remanent') -> NDArray:
        """Return the aligned (H, W) field map for entry idx."""
        entry = self.entries[idx]
        raw = entry.result.b111[component].values  # or b111_remanent etc.
        if entry.alignment is None:
            return raw
        return entry.alignment.apply(raw, self.scan_dimensions)

    def iter_b111(self, component: str = 'remanent') -> Iterator[NDArray]:
        """Yield aligned field maps in stack order."""
        for i in range(len(self.entries)):
            yield self.get_b111(i, component)

    def stack_b111(
        self,
        component: str = 'remanent',
        reducer: Literal['mean', 'median', 'std'] = 'mean',
        mask: NDArray | None = None,
    ) -> NDArray:
        """Stack all aligned maps and reduce along the entry axis.

        Pixels outside any entry's valid region are NaN before reducing.
        """
        maps = np.stack(list(self.iter_b111(component)), axis=0)  # (N, H, W)
        if mask is not None:
            maps[:, ~mask] = np.nan
        return getattr(np, f'nan{reducer}')(maps, axis=0)

    def as_dataset(self, component: str = 'remanent') -> xr.Dataset:
        """Return stacked maps as an xr.Dataset with labelled coordinates.

        Dimensions: (entry, y, x)
        Coordinates: entry = [entry.label for entry in self.entries]
        """
```

---

## Data Flow

```
LED images (H_led, W_led)
    ↓  AlignmentEstimator.estimate(fixed_led, moving_led)
AlignmentTransform (3×3 matrix, LED space)
    ↓  .scale_to(b111_shape)
AlignmentTransform (3×3 matrix, B111 space)
    ↓  stored in StackEntry.alignment

ResultStack.get_b111(i)
    → entry.result.b111_remanent       (H, W) raw
    → entry.alignment.apply(raw)       (H, W) aligned
ResultStack.stack_b111()
    → np.nanmean(aligned_maps, axis=0) (H, W)
```

---

## Integration: Extracting LED/Laser Images from Measurement

`ResultStack.add()` requires `led_image` and `laser_image` as explicit parameters.
These are not stored on `FitResult`; they live on `Measurement`:

```python
measurement.light_image      # (H_led, W_led) — the LED reference image
measurement.laser_image      # (H_led, W_led) — the laser reference image (optional)
```

**Caller responsibility:** Code that constructs a `ResultStack` must extract these
from the `Measurement` object and pass them to `stack.add()`. Example workflow:

```python
from QDMpy.result_stack import ResultStack

stack = ResultStack()
for measurement in measurements:
    fit_result = measurement.fit_odmr(model_name='ESR14N')
    stack.add(
        result=fit_result,
        led_image=measurement.light_image,
        laser_image=measurement.laser_image,
        label=f"sample_{measurement.folder_name}",
        metadata={'applied_field': 100, 'timestamp': measurement.timestamp},
    )

stack.compute_alignments(method='phase_correlation', source='led')
```

---

## Key Implementation Details

### LED vs B111 resolution

QDM typically bins 4×4 (or more). LED images (~5 MP) are much higher resolution than
B111 maps (~300k pixels). Alignment is always computed in LED space (better SNR for
feature matching), then the affine matrix is rescaled:

```python
b111_shape = result.scan_dimensions
led_shape = led_image.shape
transform_b111 = transform_led.scale_to(b111_shape)
```

For a pure translation `[dy, dx]` in LED pixels and bin factor `b`:
```
translation_b111 = [dy / b, dx / b]
```
For a general affine: `T_b111 = S⁻¹ @ T_led @ S` where `S = diag(b, b, 1)`.

### Transform serialization (JSON)

```json
{
  "matrix": [[1.0, 0.0, -3.2], [0.0, 1.0, 1.7], [0.0, 0.0, 1.0]],
  "source_shape": [1200, 1920],
  "method": "phase_correlation"
}
```

Keyed by entry label in a dict, so partial results can be loaded/saved without
re-running the whole alignment.

### NaN handling during stacking

`affine_transform` with `cval=np.nan` produces NaN for any output pixel that maps
outside the input image boundary. `np.nanmean` then naturally ignores missing pixels
per the `common_mask` logic.

---

## Alternatives Considered

### A. Store B111 maps directly instead of FitResult
Rejected. Keeping `FitResult` allows re-extracting any parameter (chi2, width, contrast)
through the same pipeline, not just B111.

### B. Compute alignment on B111 maps rather than LED images
Rejected. B111 maps are noisier than optical images and often have uniform regions
(non-magnetic background) that confuse feature detectors. QDMlab always aligns on LED.

### C. Use `skimage.transform.warp` instead of `scipy.ndimage.affine_transform`
Either works. `scipy.ndimage` avoids an extra dependency (skimage already needed for
feature detection); `skimage.transform.warp` is marginally more flexible. Use scipy.

### D. Store transforms as skimage `AffineTransform` objects
Rejected. The plain 3×3 numpy matrix is self-contained, trivially serializable, and
language-agnostic if we ever need to exchange with MATLAB.

---

## Dependencies

**Required additions to `pyproject.toml`:**

```toml
"scikit-image>=0.21",
```

`scikit-image` is needed for:
- `skimage.registration.phase_cross_correlation` (phase correlation alignment)
- `skimage.feature.ORB` (feature-based fallback alignment)
- `skimage.measure.ransac` (outlier rejection)

`scipy.ndimage.affine_transform` is already available (scipy is a core dependency).

After adding scikit-image, run `uv lock` to update the lock file.

---

## Migration

No existing public API is broken. `FitResult` gains no new interface. The entire
`ResultStack`/`AlignmentTransform` surface is new.
