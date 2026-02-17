# QEP-011: Spectral Folding for Model-Free Parameter Extraction

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Priority** | P2 |
| **Complexity** | L |
| **Depends on** | QEP-005 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |

## Motivation

### The technique: spectral folding in Mossbauer spectroscopy

In Mossbauer spectroscopy, a triangular velocity drive produces two mirror-image
halves of the absorption spectrum per oscillation cycle: one from the ascending
velocity ramp and one from the descending ramp. **Spectral folding** is the
standard technique of combining these two halves by:

1. Finding the optimal **folding point** (the channel of mirror symmetry)
2. Reversing one half-spectrum
3. Adding the two halves together

This yields three critical benefits:

- **sqrt(2) SNR improvement** from doubling the effective count time at each velocity
- **Parabolic baseline cancellation** (the "cosine effect" from changing
  source-detector distance cancels between the two halves)
- **Model-free calibration**: the folding point itself encodes the velocity
  zero without requiring a full spectral fit

The folding point is found by minimizing the sum of squared differences between
symmetric channel pairs: `sum[(C[f-i] - C[f+i])^2]` over candidate fold
centers `f`. More sophisticated approaches use DFT-based symmetry analysis
(PyMossFit, Saccone 2024).

### The analogy to QDM ODMR

NV center ODMR spectra have a strikingly similar structure. The ground-state
spin Hamiltonian gives two resonance frequencies:

```
f+ = D + gamma * B * cos(theta) + delta_strain
f- = D - gamma * B * cos(theta) + delta_strain
```

where:
- `D` = zero-field splitting (~2.870 GHz), varies with temperature and strain
- `gamma` = 28.024 GHz/T (NV gyromagnetic ratio)
- `B * cos(theta)` = projected magnetic field along the NV axis
- `delta_strain` = common-mode shift from crystal strain

The two frequency ranges measured in a QDM experiment (low ~2.82-2.87 GHz and
high ~2.87-2.92 GHz) capture `f-` and `f+` respectively. These are **mirror
images about D**, exactly analogous to the two halves of a Mossbauer spectrum
about the velocity zero.

### What QDMpy currently does (and doesn't do)

The existing pipeline fits each frequency range independently, then computes
magnetic field from the difference of fitted centers:

```python
# result.py: _compute_delta_resonance
freq_diff = resonance[:, 1] - resonance[:, 0]  # f+ - f-
delta_resonance = freq_diff / 2 / GAMMA_NV * 1e6 * d
```

This approach:
- Fits each half independently (no information sharing between the two ranges)
- Requires a successful fit before any parameter extraction
- Assumes D_ZFS is a global constant (2.870 GHz everywhere)
- Cannot separate temperature/strain shifts from magnetic field effects
- Provides no model-free quality metric for raw data

Spectral folding addresses all of these limitations.

## Specification

### 1. Core folding algorithm

The `SpectralFolder` class operates on raw ODMR data (before fitting) to
combine the two frequency ranges.

```python
class SpectralFolder:
    """Fold ODMR spectra by exploiting f+/f- mirror symmetry about D_ZFS.

    Analogous to Mossbauer spectral folding, where two halves of the
    absorption spectrum are combined about the velocity zero-point.
    """

    def __init__(
        self,
        data: xr.DataArray,
        d_zfs_initial: float = D_ZFS,
        search_range: float = 0.01,
        search_steps: int = 201,
    ) -> None:
        """Initialize the folder.

        Args:
            data: ODMR data with dims (polarity, freq_range, y, x, freq_idx).
                  Must have exactly 2 freq_ranges.
            d_zfs_initial: Initial guess for zero-field splitting (GHz).
            search_range: Half-width of D_ZFS search window (GHz).
            search_steps: Number of candidate D values to test.
        """

    def find_fold_point(self) -> xr.DataArray:
        """Find optimal D_ZFS per pixel by minimizing fold residual.

        For each pixel, tests candidate D values and finds the one that
        minimizes the squared difference between the reflected high-frequency
        spectrum and the low-frequency spectrum.

        Returns:
            DataArray of optimal D_ZFS values with dims (polarity, y, x),
            in GHz.
        """

    def fold(self, d_zfs_map: xr.DataArray | None = None) -> xr.DataArray:
        """Fold the two frequency ranges into a single spectrum.

        Reflects the high-frequency range about D_ZFS and adds it to the
        low-frequency range, producing a folded spectrum with improved SNR.

        Args:
            d_zfs_map: Per-pixel D_ZFS values. If None, uses find_fold_point().

        Returns:
            Folded DataArray with dims (polarity, y, x, freq_idx) and
            frequency coordinates relative to D_ZFS (i.e., in units of
            gamma * B * cos(theta)).
        """

    def fold_residual(self, d_zfs_map: xr.DataArray | None = None) -> xr.DataArray:
        """Compute per-pixel residual between the two reflected halves.

        The fold residual is a model-free quality metric. High values
        indicate:
        - Strong transverse strain (breaks f+/f- symmetry)
        - Multiple NV orientations contributing
        - Data artifacts or poor signal

        Returns:
            DataArray with dims (polarity, y, x), values in [0, 1]
            normalized by total spectral variance.
        """
```

### 2. Mathematical details

#### 2.1 Fold-point optimization (per pixel)

Given low-frequency spectrum `S_low(f)` and high-frequency spectrum `S_high(f)`:

```
S_high_reflected(f; D) = S_high(2D - f)
```

The optimal fold point minimizes:

```
chi2(D) = sum_f [ S_low(f) - S_high_reflected(f; D) ]^2
```

Since the two frequency ranges may have different frequency grids, reflection
requires interpolation. We use `scipy.interpolate.interp1d` (linear) to
map `S_high` onto the reflected grid.

For a 2000x2000 image, the brute-force approach (201 candidate D values per
pixel) is tractable because the inner loop is a simple dot product. Estimated
cost: ~4M pixels x 201 candidates x 50 freq points = ~40 GFLOP, feasible in
<10 seconds with numba parallelization.

#### 2.2 DFT-based fold-point refinement (optional)

Following PyMossFit (Saccone 2024), the folding channel can be refined using
the discrete Fourier transform. The DFT of a perfectly folded (symmetric)
spectrum has zero imaginary components. The fold point is the shift that
minimizes the imaginary energy:

```
D_opt = argmin_D  sum_k |Im[ DFT( S_shifted(f - D) ) ]|^2
```

This is more robust than brute-force minimization for noisy data and naturally
handles sub-channel (sub-frequency-bin) precision.

#### 2.3 Folded spectrum construction

Once `D_opt` is known per pixel:

```
S_folded(delta_f) = S_low(D - delta_f) + S_high(D + delta_f)
```

where `delta_f = gamma * B * cos(theta)` is the frequency offset from D_ZFS.
The folded spectrum has:
- **Double the effective measurement time** at each frequency offset (sqrt(2) SNR improvement)
- **Common-mode rejection** of baseline variations that are symmetric about D
  (analogous to cosine-effect cancellation in Mossbauer)
- A natural frequency axis in **magnetic field units** rather than absolute GHz

#### 2.4 Symmetry decomposition

The fold naturally decomposes the spectrum into symmetric and antisymmetric
components:

```
S_symmetric(delta_f) = [S_low(D - delta_f) + S_high(D + delta_f)] / 2
S_antisymmetric(delta_f) = [S_low(D - delta_f) - S_high(D + delta_f)] / 2
```

- **S_symmetric**: Contains the magnetic field information (resonance dips).
  This is the "folded spectrum" used for fitting.
- **S_antisymmetric**: Contains residual asymmetry from transverse strain,
  off-axis fields, or systematic errors. Should be ~zero for ideal data.
  Its magnitude is the fold residual.

### 3. Polarity folding (second-level fold)

The two polarities (positive/negative applied field) provide another folding
axis. Currently, the code computes B111 components post-fit:

```python
b111_remanent = (neg_difference + pos_difference) / 2
b111_induced = (neg_difference - pos_difference) / 2
```

This can be extended to the raw spectral level. Before fitting, the two
polarity spectra can be decomposed:

```
S_remanent(f) = [S_pol0(f) + S_pol1(f)] / 2   # field-independent signal
S_induced(f) = [S_pol0(f) - S_pol1(f)] / 2    # field-dependent signal
```

Fitting `S_remanent` and `S_induced` separately can improve sensitivity to
weak induced fields that would otherwise be masked by a strong remanent signal.

### 4. Integration with existing pipeline

The folder slots into the processing pipeline between `Processor` and
`FitManager`:

```
Raw Data  ->  Processors  ->  SpectralFolder  ->  FitManager  ->  FitResult
                                   |
                                   +-> D_ZFS map (model-free)
                                   +-> fold residual map (quality metric)
                                   +-> folded spectrum (for improved fitting)
```

#### 4.1 New processor: FoldingProcessor

```python
class FoldingProcessor(BaseProcessor):
    """Processor that folds ODMR spectra about D_ZFS.

    Produces a folded DataArray with reduced dimensionality
    (freq_range dimension is eliminated) and improved SNR.
    """

    def process(self, data: ODMRData) -> ODMRData:
        """Fold the data and return folded ODMRData.

        The returned ODMRData has:
        - freq_range dimension eliminated (folded into single range)
        - frequency coordinate relative to D_ZFS
        - metadata containing d_zfs_map and fold_residual
        """
```

#### 4.2 Folded-spectrum fitting

After folding, the spectrum has a single frequency range centered on D_ZFS.
The existing models work unchanged: the center parameter now represents the
Zeeman shift `gamma * B * cos(theta)` directly, rather than an absolute
frequency. This simplifies constraint specification and interpretation.

The constraint for center shifts from `(2.0, 3.1) GHz` to `(0.0, 0.15) GHz`
(typical magnetic field range in QDM experiments).

#### 4.3 Settings extension

```python
class FoldingSettings(BaseSettings):
    enabled: bool = False
    d_zfs_initial: float = 2.870        # GHz
    search_range: float = 0.01          # GHz (+/- around d_zfs_initial)
    search_steps: int = 201
    use_dft_refinement: bool = True
    fold_polarities: bool = False       # Also fold polarity dimension
    min_fold_quality: float = 0.8       # Minimum quality to accept fold
```

### 5. New outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `d_zfs_map` | `(n_pol, y, x)` | Per-pixel zero-field splitting (GHz). Encodes temperature and strain variations across the diamond. |
| `fold_residual` | `(n_pol, y, x)` | Normalized residual in [0,1]. Model-free data quality metric. |
| `folded_spectrum` | `(n_pol, y, x, freq_idx)` | SNR-improved spectrum for fitting. |
| `antisymmetric_spectrum` | `(n_pol, y, x, freq_idx)` | Residual asymmetry. Diagnostic for strain, off-axis fields. |

The `d_zfs_map` is a genuinely new output that is not available from the
current pipeline. Spatial variations in D_ZFS are scientifically valuable:
- Temperature mapping (dD/dT ~ -74 kHz/K near room temperature)
- Strain mapping (sensitivity depends on crystal orientation)
- Diamond quality assessment

## Impact Assessment

### Performance

- **Fold-point search**: ~10s for 2000x2000 image (numba-parallelized). One-time
  cost, amortized over subsequent fitting.
- **Spectrum folding**: Negligible (<1s). Simple array operations on xarray.
- **Fitting improvement**: Fitting a single folded range vs. two separate ranges
  reduces the number of fits by 2x. Net speedup depends on whether the
  fold-point search time is offset by the fitting speedup.
- **Memory**: The folded spectrum is half the size of the original (one freq_range
  instead of two). D_ZFS map and residual add ~32 MB for 2000x2000.

### Signal-to-noise

- **Direct improvement**: sqrt(2) ~ 41% SNR improvement in the folded spectrum,
  translating to better-constrained fit parameters.
- **Indirect improvement**: Model-free D_ZFS determination removes one degree
  of freedom from the fit (the absolute frequency), further improving
  convergence and reducing parameter correlations.
- **Weak-field sensitivity**: For small B-fields where the two resonances
  barely separate, folding concentrates the signal and makes the splitting
  easier to detect.

### Scientific capability

- **D_ZFS mapping**: Entirely new capability. Enables temperature and strain
  imaging from the same dataset, without additional measurements.
- **Data quality assessment**: The fold residual provides a per-pixel quality
  metric before any fitting is attempted, enabling early rejection of bad
  regions.
- **Strain detection**: Strong antisymmetric components in the fold residual
  indicate transverse strain, which breaks the f+/f- symmetry. This is a
  model-free strain indicator.

### Risks

- **Asymmetric frequency ranges**: If the two measured frequency ranges have
  very different widths or are not centered on D_ZFS, folding requires
  extrapolation or produces a reduced frequency range. The implementation
  must handle partial overlap gracefully.
- **Different contrasts**: The f+ and f- dips may have different contrasts
  (e.g., due to microwave power variation across frequency). Folding averages
  these, which may not be desirable for all analyses.
- **Non-Lorentzian lineshapes**: If the two dips have different shapes
  (e.g., due to inhomogeneous broadening or multiple NV orientations),
  folding may smear features. The fold residual serves as a diagnostic
  for this case.
- **Complexity**: Adds a new processing step with several tunable parameters.
  Keeping it optional (disabled by default) mitigates adoption risk.

## Files Affected

- `src/QDMpy/odmr/processors.py` (add `FoldingProcessor`)
- `src/QDMpy/odmr/folding.py` (new: `SpectralFolder` class)
- `src/QDMpy/settings.py` (add `FoldingSettings`)
- `src/QDMpy/result.py` (expose `d_zfs_map`, `fold_residual` from metadata)
- `src/QDMpy/constants.py` (add `D_ZFS_TEMP_COEFFICIENT = -74e-6` GHz/K)
- `tests/test_folding.py` (new: comprehensive tests)
- `tests/odmr/test_processors.py` (test `FoldingProcessor` integration)

## Backwards Compatibility

Folding is **disabled by default** (`FoldingSettings.enabled = False`). The
existing pipeline is completely unchanged unless the user opts in. No existing
APIs are modified.

When folding is enabled, the `ODMRData` returned by the processor pipeline
has a different shape (no `freq_range` dimension). Downstream code that
explicitly indexes `freq_range` will need to handle this. The `FitManager`
already iterates over `freq_range` in a loop, so a single-range folded
spectrum works without modification.

## Verification

```bash
uv run pytest tests/test_folding.py -v
uv run pytest tests/odmr/test_processors.py -v -k "folding"
uv run ruff check src/QDMpy/odmr/folding.py
uv run mypy src/QDMpy/odmr/folding.py

# Integration test with synthetic data:
uv run pytest tests/test_folding.py -v -k "synthetic_symmetric"
# Verify fold residual is ~0 for symmetric synthetic spectra

# Verify D_ZFS recovery from synthetic data with known D shift:
uv run pytest tests/test_folding.py -v -k "d_zfs_recovery"
```

## Rejection Alternatives

**Alternative: Post-fit parameter averaging only.** This is what the current
code does (compute `(f+ - f-)` after independent fits). Rejected because it
does not improve fit quality, cannot extract D_ZFS per pixel, and provides
no model-free quality metric.

**Alternative: Joint fitting of both frequency ranges with shared D_ZFS.**
This would add D_ZFS as a shared parameter in a coupled fit. Rejected because
it requires major changes to the fitting infrastructure (pygpufit does not
natively support shared parameters across separate spectra), and the
model-free folding approach is simpler, faster, and more robust.

**Alternative: Full Bayesian inference with marginalization over D.**
Rejected as disproportionately complex for the benefit. The folding approach
achieves the key goals (D_ZFS mapping, SNR improvement) with much simpler
implementation.

## References

1. Lin, T.M. and Preston, R.S. (1974). "Comparison of Techniques for Folding
   and Unfolding Mossbauer Spectra for Data Analysis." Mossbauer Effect
   Methodology, Vol. 9, pp. 205-224. Plenum Press.

2. Saccone, F.D. (2024). "PyMossFit: A Google Colab Option for Mossbauer
   Spectra Fitting." Spectroscopy Journal, 3(4), 29.
   DFT-based folding implementation.

3. Spiering, H. et al. (2020). "Non-linearity correction of the velocity
   scale of a Mossbauer spectrum." NIMB, 480, 98.

4. Acosta, V.M. et al. (2010). "Temperature Dependence of the Nitrogen-Vacancy
   Magnetic Resonance in Diamond." Physical Review Letters, 104, 070801.
   Temperature coefficient of D_ZFS: dD/dT ~ -74 kHz/K.

5. Kehayias, P. et al. (2019). "Imaging crystal stress in diamond using
   ensemble nitrogen-vacancy centers." Physical Review B, 100, 174103.
   Strain effects on NV center resonance frequencies.

6. Levine, E.V. et al. (2019). "Principles and techniques of the quantum
   diamond microscope." Nanophotonics, 8(11), 1945-1973.
