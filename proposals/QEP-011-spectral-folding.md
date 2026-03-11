# QEP-011: Spectral Folding for Model-Free Parameter Extraction

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Priority** | P2 |
| **Complexity** | L |
| **Depends on** | QEP-005 |
| **Blocks** | None |
| **Author** | QDMpy Team |
| **Created** | 2026-02-15 |
| **Revised** | 2026-02-27 |

---

## Motivation

### The technique: spectral folding in Mössbauer spectroscopy

In Mössbauer spectroscopy, a triangular velocity drive produces two mirror-image
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

The two frequency ranges measured in a QDM experiment (low ~2.82–2.87 GHz and
high ~2.87–2.92 GHz) capture `f-` and `f+` respectively. These are **mirror
images about D**, exactly analogous to the two halves of a Mössbauer spectrum
about the velocity zero.

**Critical clarification (from prototype, 2026-02-27):** the correct fold axis
is `D` — the zero-field splitting, which lies at the boundary between the two
frequency ranges. Folding must combine the *low and high frequency ranges* about
`D`. Folding within a single frequency range around that range's own dip center
does NOT work for per-pixel field mapping: it merely symmetrises the noise and
destroys all spatial field variation (see Empirical Findings below).

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

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Empirical Findings from Prototype (2026-02-27)

A research prototype was implemented in `research/spectral_folding.py` and
`research/fold_fit.py` and validated against `tests/data/MIL2_FOV1` (4×4
binned, 300×480 pixels).

### Finding 1 — Mean-spectrum fold gives sub-grid D_ZFS precision

Folding the spatially-averaged mean spectrum per (polarity, freq_range):

```
pol=neg  frange=low:   fold=2.844448 GHz   argmin=2.844555 GHz   Δ=−0.11 MHz
pol=neg  frange=high:  fold=2.895665 GHz   argmin=2.895632 GHz   Δ=+0.03 MHz
```

The fold centre is within 0.03–0.12 MHz of argmin, with sub-grid-point
resolution. Note that what was found here is the centre of the dip within
each range, not D itself. D = (f_high + f_low) / 2 ≈ 2.870 GHz — confirming
the data are correctly centred.

### Finding 2 — Fold of mean spectrum gives a good global B111 estimate

```
mean B111_ind (fold):   −913.96 µT
mean B111_ind (argmin): −911.31 µT
per-pixel argmin median: −911.31 µT
```

The ~2.65 µT discrepancy between fold and argmin comes from the fold finding
the symmetry centre of the dip envelope (more robust against asymmetry) while
argmin finds the grid minimum. Both are within 0.3% of each other.

### Finding 3 — Within-range symmetrisation destroys per-pixel field variation

When pixels were symmetrised within each frequency range around the global
(mean-spectrum) fold centre and then GPU-fitted:

```
                  normal fit    fold-sym fit
  remanent std:   4.03 µT       0.08 µT
  induced  std:   8.89 µT       0.11 µT
  RMS diff:       —             4.04 µT (remanent), 8.86 µT (induced)
```

The fold-symmetrised fit produced a nearly constant B111 map (std ≈ 0.1 µT)
because symmetrising each pixel around the GLOBAL fold centre forces all
spectra to look the same — spatial field variation is entirely washed out.

**This is the key negative result**: per-pixel symmetrisation around a global
centre is not the right approach.

### Finding 4 — The correct approach is cross-range folding about D

The fold must be applied **across** the two frequency ranges about D_ZFS, not
within each range. Specifically, for the folded spectrum:

```
S_folded(delta_f) = S_low(D - delta_f) + S_high(D + delta_f)
```

where `delta_f = gamma * B * cos(theta)`. This:
- Preserves per-pixel field variation (each pixel's `delta_f` remains intact)
- Achieves sqrt(2) SNR improvement (two measurements per Zeeman shift)
- Makes D_ZFS the fold parameter (per-pixel temperature/strain map)

This is exactly what the original QEP-011 design specifies. The prototype
empirically confirmed the failure mode of the alternative (within-range fold),
providing experimental justification for the cross-range design.

---

## Specification

### 1. Core folding algorithm

The `SpectralFolder` class operates on raw ODMR data (before fitting) to
combine the two frequency ranges.

```python
class SpectralFolder:
    """Fold ODMR spectra by exploiting f+/f- mirror symmetry about D_ZFS.

    Analogous to Mössbauer spectral folding, where two halves of the
    absorption spectrum are combined about the velocity zero-point.

    The fold axis is D_ZFS (zero-field splitting), which lies at the
    boundary between the low and high frequency ranges. After folding:
    - A single spectrum is produced per (polarity, pixel)
    - The frequency axis is delta_f = gamma * B (in GHz or µT)
    - D_ZFS per pixel is the fold parameter (temperature/strain map)
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
requires interpolation. Use vectorized fractional-index interpolation (same
approach as `research/fold_fit.py:_symmetrize_band`) to map `S_high` onto
the reflected grid efficiently for all pixels at once.

For a 2000×2000 image with 201 candidate D values and 50 freq points:
~40 GFLOP — feasible in < 10 s with numpy vectorisation over the spatial axes.

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
- **Double the effective measurement time** at each frequency offset (sqrt(2) SNR)
- **Common-mode rejection** of baseline variations that are symmetric about D
- A natural frequency axis in **field units** (delta_f in GHz, or µT after
  dividing by GAMMA_NV)

Note on linewidth/contrast: the folded dip shape is the sum of the two
individual dip shapes (low and high range). If microwave power or contrast
differs between the two ranges, the folded shape is an average. This is
acceptable for field estimation; linewidth/contrast should be interpreted
as range-averaged values.

#### 2.4 Symmetry decomposition

The fold naturally decomposes the spectrum into symmetric and antisymmetric
components:

```
S_symmetric(delta_f)     = [S_low(D - delta_f) + S_high(D + delta_f)] / 2
S_antisymmetric(delta_f) = [S_low(D - delta_f) - S_high(D + delta_f)] / 2
```

- **S_symmetric**: Contains the magnetic field information (resonance dips).
  Used for fitting.
- **S_antisymmetric**: Contains residual asymmetry from transverse strain,
  off-axis fields, or systematic errors. Should be ~zero for ideal data.
  Its magnitude is the fold residual quality metric.

### 3. Polarity folding (second-level fold)

The two polarities provide a second folding axis. The polarity decomposition:

```
S_remanent(f) = [S_pol0(f) + S_pol1(f)] / 2   # field-independent signal
S_induced(f)  = [S_pol0(f) - S_pol1(f)] / 2   # field-dependent signal
```

Fitting `S_remanent` and `S_induced` separately can improve sensitivity to
weak induced fields masked by a strong remanent signal. This is a Phase 2
feature.

### 4. Integration with existing pipeline

The folder uses a **two-scale architecture**: D_ZFS is estimated at coarse
spatial resolution (high SNR) then interpolated to full resolution, before
the per-pixel fold is applied.

```
processed_data
      │
      ├─ spatially binned (coarse) ──► fold-point search ──► coarse D_ZFS map
      │   (d_zfs_bin_factor² pixels                                   │
      │    averaged per super-pixel)                      bicubic interpolation
      │                                                               │
      └─ full resolution ──────────────► per-pixel fold ◄── D_ZFS(y,x)
                                               │
                                          FoldedODMR ──► FitManager ──► QDMResult
                                          + d_zfs_map (GHz)
                                          + fold_residual ([0,1])
                                          + folded_spectrum
                                          + antisymmetric_spectrum
```

#### 4.1 New module: `src/qdmpy_core/odmr/folding.py`

Contains `SpectralFolder` and helper functions.

#### 4.2 New processor: `FoldingProcessor`

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
        - frequency coordinate relative to D_ZFS (delta_f in GHz)
        - metadata containing d_zfs_map and fold_residual
        """
```

#### 4.3 Folded-spectrum fitting

After folding, the spectrum has a single frequency range centred on D_ZFS.
The existing models work unchanged — the center parameter now represents
`gamma * B * cos(theta)` directly rather than an absolute GHz frequency.

Constraint update: center shifts from `(2.0, 3.1) GHz` to `(0.0, 0.15) GHz`
(typical magnetic field range in QDM experiments).

#### 4.4 Settings extension

```python
class FoldingSettings(BaseSettings):
    enabled: bool = False
    d_zfs_initial: float = 2.870        # GHz
    search_range: float = 0.01          # GHz (+/- around d_zfs_initial)
    search_steps: int = 201
    use_dft_refinement: bool = True
    fold_polarities: bool = False       # Phase 2: also fold polarity dimension
    min_fold_quality: float = 0.8       # Minimum quality to accept fold
    d_zfs_bin_factor: int = 16          # extra spatial binning for D_ZFS estimation
    d_zfs_interpolation: str = 'bicubic'  # method for upsampling D_ZFS map
```

The `d_zfs_bin_factor` controls the two-scale resolution trade-off: larger
values give better SNR (and thus more precise D_ZFS) at coarser spatial
resolution. Since temperature and strain vary on millimetre scales and the
pixel spacing is typically 4 µm, a factor of 16 (64 µm super-pixels) is
physically appropriate for most QDM experiments.

### 5. New outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `d_zfs_map` | `(n_pol, y, x)` | Per-pixel zero-field splitting (GHz). Estimated at coarse resolution (`d_zfs_bin_factor²` pixels averaged per super-pixel) then bicubic-interpolated to full resolution. Converts to temperature via dD/dT = −74 kHz/K. |
| `fold_residual` | `(n_pol, y, x)` | Normalised residual in [0,1]. Model-free data quality metric. |
| `folded_spectrum` | `(n_pol, y, x, freq_idx)` | SNR-improved spectrum for fitting. |
| `antisymmetric_spectrum` | `(n_pol, y, x, freq_idx)` | Residual asymmetry. Diagnostic for strain, off-axis fields. |

The `d_zfs_map` is a genuinely new output not available from the current
pipeline. Spatial variations in D_ZFS are scientifically valuable:
- Temperature mapping (dD/dT ~ −74 kHz/K near room temperature)
- Strain mapping
- Diamond quality assessment

---

## Impact Assessment

### Performance

- **Coarse fold-point search**: runs on a (ny/16 × nx/16) array — for
  2000×2000 this is 125×120 = 15,000 super-pixels. Negligible cost (< 1 s).
- **Full-resolution fold**: ~10 s for 2000×2000 (numpy-vectorised). One-time
  cost, amortised over subsequent fitting.
- **Spectrum folding**: < 1 s. Simple array operations.
- **Fitting improvement**: Fitting a single folded range vs two separate ranges
  halves the number of GPU fit calls. Net speedup depends on whether the
  fold-point search is offset by the fitting speedup.
- **Memory**: Folded spectrum is half the size (one freq_range vs two).
  D_ZFS map and residual add ~32 MB for 2000×2000.

### Signal-to-noise

- **Direct improvement**: sqrt(2) ~ 41% SNR improvement in the folded spectrum,
  translating to better-constrained fit parameters.
- **Indirect improvement**: Model-free D_ZFS determination removes one degree
  of freedom from the fit, improving convergence and reducing parameter
  correlations.
- **Weak-field sensitivity**: For small B-fields the two resonances barely
  separate; folding concentrates the signal.
- **Two-scale D_ZFS precision**: per-pixel fold-point search gives ~150 K
  temperature precision (limited by single-pixel SNR). The two-scale approach
  bins by `d_zfs_bin_factor = 16`, gaining sqrt(256) = 16× SNR and improving
  precision to ~9 K — physically useful for detecting laser heating or strain
  gradients. Temperature and strain vary on millimetre scales so the coarse
  D_ZFS map captures all meaningful variation.

### Scientific capability

- **D_ZFS mapping**: Entirely new capability. Enables temperature and strain
  imaging from the same dataset. Also corrects a systematic error in the
  B111 pipeline: assuming D = 2.870 GHz everywhere makes any real D variation
  (from temperature or strain) appear as a spurious B111 offset of
  ΔB = ΔD / GAMMA_NV ≈ 13 µT/MHz. The interpolated D_ZFS map removes this
  confound without additional measurements.
- **Data quality assessment**: The fold residual provides a per-pixel quality
  metric before any fitting, enabling early rejection of bad regions.
- **Strain detection**: Strong antisymmetric components indicate transverse
  strain breaking f+/f- symmetry.

### Risks

- **Asymmetric frequency ranges**: If the two ranges have very different widths
  or are not centred on D_ZFS, folding requires extrapolation or produces a
  reduced overlap window. Handle gracefully by computing the frequency overlap
  and restricting the folded range accordingly.
- **Different contrasts per range**: The two dips may have different contrasts
  due to microwave power variation across frequency. The folded dip shape is
  their average — acceptable for field estimation.
- **Non-Lorentzian lineshapes**: Inhomogeneous broadening or multiple NV
  orientations may produce asymmetric dips; the fold residual serves as a
  diagnostic.
- **Complexity**: Adds a new processing step with tunable parameters. Keep
  disabled by default (`FoldingSettings.enabled = False`).

---

## Implementation Plan

### Phase 1 — Core fold algorithm (no fitting integration yet)

1. Create `src/qdmpy_core/odmr/folding.py`:
   - `SpectralFolder` class with `find_fold_point()`, `fold()`, `fold_residual()`
   - Two-scale D_ZFS estimation: `_estimate_d_zfs_coarse()` + bicubic interpolation
   - Vectorized fold-point search over all pixels simultaneously (numpy)
   - Linear interpolation for reflecting `S_high` onto the `S_low` grid
2. Add `D_ZFS_TEMP_COEFFICIENT = -74e-6` to `constants.py`
3. Tests in `tests/odmr/test_folding.py`:
   - Synthetic symmetric spectrum → fold residual ≈ 0
   - Synthetic D shift → `find_fold_point()` recovers known D
   - **Two-scale interpolation**: inject a known smooth D_ZFS field (e.g. Gaussian
     hotspot), run coarse estimation + bicubic interpolation, verify recovered
     map matches injected field to within grid precision
   - Real data smoke test: `d_zfs_map` values in plausible range [2.86, 2.88] GHz
4. Expose `d_zfs_map` and `fold_residual` as outputs (no fitting changes yet)

### Phase 2 — Pipeline integration

5. Add `FoldingProcessor` to `src/qdmpy_core/odmr/processors.py`
6. Add `FoldingSettings` to `src/qdmpy_core/settings.py`
7. Update `FitManager` to accept single-range folded data with relative
   frequency axis (the fold removes the `freq_range` dimension)
8. Update constraint defaults for folded fitting
9. Expose `folded_spectrum` and `antisymmetric_spectrum` from `QDMResult`

### Phase 3 — Polarity fold (optional)

10. Add `fold_polarities` option to decompose into remanent/induced spectra
    before fitting

---

## Files Affected

- `src/qdmpy_core/odmr/folding.py` (new)
- `src/qdmpy_core/odmr/processors.py` (add `FoldingProcessor`)
- `src/qdmpy_core/settings.py` (add `FoldingSettings`)
- `src/qdmpy_core/result.py` (expose `d_zfs_map`, `fold_residual` from metadata)
- `src/qdmpy_core/constants.py` (add `D_ZFS_TEMP_COEFFICIENT`)
- `tests/odmr/test_folding.py` (new)
- `tests/odmr/test_processors.py` (add `FoldingProcessor` integration tests)

---

## Backwards Compatibility

Folding is **disabled by default** (`FoldingSettings.enabled = False`). The
existing pipeline is completely unchanged unless the user opts in. No existing
APIs are modified in Phase 1.

In Phase 2, when folding is enabled, the `ODMRData` returned by the processor
pipeline has a different shape (no `freq_range` dimension). The `FitManager`
already iterates over `freq_range` in a loop, so single-range folded data
works without structural changes to the fitting infrastructure.

---

## Verification

```bash
uv run pytest tests/odmr/test_folding.py -v
uv run pytest tests/odmr/test_processors.py -v -k "folding"
uv run ruff check src/qdmpy_core/odmr/folding.py
uv run ty check src/qdmpy_core/odmr/folding.py

# Verify fold residual ≈ 0 for synthetic symmetric spectra
uv run pytest tests/odmr/test_folding.py -v -k "symmetric"
# Verify D_ZFS recovery from synthetic data with known D shift
uv run pytest tests/odmr/test_folding.py -v -k "d_zfs_recovery"
```

---

## Alternatives Rejected

**Fold within each frequency range separately.** Prototype (2026-02-27) showed
that symmetrising each pixel around the global within-range fold centre
destroys all spatial field variation — the resulting B111 map has std ≈ 0.1 µT
instead of the expected 4–9 µT. The cause: symmetrisation around a fixed
global centre forces all pixels to look the same. Only cross-range folding
about D preserves per-pixel field information.

**Post-fit parameter averaging only.** Current approach. Rejected because it
does not improve fit quality, cannot extract D_ZFS per pixel, and provides no
model-free quality metric.

**Joint fitting of both ranges with shared D_ZFS.** Would add D_ZFS as a shared
parameter in a coupled fit. Rejected because pygpufit does not natively support
shared parameters across separate spectra, and the model-free folding approach
is simpler and faster.

**Full Bayesian inference with marginalization over D.** Disproportionately
complex. Rejected.

---

## References

1. Lin, T.M. and Preston, R.S. (1974). "Comparison of Techniques for Folding
   and Unfolding Mossbauer Spectra for Data Analysis." Mössbauer Effect
   Methodology, Vol. 9, pp. 205–224. Plenum Press.

2. Saccone, F.D. (2024). "PyMossFit: A Google Colab Option for Mossbauer
   Spectra Fitting." Spectroscopy Journal, 3(4), 29. DFT-based folding.

3. Spiering, H. et al. (2020). "Non-linearity correction of the velocity
   scale of a Mossbauer spectrum." NIMB, 480, 98.

4. Acosta, V.M. et al. (2010). "Temperature Dependence of the Nitrogen-Vacancy
   Magnetic Resonance in Diamond." Physical Review Letters, 104, 070801.
   Temperature coefficient of D_ZFS: dD/dT ~ −74 kHz/K.

5. Kehayias, P. et al. (2019). "Imaging crystal stress in diamond using
   ensemble nitrogen-vacancy centers." Physical Review B, 100, 174103.

6. Levine, E.V. et al. (2019). "Principles and techniques of the quantum
   diamond microscope." Nanophotonics, 8(11), 1945–1973.

---

## Post-Implementation Findings (2026-03-11)

### SNR improvement is conditional on signal strength

Empirical benchmarking on 6 test fixtures (3 x ESR15N, 3 x ESR14N) revealed
that the sqrt(2) noise reduction in the folded spectrum does not always
translate to improved B111 accuracy. The D_ZFS estimation step introduces
per-pixel errors that propagate through the folded fit:

| Signal regime | D_ZFS error impact | Folded vs normal B111 |
|---|---|---|
| Strong (B111 std >> 2-3 uT) | Negligible | Comparable accuracy, 2x faster |
| Weak (B111 std < 2 uT) | Dominant | Normal fit is more accurate |

**Key numbers (centroid D_ZFS estimator, best tested method):**

- D_ZFS estimation error: ~0.05-0.16 MHz RMSE (varies by method and linewidth)
- ESR15N (narrow linewidth ~0.6 MHz): D_ZFS error is ~10-25% of linewidth
- ESR14N (wider linewidth ~1.2 MHz): D_ZFS error is ~5-10% of linewidth
- For weak-signal FOV18x: folded B111 RMSE = 0.6 uT vs normal fit (corr 0.92)
- For strong-signal FOV1: folded B111 RMSE = 0.36-0.84 uT vs normal (corr 0.99)

### D_ZFS estimation methods compared

Several D_ZFS estimation methods were prototyped and benchmarked:

| Method | D_ZFS RMSE | Speed (128x128) | Notes |
|---|---|---|---|
| Brute-force (bin_factor=8) | ~1 MHz | 200 ms | Current production; coarse spatial resolution |
| FFT cross-correlation + refine | ~0.05 MHz | 3500 ms | Best accuracy, per-pixel loop is slow |
| FFT zero-padded (8x oversample) | ~0.13 MHz | 480 ms | Good accuracy, fully vectorised |
| Absorption centroid | ~0.06-0.16 MHz | 35 ms | Best speed-accuracy trade-off |
| Phase correlation | ~0.12-0.40 MHz | 550 ms | No advantage over standard xcorr |

The absorption centroid method (weighted mean frequency of the dip) emerged
as the best overall option: trivially vectorised, O(N), and competitive or
superior accuracy. Scripts: `scripts/prototype_fft_dzfs.py`,
`scripts/prototype_fft_v2.py`, `scripts/compare_folded_fits.py`.

### Recommendation

Documentation and docstrings have been updated to reflect the conditional
nature of the SNR benefit. The unqualified "sqrt(2) SNR improvement" claim
has been replaced with guidance on when folding helps (strong signals, speed,
D_ZFS mapping) vs when normal fitting is preferred (weak signals, maximum
B111 accuracy).
