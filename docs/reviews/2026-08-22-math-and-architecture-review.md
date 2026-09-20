# Math and Architecture Review -- qdmpy-core

| Field       | Value                                              |
|-------------|----------------------------------------------------|
| Date        | 2026-08-22                                         |
| Branch      | `develop` @ `6a5a6c6`                              |
| Scope       | `src/qdmpy/` (~11k LOC), full read                 |
| Suite state | 918 passed, 87.20% coverage, `ty` clean            |
| Lint state  | `ruff check src/ tests/` clean (168 hits are notebooks only) |

Numeric claims below were verified by execution, not by reading. Reproduction
snippets are inline where relevant.

---

## Part 1 -- Math review

### 1.1 Verified correct

The sign and unit chain was traced end-to-end rather than taken from docstrings.

**B111 extraction** -- `fitting/result.py:330-401`.

```
d       = [-1, +1]                       # neg, pos
delta   = (f_high - f_low) / 2 / GAMMA_NV * 1e6 * d      [uT]
remanent = (delta_neg + delta_pos) / 2
induced  = (delta_neg - delta_pos) / 2
```

Matches `CLAUDE.md` and the QDMlab convention. GHz / (GHz/T) -> T -> uT is right.

**Fourier inversion** -- `magnetic_map.py:74-125`. Derived independently. Above
the sources with `phi ~ exp(-k z)`:

```
Bx~ = -i (kx/k) Bz~
Bu~ = Bz~ [ uz - i (ux kx + uy ky) / k ]
```

so `Bz~ = Bu~ * k / (uz*k - i*ux*kx - i*uy*ky)`, which is exactly
`denom = uz*k - uy*1j*ky - ux*1j*kx`. The transfer function magnitude is bounded
in `[1, 1/uz]`, so this inversion does **not** amplify high-k noise. Good.

**Upward continuation** -- `H = exp(-dz * k)` with k in rad/m. Correct sign for
`dz > 0` = upward = attenuation.

**Folded -> B111 consistency.** Folding produces dips at `df = gamma*B`;
`to_fit_inputs` shifts to `D_ZFS + df`; `_calc_delta_from_single_center`'s
`n_frange < 2` branch divides by gamma *without* the factor 2. Correct, because
`(f_high - f_low)/2 == df`.

Additionally: the fold is robust to D_ZFS estimation error to first order. If
`D_est = D + eps`, the two folded halves place dips at `B*gamma + eps` and
`B*gamma - eps`; the average is centred at `B*gamma` and merely broadened. The
`folding.py` module docstring is more pessimistic than the math warrants.

**Other checks that passed:**

- `NvSettings.axis = (0, 0.8164966, 0.5773503)` is a unit vector, `sqrt(2/3)`
  and `1/sqrt(3)`.
- Lorentzian HWHM parameterization is consistent between `models.py`
  (`width^2 / ((x-c)^2 + width^2)`) and `halfpower_width` (returns FWHM/2).
- **float32 at the gpufit boundary.** ULP at 2.87 GHz is ~0.34 kHz, giving a
  ~12 nT B111 floor. QEP-059 already measured delta-parameterization
  empirically: <0.35 uT difference, ~1% *worse* chi2. Confirmed non-issue --
  do not revisit.

### 1.2 Findings

Ordered by severity.

---

#### F1 -- CRITICAL -- `OutlierProcessor` default annihilates the data

`odmr/processors.py:182` -- `z_score_threshold: float = 0.003`.

Z-scores along `freq_idx` have RMS ~= 1 by construction, so a threshold of 0.003
masks essentially every point. Measured:

```
DEFAULT OutlierProcessor -> NaN fraction: 0.9994
  thr=0.003  NaN frac = 0.9994
  thr=1.0    NaN frac = 0.2350
  thr=2.0    NaN frac = 0.0013
  thr=3.0    NaN frac = 0.0000
```

`docs/tutorials/processor_tutorial.ipynb:358` recommends this exact value with
"Always -- remove sensor glitches", and `docs/migration.md:155` puts it in the
recommended pipeline. It is saved from being a live disaster only because
`Measurement.from_folder` does not wire it in.

Three separate defects:

1. The default is off by roughly three orders of magnitude. It reads like a
   leftover from an absolute-intensity threshold.
2. **The axis is wrong.** A z-score across the frequency sweep flags the
   resonance dip -- that is the signal. Even at `thr=2` it removes the deepest
   points of the deepest dips. Glitch rejection belongs in the spatial or
   repeat domain.
3. Downstream, `absorption_centroid` and `halfpower_width` in
   `fitting/guess.py` have **no NaN handling**, so masked points propagate NaN
   into gpufit initial guesses. (`top3_contrast` does skip NaN.)

**Action:** fix the default to ~4-5, add NaN guards to the numba guessers, and
either re-scope the processor to spatial outliers or delete it and correct the
docs.

---

#### F2 -- HIGH -- refit accepts strictly worse fits, and always fires

`fitting/refit.py:312-315` writes refit results back unconditionally. A refit
that lands in a worse local minimum silently degrades the map.

**Action:** guard the write-back on `new_chi2 < old_chi2`. Three lines, and the
single highest-value change in the file.

Compounding it: `identify_outlier_pixels` uses a *relative* percentile
(`chi2_percentile = 90`), so on a perfect fit it still declares the top 10% of
pixels outliers and refits them. With `max_iterations > 1` each pass re-flags a
fresh 10%. Because the initial guesses are neighbour medians, iterating biases
the parameter map toward spatial smoothness -- it will quietly erase real
small-scale structure while making the chi2 map look better.

**Action:** add an absolute floor (e.g. `chi2 > k * median(chi2)`) or make the
percentile path explicit opt-in.

---

#### F3 -- HIGH -- the refit path bypasses `fit()`'s preprocessing

`refit.py:304` calls `fit_manager.fit_frange(...)` directly, skipping both:

- `_apply_freq_cutoff_for_range` -- if the user passed `freq_cutoff`, the
  initial fit used a masked frequency axis and the refit uses the full one.
  `measurement_workflows.refit_measurement_result` explicitly threads
  `freq_cutoff` through to `build_fit_manager`, so the caller has every reason
  to believe it applied. It silently does not.
- `_apply_mt_center_window_for_range` -- mT-mode per-branch centre windows are
  absent on refit.

Additionally, refitting a **folded** result constructs a fresh `FitManager` from
settings, losing the contrast/offset overrides that `fit_folded`
(`manager.py:762-774`) applies.

Root cause is structural -- three fit entry points with three different
preprocessing sets. See A1 in Part 2.

---

#### F4 -- HIGH -- folding silently clamps out-of-range interpolation

`odmr/folding.py:253-284`. `_interp_batch` clips both the index and the
fraction, so a query outside the measured axis returns the edge value with no
NaN and no warning:

```
axis spans 2.86 - 2.88 GHz
queries: [2.855 2.86  2.87  2.88  2.89]
interp : [0.    0.   0.5   1.    1.   ]
```

The `df` axis is built from the **median** D_ZFS (`_fold_spectra:645`), but the
queries use the **per-pixel** D_ZFS. Pixels whose D deviates from the median
query outside the axis and get a flat clamped tail grafted onto the spectrum --
no error, just a distorted line shape feeding the fit. `_overlap_range` only
guards `d_ref`.

**Action:** emit NaN outside the range and track a validity mask, or shrink the
common `df` window to cover the whole D_ZFS distribution rather than the median.

Related, smaller: `_find_d_zfs_coarse:564-567` computes `step` from `f_low` and
applies it to `f_high` index arithmetic. Fine while both branches share
spacing, wrong the moment they do not.

---

#### F5 -- MEDIUM -- centre constraint may be tighter than the bias field, with no diagnostic

`center_max_mt = 1.1` gives `+-30.8 MHz` about D_ZFS. QEP-059's own draft
proposed 6.0 mT; it was tightened to 1.1 later. With a 2 mT bias (common) the
true resonance sits ~56 MHz out and gpufit will pin the centre at the bound.

**Nothing anywhere checks for parameters pinned at their constraint bounds.** A
constrained fit returns the boundary value with a plausible-looking chi2, and
percentile-based outlier detection will not reliably catch a systematic clip.

**Action:** add an `n_at_bound` count per parameter to `quality_metrics`. Cheap,
and it would already have surfaced this class of problem. QEP-066 gestures at
this but is still Draft.

---

#### F6 -- MEDIUM -- `_resolve_spatial_dims` guesses an image shape

`fitting/result.py:234-271`. When the pixel count does not match
`scan_dimensions`, it factorizes `n_pixels`, picks the factor pair closest to
the original aspect ratio, logs at DEBUG, and carries on. A shape mismatch here
is an upstream bug; silently inventing a 2D layout turns it into a scrambled
B111 map.

**Action:** raise `DataShapeError`. Same treatment for
`_normalize_resonance_shape`'s hardcoded `n_pol = 2` in the 2D branch.

---

#### F7 -- MEDIUM -- `HotPixelFilter` will eat real magnetic features

`field_processing.py:138-142` uses global `nanmedian` / `nanstd` over the whole
map. Two problems:

- `std` is inflated by the very outliers being detected (non-robust estimator).
- A genuine strong dipole is exactly a cluster of `|B - median| > 5 sigma`
  pixels. Applied to a B111 map before source fitting, this removes the thing
  being measured.

**Action:** use MAD; consider a local (median-filter-residual) criterion rather
than a global one.

Secondary: the replacement loop is a Python double loop over every pixel
(~2.3M iterations for an unbinned FOV) -- use `np.argwhere` on the mask.
Replacements also feed into subsequent windows, so the result is order-dependent.

---

#### F8 -- MEDIUM -- Fourier reconstruction has no padding or taper

`_reconstruct_bxyz` FFTs the raw B111 map. Any DC offset or edge mismatch
creates a wrap-around discontinuity and Gibbs ringing across the whole
reconstructed Bz -- and Bz is what `fit_sources` fits dipoles to.

`UpwardContinuation` *does* pad (3x), so the pattern already exists in the
codebase; it is just missing here. Note that `UpwardContinuation` pads with
**zeros**, which for a non-zero-mean field creates the same step problem --
subtract the mean before padding, or taper.

---

#### F9 -- MEDIUM -- D_ZFS map is shrunk toward the nominal value

`_estimate_d_zfs_centroid:326` uses `spectrum.max()` as the absorption baseline.
The sibling `absorption_centroid` in `guess.py` correctly uses an edge-mean
baseline.

Using `max` adds a noise-dependent constant to every weight, dragging each
branch's centroid toward its own axis midpoint. B111 is largely protected
(errors in the two branches partly cancel), but `d_zfs_map` is advertised as a
temperature/strain map and will systematically under-report variation.

**Action:** use the edge baseline for consistency with `guess.py`.

---

#### F10 -- MEDIUM -- `QuadraticBackgroundSubtractor` is NaN-unsafe

`field_processing.py:249`. `np.linalg.lstsq` on data containing a single NaN
returns all-NaN coefficients, and the whole map becomes NaN with no error.
Reachable, because `HotPixelFilter(replacement='nan')` is a supported upstream
step in the same pipeline.

**Action:** mask non-finite values out of the fit.

---

#### F11 -- LOW -- `FitManager` mutates itself during `fit()`

`_apply_mt_center_window_for_range` calls
`self._constraint_manager.set_constraint(...)` inside the fit loop. The class
docstring says "keeping the instance stateless between calls". After a two-range
fit the manager is left holding the *high* branch window, so reusing the
instance -- which the docstring invites -- gives different results on the
second call.

---

#### F12 -- LOW -- fluorescence correction reduces contrast and does not renormalize

Measured with `correction_factor = 0.2`:

```
dip depth before = 0.15225   after = 0.12238   -> SHALLOWER
```

Subtracting `alpha * (mean_spectrum - baseline)` removes a scaled *dip*, whereas
the standard background model `S = (1-alpha) S_NV + alpha B` with flat `B` calls
for `S_NV = (S - alpha B) / (1 - alpha)` -- a *deepening* plus renormalization.

In fairness this is a faithful port of the legacy behaviour
(`old_reference/QDMpy_old/src/QDMpy/_core/odmr.py:636`), not a regression. But it
is applied by default (`fluorescence_correction = 0.2` in `qdmpy.load`) and
biases every contrast map by ~20%. Because it is spatially uniform it does not
move centres, so **B111 is unaffected** -- contrast maps are not.

**Action:** confirm the intended semantics before contrast is treated as
quantitative.

---

#### F13 -- LOW -- docstring/code drift in physics-facing places

- `FitResult.linewidths` (`result.py:114`): "Get ODMR linewidths in Hz." They
  are GHz.
- `source_fitting._build_p0:57`: "Converts the qdmpy declination convention
  (+Y = 0) to the pypole convention (-Y = 0)" -- the module header and the code
  both say no conversion happens. One of the two is wrong, on a coordinate
  convention that matters.
- `guesser.ParameterGuesser` ASCII diagram says `cumsum_contrast`; the code
  calls `top3_contrast`.
- `odmr/analysis._validate_b111_coords` raises `KeyError`, not
  `DataValidationError`, when the `polarity` coord is missing entirely.
- `ParameterGuesser.guess()` caches on first call and ignores its argument
  thereafter -- a pure-looking signature that is not. Currently safe (one
  guesser per range) but a live trap.

---

## Part 2 -- Architecture review

### 2.1 What is working

Not faint praise -- the bones are solid:

- Named-dimension `xarray` throughout eliminates the axis-transposition bug
  class endemic to QDM code.
- `Model` / `ModelRegistry` and the `Processor` / `FieldSource` extension points
  are honest Open/Closed. New models and processors genuinely require no edits
  to existing classes.
- `FitResult` freezes its parameter arrays (`flags.writeable = False`) to
  protect its own caches.
- Pickle-free NPZ with a JSON `__meta__` block is the right serialization call.
- 87% coverage on a scientific codebase is well above par.
- The QEP process means design intent is actually recoverable months later.

### 2.2 Issues

---

#### A1 -- One fit path, not three

Structural root of F3. `fit()`, `fit_folded()`, and refit's direct
`fit_frange()` each assemble their own preprocessing. `fit_folded` even builds a
*second* `FitManager` internally (`manager.py:776`) to obtain different
constraints -- a strong signal that constraints want to be a per-call argument
rather than construction state.

**Action:** extract a single `_fit_pixels(data, freq, guesses, constraints)`
that owns cutoff masking, the mT centre window, and the gpufit call. `fit`,
`fit_folded`, and `refit` become thin callers. Fixes F3 and F11 structurally
rather than by patching each site.

---

#### A2 -- Layering violation: domain objects import the presentation layer

`memory/architecture.md` states "Layers only call downward", and `QDMResult`'s
docstring states "Pure data container. All plotting is in `qdmpy.plotting`."
`QDMResult` honours that. Two layers below it, five modules do not:

```
fitting/result.py:633   -> from qdmpy.plotting import plot_fit_result_overview
odmr/folding.py:210     -> from qdmpy.plotting import plot_folding_overview
odmr/manager.py:160     -> from qdmpy.plotting import plot_odmr_spectra
fitting/guess.py:168    -> from qdmpy.plotting import plot_model_detection
odmr/processors.py:294  -> from qdmpy.plotting import plot_fluorescence_correction
```

The deferred imports dodge the cycle at runtime, but the dependency is real: the
fitting layer cannot ship without the plotting layer.

**Action:** either drop the rule (the convenience API is defensible) or move
`.plot()` / `.show()` up to `QDMResult` and use an accessor pattern below. The
current state has the documentation saying one thing and five modules doing
another, which is exactly the drift `.claude/rules/common/architecture.md`
exists to prevent.

---

#### A3 -- `import qdmpy` costs 1.8 seconds

```
1713853  qdmpy
 699598    qdmpy.fitting        (672892 -> guess -> scipy.signal 549231)
 454305    qdmpy.field_processing
 414287    qdmpy.source_fitting (412591 -> pypole.convert, numba)
 305473    xarray
```

`__init__.py` eagerly imports every subsystem: `scipy.signal` for a single
`find_peaks` call, `pypole` + numba only needed for dipole fitting, matplotlib
via `io/images.py`. Meanwhile `qdmpy.load` already defers its own import.

This will hurt `qdmpy-server` startup and CLI responsiveness.

**Action:** PEP 562 module `__getattr__` keeps the flat public surface with lazy
resolution. Move `find_peaks` inside `guess_n_peaks`.

---

#### A4 -- Bulk arrays as Pydantic tuple fields

```python
BlankSubtractor.blank: tuple[tuple[float, ...], ...]
QuadraticBackgroundSubtractor.mask: tuple[tuple[int, ...], ...]
```

For a 1200x1920 map that is a 2.3M-element nested tuple that Pydantic validates
element-by-element at construction. A serialization concern (frozen models must
be hashable) is being solved at the cost of the hot path.

**Action:** use `NDArray` with `arbitrary_types_allowed` and handle persistence
explicitly. `FieldSource.field_map` already takes exactly that approach in the
same codebase.

---

#### A5 -- Memory profile at the stated target resolution

At 1200 x 1920 x 2 x 2 x 50, `data.values` alone is 3.7 GB float64. Per range,
on top of that:

- `np.ascontiguousarray(data, dtype=np.float32)` -- ~0.9 GB copy.
- `ConstraintManager.to_array` does `np.tile(constraints_list, (n_pixel, 1))`
  with `n_pixel = n_pol * n_pixels` ~= 4.6M rows x 12 float64 = **442 MB to
  replicate twelve identical numbers**, then a further 221 MB for the float32
  copy.

The tile is pure waste.

**Action:** build it float32 directly, or check whether pygpufit accepts a
broadcast / single-row constraint spec. More broadly there is no chunking
anywhere; unbinned fitting at the documented target size needs ~8 GB peak.

---

#### A6 -- Error-handling consistency

A well-designed `QDMpyError` hierarchy exists, but `magnetic_map.py:175`,
`field_processing.py:57,236`, and every `field_source` validator raise bare
`ValueError`. Callers cannot catch on the domain root. (Pydantic validators must
raise `ValueError` -- that is the framework contract -- but the hand-written
checks need not.)

Relatedly, `QuadraticBackgroundSubtractor` validates `degree in {0,1,2}` inside
`process()` rather than as a field validator, so a misconfigured pipeline fails
halfway through a run instead of at construction.

---

#### A7 -- Proposal debt

75 QEPs, of which ~15 are `Draft` and QEP-001 through QEP-017 carry **no status
field at all** -- there is no way to distinguish implemented from abandoned
without reading each one.

Several open drafts (QEP-066 constraints, QEP-063 folded centre estimation,
QEP-065 initial guesses) target exactly the areas where live issues were found
above.

**Action:** finish the untracked `proposals/README.md` as a status table and
backfill statuses on 001-017. Per the Gitflow rule, that file needs a
`feature/*` branch and a PR rather than landing on `develop`.

---

## Part 3 -- Recommended order of work

| # | Finding | Change | Size |
|---|---------|--------|------|
| 1 | F2 | Chi2-guarded write-back in `refit._refit_pass` | 3 lines |
| 2 | F4 | NaN instead of edge-clamp in `_interp_batch` | small |
| 3 | F1 | `OutlierProcessor` default + NaN guards in numba guessers + tutorial fix | small |
| 4 | F5 | `n_at_bound` diagnostic in `quality_metrics` | small |
| 5 | F3 / A1 | Unify the fit path so `freq_cutoff` cannot be silently dropped | QEP |
| 6 | F6, F10, A6 | Fail loudly instead of guessing / NaN-poisoning | small |
| 7 | F7, F8, F9 | Field-processing and reconstruction robustness | medium |
| 8 | A3, A4, A5 | Import cost, bulk-data modelling, memory | medium |
| 9 | A2, A7 | Layering decision, proposal status hygiene | governance |

Items 5 and 7 warrant QEPs before implementation, per the project's proposal
policy.
