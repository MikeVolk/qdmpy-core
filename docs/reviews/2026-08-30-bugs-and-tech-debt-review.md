# Bugs & Technical Debt Review -- qdmpy-core

| Field       | Value                                                             |
|-------------|-------------------------------------------------------------------|
| Date        | 2026-08-30                                                        |
| Branch      | `develop` @ `c68210e`                                             |
| Scope       | `src/qdmpy/` (~13.2k LOC), focus on bugs and technical debt        |
| Suite state | `uv run pytest` passes (exit 0)                                    |
| Lint state  | `ruff check src/ tests/` clean; `ty check src/qdmpy` -- 3 stale `# ty: ignore` suppressions, no real diagnostics |

This follows [`2026-08-30-correctness-and-solid-review.md`](2026-08-30-correctness-and-solid-review.md),
whose findings were largely fixed in `31b6c26` (11 correctness bugs) and
`c4ec276` (5 SOLID fixes). Those fixes were spot-checked and hold. This pass
targets areas both prior reviews covered lightly -- `settings.py`,
`io/qdm.py`, `field_processing.py`, `measurement.py`, `guess.py` -- plus
regressions introduced by the fixes themselves.

Every numeric claim below was verified by execution, not asserted from
reading. Two findings carried over from 2026-08-22 are still open and are
re-stated with fresh evidence.

---

## Part 1 -- Bugs

### 1. HIGH (carried over, F1, open since 2026-08-22) -- `OutlierProcessor()`'s default still destroys the data

`odmr/processors.py:277` -- `z_score_threshold: float = Field(default=0.003, gt=0)`.
A z-score threshold of 0.003 flags every point more than 0.003 standard
deviations from its pixel mean, i.e. essentially all of them. Verified on
synthetic data:

```
default z_score_threshold: 0.003
fraction NaN after default OutlierProcessor: 0.9990625
```

What makes this worse than at the last review: **the published tutorial tells
users to make exactly this call.** `docs/tutorials/processors.md:38,53` both
say `OutlierProcessor(threshold=3.0)`. There is no `threshold` field, and
because the model does not forbid extra kwargs (finding 2), the call
constructs silently with the lethal 0.003 default:

```
>>> OutlierProcessor(threshold=3.0)
type='OutlierProcessor' z_score_threshold=0.003
```

A user following the tutorial NaNs out their entire dataset and gets no
error, no warning, and a plausible-looking constructor repr.

Fix: pick a real default (`3.0` is the conventional sigma-clip value for
this statistic), and correct the three tutorial rows. `docs/migration.md`
also recommends `z_score_threshold=0.003` in three places (:155, :234, :495).

### 2. HIGH -- no Pydantic model in the package forbids unknown fields

Root cause of the tutorial failure above, and a systemic input-validation
gap. `odmr/processors.py:87` is `ConfigDict(frozen=True)` -- extras default
to `"ignore"`. The same holds for `field_processing.py`, `refit.py`,
`folding.py`, `freq_cutoff.py`, `backends.py`. `settings.py` goes further and
sets `extra="ignore"` **explicitly in nine places** (`:97,111,130,144,157,175,196,214,238`).

Consequences:

- Every processor/settings constructor silently swallows typos. The tutorial
  also has `FluorescenceCorrectionProcessor(factor=0.2)` (:37,52) -- the field
  is `correction_factor`; `factor` is dropped.
- A typo in `~/.config/QDMpy/settings.toml` (`center_min_ml`, `width_max_mT`,
  `estimater`) silently reverts that setting to its default. The user gets a
  fit run under constraints they did not ask for and never finds out.

For a scientific package this is the highest-leverage single change in this
review: `extra="forbid"` on `BaseProcessor`, `BaseFieldProcessor`, and the
settings tree. The project's own rules mandate "validate all user input",
"fail fast with clear error messages", "never trust external data (... file
content)".

### 3. HIGH -- `.qdm` round-trip silently loses every fit state

`io/qdm.py:115-130` writes `parameters["states"]` to the dataset
`fit/fit_states`. `io/qdm.py:244` (`_read_fit_parameters`) then explicitly
**skips** that key and never restores it under its original name. Verified
round-trip:

```
fit/ datasets on disk : ['center', 'chi2', 'fit_states']
parameters after load : ['center', 'chi2']
'states' survived     : False
n non-converged before: 45
n non-converged after : 0
quality metrics after : ['mean_chi2', 'median_chi2', 'n_pixels', 'std_chi2']
```

Downstream effects, all silent:

- `FitResult.fit_states` falls back to `np.zeros_like(centers)` (`result.py:177`)
  -- every pixel reports as converged.
- `get_fit_quality_metrics()` drops `convergence_rate` and `n_converged`.
- `refit_outliers(include_non_converged=True)` becomes a no-op on any loaded
  result, because `states` is all zeros.

`.qdm` is the package's primary persistence format and the one `QDMResult.save()`
routes to. Fix: `parameters["states"] = np.array(fit_grp["fit_states"])` in
`_read_fit_parameters`.

Related, same file: the module docstring's layout diagram (`io/qdm.py:25`)
advertises `fit/frequencies (n_frange, n_freq) float64`, and `_write_fit:109`
has a branch for it -- but `FitResult.parameters` never contains a
`frequencies` key (see `manager.py:471-475`), so the branch is dead and the
dataset is never written. A `.qdm` file therefore cannot be refit or replotted
against its own frequency axis.

### 4. HIGH -- `get_settings()` destroys the host application's logging

`settings.py:294` -- `_configure_logging()` calls `logger.remove()`, which
removes **all** loguru handlers, including ones the embedding application
installed, then adds qdmpy's own. It runs as a side effect of
`get_settings()` (`:333`), which `FitManager.__init__` calls
unconditionally (`manager.py:157`). Verified:

```
HOST | Registered model: ESRSINGLE            <- host format still active
HOST | Registered processor type: ...
2026-08-30 15:15:07.962 | INFO | __main__:<module>:10 - after get_settings   <- host format gone
```

For `qdmpy-gui` and `qdmpy-server`, whose logging is configured at startup,
the first fit silently re-routes all logging to qdmpy's defaults.

Two further surprises in the same function: `enable_structured_logging`
defaults to `True`, so importing and using the library creates `~/logs/` and
starts writing rotating JSON logs to the user's home directory without being
asked (`:300-312`); and `get_settings()` also does `mkdir` via
`make_configfile()` (`:331`). A settings accessor should not have filesystem
or global-state side effects at all. Fix: split `configure_logging()` out as
an explicit opt-in call (the CLI can call it), default structured logging to
off, and never call `logger.remove()` from library code.

### 5. MEDIUM -- the chi2-guarded refit never converges early, and over-reports what it did

`refit.py:320` records `"n_refitted": n_refit` -- pixels **attempted**, not
accepted. `_refit_pass` returns a new `FitResult` whenever `n_total_refitted > 0`,
so `refit_outliers`'s convergence check (`refit.py:444`, `if nxt is current`)
never fires. And because `identify_outlier_pixels` uses a bare percentile
(`refit.py:129-130`, `chi2 > percentile(chi2, 90)`), roughly 10% of pixels are
always flagged, no matter how good the fit is -- there is no absolute floor.

Verified on a clean synthetic 8x8 fit with `max_iterations=4`:

```
pass 1: returned-same=False  metadata n_refitted=28  chi2 values actually changed=0
pass 2: returned-same=False  metadata n_refitted=28  chi2 values actually changed=0
pass 3: returned-same=False  metadata n_refitted=28  chi2 values actually changed=0
pass 4: returned-same=False  metadata n_refitted=28  chi2 values actually changed=0
```

Four full refit passes, 112 pixel-fits, **zero** parameters changed, and
`metadata["refit_info"]["n_refitted"]` claims 28 each time. The
`max_iterations` docstring's promise ("The loop stops early when no further
pixels can be refitted") no longer holds after the chi2 guard landed in
`31b6c26`.

The same log line is also internally inconsistent (`refit.py:369-376`):
"refitted 14 of 14 outlier pixels (0 accepted, 28 rejected as worse)" -- the
first two counts are per-pixel, the last two per-(polarity, pixel).

Fix: count accepted, not attempted, in `per_frange_info`; use that for the
convergence check; reconcile the log units. Separately, consider an absolute
chi2 floor alongside the percentile so a good fit isn't refit at all.

### 6. MEDIUM (carried over) -- the numba guessers are NaN-unsafe, and NaN is now more reachable

`guess.py` splits cleanly in two. The contrast estimators handle NaN
deliberately -- `cumsum_contrast:203` uses `nanmax`/`nanmin`, `top3_contrast:237`
skips NaN explicitly. The centre and width estimators do not:

- `normalize_pixel:61` -- `np.mean` over a slice containing NaN gives NaN;
  the whole normalised curve is NaN, and `argmin(abs(norm - 0.5))` returns 0,
  so `cumsum_center` and `cumsum_width` silently report `freq[r, 0]`.
- `argmin_center:315` -- `np.argmin` on a NaN-containing spectrum returns the
  NaN index, placing the centre guess at an arbitrary dead frequency.
- `halfpower_width:443` -- NaN comparisons are always False, so the minimum
  stays pinned at `spectrum[0]`.
- `guess_n_peaks:139` uses `np.median` (not `nanmedian`) across pixels, so
  one NaN pixel poisons the model auto-detection spectrum for that
  (pol, frange).

NaN pixels are more reachable now than at the last review, not less:
`NormalizationProcessor` was just changed to *deliberately* emit NaN for
zero-factor pixels (`processors.py:225`), `OutlierProcessor` emits them by
design, and `HotPixelFilter(replacement='nan')` exists. `ScipyBackend` was
hardened for NaN in `31b6c26` (`backends.py:273-282`) -- but the guesser
stage upstream of it still produces silent garbage initial parameters for the
same pixels.

### 7. MEDIUM -- `Measurement.__init__` allocates 460 MB of dead memory

`measurement.py:138`:

```python
self._outliers: NDArray | None = np.ones(self.odmr.raw_data.shape, dtype=bool)
```

`raw_data.shape` is the full 5D `(n_pol, n_frange, y, x, freq_idx)`. At the
target resolution documented in `CLAUDE.md` (2 x 2 x 1200 x 1920 x 50) that is
**460.8 MB** of booleans, allocated on every `Measurement` construction.

`_outliers` is never read anywhere in `src/` -- only `tests/test_measurement.py`
asserts it exists. Either delete it, or (if the outlier mask is coming back)
make it a lazily-built `(y, x)` mask.

### 8. MEDIUM (carried over, F10) -- `QuadraticBackgroundSubtractor` is still NaN-unsafe

`field_processing.py:249` -- `np.linalg.lstsq(features[active], data.ravel()[active])`.
A single NaN in the field map (which `HotPixelFilter(replacement='nan')`,
the fit path, and `OutlierProcessor` all produce) makes every coefficient
NaN, and the subtracted surface NaNs the entire output map. Fix: fold
`np.isfinite(data.ravel())` into the `active` mask.

Same class, `:243` -- `mask_rows, mask_cols = self.mask` raises a bare unpack
error for any mask tuple that isn't exactly length 2; the field is typed
`tuple[tuple[int, ...], ...]`, which permits any length.

### 9. MEDIUM -- chi2 is not comparable across fit backends, and the refit guard now depends on it

`GpufitBackend` passes `estimator_id=ESTIMATOR_ID[options.estimator]`
(`backends.py:176`) with the package default `estimator="MLE"`
(`settings.py:117`), so gpufit returns a Poisson-likelihood deviance.
`ScipyBackend:216-220` and `TorchBackend` ignore the estimator (warning only)
and always return the plain sum of squared residuals (`backends.py:298`,
`torch_backend.py:419`).

So `chi2` is a different quantity on a different scale depending on backend.
That was tolerable when chi2 was only used for percentile ranking, but
`31b6c26` added an absolute comparison: `_accept_improved_refits` compares
`new_chi2_arr < old_chi2` (`refit.py:229`). `Measurement.refit_outliers(backend=...)`
lets you refit with a different backend than you fit with, and in that case
the guard compares an MLE deviance against an SSR and makes systematically
wrong accept/reject decisions.

Fix: record the estimator and backend in `FitResult.metadata`, and either
refuse a cross-backend refit or renormalise. Minimum viable: document that
`chi2` is backend-relative.

### 10. LOW -- a loguru call uses printf placeholders, losing all its data

`manager.py:765` -- `logger.debug("Setting constraints for %s: vmin=%s, vmax=%s, type=%s", param, vmin, vmax, constraint_type)`.
loguru formats with `str.format`, so the `%s` are never substituted and the
arguments are discarded. Verified output:

```
Setting constraints for %s: vmin=%s, vmax=%s, type=%s
```

The sibling branch eight lines above (`:755`) uses `{}` correctly. Single
occurrence in the package; the project's logging rule names this exact
anti-pattern.

### 11. LOW -- constructing a `FitResult` freezes the caller's arrays in place

`result.py:93-95` sets `param_array.flags.writeable = False` on the arrays it
was handed, not on copies. Verified:

```python
p = {'center': ..., 'chi2': ...}
FitResult(parameters=p, ...)
p['center'].flags.writeable   # -> False
```

The immutability guarantee is purchased by mutating data the caller still
owns. Internal call sites happen to pass freshly-built arrays, so nothing
breaks today, but any user who builds a `FitResult` from arrays they intend to
keep using gets a surprising read-only failure later, far from the cause.
Fix: copy on ingest, or document the transfer-of-ownership contract loudly.

### 12. LOW -- `FitResult.linewidths` docstring claims Hz

`result.py:114` -- "Get ODMR linewidths in Hz." Every frequency in the
package is GHz (`CLAUDE.md`, `manager.py:3`, `result.py:3`). Physics-facing
docstring drift of the kind flagged as F13 on 2026-08-22.

### 13. LOW -- folding uses the low branch's frequency step for the high branch

`folding.py:520` computes `step = median(diff(f_low))` and then uses it for
both branches' index arithmetic -- `idx_h = (d + delta_f - f_high[0]) / step`
(`:567`) and `delta_f = np.arange(df_inner, df_outer, step)` (`:563`). Correct
only while both ranges are swept with identical spacing. `_interp_batch` does
the right thing (it derives the step per array, `:274`), so the two code paths
disagree. Latent, but it is exactly the kind of assumption that breaks
silently on a new acquisition script.

Same function duplicates `_overlap_range`'s body inline at `:556-557` rather
than calling it.

### 14. LOW -- `Measurement.refit_outliers` docstring promises constraint inheritance it doesn't do

`measurement.py:366` -- "constraints: ... Defaults to the same constraints
used in the original fit." It does not: `None` flows through
`refit_measurement_result` to `build_fit_manager(constraints=None)`
(`measurement_workflows.py:297-304`), which builds a `FitManager` on library
defaults. Only the `fit_odmr(..., refit_outliers=True)` path forwards the
original constraints, because it still has them in scope. Same for
`freq_cutoff`. A standalone `m.refit_outliers(result)` after a constrained fit
refits under different constraints than the fit it is correcting.

---

## Part 2 -- Technical debt

### Dead configuration users can set and the library ignores

Four settings sections are user-settable via `settings.toml` or `QDMPY_*`
env vars, documented with descriptions, and read by nothing in `src/`:

| Setting | Lines | Consumers in `src/` |
|---|---|---|
| `default_paths.data_path` | `settings.py:29-32` | none |
| `odmr.norm_method` | `settings.py:35-40` | none |
| `model.find_peaks.prominence` | `settings.py:43-46` | none -- `guess.py:34` hardcodes `_RELATIVE_PROMINENCE = 0.03` |
| `outlier_detection.*` (3 classes, ~45 lines) | `settings.py:133-175` | none -- `docs/migration.md:494` confirms sklearn detection was never ported |

`prominence` is the worst of these: it advertises a tunable peak-detection
threshold that silently does nothing, on the code path that picks the ESR
model. Delete the dead sections or wire them up; leaving them is worse than
either.

### Error-handling inconsistency (A6, still open)

`exceptions.py` defines the package's exception hierarchy, and most modules
use it -- but ~20 sites across seven modules still raise bare `ValueError`:
`field_processing.py:57,236`, `field_source.py:69,76,83,116,123,177`,
`magnetic_map.py:175,186,217`, `refit.py:74,83,92,101`,
`odmr/processors.py:200`, `plotting/_common.py:120`, `plotting/fields.py:43`,
`plotting/fit.py:257`. Callers cannot distinguish a qdmpy validation failure
from any other `ValueError` in the stack.

### Two processor frameworks with divergent capability

`odmr/processors.py` now has `Processor` (protocol), `BaseProcessor`,
`ProcessorRegistry`, and `to_config()`/`from_config()` round-tripping --
good, that was the `c4ec276` fix. `field_processing.py` has
`BaseFieldProcessor` and `FieldProcessingPipeline` with none of it: no
protocol, no registry, no serialisation, no `type` tag. A field-processing
pipeline cannot be saved or restored, and a custom field processor has no
documented contract. The two should converge on one pattern.

### Immutability is enforced three incompatible ways

- `FitResult` -- Pydantic model, arrays frozen via `flags.writeable = False`
  (mutating the caller's arrays, finding 11).
- `ODMRData` -- `ConfigDict(frozen=True)`, but the wrapped `xr.DataArray`'s
  buffer stays writeable, so the guarantee is shallow.
- `BaseFieldProcessor` / `FoldingSettings` / `RefitSettings` -- frozen
  Pydantic config objects, which is the only case where frozen means what it
  says.

`FoldedODMR` (`folding.py:174`) is `frozen=True` while holding mutable
`NDArray` and `xr.DataArray` fields -- same shallow guarantee as `ODMRData`.
Pick one story and state it in `architecture.md`.

### Fourier processing has no windowing

- `magnetic_map.py:_reconstruct_bxyz` (`:100-125`) FFTs the raw B111 map with
  no padding and no taper. F8 from 2026-08-22, still open. Edge
  discontinuities ring across the whole reconstructed Bz/Bx/By.
- `field_processing.py:UpwardContinuation` (`:299-305`) does pad, but with a
  hard zero step: the map's mean level is not removed first, so the pad
  boundary is a discontinuity of size `mean(data)` and produces Gibbs ringing
  that the continuation filter then smears inward. Removing the mean before
  padding (and adding it back) is a two-line fix.
- `UpwardContinuation.padding_factor` (`:271`) has no lower bound; any value
  below 1.0 makes `offset_h` negative and the assignment at `:305` raises a
  broadcast error.

### Smaller items

- `HotPixelFilter` (`field_processing.py:138-142`) thresholds on the global
  `nanstd`, which is inflated by the very outliers being detected -- so the
  threshold loosens exactly when there are more hot pixels. MAD is the robust
  alternative. (Runtime is fine: ~0.2 s at 1200x1920, measured.)
- `_resolve_spatial_dims` (`result.py:234-271`) still guesses an image shape
  from factor pairs when the pixel count doesn't match `scan_dimensions` (F6,
  open). It should raise.
- `get_model_by_peaks` (`guess.py:173-180`) returns the first registry entry
  matching a peak count; with the Open/Closed registry the package now
  advertises, two models with the same `n_peaks` resolve by dict insertion
  order.
- `ODMRProcessorManager.add_processor` (`processors.py:396`) is typed
  `BaseProcessor`, contradicting the `Processor` protocol's documented
  duck-typing contract eleven lines up. Should be `Processor`.
- `ODMRProcessorManager.process:406-409` duplicates the `pipeline_config`
  property's comprehension verbatim (`:429-432`).
- `_compute_fold_residual` (`folding.py:729`) clips the quality metric to
  `[0, 1]`, so every badly-asymmetric pixel saturates at exactly 1.0 and the
  map loses all discrimination in the range that matters.
- `ScipyBackend` reports `result.nfev` as `iterations` (`backends.py:299`) and
  maps `max_number_iterations` onto `max_nfev` (`:292`) -- function
  evaluations, not iterations. Cross-backend `iterations` values are not
  comparable.
- Three stale `# ty: ignore` suppressions remain (`torch_backend.py:230` and
  two others); `ty check` reports them as removable.
- `_UNSET` sentinel (`measurement.py:43,160`) is typed
  `float | None = _UNSET  # type: ignore[assignment]`.

---

## Part 3 -- Recommended order of work

| # | Finding | Change | Size |
|---|---------|--------|------|
| 1 | 1 | `OutlierProcessor` default -> 3.0; fix `docs/tutorials/processors.md` (3 rows) and `docs/migration.md` (3 rows) | small -- open since 2026-08-22, and the tutorial actively points users at it |
| 2 | 2 | `extra="forbid"` on `BaseProcessor`, `BaseFieldProcessor`, settings tree | small -- highest leverage in this review |
| 3 | 3 | Restore `states` in `_read_fit_parameters`; write or drop `fit/frequencies` | ~3 lines + a round-trip test |
| 4 | 4 | Split `configure_logging()` out of `get_settings()`; default structured logging off | small, but touches gui/server startup |
| 5 | 5 | Count accepted (not attempted) refits; use it for convergence; fix log units | small |
| 6 | 6, 8 | NaN guards in the centre/width guessers and `QuadraticBackgroundSubtractor` | small-medium |
| 7 | 7 | Delete `Measurement._outliers` (460 MB, unread) | 1 line + test update |
| 8 | 9 | Record backend/estimator in `FitResult.metadata`; guard cross-backend refit | small |
| 9 | 10-14 | loguru `%s`, `FitResult` array ownership, Hz docstring, folding step, refit constraint docstring | small each |
| 10 | Part 2 | Delete dead settings; unify the two processor frameworks; add mean-removal/taper to the Fourier paths | medium |
