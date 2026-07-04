# qdmpy-core Architecture Review — Findings

- **Date:** 2026-07-03
- **Branch:** `develop` @ `6a5a6c6`
- **Focus:** maintainability, usability, extensibility
- **Companion:** `architecture-review-2026-07-03.html` (same content with before/after diagrams)

Vocabulary: a *module* is an interface plus an implementation; it is *shallow* when the
interface is nearly as complex as the implementation; a *seam* is a place where behaviour
can be swapped without editing code in place; *locality* means change/bugs/knowledge
concentrated in one place. All QEP statuses were audited; implemented QEPs are treated as
settled decisions.

---

## Candidates (ranked)

| # | Candidate | Strength | Existing QEP |
|---|-----------|----------|--------------|
| 1 | FitBackend seam (GPU / CPU / fake) | **Strong** | none — new (→ QEP-068) |
| 2 | One fit pipeline inside FitManager | **Strong** | FIT-003, 060 (drafts) |
| 3 | Collapse wrapper chain; FitOptions value object | **Strong** | 057 (draft, partial) |
| 4 | Models that fully describe themselves | **Strong** | FIT-005 (draft, partial) |
| 5 | One owner for fit-result serialization | Worth exploring | 056-IO (draft) |
| 6 | Pluggable loaders at the front door | Worth exploring | ODMR-003 (draft, partial) |
| 7 | One processor-pipeline framework | Worth exploring | ODMR-002 (draft, partial) |
| 8 | Downward-only layering & delegation hygiene | Speculative | FIT-004, CORE-003 (drafts) |

**Top recommendation: #1.** Only Strong candidate with no QEP coverage; keystone for
#2–#4 (fake backend makes the pipeline testable, kwarg collapse deletes `gpu_available`
plumbing, CPU backend makes pure-Python custom models fittable); converts ~85 monkeypatch
sites in fit-adjacent test suites into ordinary dependency injection.

Suggested sequencing: #1 → #2 (activate QEP-FIT-003 + QEP-060) → #3 → #4; #5–#8
independent.

---

## Detailed friction findings

### 1. Entry-path multiplicity through pass-through wrappers
`qdmpy.load()` (`__init__.py:18`) is a pure forwarder to `Measurement.from_folder()`.
A fit call descends four layers: `Measurement.fit_odmr` → `fit_measurement_odmr` →
`build_fit_manager` → `FitManager.fit`. `build_fit_manager`
(`measurement_workflows.py:180`) and `build_qdm_result` (`:200`) are 3-line thunks that
fail the deletion test. The kwarg tuple `(constraints, freq_cutoff, settings,
gpu_available)` is copy-pasted across seven signatures.
**Covered by:** QEP-057 (draft) partially; wrapper thinness not itemized anywhere.

### 2. FitManager.fit() is a god method; fit_folded() re-implements the pipeline
`fit()` (`fitting/manager.py:269-382`) does nine jobs. `fit_folded()` (`:680-794`)
hand-builds the folded inputs and **constructs a second FitManager inside itself**
(`:776`). `_apply_mt_center_window_for_range()` (`:384`) mutates the shared
ConstraintManager mid-fit, contradicting the documented "stateless between calls" claim.
~110 lines of `freq_cutoff` dict parsing (`:140-254`) belong in a frozen value object.
**Covered by:** QEP-FIT-003 + QEP-060 (drafts); mutation bug and self-construction not
itemized.

### 3. Folded fit-input conversion duplicated
`FoldedODMR.to_fit_inputs()` (`odmr/folding.py:185`) exists and is used by the refit path
(`measurement_workflows.py:267`), but `FitManager.fit_folded` re-derives the same thing by
hand (`manager.py:713-732`). One consumer uses the seam, the other bypasses it.

### 4. Model registry is not truly open/closed
Adding a model touches ≈3 files: `guesser.py:136-147` branches on `n_peaks == 3 / == 2`
with hardcoded AHYP width-correction factors; `folding.py:69-73` keys `_CENTROID_POWER`
on model-name strings (unknown models silently defaulted); `guess.py:173-180` matches
models by peak count only; `result.py:121-159` probes parameter-name strings;
`manager.py:586-587` carries undocumented aliases (`resonance`→`center`,
`mean_contrast`→`contrast`). `ModelRegistry._registry` is class-level mutable global
state. **Covered by:** QEP-005 (implemented, partial), QEP-FIT-005 (draft, registry only).

### 5. No pygpufit/CPU seam — documented custom-model contract is dead
`is_pygpufit_available()` checked in six places; `gpu_available: bool` threaded through
every fit-adjacent signature; `pygpufit.gpufit.fit_constrained` called inline
(`manager.py:640`). `Model` docs (`models.py:206`) promise `model_id = -1  # CPU-only`,
but a registered pure-Python model crashes in gpufit — `Model.func()` is never used for
fitting, only residual logging and plotting. **Covered by:** no QEP. → **QEP-068**.

### 6. Three overlapping fit-serialization codecs; "pure container" claims are fictional
`FitResult.save_results/load_results` (+legacy pickle migration,
`fitting/result.py:547-804`); `io/npz.py` reaching into FitResult **privates**
(`_build_save_dict` at `npz.py:38`, `_from_npz` at `npz.py:73`); `io/qdm.py` (HDF5)
re-implementing cache injection. Both containers claim "all I/O is in qdmpy.io".
**Covered by:** QEP-051 (implemented, incomplete split), QEP-056-IO + QEP-057 (drafts).

### 7. Two parallel processor frameworks; ODMR side has three abstractions
ODMR: `Processor` Protocol ("no base class needed"), `BaseProcessor` ABC, hand-maintained
`ProcessorSpec` union (`odmr/processors.py:24, 77, 218-225`) — Protocol-only processors
fail `from_config` round-tripping. Field side: fourth framework with different API
(`.add()` vs `.add_processor()`, no serialization). Five duplicated inside-function
`ODMRData` imports with no actual cycle. **Covered by:** QEP-ODMR-002 (draft, ODMR side
only).

### 8. ODMR manager is mutable with a redundant flag
`ODMR.is_processed` (`odmr/manager.py:37`) is derivable from `_processed_data`; every
mutator must sync both; tests already bypass the invariant. `ODMRData` itself is properly
frozen. **Covered by:** QEP-ODMR-001 (draft).

### 9. Config arrives by two mechanisms; defaults duplicated across ≥3 layers
Global singleton `get_settings()` (`settings.py:315`) consumed by FitManager default,
plus explicitly threaded `FoldingSettings`/`RefitSettings`/`QDMpySettings`.
`fluorescence_correction=0.2` declared in 3 places; `pixel_spacing=4e-6` in 3;
`model` default disagrees (`"auto"` at entry, `"ESR14N"` in FitManager).

### 10. QDMResult is a shallow wrapper
~18 hand-written one-line forwards to FitResult (`result.py:87-171`); genuine added value
is ~4 members (lazy magnetic_map, images, save/load dispatch). Every new FitResult
property must be mirrored or is silently missing.

### 11. Layering inversion via inside-function imports
~30 inside-function `from qdmpy...` imports, mostly lower layers calling *up* into
plotting (`fitting/guess.py:168`, `fitting/result.py:617-633`, `odmr/folding.py:210`,
`odmr/manager.py:160`). Plotting also reaches into fitting privates
(`plotting/odmr.py:426` imports `_relative_prominence`).

### 12. B111 physics duplicated
`fitting/result.py:330-419` and `odmr/analysis.py:35-83` implement the same equations and
have already diverged once (QEP-025 relabelling). **Covered by:** QEP-FIT-004 (draft).

### 13. Loaders can't reach the front door
`BaseLoader` is a clean seam, but `load()`/`from_folder()` hardcode `MatlabLoader`
(`measurement_workflows.py:115`). A new format means hand-assembling
`ODMRData → ODMR → Measurement`.

### 14. MatlabLoader.load() is a 105-line method with lint suppressions
`# noqa: C901, PLR0912, PLR0915` (`odmr/io.py:65`); load-bearing y-axis flip buried
mid-method. **Covered by:** QEP-ODMR-003 (draft).

### 15. Catch-alls
`utils.py` mixes string formatting, coordinate transforms, math. Two magnetic-map modules
(`magnetic_map.py` vs 20-line `io/magnetic_map.py`). Stale comments.

### 16. Field-domain module naming gives little navigation signal
`field_processing.py` / `field_source.py` / `source_fitting.py` / `magnetic_map.py` are
individually clean but hard to tell apart; `source_fitting.fit_sources` silently skips
non-`MagneticSource` field sources (`source_fitting.py:199`) — open/closed gap in the
discriminated union.

### 17. Mock density tracks weak seams exactly
46 mock/patch sites in `tests/test_measurement.py`, 28 in `test_refit.py`, 18 in
`test_load.py`, 11 in `test_folded_fit.py` — all fit-adjacent, all forced by findings
#1/#5. Leaf physics modules (models, guess, constraints, analysis, field_source) have
clean focused unit tests.

---

## Extensibility scorecard

| Extension | Today | Target |
|---|---|---|
| New ESR model | ≈3 files (`models.py`, `guesser.py`, `folding.py`) + un-fittable without a CUDA kernel | 1 file, fittable via CPU backend |
| New file-format loader | 1 file, but front door (`load()`) unreachable | 1 file + `load(path, loader=...)` |
| New ODMR processor | 2 edit points (class + hand-maintained union) | 1 file (registry-derived union) |
| New field-map processor | different framework, no serialization | same pipeline as ODMR |
