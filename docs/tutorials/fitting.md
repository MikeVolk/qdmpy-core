# Fitting Quality & Optimization

**Audience:** Lila &nbsp;|&nbsp; **Time:** ~15 min &nbsp;|&nbsp; **Prerequisites:** [02 · Exploring Data](02-exploration.ipynb)

---

## What you'll learn

- Which ESR model to choose and how auto-detection works
- How to set parameter constraints and why they help
- How to interpret chi2 and fit_states to assess quality
- When to use GPU vs CPU fitting

---

## Setup

```python
import qdmpy
import numpy as np

result = qdmpy.make_synthetic_qdm_result(shape=(32, 32))
fit = result.fit_result
```

---

## Model selection

qdmpy includes three ESR models for the three common NV configurations:

| Model | Dips | Use when |
|-------|------|----------|
| `ESR14N` | 3 (hyperfine triplet) | Standard ^14^N diamond — most common |
| `ESR15N` | 2 (hyperfine doublet) | Isotopically enriched ^15^N diamond |
| `ESRSINGLE` | 1 | Low-field or heavily broadened spectra |

### Auto-detection

Pass `model='auto'` (the default) to let qdmpy inspect the spectrum and pick
the best model:

```python
result = qdmpy.load('/data/FOV18x').fit_odmr()   # model='auto' by default
print(result.model_name)   # e.g. 'ESR14N'
```

### Explicit selection

Override auto-detection when you know your sample:

```python
result = qdmpy.load('/data/FOV18x', model='ESR15N').fit_odmr()
```

---

## Constraints

Constraints prevent the optimizer from exploring physically unreasonable
regions of parameter space. They often eliminate spurious convergence and
reduce fit time.

### Constraint types

| Type | Effect |
|------|--------|
| `FREE` | No bounds (default) |
| `LOWER` | Parameter >= vmin |
| `UPPER` | Parameter <= vmax |
| `LOWER_UPPER` | vmin <= parameter <= vmax |

### Setting constraints

Build a `FitManager` directly:

```python
from qdmpy import FitManager

fm = FitManager(model_name='ESR14N')

# Restrict center frequency near ZFS (absolute GHz override)
fm.set_constraints('center', vmin=2.85, vmax=2.89, constraint_type='LOWER_UPPER')

# Prevent unrealistically narrow linewidths (GHz)
fm.set_constraints('width', vmin=0.001, constraint_type='LOWER')

# Cap contrast to avoid over-fitting noise
fm.set_constraints('contrast', vmax=0.3, constraint_type='UPPER')

# Remove all constraints
fm.set_free_constraints()
```

!!! note "Constraint units"
    The optimizer always runs in **absolute GHz** internally, but user-facing
    defaults come from `settings.model.constraints.constraint_units = 'mt'`.
    In `mt` mode, `center_max_mt` / `width_max_mt` are converted to absolute-GHz
    bounds around `D_ZFS`. If `center_min_mt > 0`, qdmpy enforces a true
    per-branch center window (e.g., `2-7 mT`):
    low branch uses `D_ZFS-delta_max .. D_ZFS-delta_min`, high branch uses
    `D_ZFS+delta_min .. D_ZFS+delta_max`.
    Use `constraint_units='absolute_ghz'` if you prefer entering
    `center_min/max` and `width_min/max` directly in GHz.

---

## Fit quality metrics

### chi2 (reduced chi-squared)

Available as a per-pixel map:

```python
chi2 = fit.chi2   # shape (n_pol, n_frange, H, W)
print(f"Mean chi2: {chi2.mean():.3f}")
```

Interpretation:

| chi2 range | Meaning |
|---|---|
| ~1 | Excellent — model matches data within noise |
| 2–5 | Acceptable for most applications |
| > 10 | Poor — check noise, model choice, or constraints |
| < 0.5 | Suspiciously low — noise may be overestimated |

### fit_states

Encodes optimizer convergence status per pixel:

```python
states = fit.fit_states   # shape (n_pol, n_frange, H, W), dtype int32
converged_fraction = (states == 0).mean()
print(f"Convergence rate: {converged_fraction:.1%}")
```

| State | Meaning |
|-------|---------|
| 0 | Converged (target) |
| 1 | Generic failure / not converged |
| 2 | Max iterations reached |
| 3 | Singular Hessian / numerical issue |
| 4 | Negative curvature / gpufit internal |

Aim for >95 % of pixels in state 0. Low convergence usually indicates:

- Model mismatch (try `model='auto'` or change explicitly)
- Constraints too tight (widen bounds or use `FREE`)
- Noisy data (increase `bin_factor`)

---

## GPU vs CPU

qdmpy automatically uses GPU fitting when `pyGpufit` is installed and CUDA is
available. No code change is needed — the same `FitManager` API is used for
both backends.

```python
print(qdmpy.is_pygpufit_available())   # True = GPU will be used
```

**When GPU matters:** For scans larger than ~200 × 200 pixels the GPU backend
is 10-100x faster. For small synthetic datasets or quick tests, CPU is fine.

**Fallback:** If `pyGpufit` is not installed or no GPU is found, qdmpy falls
back to SciPy least-squares automatically and logs a warning.

---

## Key takeaways

- `model='auto'` works for most samples; override explicitly for 15N or SINGLE
- Constraints on `center`, `width`, and `contrast` are the most impactful
- Target chi2 ~ 1 and >95 % convergence in state 0
- GPU fitting kicks in automatically — no code change required

---

## What's next

- **Lila** — [04 · Spectral Folding](04-spectral-folding.ipynb) to further
  improve SNR and extract D_ZFS maps
- **Professor** — [03 · Extending](03-extending.ipynb) to register custom ESR
  models in the `ModelRegistry`
