# Notebook Audit Report

**Date:** 2026-03-01
**Summary:** 5 notebooks reviewed; 4 current, 1 significantly outdated

## Status by Notebook

### ✅ 01-quickstart.ipynb — CURRENT

**Target audience:** "I want to fit and be done" (minimal API)

**APIs:** `qdmpy.make_synthetic_qdm_result()`, `result.b111_remanent/induced`, `result.b111` (xarray), `result.magnetic_map`, save/load

**Status:** ✅ All current, in sync

---

### ⚠️ 02-exploration.ipynb — MIXED (IMPORT ISSUES)

**Target audience:** "I want to play around with the data"

**Issues:**
1. Uses `from qdmpy import ODMR, BinningProcessor, ...` — imports may be wrong (should be `qdmpy_core`?)
2. `from qdmpy.testing import make_synthetic_*` — verify this location
3. Core concepts sound but **imports prevent execution**

**Status:** ⚠️ Fix imports before running

---

### ✅ 03-extending.ipynb — CURRENT

**Target audience:** "I want to develop my own algorithms"

**APIs:** `@ModelRegistry.register`, `Processor` protocol, `FieldReconstructor` protocol, standalone `FitManager`

**Status:** ✅ All extension patterns current and well-documented

---

### ✅ 04-spectral-folding.ipynb — CURRENT (SPECIALIZED)

**Target audience:** "I want to fold spectra for SNR improvement"

**APIs (Quick):** `m.fold_odmr()`, `folded.plot()`, `m.fit_folded_odmr()`

**APIs (Detailed):** `FoldingSettings`, `SpectralFolder`, diagnostic plots, D_ZFS maps, temperature mapping

**Status:** ✅ All current, tested on real data (MIL2_FOV1)

---

### ❌ load_mil2_fov1.ipynb — OUTDATED (MAJOR ISSUES)

**Target audience:** "I want to load and explore ODMR data from scratch"

**Critical Issues:**

| Issue | Severity |
|-------|----------|
| Uses `from qdmpy.fitting import FitManager` | CRITICAL |
| Uses `from qdmpy import ODMR, ...` | CRITICAL |
| Calls `b111_from_dip_positions(data)` — may not be public API | HIGH |
| References undefined `res.contrasts` attribute | HIGH |
| Import path for `make_synthetic_*` unclear | MEDIUM |

**Status:** ❌ **REQUIRES MAJOR FIXES**

---

## Quick Fixes

**02-exploration.ipynb:**
```python
# Change FROM:
from qdmpy import ODMR, BinningProcessor, NormalizationProcessor
from qdmpy.testing import make_synthetic_fit_result, make_synthetic_odmr_data

# Change TO:
from qdmpy_core.odmr.data import ODMR, ...  # confirm path
from qdmpy_core.testing import make_synthetic_*  # confirm path
```

**load_mil2_fov1.ipynb:**
1. Update all `qdmpy` → `qdmpy_core` imports
2. Check if `b111_from_dip_positions()` exists in public API
   - If yes: add proper import
   - If no: reimplement 3-line function locally or link to 03-exploration version
3. Remove or fix `res.contrasts` reference at end
4. Test execution end-to-end

---

## Consolidation Recommendation

After fixing imports:

**Option A (Minimal):** Keep all 5 as-is
- Pros: Users can pick their learning path (quick vs detailed)
- Cons: Redundancy (02 and load_mil2_fov1 both do exploration)

**Option B (Recommended):** Merge into 4
- Combine 02-exploration + load_mil2_fov1 → one "Data Loading & Exploration" guide
- Keep 01-quickstart, 03-extending, 04-spectral-folding separate
- Reduces cognitive load, clearer purpose per notebook

**Option C (Curated):** Publish only in /docs
- Move notebooks into `/docs/tutorials/` (versioned, under version control)
- Keep `/notebooks/` for experimental/scratch work only
- Update mkdocs.yml nav to list all published notebooks

Pick your direction and I can implement the fixes + restructuring.
