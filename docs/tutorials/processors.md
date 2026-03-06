# ODMR Processors

**Audience:** Lila &nbsp;|&nbsp; **Time:** ~15 min &nbsp;|&nbsp; **Prerequisites:** [Quickstart](../quickstart.md)

---

## What you'll learn

- What each built-in processor does and when to use it
- How to build and execute a processing pipeline
- The critical ordering rules that affect correctness
- How to diagnose processor effects with plots

---

## Setup

```python
import qdmpy
from qdmpy import (
    BinningProcessor,
    NormalizationProcessor,
    FluorescenceCorrectionProcessor,
    OutlierProcessor,
    ODMR,
)
```

---

## Available processors

| Processor | Effect | When to use |
|-----------|--------|-------------|
| `BinningProcessor(bin_factor=N)` | 2×2 spatial averaging (reduces resolution, improves SNR) | Always for noisy data; try bin_factor=2 first |
| `NormalizationProcessor(method='max')` | Normalises each pixel spectrum to [0, 1] | Almost always — required before fluorescence correction |
| `FluorescenceCorrectionProcessor(factor=0.2)` | Removes global fluorescence variations using LED reference | When LED image is available; reduces baseline drift |
| `OutlierProcessor(threshold=3.0)` | Flags and removes anomalous pixels (sigma-clipping) | Optional; useful for samples with debris or damage |

---

## Building a pipeline

Access the processor manager through `ODMR.processor_manager`:

```python
odmr_data = qdmpy.make_synthetic_odmr_data(shape=(64, 64))
odmr = qdmpy.ODMR(odmr_data)

odmr.processor_manager.add_processor(BinningProcessor(bin_factor=2))
odmr.processor_manager.add_processor(NormalizationProcessor(method='max'))
odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor(factor=0.2))
odmr.processor_manager.add_processor(OutlierProcessor(threshold=3.0))

odmr.process_data()
```

Or use `qdmpy.load()` which applies a standard pipeline automatically:

```python
result = qdmpy.load(
    '/data/FOV18x',
    bin_factor=2,                  # BinningProcessor
    normalize=True,                # NormalizationProcessor
    fluorescence_correction=0.2,   # FluorescenceCorrectionProcessor
).fit_odmr()
```

---

## CRITICAL: Processor order matters

!!! danger "Ordering rule — read before building a custom pipeline"
    The processors are **not commutative**. Incorrect ordering produces
    silently wrong results.

    **Correct order:**

    1. `BinningProcessor` — do spatial averaging first so later processors
       see the final pixel grid
    2. `NormalizationProcessor` — must come before fluorescence correction;
       fluorescence correction assumes a normalised baseline
    3. `FluorescenceCorrectionProcessor` — requires a normalised spectrum to
       identify the baseline correctly
    4. `OutlierProcessor` — flag anomalous pixels last, after the data has
       been corrected and normalised

    **Wrong order (example):**
    ```python
    # BAD: fluorescence correction before normalisation
    odmr.processor_manager.add_processor(FluorescenceCorrectionProcessor(0.2))
    odmr.processor_manager.add_processor(NormalizationProcessor('max'))   # too late
    ```

---

## Custom processors

Implement the `Processor` protocol to add your own processing step:

```python
import numpy as np
from qdmpy import Processor

class SpectralSmoothingProcessor(Processor):
    """Apply Gaussian smoothing along the frequency axis."""

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma

    def process(self, data):   # data is ODMRData; return new ODMRData
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(data.data.values, sigma=self.sigma, axis=-1)
        new_da = data.data.copy(data=smoothed)
        return data.model_copy(update={'data': new_da})
```

Plug it into the pipeline like any built-in processor:

```python
odmr.processor_manager.add_processor(SpectralSmoothingProcessor(sigma=1.5))
```

---

## Diagnosing processor effects

After processing, compare mean spectra before and after each stage to verify
the pipeline is working as expected. Use the plotting functions from
`qdmpy.plotting`:

```python
from qdmpy import plotting

# Fluorescence correction diagnostic
fig = plotting.plot_fluorescence_correction(odmr)
fig.savefig('fluorescence_check.png')
```

The diagnostic shows the LED reference image and the correction factor map,
making it easy to identify regions where the correction is too aggressive
(factor too high) or has no effect (factor near zero).

---

## Key takeaways

- Recommended pipeline order: Binning → Normalisation → Fluorescence
  Correction → Outlier
- Skipping normalisation before fluorescence correction will silently
  produce wrong results
- `qdmpy.load()` applies a safe default pipeline; only build a custom
  pipeline when you need more control

---

## What's next

- [02 · Exploring Data](02-exploration.ipynb) — interactive notebook showing
  before/after comparisons for each processor
- [Fitting Quality](fitting.md) — once the data is processed, assess and
  improve fit quality
