# Tutorial Overview

Choose the path that matches your goal.

---

## Fry — "I just want B111 maps"

**Total time: ~15 min**

| Step | Resource | Time |
|------|----------|------|
| 1 | [Installation](../installation.md) | 2 min |
| 2 | [Quickstart guide](../quickstart.md) | 5 min |
| 3 | [01 · Quickstart notebook](01-quickstart.ipynb) | 10 min |

After step 3 you can load any QDM dataset, fit ODMR spectra, and export B111
maps. You're done.

---

## Lila — "I want optimal results"

**Total time: ~1-2 h**

| Step | Resource | Time |
|------|----------|------|
| 1 | [Quickstart](../quickstart.md) | 5 min |
| 2 | [02 · Exploring Data](02-exploration.ipynb) | 30 min |
| 3 | [Fitting Quality guide](fitting.md) | 15 min |
| 4 | [04 · Spectral Folding](04-spectral-folding.ipynb) | 20 min |
| 5 | [Settings & Configuration](settings_configuration.md) | 15 min |

After step 5 you understand the processing pipeline, can diagnose fit quality,
apply spectral folding for SNR improvements, and tune all configuration knobs.

---

## Professor — "I want to build new tools"

**Total time: open-ended**

| Step | Resource | Time |
|------|----------|------|
| 1 | [Quickstart](../quickstart.md) | 5 min |
| 2 | [03 · Extending the Framework](03-extending.ipynb) | 45 min |
| 3 | [06 · Source Fitting](06-source-fitting.ipynb) | 30 min |
| 4 | [Extending guide](../extending.md) | 15 min |
| 5 | [API Reference](../api/index.md) | reference |

After step 4 you can register custom ESR models, implement `Processor` and
`FieldReconstructor` protocols, and run magnetic dipole source fitting.

---

## All tutorials at a glance

| # | Notebook | Audience | Description |
|---|----------|----------|-------------|
| 01 | [01-quickstart.ipynb](01-quickstart.ipynb) | Fry | Load → fit → B111 maps → save |
| 02 | [02-exploration.ipynb](02-exploration.ipynb) | Lila | Pipeline, spectrum inspection, iteration |
| 03 | [03-extending.ipynb](03-extending.ipynb) | Professor | Custom model / processor / reconstructor |
| 04 | [04-spectral-folding.ipynb](04-spectral-folding.ipynb) | Lila/Professor | SNR improvement via spectral folding |
| 05 | [05-plotting.ipynb](05-plotting.ipynb) | All | Full plotting API walkthrough |
| 06 | [06-source-fitting.ipynb](06-source-fitting.ipynb) | Professor/Lila | Magnetic dipole source fitting |

---

## Guides (reference)

| Guide | Audience | Description |
|-------|----------|-------------|
| [Processors](processors.md) | Lila | Pipeline order, available processors, diagnostics |
| [ESR Models](models.md) | Professor | 14N, 15N, SINGLE — model registry, custom models |
| [Fitting Quality](fitting.md) | Lila | chi2, fit_states, constraints, GPU vs CPU |
| [Spectral Folding](spectral-folding.md) | Lila/Professor | Quick path + FoldingSettings reference |
| [Settings](settings_configuration.md) | Lila | NvSettings, global defaults |
| [Extending](../extending.md) | Professor | Protocols for custom algorithms |
