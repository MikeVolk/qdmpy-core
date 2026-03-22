# QEP-049 — Pipeline Definition Schema

**Status:** Draft
**Created:** 2026-03-04

---

## Motivation

Running qdmpy today requires writing Python for every measurement: choose
processors, set fit parameters, decide output formats.  In practice most lab
sessions reuse the same chain.  There is nowhere to encode that chain once and
share it across scripts, the GUI, or an automated watcher.

A `pipeline.toml` file solves this: it describes the full processing + fitting +
output chain in a human-writable, version-controllable format.  Validated Pydantic
models in `qdmpy-core` give every consumer a single, typed representation of that
file.

The *execution* layer (folder watcher, daemon, status server, CLI) lives in a
separate `qdmpy-server` package that depends on this schema.  This QEP covers only
the schema.

---

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

## Design

### `pipeline.toml` format

```toml
# pipeline.toml — describes how to process a QDM measurement.
# All sections except [watcher] have sensible defaults.

[watcher]
# Consumed by qdmpy-server; present here so the full config lives in one file.
watch_dir      = "/data/qdm-measurements"
folder_pattern = "*"       # glob; use "FOV*" to restrict
debounce_s     = 10.0      # seconds after last file change before triggering
output_subdir  = "results" # relative to each measurement folder

[processors]
# Verbatim QEP-030 serialised processor list.
steps = [
    {type = "BinningProcessor",                bin_factor = 2},
    {type = "NormalizationProcessor",          method = "max"},
    {type = "FluorescenceCorrectionProcessor", correction_factor = 0.2},
]

[fit]
model         = "ESR14N"
pixel_spacing = 4.0e-6   # metres
use_gpu       = true

[fit.constraints]
center_low  = {lower = 2.82, upper = 2.87}
center_high = {lower = 2.87, upper = 2.93}

[output]
formats   = ["npz", "png"]
maps      = ["b111_remanent", "b111_induced", "chi2"]
overwrite = false

[server]
# Consumed by qdmpy-server only.
enabled = true
host    = "localhost"
port    = 8765
```

All keys are optional except `[watcher].watch_dir`.  Missing keys fall back to
code-level defaults on the `PipelineConfig` model.

### Priority (highest -> lowest)

When `qdmpy-server` runs a folder, parameters are resolved in this order:

1. Explicit keyword argument in `PipelineRunner.run()` (caller wins)
2. Per-folder `metadata.toml` value (QEP-047)
3. `pipeline.toml` value
4. Code default on the Pydantic model

---

### Pydantic models (`qdmpy/pipeline/config.py`)

```python
from __future__ import annotations
import tomllib
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ProcessorStep(BaseModel):
    """Single processor entry; type discriminator mirrors QEP-030 format."""
    model_config = ConfigDict(extra='allow')  # forwards unknown keys to processor ctor
    type: str


class ProcessorsConfig(BaseModel):
    steps: list[ProcessorStep] = Field(default_factory=list)


class ConstraintSpec(BaseModel):
    lower: float | None = None
    upper: float | None = None


class FitConfig(BaseModel):
    model: str = 'ESR14N'
    pixel_spacing: float = 4e-6
    use_gpu: bool = True
    constraints: dict[str, ConstraintSpec] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    formats: list[str] = Field(default=['npz'])
    maps: list[str] = Field(default=['b111_remanent', 'b111_induced'])
    overwrite: bool = False


class WatcherConfig(BaseModel):
    watch_dir: Path
    folder_pattern: str = '*'
    debounce_s: float = Field(default=10.0, gt=0)
    output_subdir: str = 'results'


class ServerConfig(BaseModel):
    enabled: bool = False
    host: str = 'localhost'
    port: int = Field(default=8765, ge=1024, le=65535)


class PipelineConfig(BaseModel):
    """Root config object.  Load from pipeline.toml via from_file()."""
    watcher: WatcherConfig
    processors: ProcessorsConfig = Field(default_factory=ProcessorsConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @classmethod
    def from_file(cls, path: Path) -> PipelineConfig:
        """Load and validate a pipeline.toml file."""
        with path.open('rb') as f:
            raw = tomllib.load(f)
        return cls.model_validate(raw)

    def to_file(self, path: Path) -> None:
        """Serialise back to TOML (round-trip)."""
        import tomli_w
        with path.open('wb') as f:
            tomli_w.dump(self.model_dump(mode='json'), f)
```

`tomllib` is stdlib >= 3.11.  `tomli_w` (write support) is a small optional
dependency; `to_file()` is a convenience method for tooling, not required for
normal load paths.

---

### Module layout

```
src/qdmpy/pipeline/
    __init__.py      # exports PipelineConfig and sub-models
    config.py        # all Pydantic models + from_file() / to_file()
```

No other modules.  The `pipeline/` subpackage in core is intentionally a single
file of schema definitions; execution lives in `qdmpy-server`.

---

## Dependencies

| Package | Use | Status |
|---------|-----|--------|
| `tomllib` | Parse `.toml` | stdlib >= 3.11 (already required) |
| `pydantic` | Config validation | already required |
| `tomli_w` | Serialise to `.toml` (optional) | new optional dep |

`tomli_w` is listed as `qdmpy-core[pipeline]` optional extra.  The core load path
has no new hard dependencies.

---

## Interactions

- **QEP-030 (Serializable Processor Pipeline):** `ProcessorsConfig.steps` uses
  the same discriminated-union format.  `qdmpy-server` passes
  `[s.model_dump() for s in config.processors.steps]` directly to
  `ODMRProcessorManager.from_config()`.
- **QEP-029 (FitManager config/execution split):** `FitConfig` maps directly to
  `FitManager.__init__` kwargs.
- **QEP-047 (Measurement metadata file):** `metadata.toml` per-folder values are
  merged by `qdmpy-server` at lower priority than `pipeline.toml`.
- **`qdmpy-server` (future package):** imports `PipelineConfig` and provides the
  watcher daemon, pipeline runner, status server, and CLI.

---

## Alternatives Considered

### A. YAML instead of TOML
Requires `pyyaml`; no stdlib support.  TOML is human-writable with typed values
and comments.  No new dependency for the read path.

### B. Python config file (`.py`)
Flexible, but not safely loadable by a daemon, not shareable with non-programmers,
and not serialisable back to disk without `ast` gymnastics.

### C. Keep schema inside `qdmpy-server`
Then the GUI and user scripts that want to load a `pipeline.toml` must depend on
`qdmpy-server`.  Keeping the schema in `qdmpy-core` means any consumer has access
to it without pulling in server dependencies.

### D. JSON instead of TOML
No comments; poor human UX for a file lab members are expected to edit by hand.

---

## Migration

Purely additive.  No existing module is changed.

---

## Implementation Plan

1. Add `src/qdmpy/pipeline/__init__.py` — re-export `PipelineConfig` and sub-models
2. Add `src/qdmpy/pipeline/config.py` — Pydantic models + `from_file()` / `to_file()`
3. Add `tomli_w` to `[project.optional-dependencies] pipeline = ["tomli_w"]` in
   `pyproject.toml`
4. Add `tests/pipeline/test_config.py` (see Test Plan)
5. Add `examples/pipeline.toml` — annotated reference file
6. Update CHANGELOG

---

## Test Plan

- [ ] `PipelineConfig.from_file()` loads a minimal TOML (only `[watcher].watch_dir`)
- [ ] `PipelineConfig.from_file()` loads a full TOML; all fields match expected values
- [ ] Missing `watch_dir` raises `ValidationError` with a clear field path
- [ ] `debounce_s = 0` raises `ValidationError` (gt=0 constraint)
- [ ] `port = 80` raises `ValidationError` (ge=1024 constraint)
- [ ] `ProcessorStep` with unknown extra fields round-trips via `model_dump()`
- [ ] `to_file()` -> `from_file()` round-trip produces an equal `PipelineConfig`
- [ ] `FitConfig.constraints` parses `{lower = 2.82, upper = 2.87}` correctly
