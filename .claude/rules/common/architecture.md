# Clean Code & SOLID Architecture

## Clean Code Principles

### Naming
- Names must reveal intent: `compute_b111_remanent()` not `calc()`
- No abbreviations except established domain terms (e.g. `odmr`, `nv`, `b111`, `gamma`)
- Boolean names start with `is_`, `has_`, `can_` (e.g. `is_fitted`, `has_two_polarities`)

### Functions
- Do **one** thing only — if you need "and" in the name, split it
- Max ~30 lines; functions > 50 lines are a smell
- Max 4 parameters; prefer dataclasses/Pydantic models for grouped config
- No flag arguments (bool params that change behavior) — split into two functions

### Files / Modules
- 200–400 lines typical, 600 max
- One primary class or cohesive group of functions per module
- No `utils.py` catch-alls — name by what they do (`odmr/loaders.py`, `fitting/models.py`)

### Comments
- Code should be self-documenting; comments explain **why**, not **what**
- No commented-out dead code; delete it (git remembers)
- Docstrings on public API (Google style, as per CLAUDE.md)

## SOLID Principles

### S — Single Responsibility
Each class/module has exactly one reason to change:
- `FitManager` — orchestrates fitting; does not do I/O or plotting
- `FitResult` — holds results and computes derived quantities; does not fit
- `ODMRData` — holds raw data; does not fit or plot

### O — Open/Closed
Extend via **new classes**, not by modifying existing ones:
- New ESR models → subclass `Model`; register via `ModelRegistry`
- New processors → implement `Processor` protocol; plug into pipeline
- Never add `if model_name == "new_thing"` branches inside existing classes

### L — Liskov Substitution
Subclasses must be drop-in replacements:
- All `Model` subclasses must accept `(x: NDArray, params: NDArray) → NDArray`
- All `Processor` implementations must honour `process(data: ODMRData) → ODMRData`

### I — Interface Segregation
Prefer narrow protocols over fat base classes:
- A class that only reads data shouldn't implement a write interface
- Use `typing.Protocol` for structural typing rather than deep inheritance hierarchies

### D — Dependency Inversion
Depend on abstractions, not concretions:
- `FitManager` accepts `model_name: str`; resolves via `ModelRegistry` (not hardcoded)
- `Measurement` accepts a loader; concrete MATLAB/CSV loaders are injected
- Use constructor injection; avoid globals and module-level singletons

## QDMpy-Specific Patterns

### Data flow: immutable results
`fit_manager.fit(data, freq)` returns a **new** `FitResult` — never mutates in place.
`ODMRData` processors return a new `ODMRData`; never overwrite the original.

### Layering
```
Measurement (orchestration)
  └── ODMRData (raw data + processors)
  └── FitManager (stateless fitting config)
        └── FitResult (computed results, lazy properties)
```
Layers only call **downward**. `FitResult` never imports `Measurement`.

### Registry pattern
`ModelRegistry` and processor pipelines follow Open/Closed:
```python
@ModelRegistry.register
class ESRNewModel(Model):
    name: ClassVar[str] = 'ESRNew'
    ...
```
