# QEP-047 — Measurement Metadata File

**Status:** Implemented (2026-03-06)
**Created:** 2026-02-21

---

## Motivation

Each QDM measurement folder currently carries no machine-readable description of
itself. Parameters like `pixel_spacing`, `model`, and `bin_factor` must be passed
every time `Measurement.from_folder()` is called, and there is nowhere to record
sample name, date, or free-form notes. The `Measurement.metadata` dict exists but
is never populated.

A per-folder `metadata.toml` file solves both problems: it provides persistent
defaults that travel with the data, and a place to annotate the measurement.

---

## Design

### File location and name

```
/data/FOV18x/
    run_00000.mat
    run_00001.mat
    light.png
    laser.png
    metadata.toml          ← new, optional
```

### Format — TOML

TOML is in the Python 3.11+ stdlib (`tomllib`) — no new dependency.
It is human-writable, comment-friendly, and well-typed.

### Schema

```toml
# metadata.toml — optional fields, all have code-level defaults

[measurement]
sample_name    = "FOV18x"
date           = "2025-11-03"
operator       = "alice"
notes          = "High fluence region, diamond batch C"
diamond_type   = "14N"          # informational only

[acquisition]
pixel_spacing  = 4e-6           # metres; overrides from_folder() default
bin_factor     = 1              # applied if > 1
model          = "ESR14N"       # overrides from_folder() default
normalize      = true
fluorescence_correction = 0.2   # set to 0.0 to disable
```

All keys are optional. Missing keys fall back to `from_folder()` parameter
defaults, which still take precedence over the file when passed explicitly by
the caller.

### Priority (highest → lowest)

1. Explicit keyword argument to `from_folder()` (caller wins)
2. `metadata.toml` value
3. `from_folder()` default

This means existing call sites that pass explicit values are unaffected.

### Integration point

`from_folder()` gains a private helper `_load_metadata_file(path)` that returns
a dict. Values from `[acquisition]` are used as fallback defaults; values from
`[measurement]` are stored verbatim in `Measurement.metadata`.

```python
@staticmethod
def _load_metadata_file(path: Path) -> dict:
    """Load metadata.toml from path if present; return empty dict otherwise."""
    toml_path = path / 'metadata.toml'
    if not toml_path.exists():
        return {}
    import tomllib
    with toml_path.open('rb') as f:
        return tomllib.load(f)
```

Signature of `from_folder()` is unchanged — no new parameters needed.

---

## Alternatives Considered

| Format | Verdict |
|--------|---------|
| YAML | Requires `pyyaml` dependency; TOML preferred |
| JSON | No comments; poor human UX |
| `.ini` / `configparser` | No typed values (everything is a string) |
| Markdown | Not machine-parseable |
| `metadata.toml` vs `qdmpy.toml` | `metadata.toml` is self-describing in a file browser |

---

## Migration

No breaking changes. The file is optional; existing folders without it behave
identically. No changes to `Measurement.__init__()` or the public API.

---

## Implementation Plan

1. Add `Measurement._load_metadata_file(path)` static method
2. Update `from_folder()` to call it and merge acquisition defaults
3. Store `metadata['measurement']` section in `self.metadata`
4. Add `tests/test_measurement.py` cases: missing file, partial file, full file,
   explicit-arg overrides file value
5. Add example `metadata.toml` to `tests/data/` fixture folder
6. Update CHANGELOG
