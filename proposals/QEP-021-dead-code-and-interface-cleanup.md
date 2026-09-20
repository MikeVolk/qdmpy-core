# QEP-021: Dead Code, Broken Interfaces, and Hygiene Cleanup

**Status:** Implemented (QEP-032, 2026-02-20)
**Priority:** Medium
**Affects:** Multiple modules

## Problem

The codebase has accumulated several pieces of dead code, incomplete
implementations, and interface inconsistencies that add confusion without
providing value.

### 1. CLI is a non-functional skeleton

Two of three CLI commands raise `NotImplementedError`:

```python
# cli/qdmpy_cli.py
def process_command_handler(args):
    raise NotImplementedError("The 'process' command is not yet available...")

def info_command_handler(args):
    raise NotImplementedError("The 'info' command is not yet available...")
```

The `models` command instantiates model objects but **never prints anything**:

```python
def models_command_handler(args):
    for name, _info in models.items():
        ModelRegistry.get(name)  # Instantiated, then discarded
    return 0
```

Users who run `qdmpy models` get zero output and exit code 0.

### 2. `__main__` blocks with hardcoded paths

`measurement.py:323-354` has a `__main__` block with:
```python
data_folder = "/home/mike/git/QDMpy/tests/data/FOV18x"
```
This is a developer-specific path that won't work for anyone else and isn't
a proper entry point.

### 3. `utils.py:main()` does nothing

```python
def main() -> None:
    # Example of millify function
    # Example of double_norm
    np.array([1, 2, 5, 10])  # Created and discarded
```

### 4. Unused `_B111` attribute in Measurement

`Measurement.__init__` initializes `self._B111 = None` (line 130) but it is
never read or written anywhere in the class. B111 calculations live entirely
in `FitResult`.

### 5. Deprecated aliases in exceptions.py

```python
CantImportError = DependencyError
WrongFileNumberError = DataValidationError
```

Per CLAUDE.md: "Avoid backwards-compatibility hacks." If nothing uses these
aliases, they should be removed.

### 6. `ModelRegistry.register` creates throwaway instance

```python
@classmethod
def register(cls, model_cls):
    instance = model_cls()          # Creates a full instance
    cls._registry[instance.name] = model_cls  # Just to read .name
    return model_cls
```

This instantiates the model class solely to read its `name` attribute. If
`__init__` ever has side effects (GPU allocation, file I/O), this pattern
breaks.

### 7. `ModelRegistry._initialize_constraints` is duplicated logic

`ModelRegistry._initialize_constraints()` (models.py:389-414) duplicates
the same logic as `ConstraintManager.__init__()` (fit.py:61-73). Both
iterate over `model.parameters_unique` and call `getattr(settings, ...)`.
The ModelRegistry version appears unused.

### 8. FitResult serialization uses allow_pickle=True

```python
# result.py:636
data = np.load(filepath, allow_pickle=True)
```

`allow_pickle=True` is a known security risk (arbitrary code execution via
crafted `.npz` files). The serialization also stores Python dicts as numpy
object arrays, which is fragile and non-portable.

## Proposed Fix

### Fix 1: Make the models CLI command functional

```python
def models_command_handler(args):
    models = ModelRegistry.all()
    if args.model_name:
        model = ModelRegistry.get(args.model_name)
        print(f"Model: {model.name}")
        print(f"Parameters: {model.n_parameters}")
        print(f"Peaks: {model.n_peaks}")
        for p in model.parameters_unique:
            print(f"  {p}: {model.parameter_types[p]} ({model.units[p]})")
    else:
        for name in models:
            model = ModelRegistry.get(name)
            print(f"  {model.name}: {model.n_parameters} params, {model.n_peaks} peaks")
    return 0
```

Mark `process` and `info` as planned but not hide them behind
`NotImplementedError` — either remove them from the parser or implement them.

### Fix 2: Remove dead code

- Delete `measurement.py` `__main__` block.
- Delete `utils.py` `main()` function and `__main__` block.
- Delete `Measurement._B111` attribute.
- Delete `CantImportError` and `WrongFileNumberError` aliases (verify no usage).
- Delete `ModelRegistry._initialize_constraints` (verify no usage).

### Fix 3: Use class attribute for model name in registry

```python
class Model(ABC):
    name: ClassVar[str]  # Require as class variable

@classmethod
def register(cls, model_cls):
    cls._registry[model_cls.name] = model_cls  # No instantiation needed
    return model_cls
```

Or use a simpler approach: pass the name to the decorator.

### Fix 4: Replace pickle-based serialization

Use a structured format that doesn't require `allow_pickle`:

```python
def save_results(self, filepath):
    arrays = {k: v for k, v in self.parameters.items()}
    arrays['__model_name'] = np.array(self.model_name)
    arrays['__scan_dimensions'] = np.array(self.scan_dimensions)
    arrays['__pixel_spacing'] = np.array(self.pixel_spacing)
    np.savez_compressed(filepath, **arrays)

@classmethod
def load_results(cls, filepath):
    data = np.load(filepath)  # No allow_pickle needed
    ...
```

## Validation

- `grep -r "CantImportError\|WrongFileNumberError" src/ tests/` to verify
  aliases are unused.
- `grep -r "_B111" src/` to verify attribute is unused.
- `grep -r "_initialize_constraints" src/` to verify method is unused.
- CLI tests: verify `qdmpy models` produces output.

## Files to change

| File | Change |
|------|--------|
| `src/QDMpy/cli/qdmpy_cli.py` | Fix models command output, remove or stub process/info |
| `src/QDMpy/measurement.py` | Remove __main__ block, remove _B111 |
| `src/QDMpy/utils.py` | Remove main() and __main__ |
| `src/QDMpy/exceptions.py` | Remove deprecated aliases |
| `src/QDMpy/models.py` | Fix register pattern, remove _initialize_constraints |
| `src/QDMpy/result.py` | Remove allow_pickle, use structured serialization |
