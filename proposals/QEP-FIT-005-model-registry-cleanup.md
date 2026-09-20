# QEP-FIT-005 — Model ABC & Registry Cleanup

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-3) + LOW (L-2, L-3, L-7)
**Module:** `fitting/models.py`
**Revised:** 2026-09-19 -- drift update; Phase 2 rejected (tests are already isolated, and it would break the GUI); Phase 1 now also removes three ty suppressions

---

## Motivation

Four issues in `fitting/models.py` reduce testability and add dead code:

1. **H-3: `ModelRegistry` is a global mutable singleton.** `_registry` is a
   `ClassVar[dict]` on the class itself. Tests that register custom models
   pollute the global state. Parallel tests or test isolation is impossible
   without manual cleanup. This also means `import QDMpy` has the side effect
   of populating the registry (via `@ModelRegistry.register` decorators at
   module level).

2. **L-2: `Model.func()` has both `@abstractmethod` and `raise NotImplementedError`.**
   The `raise` is unreachable because `@abstractmethod` prevents instantiation
   of classes that don't override `func()`. Dead code.

3. **L-3: `Model.__init__` duplicates `name` from `ClassVar`.** Each concrete
   model declares `name: ClassVar[str] = "ESR14N"` *and* passes `"ESR14N"` to
   `super().__init__("ESR14N", ...)`. The `__init__` parameter is redundant
   and creates a divergence risk.

4. **L-7: `_main_demo()` is dead code.** The `if __name__ == "__main__"` block
   at the bottom is never exercised and has no test.

### Status (2026-09-19)

- L-2, L-3 and L-7 are still open: `models.py:444` (`raise
  NotImplementedError`), `models.py:798-807` (`_main_demo`), and every
  concrete model still passes its name to `Model.__init__`.
- L-3 now costs type-checker suppressions. The registry instantiates models
  with no arguments (`ModelRegistry.get`, `get_model_by_peaks`), while
  `Model.__init__` requires `name, n_peaks, parameter_names`. ty 0.0.82
  reports this, so three sites carry `# ty: ignore[...]  # QEP-FIT-005`
  (`models.py:609,631`, `guess.py:224,226`). Phase 1 removes them.
- `ModelRegistry.register` has been generic since 2026-09-19
  (`def register[M: type[Model]](cls, model_cls: M) -> M`).
- H-3 is mitigated in tests: an autouse fixture in `tests/conftest.py`
  (`_isolate_model_registry`) snapshots and restores `_registry` around
  every test. It was added after three test modules were found leaking
  custom models into the global registry.

## GUI Integration Requirements

1. **Touchpoints.** `app/measurement_vm.py` calls `ModelRegistry.get(name)`
   (3 sites) and reads `model.func`, `parameter_names` and `name`. With Phase
   2 rejected, all of these are unchanged.
2. **Persisted data.** None: model names in `.qdm` / result metadata do not
   change.
3. **User-facing behaviour.** None.
4. **Acceptance.** GUI test suite passes; load -> fit (auto model) ->
   per-pixel spectrum overlay renders the model curve.

## Proposed Changes

### Phase 1: Low-hanging fruit (L-2, L-3, L-7)

**Remove `raise NotImplementedError` from `Model.func()`:**

```python
@abstractmethod
def func(self, x: NDArray, parameters: NDArray) -> NDArray:
    """Evaluate the model."""
    ...  # @abstractmethod is sufficient
```

**Derive `name` from `ClassVar` in `__init__`:**

```python
class Model(ABC):
    name: ClassVar[str]  # must be set by subclasses

    def __init__(self, n_peaks: int, parameter_names: list[str]) -> None:
        self.parameter_names = parameter_names
        self.n_peaks = n_peaks
        # self.name is already set via ClassVar — no parameter needed
```

*Revised 2026-09-19:* dropping only `name` is not enough. The registry
instantiates `type[Model]` with no arguments, and a type checker checks that
call against the **base** `__init__`, which would still require
`n_peaks, parameter_names`. Every concrete model sets all three from
constants, so they all become ClassVars and the base `__init__` takes no
arguments:

```python
class Model(ABC):
    name: ClassVar[str]
    n_peaks: ClassVar[int]
    parameter_names: ClassVar[tuple[str, ...]]
    model_id: ClassVar[int] = -1          # -1 = no gpufit kernel

@ModelRegistry.register
class ESR14N(Model):
    name = "ESR14N"
    n_peaks = 3
    parameter_names = ("center", "width", "contrast_0", "contrast_1", "contrast_2", "offset")
    model_id = _GPUFIT_MODEL_ID_ESR14N
    ahyp: ClassVar[float] = AHYP_14N
```

`ModelRegistry.register` checks that the three required ClassVars are set
and raises `TypeError` at registration (import) time otherwise.
`parameter_names` becomes a tuple so the class-level value can't be mutated
through an instance; callers that index or iterate it are unaffected.

This removes the three `ty: ignore ... QEP-FIT-005` suppressions.

Update concrete models:

```python
@ModelRegistry.register
class ESR14N(Model):
    name: ClassVar[str] = 'ESR14N'

    def __init__(self) -> None:
        super().__init__(n_peaks=3, parameter_names=[...])
        self.ahyp = AHYP_14N
        self.model_id = 13
```

**Delete `_main_demo()` and `if __name__ == "__main__"` block.**

### Phase 2: Registry dependency inversion (H-3) -- rejected 2026-09-19

*Rejected.* The motivating problem, test pollution, is solved by the
`tests/conftest.py` snapshot fixture. qdmpy-gui calls the classmethod API
directly (`ModelRegistry.get`, `app/measurement_vm.py:467,605,702`), so
moving to instances would force a GUI change for no user-facing benefit. An
injectable registry can be revisited if a real caller needs two registries
in one process. Original proposal kept below for the record.

Replace `ClassVar[dict]` with instance-based registry that can be injected:

```python
class ModelRegistry:
    """Instance-based model registry. A default singleton is provided."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Model]] = {}

    def register(self, model_cls: type[Model]) -> type[Model]:
        self._registry[model_cls.name] = model_cls
        return model_cls

    def get(self, name: str) -> Model:
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found")
        return self._registry[name]()

    def all(self) -> dict[str, type[Model]]:
        return dict(self._registry)

    def available_models(self) -> list[str]:
        return sorted(self._registry.keys())


# Module-level default instance
default_registry = ModelRegistry()
```

Built-in models register on the default instance:

```python
@default_registry.register
class ESR14N(Model):
    ...
```

`FitManager` accepts an optional `registry` parameter:

```python
class FitManager:
    def __init__(self, model_name='ESR14N', *, registry=None, ...):
        self._registry = registry or default_registry
        self._model = self._registry.get(model_name.upper())
```

Tests create isolated registries:

```python
def test_custom_model():
    reg = ModelRegistry()
    reg.register(MyTestModel)
    fm = FitManager('MYTEST', registry=reg)
```

## Migration

- Phase 1 is backward compatible — `Model.__init__` signature changes but
  all concrete models are internal. External custom models that call
  `super().__init__(name, ...)` need to drop the `name` arg.
- ~~Phase 2 changes `ModelRegistry` from a class with classmethods to an
  instance with methods.~~ Rejected; see Phase 2.
- Breaking for external custom models only: `super().__init__(name, n_peaks,
  parameter_names)` goes away and the three values become ClassVars (see
  Phase 1). A model registered without them raises `TypeError` at import,
  not at first use. `docs/extending.md`, the `Model` docstring example and
  the custom models in the test suite must be updated.

## Test Plan

- [ ] Verify `@abstractmethod` alone prevents `Model()` instantiation
- [ ] Verify concrete model `.name` matches `ClassVar` (no `__init__` param)
- [ ] Verify `_main_demo` is gone
- [ ] ~~Verify isolated `ModelRegistry()` instances don't share state~~ (Phase 2 rejected)
- [ ] ~~Verify `FitManager(registry=...)` uses injected registry~~ (Phase 2 rejected)
- [ ] Verify `ModelRegistry.all()` contains ESR14N, ESR15N, ESRSINGLE after import
- [ ] Verify registering a model missing `name`/`n_peaks`/`parameter_names` raises `TypeError`
- [ ] Verify `uv run ty check src/qdmpy` passes with the QEP-FIT-005 ignores removed
