# QEP-FIT-005 — Model ABC & Registry Cleanup

**Status:** Draft
**Created:** 2026-02-22
**Severity:** HIGH (H-3) + LOW (L-2, L-3, L-7)
**Module:** `fitting/models.py`

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

## GUI Integration Requirements

1. List the exact core API/data contract touchpoints used by `qdmpy-gui` (view-model calls, settings keys, map/result fields).
2. Define GUI state/settings migration behavior for any changed defaults, renamed keys, or persisted session/config data.
3. Specify expected user-facing behavior in the GUI for progress, warnings, and errors introduced by this QEP.
4. Include explicit GUI acceptance checks for this QEP scope: `load -> run action -> inspect outputs -> save/reload`, and verify no GUI-only workaround is required.
5. If impact is expected to be none, state the rationale and include a smoke check confirming no `qdmpy-gui` regression.

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

### Phase 2: Registry dependency inversion (H-3)

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
- Phase 2 changes `ModelRegistry` from a class with classmethods to an
  instance with methods. Code calling `ModelRegistry.get(...)` must change to
  `default_registry.get(...)` or accept an injected registry. A module-level
  `get = default_registry.get` alias can ease transition.

## Test Plan

- [ ] Verify `@abstractmethod` alone prevents `Model()` instantiation
- [ ] Verify concrete model `.name` matches `ClassVar` (no `__init__` param)
- [ ] Verify `_main_demo` is gone
- [ ] Verify isolated `ModelRegistry()` instances don't share state
- [ ] Verify `FitManager(registry=...)` uses injected registry
- [ ] Verify `default_registry` contains ESR14N, ESR15N, ESRSINGLE after import
