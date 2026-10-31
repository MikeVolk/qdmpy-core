"""Guard that the pytest marker vocabulary stays honest.

Until QEP-TEST-001, eleven markers were registered in pyproject.toml and not one
was applied to a single test. Nothing failed: pytest does not warn about a
registered marker with no members, so every ``-m`` command documented in
tests/integration/README.md silently selected zero tests and looked like a
suite with nothing to run.

These checks close that loop in both directions -- a registered marker with no
users, and a marker applied but never registered. The second case is also
covered at runtime by ``--strict-markers`` in addopts, which turns a typo such
as ``pytest.mark.fiting`` into a collection error; this module additionally
catches the reverse, which ``--strict-markers`` cannot see.

The analysis is static (ast + tomllib) rather than a pytest introspection run,
so it neither re-enters pytest nor depends on collection succeeding.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).parent.parent
TESTS_ROOT = REPO_ROOT / "tests"

# Files that are test modules by name but carry no tests of their own.
EXEMPT_FROM_PYTESTMARK: frozenset[str] = frozenset()

# Exactly one of these must be present on every test module: it is the axis that
# decides whether a test runs in a plain `pytest` invocation.
EXCLUSIVE_MARKERS = {"unit", "integration"}


def _test_modules() -> list[Path]:
    return sorted(
        path for path in TESTS_ROOT.rglob("test_*.py") if path.name not in EXEMPT_FROM_PYTESTMARK
    )


def _registered_markers() -> set[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    entries = config["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip() for entry in entries}


def _markers_in(path: Path) -> set[str]:
    """Every pytest.mark.<name> referenced anywhere in one module."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "mark"
            and isinstance(value.value, ast.Name)
            and value.value.id == "pytest"
        ):
            found.add(node.attr)
    return found


def _module_level_markers(path: Path) -> set[str] | None:
    """Marker names in the module's `pytestmark`, or None if absent."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if getattr(node.targets[0], "id", None) != "pytestmark":
            continue
        return _markers_in_expression(node.value)
    return None


def _markers_in_expression(node: ast.expr) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Attribute)
            and sub.value.attr == "mark"
        ):
            found.add(sub.attr)
    return found


def test_every_test_module_declares_pytestmark() -> None:
    """A module without pytestmark is invisible to every -m selection."""
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _test_modules()
        if _module_level_markers(path) is None
    ]

    assert not missing, f"test modules without a module-level pytestmark: {missing}"


def test_every_module_picks_exactly_one_of_unit_or_integration() -> None:
    """The run/skip axis must be unambiguous."""
    offenders = {}
    for path in _test_modules():
        markers = _module_level_markers(path) or set()
        chosen = markers & EXCLUSIVE_MARKERS
        if len(chosen) != 1:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = sorted(chosen)

    assert not offenders, (
        f"each module needs exactly one of {sorted(EXCLUSIVE_MARKERS)}: {offenders}"
    )


def test_no_marker_is_registered_without_users() -> None:
    """The failure mode this whole module exists to prevent."""
    used: set[str] = set()
    for path in _test_modules():
        used |= _markers_in(path)

    unused = sorted(_registered_markers() - used)

    assert not unused, (
        f"markers registered in pyproject.toml but applied to no test: {unused}. "
        f"Either apply them or remove the registration -- a marker that selects "
        f"nothing is worse than no marker, because it reads as documentation."
    )


def test_no_marker_is_used_without_registration() -> None:
    """Belt and braces alongside --strict-markers."""
    builtin = {"parametrize", "skipif", "skip", "xfail", "usefixtures", "filterwarnings"}
    used: set[str] = set()
    for path in _test_modules():
        used |= _markers_in(path)

    unregistered = sorted(used - _registered_markers() - builtin)

    assert not unregistered, f"markers used but not registered in pyproject.toml: {unregistered}"
