"""Shared pytest configuration for the QDMpy test suite.

Data constructors live in ``tests/helpers.py`` and are imported directly; they
are plain functions rather than fixtures, so they work under any pytest import
mode and can be called from inside ``@pytest.mark.parametrize`` bodies.

This module previously also defined MOCK_SETTINGS, make_xr_data, rng,
sample_numpy_data, sample_frequencies, sample_data, sample_parameters and
sample_fit_result. Every one of them was shadowed by a local fixture in each
test module, so none was ever resolved from here. Worse, three different test
modules defined incompatible fixtures all named ``sample_data`` -- returning a
3-tuple, a bare DataArray, and a numpy tuple respectively -- so whichever one a
new test file inherited was a coin toss. They were removed by QEP-TEST-001.
"""

from __future__ import annotations

import pytest

from qdmpy.fitting.models import ModelRegistry


@pytest.fixture(autouse=True)
def _isolate_model_registry():
    """Restore the global ModelRegistry after every test.

    Several tests register custom models (e.g. extension and backend tests).
    Without cleanup they leak into every later test, and registry-wide lookups
    such as ``get_model_by_peaks`` then see models that only existed for one
    test -- the old first-match lookup hid this by depending on insertion order.
    """
    snapshot = dict(ModelRegistry._registry)
    yield
    ModelRegistry._registry.clear()
    ModelRegistry._registry.update(snapshot)
