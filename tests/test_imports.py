"""Smoke tests verifying every name in QDMpy.__all__ is importable and is the right type.

These tests catch regressions where a refactor moves or renames something that was
previously exported. They run fast (no data, no GPU) and serve as the first line of
defence for the public API contract.
"""

from __future__ import annotations

import importlib

import pytest


ALL_NAMES = [
    # Entry points
    ('load', 'callable'),
    ('Measurement', 'class'),
    ('QDMResult', 'class'),
    # Data loading
    ('MatlabLoader', 'class'),
    ('ODMRData', 'class'),
    ('ODMR', 'class'),
    # Processing
    ('BinningProcessor', 'class'),
    ('FluorescenceCorrectionProcessor', 'class'),
    ('NormalizationProcessor', 'class'),
    ('OutlierProcessor', 'class'),
    ('Processor', 'class'),
    # Fitting
    ('FitManager', 'class'),
    ('FitResult', 'class'),
    ('Model', 'class'),
    ('ModelRegistry', 'class'),
    # Magnetic reconstruction
    ('FieldReconstructor', 'class'),
    ('MagneticMap', 'class'),
    # Settings
    ('NvSettings', 'class'),
    ('get_settings', 'callable'),
    ('is_pygpufit_available', 'callable'),
    ('reset_settings', 'callable'),
    # Testing / tutorial utilities
    ('make_synthetic_fit_result', 'callable'),
    ('make_synthetic_odmr_data', 'callable'),
    ('make_synthetic_qdm_result', 'callable'),
    # Field processing
    ('BaseFieldProcessor', 'class'),
    ('BlankSubtractor', 'class'),
    ('FieldProcessingPipeline', 'class'),
    ('HotPixelFilter', 'class'),
    ('QuadraticBackgroundSubtractor', 'class'),
    ('UpwardContinuation', 'class'),
]


@pytest.mark.parametrize('name,kind', ALL_NAMES)
def test_name_importable_from_qdmpy(name: str, kind: str) -> None:
    """Each name in __all__ must be importable directly from qdmpy_core."""
    import qdmpy_core
    assert hasattr(QDMpy, name), f'QDMpy.{name} not found'
    obj = getattr(QDMpy, name)
    if kind == 'class':
        assert isinstance(obj, type), f'QDMpy.{name} should be a class, got {type(obj)}'
    elif kind == 'callable':
        assert callable(obj), f'QDMpy.{name} should be callable'


def test_all_is_complete() -> None:
    """__all__ must contain exactly the names listed in ALL_NAMES (no surprises)."""
    import qdmpy_core
    expected = {name for name, _ in ALL_NAMES}
    actual = set(QDMpy.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f'Names missing from __all__: {missing}'
    assert not extra, f'Unexpected names in __all__ (update ALL_NAMES): {extra}'
