"""Pytest configuration for integration tests.

Intentionally empty. Each module here gates itself with its own `pytestmark`
(a CUDA probe, a torch import check, or a real-data path check) rather than
sharing fixtures, which keeps the reason a test skipped next to the test.

This module previously defined a PROJECT_ROOT constant that nothing imported.
"""

from __future__ import annotations
