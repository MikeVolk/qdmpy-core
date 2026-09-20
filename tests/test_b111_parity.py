"""Parity between the two independent B111 implementations.

qdmpy computes B111 twice, from the same physics:

* ``FitResult._compute_b111`` (fitting/result.py) works from fitted centers and
  applies the sign convention positionally, ``np.array([-1, 1])[:n_pol]``.
* ``b111_from_dip_positions`` (odmr/analysis.py) works from raw dip positions
  and applies it by label, ``{neg: -1, pos: +1}``.

QEP-FIT-004 records that these already diverged once, during the QEP-025
coordinate relabelling, and plans to extract the shared physics. This module is
the characterisation test that makes that extraction verifiable rather than
hopeful: it pins current agreement so a refactor that changes a sign, drops a
factor of two, or mixes up the polarity order fails here.

It is also the concrete reason QEP-TEST-001 had to land first. ``analysis.py``
selects by label (``.sel(polarity='neg')``), so it cannot consume data carrying
the pre-QEP-025 ``pol_0``/``pol_1`` labels at all -- with the old fixtures, one
of the two duplicated paths was untestable.

Tolerance is derived, not tuned. The dip-position path takes an ``argmin`` over
the frequency grid, so its resonance estimate is quantised to one grid step
while the fitted path is continuous. One step of ``df`` GHz maps to
``df / 2 / GAMMA_NV * 1e6`` uT. Measured deviation stays at 0.77-0.94 of that
bound across n_freq = 100, 200, 400 and 800, and shrinks as the grid refines,
which is quantisation rather than a physics discrepancy.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdmpy.constants import GAMMA_NV
from qdmpy.odmr.analysis import b111_from_dip_positions
from qdmpy.testing import make_synthetic_fit_result, make_synthetic_odmr_data

pytestmark = [pytest.mark.unit, pytest.mark.fitting, pytest.mark.magnetic_fields]

SHAPE = (8, 8)

# Half-width of each branch's frequency axis in make_synthetic_odmr_data.
HALF_WIDTH_GHZ = 0.013


def _grid_step_ut(n_freq: int) -> float:
    """One frequency grid step expressed as a B111 uncertainty in uT."""
    step_ghz = 2 * HALF_WIDTH_GHZ / (n_freq - 1)
    return step_ghz / 2 / GAMMA_NV * 1e6


@pytest.fixture
def paths() -> tuple[dict[str, np.ndarray], object]:
    """Both B111 implementations evaluated on the same synthetic field."""
    n_freq = 200
    odmr = make_synthetic_odmr_data(shape=SHAPE, n_freq=n_freq, noise=0.0)
    dip_based = b111_from_dip_positions(odmr.data)
    fit_based = make_synthetic_fit_result(shape=SHAPE)
    return dip_based, fit_based


def test_induced_agrees_exactly(paths) -> None:
    """The induced (bias) component must match to floating-point precision.

    Both paths see the same applied bias, and it cancels the per-pixel
    quantisation because induced = (neg - pos) / 2 subtracts two estimates that
    are quantised identically.
    """
    dip_based, fit_based = paths

    np.testing.assert_allclose(
        np.asarray(dip_based["induced"]),
        np.asarray(fit_based.b111_induced),
        atol=1e-9,
    )


def test_remanent_agrees_within_one_grid_step(paths) -> None:
    """The remanent component must match to the grid-quantisation bound."""
    dip_based, fit_based = paths

    from_dips = np.asarray(dip_based["remanent"])
    from_fit = np.asarray(fit_based.b111_remanent)

    max_deviation = np.abs(from_dips - from_fit).max()
    assert max_deviation < _grid_step_ut(200), (
        f"paths differ by {max_deviation:.3f} uT, beyond one grid step "
        f"({_grid_step_ut(200):.3f} uT) -- this is a physics divergence, "
        f"not quantisation"
    )
    assert np.corrcoef(from_dips.ravel(), from_fit.ravel())[0, 1] > 0.99


def test_remanent_deviation_shrinks_with_finer_grid() -> None:
    """Confirms the residual is quantisation: refine the grid, it halves.

    If the two paths ever diverge for a real reason -- a sign flip, a lost
    factor of two, a polarity ordering mismatch -- the deviation becomes
    grid-independent and this fails while the bound above might not.
    """
    fit_based = np.asarray(make_synthetic_fit_result(shape=SHAPE).b111_remanent)

    deviations = []
    for n_freq in (200, 800):
        odmr = make_synthetic_odmr_data(shape=SHAPE, n_freq=n_freq, noise=0.0)
        from_dips = np.asarray(b111_from_dip_positions(odmr.data)["remanent"])
        deviations.append(np.abs(from_dips - fit_based).max())

    coarse, fine = deviations
    assert fine < coarse / 2, (
        f"deviation {fine:.3f} uT at n_freq=800 did not fall below half of "
        f"{coarse:.3f} uT at n_freq=200 -- the residual is not quantisation"
    )


def test_both_paths_use_the_same_sign_convention(paths) -> None:
    """Remanent varies spatially and induced is the constant bias, in both.

    A swapped polarity order exchanges these two roles, which neither an
    allclose on a single map nor a correlation would necessarily catch.
    """
    dip_based, fit_based = paths

    for remanent, induced in (
        (np.asarray(dip_based["remanent"]), np.asarray(dip_based["induced"])),
        (np.asarray(fit_based.b111_remanent), np.asarray(fit_based.b111_induced)),
    ):
        assert remanent.std() > 1.0, "remanent should carry the dipole structure"
        assert induced.std() < 1e-6, "induced should be the constant applied bias"
        assert induced.mean() < 0, "induced sign convention changed"
