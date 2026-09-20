from __future__ import annotations

import unittest

from qdmpy import constants


class TestConstants(unittest.TestCase):
    """Tests for constants in QDMpy."""

    def test_gamma_nv(self) -> None:
        """Test NV center gyromagnetic ratio is in GHz/T."""
        assert constants.GAMMA_NV == 28.024

    def test_d_zfs(self) -> None:
        """Test zero-field splitting is in GHz."""
        assert constants.D_ZFS == 2.87

    def test_hyperfine_constants(self) -> None:
        """Test hyperfine splitting constants."""
        # Verify that the hyperfine constants are defined
        assert constants.AHYP_14N is not None
        assert constants.AHYP_15N is not None

        # Verify their values
        assert constants.AHYP_14N == 0.002158
        assert constants.AHYP_15N == 0.0015

        # Verify they are in the expected range (physical correctness)
        assert constants.AHYP_14N > 0.002
        assert constants.AHYP_14N < 0.0022
        assert constants.AHYP_15N > 0.001
        assert constants.AHYP_15N < 0.002

    def test_default_values(self) -> None:
        """Test default values for algorithms."""
        assert constants.DEFAULT_VMIN is not None
        assert constants.DEFAULT_VMAX is not None

        assert constants.DEFAULT_VMIN == 0.3
        assert constants.DEFAULT_VMAX == 0.7

        assert constants.DEFAULT_VMIN < constants.DEFAULT_VMAX
        assert constants.DEFAULT_VMIN > 0
        assert constants.DEFAULT_VMAX < 1

    def test_coordinate_labels_are_canonical(self) -> None:
        """Lock the QEP-025 coordinate vocabulary by literal value.

        These strings are a contract, not an implementation detail. Production
        selects on them (`.sel(polarity='neg')` in fitting/result.py,
        odmr/analysis.py and odmr/folding.py) and they are persisted into every
        saved .qdm file, so renaming one silently breaks reading old results.

        Test helpers import these constants rather than hardcoding them, which
        means a rename would otherwise propagate everywhere and fail nothing.
        This is the single place it fails loudly instead.
        """
        assert constants.POLARITY_LABELS == ["neg", "pos"]
        assert constants.FRANGE_LABELS == ["low", "high"]


if __name__ == "__main__":
    unittest.main()
