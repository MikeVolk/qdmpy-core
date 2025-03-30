from __future__ import annotations

import unittest

from QDMpy import constants


class TestConstants(unittest.TestCase):
    """Tests for constants in QDMpy."""

    def test_hyperfine_constants(self):
        """Test hyperfine splitting constants."""
        # Verify that the hyperfine constants are defined
        self.assertIsNotNone(constants.AHYP_14N)
        self.assertIsNotNone(constants.AHYP_15N)
        
        # Verify their values
        self.assertEqual(constants.AHYP_14N, 0.002158)
        self.assertEqual(constants.AHYP_15N, 0.0015)
        
        # Verify they are in the expected range (physical correctness)
        self.assertGreater(constants.AHYP_14N, 0.002)
        self.assertLess(constants.AHYP_14N, 0.0022)
        self.assertGreater(constants.AHYP_15N, 0.001)
        self.assertLess(constants.AHYP_15N, 0.002)

    def test_default_values(self):
        """Test default values for algorithms."""
        # Verify that the default values are defined
        self.assertIsNotNone(constants.DEFAULT_VMIN)
        self.assertIsNotNone(constants.DEFAULT_VMAX)
        self.assertIsNotNone(constants.PROMINENCE)
        
        # Verify their values
        self.assertEqual(constants.DEFAULT_VMIN, 0.3)
        self.assertEqual(constants.DEFAULT_VMAX, 0.7)
        self.assertEqual(constants.PROMINENCE, 0.0004)
        
        # Verify expected relationships
        self.assertLess(constants.DEFAULT_VMIN, constants.DEFAULT_VMAX)
        self.assertGreater(constants.DEFAULT_VMIN, 0)
        self.assertLess(constants.DEFAULT_VMAX, 1)


if __name__ == '__main__':
    unittest.main()