from __future__ import annotations

import importlib.util
import unittest

import matplotlib.pyplot as plt
import numpy as np


@unittest.skipIf(importlib.util.find_spec("QDMpy._core") is None,
                "QDMpy._core module not found - skipping plotting tests")
class TestPlotting(unittest.TestCase):
    """Tests for plotting functions in QDMpy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.frequencies = np.linspace(2.87e9, 2.9e9, 100)
        self.data = 1.0 - 0.1 * np.exp(-((self.frequencies - 2.885e9) / 1e7) ** 2)

        # Create a mock figure for testing
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Save original plt.figure method to restore later
        self.original_figure = plt.figure

    def tearDown(self):
        """Clean up after tests."""
        # Close all figures
        plt.close('all')

        # Restore original plt.figure method
        plt.figure = self.original_figure

    def test_plotting_placeholder(self):
        """Placeholder test to ensure the test module runs."""
        # This test doesn't actually test anything, it's just here to make sure
        # the test module can be loaded and executed
        self.assertTrue(True)
