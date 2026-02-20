from __future__ import annotations

import unittest
from typing import NoReturn

import pytest

from QDMpy.exceptions import ModelGuessNotPossibleError, QDMpyError


class TestExceptions(unittest.TestCase):
    """Tests for custom exceptions in QDMpy."""

    def test_model_guess_not_possible(self) -> NoReturn:
        """Test ModelGuessNotPossibleError."""
        # Test that the exception can be raised with a message
        error_message = "Could not automatically determine model for data"
        with pytest.raises(ModelGuessNotPossibleError) as context:
            raise ModelGuessNotPossibleError(error_message)

        # Verify the error message
        assert str(context.value) == error_message

        # Verify inheritance
        assert isinstance(context.value, Exception)
        assert isinstance(context.value, QDMpyError)


if __name__ == "__main__":
    unittest.main()
