from __future__ import annotations

import unittest
from typing import NoReturn

import pytest

from QDMpy.exceptions import CantImportError, ModelGuessNotPossibleError, WrongFileNumberError


class TestExceptions(unittest.TestCase):
    """Tests for custom exceptions in QDMpy."""

    def test_cant_import_error(self) -> NoReturn:
        """Test CantImportError."""
        # Test that the exception can be raised with a message
        error_message = "Could not import required module 'test_module'"
        with pytest.raises(CantImportError) as context:
            raise CantImportError(error_message)

        # Verify the error message
        assert str(context.value) == error_message

        # Verify inheritance
        assert isinstance(context.value, Exception)

    def test_wrong_file_number(self) -> NoReturn:
        """Test WrongFileNumberError."""
        # Test that the exception can be raised with a message
        error_message = "Expected 2 files, got 1"
        with pytest.raises(WrongFileNumberError) as context:
            raise WrongFileNumberError(error_message)

        # Verify the error message
        assert str(context.value) == error_message

        # Verify inheritance
        assert isinstance(context.value, Exception)

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


if __name__ == "__main__":
    unittest.main()
