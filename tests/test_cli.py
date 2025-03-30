from __future__ import annotations

import unittest
import sys
import argparse
import pathlib
from unittest.mock import patch, MagicMock

# Import the modules we want to test
from QDMpy.cli import main
from QDMpy.cli.qdmpy_cli import create_parser, process_command


class TestCLI(unittest.TestCase):
    """Tests for QDMpy command-line interface."""

    def test_create_parser(self):
        """Test the parser creation for the CLI."""
        parser = create_parser("1.0.0")
        
        # Check that the parser was created
        self.assertIsNotNone(parser)
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Check version action
        for action in parser._actions:
            if isinstance(action, argparse._VersionAction):
                self.assertIn("1.0.0", action.version)
                break
        else:
            self.fail("No version action found in parser")

    @patch("QDMpy.cli.CLI_LOGGER")
    def test_main_no_args(self, mock_logger):
        """Test main function with no arguments."""
        # Mock sys.argv
        with patch("sys.argv", ["qdmpy"]):
            # Mock parser.parse_args() to return an object without 'func'
            mock_args = MagicMock()
            del mock_args.func
            
            with patch("QDMpy.cli.qdmpy_cli.create_parser") as mock_create_parser:
                mock_parser = MagicMock()
                mock_parser.parse_args.return_value = mock_args
                mock_create_parser.return_value = mock_parser
                
                # Call main and check the result
                result = main()
                
                # Should print help and return 2
                mock_parser.print_help.assert_called_once()
                self.assertEqual(result, 2)

    @patch("QDMpy.cli.qdmpy_cli.process_command")
    @patch("QDMpy.cli.CLI_LOGGER")
    def test_main_with_command(self, mock_logger, mock_process_command):
        """Test main function with a valid command."""
        # Mock successful command execution
        mock_process_command.return_value = 0
        
        # Mock sys.argv
        with patch("sys.argv", ["qdmpy", "info"]):
            # Mock parser.parse_args() to return an object with 'func'
            mock_args = MagicMock()
            mock_args.func = MagicMock()
            mock_args.debug = False
            
            with patch("QDMpy.cli.qdmpy_cli.create_parser") as mock_create_parser:
                mock_parser = MagicMock()
                mock_parser.parse_args.return_value = mock_args
                mock_create_parser.return_value = mock_parser
                
                # Call main and check the result
                result = main()
                
                # Should call process_command and return 0
                mock_process_command.assert_called_once_with(mock_args)
                self.assertEqual(result, 0)

    @patch("QDMpy.cli.CLI_LOGGER")
    def test_main_with_exception(self, mock_logger):
        """Test main function when an exception occurs."""
        # Mock sys.argv
        with patch("sys.argv", ["qdmpy", "info"]):
            # Mock parser.parse_args() to return an object with 'func'
            mock_args = MagicMock()
            mock_args.func = MagicMock()
            mock_args.debug = False
            
            with patch("QDMpy.cli.qdmpy_cli.create_parser") as mock_create_parser, \
                 patch("QDMpy.cli.qdmpy_cli.process_command") as mock_process_command:
                mock_parser = MagicMock()
                mock_parser.parse_args.return_value = mock_args
                mock_create_parser.return_value = mock_parser
                
                # Make process_command raise an exception
                mock_process_command.side_effect = ValueError("Test error")
                
                # Call main and check the result
                result = main()
                
                # Should log the error and return 1
                mock_logger.error.assert_called_once()
                self.assertEqual(result, 1)

    @patch("QDMpy.cli.CLI_LOGGER")
    def test_main_with_keyboard_interrupt(self, mock_logger):
        """Test main function when KeyboardInterrupt occurs."""
        # Mock sys.argv
        with patch("sys.argv", ["qdmpy", "info"]):
            # Mock parser.parse_args() to return an object with 'func'
            mock_args = MagicMock()
            mock_args.func = MagicMock()
            mock_args.debug = False
            
            with patch("QDMpy.cli.qdmpy_cli.create_parser") as mock_create_parser, \
                 patch("QDMpy.cli.qdmpy_cli.process_command") as mock_process_command:
                mock_parser = MagicMock()
                mock_parser.parse_args.return_value = mock_args
                mock_create_parser.return_value = mock_parser
                
                # Make process_command raise KeyboardInterrupt
                mock_process_command.side_effect = KeyboardInterrupt()
                
                # Call main and check the result
                result = main()
                
                # Should log a warning and return 130 (SIGINT)
                mock_logger.warning.assert_called_once()
                self.assertEqual(result, 130)




if __name__ == '__main__':
    unittest.main()