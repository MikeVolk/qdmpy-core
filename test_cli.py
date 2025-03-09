#!/usr/bin/env python
"""Test script for QDMpy CLI functionality.

This script simulates CLI calls to test the QDMpy command-line interface.
"""
import sys
from QDMpy.cli import main
from QDMpy.cli.calculate_QDMio import main as legacy_main

def test_main_help():
    """Test the main CLI help command."""
    print("\n=== Testing 'qdmpy --help' ===")
    sys.argv = ["qdmpy", "--help"]
    try:
        main()
    except SystemExit:
        pass

def test_models_command():
    """Test the models command."""
    print("\n=== Testing 'qdmpy models' ===")
    sys.argv = ["qdmpy", "models"]
    try:
        main()
    except SystemExit:
        pass

def test_legacy_help():
    """Test the legacy CLI help command."""
    print("\n=== Testing legacy 'calculate_QDMio --help' ===")
    sys.argv = ["calculate_QDMio", "--help"]
    try:
        legacy_main()
    except SystemExit:
        pass

if __name__ == "__main__":
    test_main_help()
    test_models_command()
    test_legacy_help()
    print("\nAll tests completed.")