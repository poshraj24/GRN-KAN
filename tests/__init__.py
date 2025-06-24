# ============================================================================
# tests/__init__.py
# ============================================================================

"""
Test suite for KAN-GRN package.

This module contains unit tests, integration tests, and test utilities
for validating the functionality of the KAN-GRN package.
"""

import os
from pathlib import Path

# Get test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


def get_test_data_path(filename):
    """
    Get path to test data file.

    Args:
        filename: Name of the test data file

    Returns:
        Path to the test data file
    """
    return TEST_DATA_DIR / filename


def get_small_expression_data():
    """Get path to small test expression dataset."""
    return get_test_data_path("small_expression.h5ad")


def get_test_network():
    """Get path to test network file."""
    return get_test_data_path("test_network.tsv")


def get_test_config():
    """Get path to test configuration file."""
    return get_test_data_path("test_config.json")


__all__ = [
    "get_test_data_path",
    "get_small_expression_data",
    "get_test_network",
    "get_test_config",
    "TEST_DATA_DIR",
]
