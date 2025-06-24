# ============================================================================
# examples/__init__.py
# ============================================================================

"""
Usage examples for KAN-GRN package.

This module provides example scripts and tutorials demonstrating
how to use the KAN-GRN package for various use cases.
"""

import os
from pathlib import Path

# Get examples directory
EXAMPLES_DIR = Path(__file__).parent


def get_example_script(script_name):
    """
    Get path to an example script.

    Args:
        script_name: Name of the example script

    Returns:
        Path to the example script
    """
    return EXAMPLES_DIR / script_name


def get_sample_data_path(filename):
    """
    Get path to sample data file.

    Args:
        filename: Name of the sample data file

    Returns:
        Path to the sample data file
    """
    return EXAMPLES_DIR / "sample_data" / filename


def list_examples():
    """
    List all available example scripts.

    Returns:
        List of example script names
    """
    return [
        f.name
        for f in EXAMPLES_DIR.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]


def list_notebooks():
    """
    List all available example notebooks.

    Returns:
        List of notebook names
    """
    notebooks_dir = EXAMPLES_DIR / "notebooks"
    if notebooks_dir.exists():
        return [
            f.name
            for f in notebooks_dir.iterdir()
            if f.is_file() and f.suffix == ".ipynb"
        ]
    return []


__all__ = [
    "get_example_script",
    "get_sample_data_path",
    "list_examples",
    "list_notebooks",
    "EXAMPLES_DIR",
]
