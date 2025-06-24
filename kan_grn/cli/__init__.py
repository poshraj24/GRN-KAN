# ============================================================================
# kan_grn/cli/__init__.py
# ============================================================================

"""
Command-line interface for KAN-GRN.

This module provides command-line tools for running KAN-GRN pipelines,
training models, and building networks from the terminal.
"""

from .commands import main, create_parser

__all__ = [
    "main",
    "create_parser",
]

# Version information for CLI
__cli_version__ = "1.0.0"
__cli_author__ = "Posh Raj Dahal"
