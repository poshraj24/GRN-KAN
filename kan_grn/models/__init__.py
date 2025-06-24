# ============================================================================
# kan_grn/models/__init__.py
# ============================================================================

"""
Model definitions and interfaces for KAN-GRN.

This module contains model architectures, wrappers, and interfaces for
Kolmogorov-Arnold Networks and related models.
"""

# Import KAN model (adjust import based on your actual KAN implementation)
try:
    from .kan_model import KANWrapper, create_kan_model

    __all__ = ["KANWrapper", "create_kan_model"]
except ImportError:
    # If you're using an external KAN library
    try:
        from kan import KAN  # Adjust this import to match your KAN library

        __all__ = ["KAN"]
    except ImportError:
        import warnings

        warnings.warn(
            "KAN model not available. Please install the required KAN library.",
            ImportWarning,
        )
        __all__ = []
