# ============================================================================
# kan_grn/pipeline/__init__.py
# ============================================================================

"""
Pipeline orchestration and configuration for KAN-GRN.

This module contains the main pipeline class and configuration management
for running complete gene regulatory network inference workflows.
"""

from .main_pipeline import KANGRNPipeline
from .config import (
    PipelineConfig,
    ModelConfig,
    TrainingConfig,
    NetworkConfig,
)

# Optional: Import validation functions if they exist
try:
    from .validation import validate_expression_file, validate_network_file

    validation_available = True
except ImportError:
    validation_available = False

__all__ = [
    "KANGRNPipeline",
    "PipelineConfig",
    "ModelConfig",
    "TrainingConfig",
    "NetworkConfig",
]

if validation_available:
    __all__.extend(["validate_expression_file", "validate_network_file"])
