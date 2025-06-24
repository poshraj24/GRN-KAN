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


__all__ = [
    "KANGRNPipeline",
    "PipelineConfig",
    "ModelConfig",
    "TrainingConfig",
    "NetworkConfig",
]
