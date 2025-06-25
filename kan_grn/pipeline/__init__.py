# ============================================================================
# kan_grn/pipeline/__init__.py
# ============================================================================

"""
Pipeline orchestration and configuration for KAN-GRN.

This module contains the main pipeline class and configuration management
for running complete gene regulatory network inference workflows.
"""

# Remove direct imports - make them lazy
# from .main_pipeline import KANGRNPipeline
# from .config import (...)


# Lazy import implementation
def __getattr__(name):
    """Implement lazy imports to avoid dependency issues during installation."""

    if name == "KANGRNPipeline":
        from .main_pipeline import KANGRNPipeline

        return KANGRNPipeline

    elif name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig

    elif name == "ModelConfig":
        from .config import ModelConfig

        return ModelConfig

    elif name == "TrainingConfig":
        from .config import TrainingConfig

        return TrainingConfig

    elif name == "NetworkConfig":
        from .config import NetworkConfig

        return NetworkConfig

    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "KANGRNPipeline",
    "PipelineConfig",
    "ModelConfig",
    "TrainingConfig",
    "NetworkConfig",
]
