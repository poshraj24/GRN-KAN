# ============================================================================
# kan_grn/__init__.py (Main package __init__.py)
# ============================================================================

"""
KAN-GRN: Gene Regulatory Network inference using Kolmogorov-Arnold Networks

A comprehensive package for inferring gene regulatory networks from expression data
using Kolmogorov-Arnold Networks (KANs) with symbolic formula generation.
"""

__version__ = "1.0.0"
__author__ = "Posh Raj Dahal"
__email__ = "dahal.poshraj24@gmail.com"
__license__ = "GPL-3.0 license"
__copyright__ = "Copyright 2024 Posh Raj Dahal"

# Package metadata
__title__ = "kan-grn"
__description__ = "Gene Regulatory Network inference using Kolmogorov-Arnold Networks"
__url__ = "https://github.com/poshraj24/GRN-KAN"


# Lazy import implementation
def __getattr__(name):
    """Implement lazy imports to avoid dependency issues during installation."""

    if name == "KANGRNPipeline":
        from .pipeline.main_pipeline import KANGRNPipeline

        return KANGRNPipeline

    elif name == "PipelineConfig":
        from .pipeline.config import PipelineConfig

        return PipelineConfig

    elif name == "ModelConfig":
        from .pipeline.config import ModelConfig

        return ModelConfig

    elif name == "TrainingConfig":
        from .pipeline.config import TrainingConfig

        return TrainingConfig

    elif name == "NetworkConfig":
        from .pipeline.config import NetworkConfig

        return NetworkConfig

    elif name == "HPCSharedGeneDataManager":
        from .core.data_manager import HPCSharedGeneDataManager

        return HPCSharedGeneDataManager

    elif name == "GeneRegulatoryNetworkBuilder":
        from .core.network_builder import GeneRegulatoryNetworkBuilder

        return GeneRegulatoryNetworkBuilder

    elif name == "KANTrainer":
        from .core.trainer import KANTrainer

        return KANTrainer

    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Define what's available for import (used by help() and IDE autocomplete)
__all__ = [
    "KANGRNPipeline",
    "PipelineConfig",
    "ModelConfig",
    "TrainingConfig",
    "NetworkConfig",
    "HPCSharedGeneDataManager",
    "GeneRegulatoryNetworkBuilder",
    "KANTrainer",
    "__version__",
    "__author__",
    "__email__",
]
