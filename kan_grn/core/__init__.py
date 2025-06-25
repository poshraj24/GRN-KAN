# ============================================================================
# kan_grn/core/__init__.py
# ============================================================================

"""
Core functionality for KAN-GRN package.

This module contains the core algorithmic components including data management,
model training, network building, and utility functions.
"""

# Remove direct imports - make them lazy
# from .data_manager import HPCSharedGeneDataManager
# from .trainer import KANTrainer
# from .network_builder import GeneRegulatoryNetworkBuilder, build_network_from_models
# from .utils import (...)

__author__ = "Posh Raj Dahal"
__email__ = "dahal.poshraj24@gmail.com"
__license__ = "GPL-3.0 license"
__copyright__ = "Copyright 2024 Posh Raj Dahal"

# Package metadata
__title__ = "kan-grn-core"
__description__ = "Core functionality for KAN-GRN"


# Lazy import implementation
def __getattr__(name):
    """Implement lazy imports to avoid dependency issues during installation."""

    if name == "HPCSharedGeneDataManager":
        from .data_manager import HPCSharedGeneDataManager

        return HPCSharedGeneDataManager

    elif name == "KANTrainer":
        from .trainer import KANTrainer

        return KANTrainer

    elif name == "GeneRegulatoryNetworkBuilder":
        from .network_builder import GeneRegulatoryNetworkBuilder

        return GeneRegulatoryNetworkBuilder

    elif name == "build_network_from_models":
        from .network_builder import build_network_from_models

        return build_network_from_models

    # Utility functions
    elif name == "r2_score":
        from .utils import r2_score

        return r2_score

    elif name == "rmse":
        from .utils import rmse

        return rmse

    elif name == "mae":
        from .utils import mae

        return mae

    elif name == "get_process_memory_info":
        from .utils import get_process_memory_info

        return get_process_memory_info

    elif name == "get_gpu_info":
        from .utils import get_gpu_info

        return get_gpu_info

    elif name == "optimize_gpu_memory":
        from .utils import optimize_gpu_memory

        return optimize_gpu_memory

    elif name == "create_feature_importance_csv":
        from .utils import create_feature_importance_csv

        return create_feature_importance_csv

    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Main classes
    "HPCSharedGeneDataManager",
    "KANTrainer",
    "GeneRegulatoryNetworkBuilder",
    # Utility functions
    "build_network_from_models",
    "r2_score",
    "rmse",
    "mae",
    "get_process_memory_info",
    "get_gpu_info",
    "optimize_gpu_memory",
    "create_feature_importance_csv",
]
