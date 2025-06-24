# ============================================================================
# kan_grn/core/__init__.py
# ============================================================================

"""
Core functionality for KAN-GRN package.

This module contains the core algorithmic components including data management,
model training, network building, and utility functions.
"""

from .data_manager import HPCSharedGeneDataManager
from .trainer import KANTrainer
from .network_builder import GeneRegulatoryNetworkBuilder, build_network_from_models

# Import commonly used utility functions
from .utils import (
    r2_score,
    rmse,
    mae,
    get_process_memory_info,
    get_gpu_info,
    optimize_gpu_memory,
    create_feature_importance_csv,
)

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
