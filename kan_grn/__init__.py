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

# Import main classes and functions for easy access
try:
    from .pipeline.main_pipeline import KANGRNPipeline
    from .pipeline.config import (
        PipelineConfig,
        ModelConfig,
        TrainingConfig,
        NetworkConfig,
    )
    from .core.data_manager import HPCSharedGeneDataManager
    from .core.network_builder import GeneRegulatoryNetworkBuilder
    from .core.trainer import KANTrainer

    # If imports succeed, define __all__
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

except ImportError as e:
    # Handle import errors gracefully during package installation
    import warnings

    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)

    # Import classes individually to find the specific problem
    try:
        from .pipeline.main_pipeline import KANGRNPipeline
    except ImportError as e2:
        print(f"Failed to import KANGRNPipeline: {e2}")
        KANGRNPipeline = None

    try:
        from .pipeline.config import (
            PipelineConfig,
            ModelConfig,
            TrainingConfig,
            NetworkConfig,
        )
    except ImportError as e2:
        print(f"Failed to import config classes: {e2}")
        PipelineConfig = ModelConfig = TrainingConfig = NetworkConfig = None

    try:
        from .core.data_manager import HPCSharedGeneDataManager
    except ImportError as e2:
        print(f"Failed to import HPCSharedGeneDataManager: {e2}")
        HPCSharedGeneDataManager = None

    try:
        from .core.trainer import KANTrainer
    except ImportError as e2:
        print(f"Failed to import KANTrainer: {e2}")
        KANTrainer = None

    # Minimal __all__ for failed imports
    __all__ = ["__version__", "__author__", "__email__"]
