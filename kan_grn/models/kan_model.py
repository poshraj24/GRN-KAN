"""
KAN Model Implementation and Wrapper for Gene Regulatory Network Inference

This module provides a wrapper around the KAN (Kolmogorov-Arnold Network) implementation
and utilities for creating and managing KAN models for gene expression prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import tempfile
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Try to import the KAN library - adjust this import based on your KAN implementation
try:
    from kan import KAN  # Adjust this import path based on your KAN library

    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False
    logging.warning("KAN library not available. Using fallback implementation.")

from ..pipeline.config import ModelConfig


class KANWrapper(nn.Module):
    """
    Wrapper class for KAN models to provide consistent interface
    """

    def __init__(
        self,
        input_size: int,
        width: List[int] = None,
        grid: int = 5,
        k: int = 4,
        seed: int = 63,
        device: str = "cuda",
        ckpt_path: Optional[str] = None,
    ):
        """
        Initialize KAN wrapper

        Args:
            input_size: Number of input features
            width: Architecture width [input, hidden, output]
            grid: Grid parameter for KAN
            k: Spline order for KAN
            seed: Random seed
            device: Device to place model on
            ckpt_path: Checkpoint path for model saving
        """
        super().__init__()

        self.input_size = input_size
        self.width = width or [input_size, 2, 1]
        self.grid = grid
        self.k = k
        self.seed = seed

        # Properly handle device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Create temporary checkpoint path if none provided
        if ckpt_path is None:
            temp_dir = tempfile.mkdtemp(prefix="kan_temp_")
            self.ckpt_path = os.path.join(temp_dir, "temp_ckpt")
            os.makedirs(self.ckpt_path, exist_ok=True)
            self._temp_dir_created = True
        else:
            self.ckpt_path = ckpt_path
            self._temp_dir_created = False

        # Ensure width has correct input size
        if self.width[0] != input_size:
            self.width[0] = input_size

        # Initialize the model
        self._initialize_model()

        # Move to device
        self.to(self.device)

        logging.info(
            f"KAN model initialized with architecture {self.width}, grid={self.grid}, k={self.k}"
        )

    def _initialize_model(self):
        """Initialize the underlying KAN model"""
        if KAN_AVAILABLE:
            try:
                # Set random seed
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)

                # Create KAN model with proper checkpoint path
                self.kan_model = KAN(
                    width=self.width,
                    grid=self.grid,
                    k=self.k,
                    seed=self.seed,
                    ckpt_path=self.ckpt_path,
                )

                self.model_type = "KAN"

            except Exception as e:
                logging.warning(f"Error initializing KAN model: {e}. Using fallback.")
                self._initialize_fallback_model()
        else:
            self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        """Initialize a fallback neural network if KAN is not available"""
        logging.info("Using fallback MLP model instead of KAN")

        layers = []
        for i in range(len(self.width) - 1):
            # Properly handle device and dtype for Linear layers
            linear_layer = nn.Linear(
                self.width[i],
                self.width[i + 1],
                dtype=torch.float32,
                device=self.device,
            )
            layers.append(linear_layer)

            if i < len(self.width) - 2:  # Don't add activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

        self.kan_model = nn.Sequential(*layers)
        self.model_type = "MLP"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.kan_model(x)

    def auto_symbolic(
        self,
        a_range: Tuple[float, float] = (-10, 10),
        b_range: Tuple[float, float] = (-10, 10),
        weight_simple: float = 0.8,
        r2_threshold: float = 0.2,
        verbose: int = 2,
    ):
        """
        Generate symbolic representation of the model

        Args:
            a_range: Range for parameter a
            b_range: Range for parameter b
            weight_simple: Weight for simplicity in symbolic regression
            r2_threshold: R² threshold for symbolic regression
            verbose: Verbosity level
        """
        if hasattr(self.kan_model, "auto_symbolic"):
            try:
                self.kan_model.auto_symbolic(
                    a_range=a_range,
                    b_range=b_range,
                    weight_simple=weight_simple,
                    r2_threshold=r2_threshold,
                    verbose=verbose,
                )
            except Exception as e:
                logging.warning(f"Error in auto_symbolic: {e}")
        else:
            logging.warning("auto_symbolic not available for this model type")

    def symbolic_formula(self) -> Any:
        """
        Get symbolic formula representation

        Returns:
            Symbolic formula (format depends on underlying implementation)
        """
        if hasattr(self.kan_model, "symbolic_formula"):
            try:
                return self.kan_model.symbolic_formula()
            except Exception as e:
                logging.warning(f"Error getting symbolic formula: {e}")
                return self._generate_fallback_formula()
        else:
            return self._generate_fallback_formula()

    def _generate_fallback_formula(self) -> str:
        """Generate a fallback formula for non-KAN models"""
        try:
            # For MLP models, create a simple linear combination formula
            if self.model_type == "MLP" and hasattr(self.kan_model, "0"):
                first_layer = self.kan_model[0]
                if hasattr(first_layer, "weight"):
                    weights = first_layer.weight.data.cpu().numpy()
                    bias = (
                        first_layer.bias.data.cpu().numpy()
                        if first_layer.bias is not None
                        else [0]
                    )

                    # Create simple formula
                    terms = []
                    for i, w in enumerate(weights[0]):  # First output neuron
                        if abs(w) > 1e-3:  # Only include significant weights
                            terms.append(f"{w:.4f}*x_{i+1}")

                    formula = " + ".join(terms)
                    if bias[0] != 0:
                        formula += f" + {bias[0]:.4f}"

                    return formula

            return "Complex nonlinear function"

        except Exception as e:
            logging.warning(f"Error generating fallback formula: {e}")
            return "Unknown function"

    @property
    def feature_score(self) -> torch.Tensor:
        """
        Get feature importance scores

        Returns:
            Feature importance tensor
        """
        if hasattr(self.kan_model, "feature_score"):
            return self.kan_model.feature_score
        else:
            # Fallback: use first layer weights as importance proxy
            return self._compute_feature_importance()

    def _compute_feature_importance(self) -> torch.Tensor:
        """Compute feature importance for non-KAN models"""
        try:
            if self.model_type == "MLP" and hasattr(self.kan_model, "0"):
                first_layer = self.kan_model[0]
                if hasattr(first_layer, "weight"):
                    weights = first_layer.weight.data
                    # Use absolute mean of weights as importance
                    importance = torch.abs(weights).mean(dim=0)
                    return importance

            # Default: equal importance
            return torch.ones(self.input_size, device=self.device)

        except Exception as e:
            logging.warning(f"Error computing feature importance: {e}")
            return torch.ones(self.input_size, device=self.device)

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance as numpy array

        Returns:
            Feature importance array
        """
        return self.feature_score.cpu().detach().numpy()

    def save_model(self, filepath: Union[str, Path]):
        """
        Save model state

        Args:
            filepath: Path to save model
        """
        try:
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "model_config": {
                        "input_size": self.input_size,
                        "width": self.width,
                        "grid": self.grid,
                        "k": self.k,
                        "seed": self.seed,
                        "model_type": self.model_type,
                    },
                    "feature_importance": self.get_feature_importance().tolist(),
                },
                filepath,
            )

            logging.info(f"Model saved to {filepath}")

        except Exception as e:
            logging.error(f"Error saving model: {e}")

    def load_model(self, filepath: Union[str, Path]):
        """
        Load model state

        Args:
            filepath: Path to load model from
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"])

            # Load config if available
            if "model_config" in checkpoint:
                config = checkpoint["model_config"]
                self.model_type = config.get("model_type", "KAN")

            logging.info(f"Model loaded from {filepath}")

        except Exception as e:
            logging.error(f"Error loading model: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.model_type,
            "architecture": self.width,
            "grid": self.grid,
            "k": self.k,
            "input_size": self.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }

    def __del__(self):
        """Cleanup temporary directories"""
        if hasattr(self, "_temp_dir_created") and self._temp_dir_created:
            try:
                import shutil

                if hasattr(self, "ckpt_path") and os.path.exists(self.ckpt_path):
                    shutil.rmtree(os.path.dirname(self.ckpt_path), ignore_errors=True)
            except:
                pass


def create_kan_model(
    input_size: int, config: Optional[Dict[str, Any]] = None, device: str = "cuda"
) -> KANWrapper:
    """
    Factory function to create KAN models

    Args:
        input_size: Number of input features
        config: Model configuration dictionary
        device: Device to place model on

    Returns:
        KANWrapper instance
    """
    if config is None:
        config = {}

    # Extract parameters from config
    width = config.get("width", [input_size, 2, 1])
    grid = config.get("grid", 5)
    k = config.get("k", 4)
    seed = config.get("seed", 63)
    ckpt_path = config.get("ckpt_path", None)

    # Ensure width has correct input size
    if isinstance(width, list) and len(width) > 0:
        width[0] = input_size

    return KANWrapper(
        input_size=input_size,
        width=width,
        grid=grid,
        k=k,
        seed=seed,
        device=device,
        ckpt_path=ckpt_path,
    )


def create_kan_from_config(
    input_size: int, model_config: ModelConfig, device: str = "cuda"
) -> KANWrapper:
    """
    Create KAN model from ModelConfig object

    Args:
        input_size: Number of input features
        model_config: ModelConfig instance
        device: Device to place model on

    Returns:
        KANWrapper instance
    """
    # Convert ModelConfig to dictionary
    config_dict = {
        "width": (
            model_config.width.copy() if model_config.width else [input_size, 2, 1]
        ),
        "grid": model_config.grid,
        "k": model_config.k,
        "seed": model_config.seed,
    }

    # Ensure width has correct input size
    if config_dict["width"][0] is None or config_dict["width"][0] != input_size:
        config_dict["width"][0] = input_size

    return create_kan_model(input_size, config_dict, device)


class KANModelManager:
    """
    Manager class for handling multiple KAN models
    """

    def __init__(self, base_output_dir: Union[str, Path]):
        """
        Initialize model manager

        Args:
            base_output_dir: Base directory for saving models
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.logger = logging.getLogger("KANModelManager")

    def create_model(
        self,
        gene_name: str,
        input_size: int,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ) -> KANWrapper:
        """
        Create and register a new KAN model for a gene

        Args:
            gene_name: Name of the target gene
            input_size: Number of input features
            config: Model configuration
            device: Device to place model on

        Returns:
            KANWrapper instance
        """
        # Create checkpoint directory for this gene
        gene_dir = self.base_output_dir / gene_name
        gene_dir.mkdir(exist_ok=True)
        ckpt_path = str(gene_dir / "temp_ckpt")

        # Update config with checkpoint path
        if config is None:
            config = {}
        config["ckpt_path"] = ckpt_path

        # Create model
        model = create_kan_model(input_size, config, device)

        # Register model
        self.models[gene_name] = {
            "model": model,
            "config": config,
            "gene_dir": gene_dir,
            "input_size": input_size,
        }

        self.logger.info(f"Created model for gene {gene_name}")
        return model

    def get_model(self, gene_name: str) -> Optional[KANWrapper]:
        """
        Get model for a specific gene

        Args:
            gene_name: Name of the target gene

        Returns:
            KANWrapper instance or None if not found
        """
        model_info = self.models.get(gene_name)
        return model_info["model"] if model_info else None

    def save_model(self, gene_name: str, filename: str = "best_model.pt") -> bool:
        """
        Save model for a specific gene

        Args:
            gene_name: Name of the target gene
            filename: Filename for the saved model

        Returns:
            True if successful, False otherwise
        """
        try:
            model_info = self.models.get(gene_name)
            if not model_info:
                self.logger.error(f"Model for gene {gene_name} not found")
                return False

            filepath = model_info["gene_dir"] / filename
            model_info["model"].save_model(filepath)
            return True

        except Exception as e:
            self.logger.error(f"Error saving model for gene {gene_name}: {e}")
            return False

    def load_model(self, gene_name: str, filename: str = "best_model.pt") -> bool:
        """
        Load model for a specific gene

        Args:
            gene_name: Name of the target gene
            filename: Filename of the saved model

        Returns:
            True if successful, False otherwise
        """
        try:
            model_info = self.models.get(gene_name)
            if not model_info:
                self.logger.error(f"Model for gene {gene_name} not registered")
                return False

            filepath = model_info["gene_dir"] / filename
            if not filepath.exists():
                self.logger.error(f"Model file not found: {filepath}")
                return False

            model_info["model"].load_model(filepath)
            return True

        except Exception as e:
            self.logger.error(f"Error loading model for gene {gene_name}: {e}")
            return False

    def cleanup_model(self, gene_name: str):
        """
        Clean up model and temporary files for a gene

        Args:
            gene_name: Name of the target gene
        """
        try:
            if gene_name in self.models:
                model_info = self.models[gene_name]

                # Clean up temporary checkpoint directory
                ckpt_path = Path(model_info["config"].get("ckpt_path", ""))
                if ckpt_path.exists():
                    import shutil

                    shutil.rmtree(ckpt_path, ignore_errors=True)

                # Remove from registry
                del self.models[gene_name]

                self.logger.info(f"Cleaned up model for gene {gene_name}")

        except Exception as e:
            self.logger.warning(f"Error cleaning up model for gene {gene_name}: {e}")

    def get_all_models(self) -> Dict[str, KANWrapper]:
        """
        Get all registered models

        Returns:
            Dictionary mapping gene names to models
        """
        return {name: info["model"] for name, info in self.models.items()}

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all models

        Returns:
            Summary dictionary
        """
        summary = {"total_models": len(self.models), "models": {}}

        for gene_name, model_info in self.models.items():
            model = model_info["model"]
            summary["models"][gene_name] = {
                "input_size": model_info["input_size"],
                "model_info": model.get_model_info(),
            }

        return summary


class KANEnsemble:
    """
    Ensemble of KAN models for improved predictions
    """

    def __init__(self, models: List[KANWrapper]):
        """
        Initialize KAN ensemble

        Args:
            models: List of KANWrapper models
        """
        self.models = models
        self.num_models = len(models)

        if self.num_models == 0:
            raise ValueError("At least one model is required for ensemble")

        # Check that all models have compatible input sizes
        input_sizes = [model.input_size for model in models]
        if len(set(input_sizes)) > 1:
            raise ValueError("All models must have the same input size")

        self.input_size = input_sizes[0]
        self.device = models[0].device

        logging.info(f"KAN ensemble initialized with {self.num_models} models")

    def predict(self, x: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """
        Make ensemble predictions

        Args:
            x: Input tensor
            method: Ensemble method ("mean", "median", "vote")

        Returns:
            Ensemble predictions
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        # Stack predictions
        stacked_preds = torch.stack(
            predictions, dim=0
        )  # [num_models, batch_size, output_size]

        # Apply ensemble method
        if method == "mean":
            return torch.mean(stacked_preds, dim=0)
        elif method == "median":
            return torch.median(stacked_preds, dim=0)[0]
        elif method == "vote":
            # For regression, vote doesn't make much sense, so use mean
            return torch.mean(stacked_preds, dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def get_prediction_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty estimates

        Args:
            x: Input tensor

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        stacked_preds = torch.stack(predictions, dim=0)

        mean_preds = torch.mean(stacked_preds, dim=0)
        std_preds = torch.std(stacked_preds, dim=0)

        return mean_preds, std_preds


def validate_kan_installation() -> Dict[str, Any]:
    """
    Validate KAN installation and return status

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "kan_available": KAN_AVAILABLE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "errors": [],
        "warnings": [],
    }

    # Test basic KAN functionality if available
    if KAN_AVAILABLE:
        try:
            # Test creating a simple KAN model
            test_model = create_kan_model(input_size=5, device="cpu")

            # Test forward pass
            test_input = torch.randn(10, 5)
            output = test_model(test_input)

            if output.shape != (10, 1):
                validation_result["errors"].append(
                    f"Unexpected output shape: {output.shape}"
                )

            validation_result["test_passed"] = True

        except Exception as e:
            validation_result["errors"].append(f"KAN test failed: {str(e)}")
            validation_result["test_passed"] = False
    else:
        validation_result["warnings"].append(
            "KAN library not available, using fallback MLP"
        )
        validation_result["test_passed"] = False

    # Check CUDA if available
    if torch.cuda.is_available():
        try:
            # Test CUDA model creation
            test_model_cuda = create_kan_model(input_size=3, device="cuda")
            test_input_cuda = torch.randn(5, 3, device="cuda")
            output_cuda = test_model_cuda(test_input_cuda)

            validation_result["cuda_test_passed"] = True

        except Exception as e:
            validation_result["errors"].append(f"CUDA test failed: {str(e)}")
            validation_result["cuda_test_passed"] = False
    else:
        validation_result["cuda_test_passed"] = False

    return validation_result


def benchmark_kan_performance(
    input_sizes: List[int] = None, batch_sizes: List[int] = None, device: str = "cuda"
) -> Dict[str, Any]:
    """
    Benchmark KAN model performance

    Args:
        input_sizes: List of input sizes to test
        batch_sizes: List of batch sizes to test
        device: Device to run benchmark on

    Returns:
        Benchmark results
    """
    if input_sizes is None:
        input_sizes = [10, 50, 100, 500]

    if batch_sizes is None:
        batch_sizes = [32, 128, 512]

    results = {"device": device, "results": [], "summary": {}}

    for input_size in input_sizes:
        for batch_size in batch_sizes:
            try:
                # Create model
                model = create_kan_model(input_size=input_size, device=device)

                # Generate test data
                test_data = torch.randn(batch_size, input_size, device=device)

                # Warm up
                for _ in range(5):
                    _ = model(test_data)

                # Benchmark forward pass
                torch.cuda.synchronize() if device.startswith("cuda") else None
                start_time = (
                    torch.cuda.Event(enable_timing=True)
                    if device.startswith("cuda")
                    else None
                )
                end_time = (
                    torch.cuda.Event(enable_timing=True)
                    if device.startswith("cuda")
                    else None
                )

                if device.startswith("cuda"):
                    start_time.record()
                else:
                    import time

                    start_time = time.time()

                # Run multiple forward passes
                for _ in range(100):
                    output = model(test_data)

                if device.startswith("cuda"):
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = (
                        start_time.elapsed_time(end_time) / 1000
                    )  # Convert to seconds
                else:
                    elapsed_time = time.time() - start_time

                # Calculate metrics
                throughput = (100 * batch_size) / elapsed_time  # samples per second
                latency = elapsed_time / 100  # seconds per batch

                result = {
                    "input_size": input_size,
                    "batch_size": batch_size,
                    "elapsed_time": elapsed_time,
                    "throughput": throughput,
                    "latency": latency,
                    "memory_used": (
                        torch.cuda.memory_allocated() / 1e9
                        if device.startswith("cuda")
                        else 0
                    ),
                }

                results["results"].append(result)

                # Cleanup
                del model, test_data, output
                torch.cuda.empty_cache() if device.startswith("cuda") else None

            except Exception as e:
                logging.warning(
                    f"Benchmark failed for input_size={input_size}, batch_size={batch_size}: {e}"
                )

    # Calculate summary statistics
    if results["results"]:
        throughputs = [r["throughput"] for r in results["results"]]
        latencies = [r["latency"] for r in results["results"]]

        results["summary"] = {
            "avg_throughput": np.mean(throughputs),
            "max_throughput": np.max(throughputs),
            "avg_latency": np.mean(latencies),
            "min_latency": np.min(latencies),
        }

    return results


# Export main classes and functions
__all__ = [
    "KANWrapper",
    "KANModelManager",
    "KANEnsemble",
    "create_kan_model",
    "create_kan_from_config",
    "validate_kan_installation",
    "benchmark_kan_performance",
    "KAN_AVAILABLE",
]
