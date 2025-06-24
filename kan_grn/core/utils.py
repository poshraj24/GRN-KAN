"""
Utility functions for KAN-GRN package.

This module contains utility functions for metrics, memory management,
and feature importance calculations.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List
from pathlib import Path


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate R2 (coefficient of determination) score"""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)

    if ss_tot < 1e-8:
        return torch.tensor(0.0, device=y_true.device)

    return 1 - ss_res / ss_tot


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate Root Mean Squared Error"""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Absolute Error"""
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    return torch.mean(torch.abs(y_true - y_pred))


def get_process_memory_info() -> Dict[str, float]:
    """Get current memory usage for the current process in GB"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "ram_used_gb": mem_info.rss / (1024**3),
            "virtual_memory_gb": mem_info.vms / (1024**3),
        }
    except Exception as e:
        logging.warning(f"Error getting process memory info: {e}")
        return {"ram_used_gb": 0, "virtual_memory_gb": 0}


def get_gpu_memory_usage_nvidia_smi(gpu_id: int = 0) -> float:
    """Get actual GPU memory usage using nvidia-smi"""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return float(result.stdout.strip()) / 1024  # Convert MB to GB

    except Exception as e:
        logging.debug(f"Error getting GPU memory via nvidia-smi: {e}")

    return 0.0


def optimize_gpu_memory():
    """Optimize GPU memory by clearing unused memory"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc

        gc.collect()
    except Exception as e:
        logging.warning(f"Error optimizing GPU memory: {e}")


def get_gpu_info(device) -> Dict[str, float]:
    """Get GPU memory information for the specified device"""
    if not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_gb": 0,
            "gpu_memory_reserved_gb": 0,
            "gpu_memory_total_gb": 0,
        }

    try:
        if isinstance(device, str):
            device = torch.device(device)

        device_id = device.index if device.index is not None else 0
        torch.cuda.synchronize(device_id)

        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9

        total = 0
        try:
            free, total_bytes = torch.cuda.mem_get_info(device_id)
            total = total_bytes / 1e9
        except:
            total = -1

        return {
            "gpu_memory_allocated_gb": allocated,
            "gpu_memory_reserved_gb": reserved,
            "gpu_memory_total_gb": total,
            "gpu_memory_free_gb": (total - reserved) if total > 0 else 0,
        }

    except Exception as e:
        logging.warning(f"Error getting GPU info: {e}")
        return {
            "gpu_memory_allocated_gb": 0,
            "gpu_memory_reserved_gb": 0,
            "gpu_memory_total_gb": 0,
            "gpu_memory_free_gb": 0,
        }


def create_feature_importance_csv(model, gene_names: List[str], output_path):
    """Create a CSV file with feature importance scores"""
    try:
        import pandas as pd

        # Set model to evaluation mode
        model.eval()

        # Get feature importance scores
        if hasattr(model, "feature_score"):
            feature_scores = model.feature_score.cpu().detach().numpy()
        elif hasattr(model, "get_feature_importance"):
            feature_scores = model.get_feature_importance()
        else:
            # Fallback: use input layer weights as proxy for importance
            with torch.no_grad():
                if hasattr(model, "kan_model"):
                    # For KANWrapper
                    if (
                        hasattr(model.kan_model, "layers")
                        and len(model.kan_model.layers) > 0
                    ):
                        first_layer = (
                            model.kan_model.layers[0]
                            if hasattr(model.kan_model, "layers")
                            else model.kan_model[0]
                        )
                        if hasattr(first_layer, "weight"):
                            weights = first_layer.weight.cpu().numpy()
                            feature_scores = np.abs(weights).mean(axis=0)
                        else:
                            feature_scores = np.ones(len(gene_names))
                    else:
                        feature_scores = np.ones(len(gene_names))
                elif hasattr(model, "layers") and len(model.layers) > 0:
                    # Direct model access
                    first_layer = model.layers[0]
                    if hasattr(first_layer, "weight"):
                        weights = first_layer.weight.cpu().numpy()
                        feature_scores = np.abs(weights).mean(axis=0)
                    else:
                        feature_scores = np.ones(len(gene_names))
                else:
                    feature_scores = np.ones(len(gene_names))

        # Ensure we have the right number of scores
        if len(feature_scores) != len(gene_names):
            min_len = min(len(feature_scores), len(gene_names))
            feature_scores = feature_scores[:min_len]
            gene_names = gene_names[:min_len]

        # Create DataFrame
        importance_df = pd.DataFrame({"Gene": gene_names, "Importance": feature_scores})

        # Sort by importance (descending)
        importance_df = importance_df.sort_values("Importance", ascending=False)

        # Save to CSV
        importance_df.to_csv(output_path, index=False)

        logging.info(f"Feature importance saved to {output_path}")
        return importance_df.set_index("Gene")["Importance"].to_dict()

    except Exception as e:
        logging.error(f"Error creating feature importance CSV: {e}")
        # Create empty CSV as fallback
        try:
            import pandas as pd

            pd.DataFrame(
                {"Gene": gene_names, "Importance": [0] * len(gene_names)}
            ).to_csv(output_path, index=False)
        except:
            pass
        return {}


__all__ = [
    "r2_score",
    "rmse",
    "mae",
    "get_process_memory_info",
    "get_gpu_memory_usage_nvidia_smi",
    "optimize_gpu_memory",
    "get_gpu_info",
    "create_feature_importance_csv",
]
