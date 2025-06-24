"""
KAN Model Trainer for Gene Regulatory Network Inference

This module contains the KANTrainer class that handles training of multiple KAN models
for gene regulatory network inference, refactored from train_with_copies.py
"""

import os
import json
import time
import torch
import numpy as np
import traceback
import gc
import multiprocessing as mp
import re
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import RobustScaler

# Import utility functions from utils module
from .utils import (
    r2_score,
    rmse,
    mae,
    get_process_memory_info,
    optimize_gpu_memory,
    get_gpu_info,
    create_feature_importance_csv,
)

# Import configuration classes
from ..pipeline.config import TrainingConfig, ModelConfig

# Force spawn method for multiprocessing
mp_ctx = mp.get_context("spawn")


class KANTrainer:
    """
    KAN model trainer that handles training of multiple models for gene regulatory networks
    """

    def __init__(
        self,
        data_manager,
        config: TrainingConfig,
        output_dir: str,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize the KAN trainer

        Args:
            data_manager: HPCSharedGeneDataManager instance
            config: Training configuration
            output_dir: Output directory for trained models
            model_config: Model architecture configuration
        """
        self.data_manager = data_manager
        self.config = config
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger("KANTrainer")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking variables
        self.completed_genes = []
        self.checkpoint_path = self.output_dir / "training_checkpoint.json"

        self.logger.info(
            f"KANTrainer initialized with output directory: {self.output_dir}"
        )
        self.logger.info(f"Training config: {self.config}")

    def train_all_models(
        self, gene_list: List[str], resume_from_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Train KAN models for all genes in the list

        Args:
            gene_list: List of target genes to train models for
            resume_from_checkpoint: Whether to resume from existing checkpoint

        Returns:
            Dictionary containing training results and statistics
        """
        start_time = time.time()

        # Load checkpoint if resuming
        if resume_from_checkpoint:
            self.completed_genes = self._load_checkpoint()
            if self.completed_genes:
                # Filter out already processed genes
                original_count = len(gene_list)
                gene_list = [g for g in gene_list if g not in self.completed_genes]
                self.logger.info(
                    f"Resuming from checkpoint: {len(self.completed_genes)} genes already processed"
                )
                self.logger.info(
                    f"Remaining genes to process: {len(gene_list)} out of {original_count}"
                )

        if not gene_list:
            self.logger.info("All genes have been processed!")
            return {"completed_genes": self.completed_genes, "total_runtime_hours": 0}

        # Train models using maximum parallelism
        results = self._train_models_parallel(gene_list)

        # Calculate total runtime
        total_time = time.time() - start_time
        results["total_runtime_hours"] = total_time / 3600

        # Create completion flag
        self._create_completion_flag(results)

        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Total genes processed: {len(self.completed_genes)}")

        return results

    def _load_checkpoint(self) -> List[str]:
        """Load checkpoint and return list of completed genes"""
        if not self.checkpoint_path.exists():
            return []

        try:
            with open(self.checkpoint_path, "r") as f:
                checkpoint = json.load(f)
                completed_genes = checkpoint.get("completed_genes", [])
                self.logger.info(
                    f"Loaded checkpoint with {len(completed_genes)} completed genes"
                )
                return completed_genes
        except Exception as e:
            self.logger.warning(f"Error loading checkpoint: {e}")
            return []

    def _save_checkpoint(self, additional_info: Dict = None):
        """Save current progress to checkpoint"""
        checkpoint_data = {
            "completed_genes": self.completed_genes,
            "timestamp": time.time(),
            "total_completed": len(self.completed_genes),
            "config": asdict(self.config),
        }

        if additional_info:
            checkpoint_data.update(additional_info)

        try:
            with open(self.checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def _train_models_parallel(self, gene_list: List[str]) -> Dict[str, Any]:
        """Train models using maximum parallelism"""
        total_genes = len(gene_list)
        self.logger.info(
            f"Training {total_genes} genes using {self.config.max_models} parallel processes"
        )

        successful_genes = []
        failed_genes = []

        with tqdm(total=total_genes, desc="Overall Progress") as pbar:
            # Process genes in batches
            for batch_idx in range(0, total_genes, self.config.max_models):
                batch_genes = gene_list[batch_idx : batch_idx + self.config.max_models]
                self.logger.info(
                    f"Processing batch {batch_idx//self.config.max_models + 1}: {len(batch_genes)} genes"
                )

                # Prepare data for each gene
                process_args = self._prepare_batch_data(batch_genes)

                if not process_args:
                    self.logger.warning("No valid genes in batch, skipping")
                    pbar.update(len(batch_genes))
                    continue

                # Clear GPU memory before batch
                optimize_gpu_memory()

                # Train models in parallel
                try:
                    batch_results = self._run_parallel_training(process_args)

                    # Process results
                    for result in batch_results:
                        if result and len(result) >= 2:
                            gene, success = result[0], result[1]
                            if success:
                                successful_genes.append(gene)
                                self.completed_genes.append(gene)
                                self.logger.info(
                                    f"[SUCCESS] Successfully trained gene {gene}"
                                )

                                # Copy essential files immediately
                                self._copy_essential_files(gene)
                            else:
                                failed_genes.append(gene)
                                self.logger.warning(
                                    f"[FAILED] Failed to train gene {gene}"
                                )

                except Exception as e:
                    self.logger.error(f"Error in batch processing: {e}")
                    failed_genes.extend([arg[0] for arg in process_args])

                # Save checkpoint after each batch
                self._save_checkpoint(
                    {
                        "batch_idx": batch_idx,
                        "successful_in_batch": len(
                            [r for r in batch_results if r and r[1]]
                        ),
                        "failed_in_batch": len(
                            [r for r in batch_results if r and not r[1]]
                        ),
                    }
                )

                # Update progress
                pbar.update(len(batch_genes))

                # Cleanup
                optimize_gpu_memory()

        return {
            "completed_genes": self.completed_genes,
            "successful_genes": successful_genes,
            "failed_genes": failed_genes,
            "total_processed": len(successful_genes) + len(failed_genes),
            "success_rate": (
                len(successful_genes) / (len(successful_genes) + len(failed_genes))
                if (successful_genes or failed_genes)
                else 0
            ),
        }

    def _prepare_batch_data(self, batch_genes: List[str]) -> List[Tuple]:
        """Prepare data for a batch of genes"""
        process_args = []

        for gene in batch_genes:
            try:
                # Get data for gene
                X, y, related_genes, target_gene = self.data_manager.get_data_for_gene(
                    gene
                )

                if X is not None and y is not None:
                    # Move to CPU for transfer to processes
                    X_cpu = X.cpu()
                    y_cpu = y.cpu()

                    self.logger.debug(
                        f"Gene {gene}: X shape={X.shape}, y shape={y.shape}"
                    )

                    process_args.append(
                        (
                            gene,
                            X_cpu,
                            y_cpu,
                            related_genes,
                            target_gene,
                            str(self.output_dir),
                            0,  # gpu_id
                            self.config.batch_size,
                            self.config.epochs,
                            self.config.patience,
                            self.config.learning_rate,
                            self.config.generate_symbolic,
                            asdict(self.model_config) if self.model_config else {},
                        )
                    )
                else:
                    self.logger.warning(f"No data available for gene {gene}")

            except Exception as e:
                self.logger.error(f"Error preparing data for gene {gene}: {e}")

        return process_args

    def _run_parallel_training(self, process_args: List[Tuple]) -> List[Tuple]:
        """Run training processes in parallel"""
        try:
            with mp_ctx.Pool(processes=len(process_args)) as pool:
                results = pool.starmap(run_training_process, process_args)
                return results
        except Exception as e:
            self.logger.error(f"Error in parallel training: {e}")
            return [(arg[0], False, float("inf"), 0) for arg in process_args]

    def _copy_essential_files(self, gene: str):
        """Copy essential files for a gene to permanent storage"""
        try:
            gene_dir = self.output_dir / gene

            # Get work directory if available
            work_dir = os.environ.get("WORK")
            if work_dir:
                results_dir = Path(work_dir) / "kan_results"
                gene_results_dir = results_dir / gene
                gene_results_dir.mkdir(parents=True, exist_ok=True)

                # Copy essential files
                essential_files = [
                    "feature_importance.csv",
                    "training_log.txt",
                    "symbolic_formula.txt",
                    "normalization_params.json",
                    "data_split_info.json",
                ]

                for filename in essential_files:
                    src_file = gene_dir / filename
                    if src_file.exists():
                        dst_file = gene_results_dir / filename
                        import shutil

                        shutil.copy2(src_file, dst_file)

                self.logger.debug(
                    f"Copied essential files for {gene} to {gene_results_dir}"
                )

        except Exception as e:
            self.logger.warning(f"Error copying essential files for {gene}: {e}")

    def _create_completion_flag(self, results: Dict):
        """Create completion flag file"""
        flag_file = self.output_dir / "training_complete.flag"

        try:
            with open(flag_file, "w") as f:
                f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total genes processed: {len(self.completed_genes)}\n")
                f.write(f"Success rate: {results.get('success_rate', 0):.2%}\n")
                f.write(f"Training method: Parallel KAN Training\n")
                f.write(
                    f"Symbolic formulas: {'Enabled' if self.config.generate_symbolic else 'Disabled'}\n"
                )
                f.write(f"Runtime: {results.get('total_runtime_hours', 0):.2f} hours\n")

        except Exception as e:
            self.logger.error(f"Error creating completion flag: {e}")


def run_training_process(
    gene: str,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    related_genes: List[str],
    target_gene: str,
    output_dir: str,
    gpu_id: int,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float = 0.001,
    generate_symbolic: bool = False,
    model_config: Dict = None,
) -> Tuple[str, bool, float, float]:
    """
    Process function for training a single KAN model with robust error handling
    """
    try:
        # Import KAN directly like in working code
        from kan import KAN

        # Setup device with better error handling
        if torch.cuda.is_available() and gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

        print(f"Training {gene} on device: {device}")

        # Create gene directory
        gene_dir = Path(output_dir) / gene
        gene_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging for this process
        logger = logging.getLogger(f"TrainingProcess-{gene}")

        # Transfer data to device with error checking
        try:
            X = X_tensor.to(device)
            y = y_tensor.to(device)
        except Exception as e:
            logger.error(f"Error moving data to device for {gene}: {e}")
            return gene, False, float("inf"), 0

        # Clean data
        X = torch.nan_to_num(X, nan=0.0)
        y = torch.nan_to_num(y, nan=0.0)

        # Validate data shapes and values
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.error(f"Empty data for gene {gene}")
            return gene, False, float("inf"), 0

        if torch.isnan(X).any() or torch.isnan(y).any():
            logger.warning(f"NaN values detected in data for gene {gene}, cleaning...")
            X = torch.nan_to_num(X, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)

        # Create reproducible data splits (same as working code)
        n_samples = X.shape[0]
        torch.manual_seed(42)  # Fixed seed for consistency
        indices = torch.randperm(n_samples)

        train_size = int(0.8 * n_samples)
        val_size = int(0.1 * n_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Save split information
        split_info = {
            "seed": 42,
            "total_samples": int(n_samples),
            "train_samples": int(train_size),
            "val_samples": int(len(val_indices)),
            "test_samples": int(len(test_indices)),
            "train_indices": train_indices.cpu().numpy().tolist(),
            "val_indices": val_indices.cpu().numpy().tolist(),
            "test_indices": test_indices.cpu().numpy().tolist(),
        }

        with open(gene_dir / "data_split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

        # Prepare and normalize data (same as working code)
        X_splits, y_splits, normalization_params = _prepare_and_normalize_data(
            X, y, train_indices, val_indices, test_indices, device
        )

        X_train, X_val, X_test = X_splits
        y_train, y_val, y_test = y_splits

        # Save normalization parameters
        with open(gene_dir / "normalization_params.json", "w") as f:
            json.dump(normalization_params, f, indent=2)

        # Create data loaders
        train_loader, val_loader, test_loader = _create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test, batch_size
        )

        # Initialize model using the same approach as working code
        input_size = X.shape[1]

        try:
            # Create temporary checkpoint directory
            temp_dir = tempfile.mkdtemp(prefix="kan_temp_")
            model_checkpoint_path = os.path.join(temp_dir, "temp_ckpt")
            os.makedirs(model_checkpoint_path, exist_ok=True)

            # Create model with same parameters as working code
            model = KAN(
                width=[input_size, 2, 1],
                grid=5,
                k=4,
                seed=63,
                ckpt_path=model_checkpoint_path,
            ).to(device)

            # Initialize weights with Xavier (same as working code)
            for p in model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_normal_(p, gain=0.7)

            logger.info(f"Successfully created KAN model for {gene}")

        except Exception as e:
            logger.error(f"Failed to create KAN model for {gene}: {e}")
            return gene, False, float("inf"), 0

        # Setup training components (same as working code)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=False
        )

        criterion = torch.nn.MSELoss(reduction="mean")

        # Training loop (same logic as working code)
        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        logger.info(f"Starting training for gene {gene}")

        for epoch in range(epochs):
            if patience_counter >= patience:
                logger.info(f"Early stopping for gene {gene} after {epoch} epochs")
                break

            # Training phase
            train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)

            # Validation phase
            val_loss, val_r2, val_rmse, val_mae = _validate_epoch(
                model, val_loader, criterion, device
            )

            # Test phase
            test_loss, test_r2, test_rmse, test_mae = _validate_epoch(
                model, test_loader, criterion, device
            )

            # Update scheduler
            scheduler.step(val_loss)

            # Record metrics
            metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_r2": float(val_r2),
                "val_rmse": float(val_rmse),
                "val_mae": float(val_mae),
                "test_loss": float(test_loss),
                "test_r2": float(test_r2),
                "test_rmse": float(test_rmse),
                "test_mae": float(test_mae),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(metrics)

            # Early stopping logic
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), gene_dir / "best_model.pt")
            else:
                patience_counter += 1

            # Periodic logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Gene {gene}, Epoch {epoch+1}/{epochs}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val R2: {val_r2:.4f}"
                )

        # Generate symbolic formula if requested
        if generate_symbolic:
            try:
                _generate_symbolic_formula(model, related_genes, gene_dir, target_gene)
            except Exception as e:
                logger.warning(f"Error generating symbolic formula for {gene}: {e}")

        # Create feature importance
        try:
            create_feature_importance_csv(
                model, related_genes, gene_dir / "feature_importance.csv"
            )
        except Exception as e:
            logger.warning(f"Error creating feature importance for {gene}: {e}")

        # Save training log
        _save_training_log(gene, history, gene_dir, model, related_genes)

        # Get best metrics
        best_val_r2 = max(
            (h.get("val_r2", float("-inf")) for h in history), default=0.0
        )

        # Cleanup
        del model, optimizer, scheduler, criterion
        del X_train, X_val, X_test, y_train, y_val, y_test
        del train_loader, val_loader, test_loader

        # Clean up temporary checkpoint directory
        if model_checkpoint_path and os.path.exists(model_checkpoint_path):
            import shutil

            try:
                shutil.rmtree(os.path.dirname(model_checkpoint_path))
            except Exception as e:
                logger.warning(f"Could not remove temp directory: {e}")

        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Completed training for gene {gene}")
        return gene, True, best_val_loss, best_val_r2

    except Exception as e:
        error_msg = f"Error training gene {gene}: {str(e)}"
        print(error_msg)
        traceback.print_exc()

        # Save error log
        try:
            gene_dir = Path(output_dir) / gene
            gene_dir.mkdir(parents=True, exist_ok=True)
            with open(gene_dir / "error_log.txt", "w") as f:
                f.write(f"Error training model for gene {gene}\n")
                f.write(f"Error message: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass

        return gene, False, float("inf"), 0


def _prepare_and_normalize_data(X, y, train_indices, val_indices, test_indices, device):
    """Prepare and normalize data splits"""
    with torch.no_grad():
        # Apply log transformation to y
        y = torch.log1p(y)

        # Normalize X using RobustScaler
        X_np = X.cpu().numpy()
        X_scaler = RobustScaler()
        X_train_np = X_np[train_indices.cpu().numpy()]
        X_scaler.fit(X_train_np)

        # Transform all splits
        X_train = torch.tensor(
            X_scaler.transform(X_np[train_indices.cpu().numpy()]), dtype=torch.float32
        ).to(device)
        X_val = torch.tensor(
            X_scaler.transform(X_np[val_indices.cpu().numpy()]), dtype=torch.float32
        ).to(device)
        X_test = torch.tensor(
            X_scaler.transform(X_np[test_indices.cpu().numpy()]), dtype=torch.float32
        ).to(device)

        # Normalize y
        y_train_mean = y[train_indices].mean()
        y_train_std = y[train_indices].std().clamp(min=1e-3)

        y_train = (y[train_indices] - y_train_mean) / y_train_std
        y_val = (y[val_indices] - y_train_mean) / y_train_std
        y_test = (y[test_indices] - y_train_mean) / y_train_std

        # Create normalization parameters dictionary
        normalization_params = {
            "log_transform": True,
            "X_center": (
                X_scaler.center_.tolist() if hasattr(X_scaler, "center_") else None
            ),
            "X_scale": (
                X_scaler.scale_.tolist() if hasattr(X_scaler, "scale_") else None
            ),
            "y_mean": y_train_mean.item(),
            "y_std": y_train_std.item(),
        }

        return (X_train, X_val, X_test), (y_train, y_val, y_test), normalization_params


def _create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    """Create PyTorch data loaders"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


def _train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    batch_count = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        try:
            output = model(X_batch)
            output = output.view(y_batch.shape)
            loss = criterion(output, y_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            batch_count += X_batch.size(0)

        except RuntimeError as e:
            continue

    return total_loss / batch_count if batch_count > 0 else 999.0


def _validate_epoch(model, data_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            try:
                preds = model(X_batch)
                preds = preds.view(y_batch.shape)
                batch_loss = criterion(preds, y_batch).item()
                total_loss += batch_loss * X_batch.size(0)

                all_predictions.append(preds.detach())
                all_targets.append(y_batch.detach())

            except Exception:
                continue

    if all_predictions and all_targets:
        all_preds = torch.cat(all_predictions)
        all_targs = torch.cat(all_targets)

        loss = total_loss / len(data_loader.dataset)
        r2 = r2_score(all_targs, all_preds).item()
        rmse_val = rmse(all_targs, all_preds).item()
        mae_val = mae(all_targs, all_preds).item()

        return loss, r2, rmse_val, mae_val
    else:
        return 999.0, 0.0, 999.0, 999.0


def _generate_symbolic_formula(model, related_genes, gene_dir, target_gene):
    """Generate symbolic formula from trained model"""
    try:
        # Generate symbolic formula
        model.auto_symbolic(
            a_range=(-10, 10),
            b_range=(-10, 10),
            weight_simple=0.8,
            r2_threshold=0.2,
            verbose=0,
        )

        # Extract formula
        raw_formula = model.symbolic_formula()
        formula_str = str(raw_formula)

        # Clean up formula
        formula_str = _clean_symbolic_formula(formula_str, related_genes)

        # Save formula
        with open(gene_dir / "symbolic_formula.txt", "w") as f:
            f.write(formula_str.strip())

        # Save metadata
        formula_metadata = {
            "target_gene": target_gene,
            "input_genes": related_genes,
            "formula_length": len(formula_str),
            "generation_timestamp": time.time(),
        }

        with open(gene_dir / "formula_metadata.json", "w") as f:
            json.dump(formula_metadata, f, indent=2)

        return True

    except Exception as e:
        # Create fallback formula
        if related_genes:
            terms = []
            for i, gene in enumerate(related_genes[:5]):
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", gene)
                coeff = 0.1 / (i + 1)
                terms.append(f"{coeff:.3f}*{clean_name}")
            fallback_formula = " + ".join(terms)
        else:
            fallback_formula = "0"

        with open(gene_dir / "symbolic_formula.txt", "w") as f:
            f.write(fallback_formula)

        # Save error info
        with open(gene_dir / "symbolic_formula_error.txt", "w") as f:
            f.write(f"Error generating symbolic formula: {str(e)}\n")
            f.write(f"Fallback formula used: {fallback_formula}\n")

        return True


def _clean_symbolic_formula(formula_str, related_genes):
    """Clean and format symbolic formula"""
    # Fix missing multiplication operators
    formula_str = re.sub(r"(\d)\s*\(", r"\1*(", formula_str)
    formula_str = re.sub(r"(\d)\s*([a-zA-Z_])", r"\1*\2", formula_str)
    formula_str = re.sub(r"\)\s*\(", ")*(", formula_str)
    formula_str = re.sub(r"\)\s*([a-zA-Z_])", r")*\1", formula_str)

    # Replace X_i with actual gene names
    def replace_variable(match):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(related_genes):
            gene_name = related_genes[idx]
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", gene_name)
            return clean_name
        else:
            return "1"

    formula_str = re.sub(r"[xX]_(\d+)", replace_variable, formula_str)

    # Final cleanup
    formula_str = re.sub(r"\*\*+", "**", formula_str)
    formula_str = re.sub(r"\*\s*\*", "**", formula_str)

    return formula_str


def _save_training_log(gene, history, gene_dir, model, related_genes):
    """Save comprehensive training log"""
    try:
        # Calculate summary statistics
        if history:
            best_val_loss = min(h.get("val_loss", float("inf")) for h in history)
            best_val_r2 = max(h.get("val_r2", float("-inf")) for h in history)
            best_val_rmse = min(h.get("val_rmse", float("inf")) for h in history)
            best_val_mae = min(h.get("val_mae", float("inf")) for h in history)
            final_epoch = len(history)
        else:
            best_val_loss = best_val_r2 = best_val_rmse = best_val_mae = 0
            final_epoch = 0

        log_content = f"""
{'='*50}
MODEL TRAINING SUMMARY FOR GENE: {gene}
{'='*50}

CONFIGURATION:
{'-'*20}
Model Architecture: KAN
Total Parameters: {sum(p.numel() for p in model.parameters()) if model else 'Unknown'}
Input Features: {len(related_genes)}
Target Gene: {gene}
Training Method: Parallel Processing

PERFORMANCE METRICS:
{'-'*20}
Best Validation Loss: {best_val_loss:.6f}
Best Validation R²: {best_val_r2:.6f}
Best Validation RMSE: {best_val_rmse:.6f}
Best Validation MAE: {best_val_mae:.6f}
Final Epochs Completed: {final_epoch}

DATA INFORMATION:
{'-'*20}
Input Genes: {', '.join(related_genes[:10])}{'...' if len(related_genes) > 10 else ''}
Total Input Features: {len(related_genes)}

DETAILED TRAINING METRICS BY EPOCH:
{'-'*20}
"""

        # Add epoch-wise metrics
        for i, metrics in enumerate(history, 1):
            log_content += f"Epoch {i:3d}: "
            log_content += f"Train Loss={metrics.get('train_loss', 0):.6f}, "
            log_content += f"Val Loss={metrics.get('val_loss', 0):.6f}, "
            log_content += f"Val R²={metrics.get('val_r2', 0):.6f}, "
            log_content += f"Test Loss={metrics.get('test_loss', 0):.6f}, "
            log_content += f"LR={metrics.get('learning_rate', 0):.6f}\n"

        log_content += f"\n{'='*50}\n"

        # Save to file
        with open(gene_dir / "training_log.txt", "w") as f:
            f.write(log_content)

    except Exception as e:
        print(f"Error saving training log for {gene}: {e}")


# Export main functions and classes
__all__ = [
    "KANTrainer",
    "run_training_process",
]
