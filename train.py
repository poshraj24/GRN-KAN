import os
import json
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import traceback
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
from kan import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dataset class for gene expression data
class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Metrics calculation functions
def r2_score(y_true, y_pred):
    total_sum_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)
    residual_sum_squares = torch.sum((y_true - y_pred) ** 2)
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    # Check if total_sum_squares is effectively zero
    if total_sum_squares < epsilon:
        # If there's no variance in the target, R² is not meaningful
        # By convention, return 0 or some other indicator
        return torch.tensor(0.0, device=y_true.device)

    return 1 - (residual_sum_squares / (total_sum_squares + epsilon))


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


# Data loader class that uses only what's needed from disk
class SimpleGeneLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_gene_data(self, gene):
        gene_path = os.path.join(self.data_dir, "gene_data", gene)

        X = np.load(os.path.join(gene_path, "X.npy"))
        y = np.load(os.path.join(gene_path, "y.npy"))

        with open(os.path.join(gene_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        return X, y, metadata["config"], metadata["related_genes"]


# Prepare data using the 80/10/10 split strategy from kan_model.py
def prepare_data(X, y, batch_size=32, seed=42):
    """
    Prepare train, validation, and test data loaders with 80/10/10 split
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Split indices using random permutation (matching kan_model.py)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    val_size = int(0.1 * len(X))

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    # Create datasets and data loaders
    train_loader = DataLoader(
        GeneDataset(X[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(GeneDataset(X[val_idx], y[val_idx]), batch_size=batch_size)
    test_loader = DataLoader(
        GeneDataset(X[test_idx], y[test_idx]), batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


# Helper functions for saving model visualizations and formulas
def log_symbolic_formula(formula_tuple, log_file, gene_names):
    """
    Log symbolic formula to training log file with actual gene names instead of X_1, X_2, etc.

    Args:
        formula_tuple: The symbolic formula from model.symbolic_formula()
        log_file: Path to the log file
        gene_names: List of gene names corresponding to input features
    """
    with open(log_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write("Symbolic Formula with Gene Names:\n")
        f.write("-" * 20 + "\n")

        # Convert tuple elements to string
        formula_str = "\n".join(str(item) for item in formula_tuple)

        # Replace X_i patterns with actual gene names
        # This regex matches X_1, X_2, etc. (case insensitive)
        import re

        replaced_formula = formula_str
        pattern = re.compile(r"[xX]_(\d+)")

        def replace_with_gene(match):
            idx = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= idx < len(gene_names):
                return f"{gene_names[idx]}"
            else:
                return match.group(0)  # Keep original if index out of range

        replaced_formula = pattern.sub(replace_with_gene, formula_str)

        f.write(replaced_formula + "\n")
        f.write("=" * 50 + "\n\n")

        # Also save the mapping for reference
        f.write("Variable Mapping:\n")
        for i, gene in enumerate(gene_names):
            if i < len(gene_names):
                f.write(f"X_{i+1} = {gene}\n")
        f.write("=" * 50 + "\n\n")


def save_plot(model, filename, dpi=3600):
    """Save the KAN architecture plot to a file."""
    try:
        model.plot(beta=10)
        plt.savefig(filename, bbox_inches="tight", dpi=dpi)
        plt.close()
        print(f"KAN architecture saved to {filename}")
    except Exception as e:
        print(f"Error saving KAN plot: {str(e)}")


def visualize_kan_node_importance(model, gene_names, filename):
    """
    Visualize KAN's feature importance scores with gene names in their original node order.
    Args:
        model: A trained KAN model
        gene_names: List of gene names corresponding to input features
        filename: Path to save the visualization
    """
    try:
        # Set model to evaluation mode
        model.eval()

        # Get feature importance scores directly from the model
        feature_scores = model.feature_score.cpu().detach().numpy()

        # Map scores to gene names (maintaining original order)
        gene_importance = {}
        for i, gene in enumerate(gene_names):
            if i < len(feature_scores):
                gene_importance[gene] = float(feature_scores[i])

        # Normalize scores
        total = sum(abs(v) for v in gene_importance.values())
        if total > 0:
            gene_importance = {k: abs(v) / total for k, v in gene_importance.items()}

        # Create visualization
        plt.figure(figsize=(12, 8))
        genes = list(gene_importance.keys())
        scores = list(gene_importance.values())

        # Limit to top 30 genes if there are too many
        if len(genes) > 30:
            sorted_idx = np.argsort(scores)[-30:]
            genes = [genes[i] for i in sorted_idx]
            scores = [scores[i] for i in sorted_idx]

        plt.bar(genes, scores)
        plt.xticks(rotation=90, ha="right")
        plt.title("Gene Importance Scores from KAN (Original Node Order)")
        plt.ylabel("Normalized Importance")
        plt.xlabel("Genes")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        # Also save as CSV
        csv_filename = filename.replace(".png", ".csv")
        with open(csv_filename, "w") as f:
            f.write("Gene,Importance\n")
            for gene, score in gene_importance.items():
                f.write(f"{gene},{score}\n")

        print(f"Feature importance saved to {filename} and {csv_filename}")

        return gene_importance
    except Exception as e:
        print(f"Error creating feature importance plot: {str(e)}")
        return {}


# Worker function for training a single model in a separate process
def train_single_model(
    gene,
    data_dir,
    output_dir,
    gpu_id,
    shared_dict,
    batch_size=32,
    epochs=50,
    patience=10,
    min_delta=1e-4,
    lr=0.001,
):
    """
    Train a single KAN model in a separate process following kan_model.py strategy

    Args:
        gene: Gene to train model for
        data_dir: Directory containing processed gene data
        output_dir: Directory to save trained model
        gpu_id: GPU device ID to use
        shared_dict: Multiprocessing shared dictionary for results
        batch_size: Batch size for training
        epochs: Number of epochs for training
        patience: Early stopping patience
        min_delta: Minimum change in validation loss to be considered improvement
        lr: Learning rate
    """
    try:
        # Set the device for this process
        torch.cuda.set_device(gpu_id)
        process_device = torch.device(f"cuda:{gpu_id}")

        # Create output directory for this gene
        gene_dir = os.path.join(output_dir, gene)
        os.makedirs(gene_dir, exist_ok=True)

        # Create log file for this gene
        log_file = os.path.join(gene_dir, "training_log.txt")

        def log(message):
            with open(log_file, "a") as f:
                f.write(f"{message}\n")
            print(message)

        log(f"Starting training for gene {gene} on GPU:{gpu_id}")

        # Load gene data
        gene_loader = SimpleGeneLoader(data_dir)
        X, y, config, related_genes = gene_loader.load_gene_data(gene)

        # Save related genes info
        with open(os.path.join(gene_dir, "related_genes.json"), "w") as f:
            json.dump(related_genes, f, indent=2)

        # Prepare data using the 80/10/10 split strategy
        train_loader, val_loader, test_loader = prepare_data(
            X, y, batch_size=batch_size
        )

        # Create KAN model
        model = KAN(
            width=config["width"],
            grid=config["grid"],
            k=config["k"],
            seed=config["seed"],
        ).to(process_device)

        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # Training variables
        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        # Start training
        start_time = time.time()

        # Create progress bar for epochs
        pbar = tqdm(range(epochs), desc=f"Training {gene}", position=gpu_id, leave=True)

        for epoch in pbar:
            # Training phase
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(process_device), y_batch.to(
                    process_device
                )

                # Forward pass
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation phase
            model.eval()
            val_loss, val_r2, val_rmse, val_mae = 0.0, 0.0, 0.0, 0.0
            test_loss, test_r2, test_rmse, test_mae = 0.0, 0.0, 0.0, 0.0

            with torch.no_grad():
                # Validation metrics
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(process_device), y_batch.to(
                        process_device
                    )
                    predictions = model(X_batch).squeeze()

                    val_loss += criterion(predictions, y_batch).item() * X_batch.size(0)
                    val_r2 += r2_score(y_batch, predictions).item()
                    val_rmse += rmse(y_batch, predictions).item()
                    val_mae += mae(y_batch, predictions).item()

                # Test metrics (matching kan_model.py)
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(process_device), y_batch.to(
                        process_device
                    )
                    predictions = model(X_batch).squeeze()

                    test_loss += criterion(predictions, y_batch).item() * X_batch.size(
                        0
                    )
                    test_r2 += r2_score(y_batch, predictions).item()
                    test_rmse += rmse(y_batch, predictions).item()
                    test_mae += mae(y_batch, predictions).item()

            # Normalize metrics
            val_loss /= len(val_loader.dataset)
            val_r2 /= len(val_loader)
            val_rmse /= len(val_loader)
            val_mae /= len(val_loader)

            test_loss /= len(test_loader.dataset)
            test_r2 /= len(test_loader)
            test_rmse /= len(test_loader)
            test_mae /= len(test_loader)

            # Store metrics with all metrics matching kan_model.py
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "val_r2": val_r2,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "time": time.time() - start_time,
            }
            history.append(epoch_metrics)

            # Update progress bar with metrics
            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_r2": f"{val_r2:.4f}",
                    "val_rmse": f"{val_rmse:.4f}",
                }
            )

            # Early stopping check with min_delta (matching kan_model.py)
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(gene_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                log(f"Saved new best model with val_loss: {val_loss:.5f}")
            else:
                patience_counter += 1
                log(f"No improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                log(f"Early stopping at epoch {epoch+1}")
                # Load the best model (matching kan_model.py)
                model.load_state_dict(torch.load(best_model_path))
                break

        # Save final model if early stopping didn't trigger
        if patience_counter < patience:
            torch.save(model.state_dict(), os.path.join(gene_dir, "final_model.pt"))

        # Save training history
        with open(os.path.join(gene_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Save model metadata with all KAN model metrics
        total_params = sum(p.numel() for p in model.parameters())
        max_memory = torch.cuda.max_memory_allocated(device=process_device) / 1e9

        metadata = {
            "gene": gene,
            "num_parameters": total_params,
            "max_gpu_memory_gb": max_memory,
            "best_val_loss": best_val_loss,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "best_val_r2": (
                max(epoch["val_r2"] for epoch in history) if history else None
            ),
            "best_val_rmse": (
                min(epoch["val_rmse"] for epoch in history) if history else None
            ),
            "best_val_mae": (
                min(epoch["val_mae"] for epoch in history) if history else None
            ),
            "total_epochs": len(history),
            "early_stopped": patience_counter >= patience,
            "training_time": time.time() - start_time,
            "config": config,
        }

        with open(os.path.join(gene_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Format detailed training log like in kan_model.py
        log_message = "=" * 50 + "\n"
        log_message += f"Configuration: width={config['width']}, grid={config['grid']}, k={config['k']}\n"
        log_message += f"Total Parameters: {total_params}\n"
        log_message += f"Maximum Memory Consumed: {max_memory:.2f} GB\n"
        log_message += f"Final Validation Loss: {best_val_loss}\n"
        log_message += "Epoch-wise Training Statistics:\n"

        for i, metrics in enumerate(history, 1):
            log_message += (
                f"Epoch {i}: Train Loss = {metrics['train_loss']:.6f}, "
                f"Val Loss = {metrics['val_loss']:.6f}, "
                f"Test Loss = {metrics['test_loss']:.6f}, "
                f"Val R2 = {metrics['val_r2']:.6f}, "
                f"Val RMSE = {metrics['val_rmse']:.6f}, "
                f"Val MAE = {metrics['val_mae']:.6f}\n"
            )

        log_message += "=" * 50 + "\n"
        log(log_message)

        # 1. Generate symbolic formula
        log("Generating symbolic formula...")
        try:

            model.auto_symbolic(
                a_range=(-20, 20),  # Wider parameter search range
                b_range=(-20, 20),
                weight_simple=0.6,  # Balance between simplicity and accuracy
                r2_threshold=0.3,  # Only use edges with reasonable fit
                verbose=2,  # More debugging output
            )
            symbolic_formula = model.symbolic_formula()
            log_symbolic_formula(symbolic_formula, log_file, related_genes)

            # Save formula to a separate file for easier access
            formula_file = os.path.join(gene_dir, "symbolic_formula.txt")
            with open(formula_file, "w") as f:
                formula_str = "\n".join(str(item) for item in symbolic_formula)
                # Replace X_i patterns with actual gene names
                import re

                pattern = re.compile(r"[xX]_(\d+)")

                def replace_with_gene(match):
                    idx = int(match.group(1)) - 1  # Convert to 0-based index
                    if 0 <= idx < len(related_genes):
                        return f"{related_genes[idx]}"
                    else:
                        return match.group(0)

                replaced_formula = pattern.sub(replace_with_gene, formula_str)
                f.write(replaced_formula)
        except Exception as e:
            log(f"Error generating symbolic formula: {str(e)}")

        # 2. Save model architecture plot
        log("Saving model architecture plot...")
        try:
            plot_path = os.path.join(gene_dir, "model_plot.png")
            save_plot(model, plot_path)
        except Exception as e:
            log(f"Error saving model plot: {str(e)}")

        # 3. Save feature importance visualization
        log("Saving feature importance visualization...")
        try:
            importance_path = os.path.join(gene_dir, "feature_importance.png")
            visualize_kan_node_importance(model, related_genes, importance_path)
        except Exception as e:
            log(f"Error saving feature importance: {str(e)}")

        # Update shared results dictionary
        shared_dict[gene] = {
            "gene": gene,
            "status": "success",
            "best_val_loss": best_val_loss,
            "best_val_r2": (
                max(epoch["val_r2"] for epoch in history) if history else None
            ),
            "best_val_rmse": (
                min(epoch["val_rmse"] for epoch in history) if history else None
            ),
            "best_val_mae": (
                min(epoch["val_mae"] for epoch in history) if history else None
            ),
            "training_time": time.time() - start_time,
        }

        log(f"Completed training for gene {gene}")
        pbar.close()

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error training model for gene {gene}: {str(e)}")

        # Create the directory if it doesn't exist yet
        gene_dir = os.path.join(output_dir, gene)
        os.makedirs(gene_dir, exist_ok=True)

        # Log the error
        with open(os.path.join(gene_dir, "error_log.txt"), "w") as f:
            f.write(f"Error training model for gene {gene}: {str(e)}\n")
            f.write(f"Traceback:\n{error_details}\n")

        # Update shared results dictionary
        shared_dict[gene] = {"gene": gene, "status": "error", "error": str(e)}

    finally:
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()


def train_kan_models_parallel(
    gene_list,
    data_dir,
    output_dir,
    batch_size=32,
    max_models=4,
    epochs=50,
    patience=10,
    min_delta=1e-4,
    lr=0.001,
):
    """
    Train multiple KAN models in parallel using multiprocessing

    Args:
        gene_list: List of genes to process
        data_dir: Directory where processed gene data is stored
        output_dir: Directory to save trained models
        batch_size: Batch size for training
        max_models: Maximum number of models to train simultaneously
        epochs: Number of epochs for training
        patience: Early stopping patience
        min_delta: Minimum change in validation loss to be considered improvement
        lr: Learning rate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Start multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    if num_gpus == 0:
        print("No GPUs available. Using CPU instead.")
        device = torch.device("cpu")
        max_models = 1  # Limit to 1 model at a time on CPU

    # Calculate GPU memory
    if num_gpus > 0:
        gpu_memory = []
        for i in range(num_gpus):
            # Set current device before querying memory
            torch.cuda.set_device(i)
            # Get free and total memory directly
            free, total = torch.cuda.mem_get_info()
            free_gb = free / 1e9
            total_gb = total / 1e9

            print(
                f"GPU {i}: Total Memory: {total_gb:.2f} GB, "
                f"Free Memory: {free_gb:.2f} GB"
            )
            gpu_memory.append(total_gb)

    # Progress tracking
    total_genes = len(gene_list)
    processed_genes = 0

    # Create a shared dictionary for results
    manager = mp.Manager()
    shared_results = manager.dict()

    # Process genes in batches
    batch_idx = 0

    with tqdm(total=total_genes, desc="Overall Progress") as pbar:
        while processed_genes < total_genes:
            # Determine batch size based on available GPUs and max_models
            current_batch_size = min(max_models, total_genes - processed_genes)

            # Get genes for this batch
            batch_genes = gene_list[
                processed_genes : processed_genes + current_batch_size
            ]

            print(
                f"\nProcessing batch {batch_idx + 1}: genes {processed_genes + 1} to {processed_genes + len(batch_genes)} of {total_genes}"
            )

            # Create processes for each gene in the batch
            processes = []

            for i, gene in enumerate(batch_genes):
                # Assign gene to a GPU (round-robin assignment)
                gpu_id = i % num_gpus if num_gpus > 0 else 0

                # Create and start process
                p = mp.Process(
                    target=train_single_model,
                    args=(
                        gene,
                        data_dir,
                        output_dir,
                        gpu_id,
                        shared_results,
                        batch_size,
                        epochs,
                        patience,
                        min_delta,
                        lr,
                    ),
                )
                processes.append(p)
                p.start()
                print(f"Started process for gene {gene} on GPU {gpu_id}")

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Update progress
            processed_genes += len(batch_genes)
            pbar.update(len(batch_genes))
            batch_idx += 1

    # Convert shared dictionary to regular dictionary
    results = dict(shared_results)

    # Save overall results with extended metrics
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        summary = {
            "total_genes": len(gene_list),
            "successful_genes": sum(
                1 for r in results.values() if r.get("status") == "success"
            ),
            "failed_genes": sum(
                1 for r in results.values() if r.get("status") != "success"
            ),
            "median_val_loss": np.median(
                [
                    r.get("best_val_loss", np.nan)
                    for r in results.values()
                    if r.get("status") == "success"
                ]
            ),
            "median_val_r2": np.median(
                [
                    r.get("best_val_r2", np.nan)
                    for r in results.values()
                    if r.get("status") == "success"
                ]
            ),
            "median_val_rmse": np.median(
                [
                    r.get("best_val_rmse", np.nan)
                    for r in results.values()
                    if r.get("status") == "success"
                ]
            ),
            "median_val_mae": np.median(
                [
                    r.get("best_val_mae", np.nan)
                    for r in results.values()
                    if r.get("status") == "success"
                ]
            ),
            "results": results,
        }
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\nTraining Summary:")
    print(f"Total genes: {len(gene_list)}")
    print(
        f"Successfully trained: {sum(1 for r in results.values() if r.get('status') == 'success')}"
    )
    print(f"Failed: {sum(1 for r in results.values() if r.get('status') != 'success')}")

    if sum(1 for r in results.values() if r.get("status") == "success") > 0:
        print(f"Median validation R2: {summary['median_val_r2']:.4f}")
        print(f"Median validation RMSE: {summary['median_val_rmse']:.4f}")

    return results
