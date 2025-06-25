# kan_grn/core/data_manager.py
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.sparse import csc_matrix
from collections import defaultdict
import os
import gc
import time
import warnings

# Suppress the specific FutureWarning from anndata
warnings.filterwarnings(
    "ignore",
    message="Importing read_csv from `anndata` is deprecated.*",
    category=FutureWarning,
)


class HPCSharedGeneDataManager:
    """
    Manages gene expression data as a shared GPU object.
    Keeps data in GPU memory for shared access by multiple models
    Filters to top 2000 highly variable genes
    """

    def __init__(self, device="cuda", scratch_dir=None, n_top_genes=2000):
        """
        Initialize the shared data manager

        Args:
            device: The device to store data on
            scratch_dir: HPC scratch directory for temporary files
            n_top_genes: Number of top highly variable genes to keep (default: 2000)
        """
        # Use SCRATCH directory if provided, otherwise use TMPDIR if available
        self.scratch_dir = scratch_dir
        if self.scratch_dir is None and "TMPDIR" in os.environ:
            self.scratch_dir = os.path.join(
                os.environ["TMPDIR"], f"shared_data_{time.time()}"
            )
            os.makedirs(self.scratch_dir, exist_ok=True)
            print(f"Using node-local storage for temporary files: {self.scratch_dir}")

        # Set device
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.is_initialized = False
        self.n_top_genes = n_top_genes

        # Shared data structures
        self.expr_tensor = None  # The full expression matrix as a torch tensor
        self.gene_names = None  # Array of gene names (filtered to HVGs)
        self.sample_names = None  # Array of sample names
        self.hvg_genes = None  # Set of highly variable gene names

        # Mapping dictionaries
        self.gene_to_idx = {}  # Maps gene names to indices (for filtered genes)
        self.gene_network = defaultdict(list)  # Maps target genes to related genes

        # Gene data dictionary
        self.gene_data_views = {}  # Stores views for each gene

        # Memory usage tracking
        self.memory_usage = {
            "total_allocated": 0,
            "expr_matrix_size": 0,
            "num_genes": 0,
            "num_samples": 0,
        }

        print(f"HPCSharedGeneDataManager initialized on {self.device}")
        print(f"Will filter to top {self.n_top_genes} highly variable genes")

    def _identify_highly_variable_genes(self, adata) -> List[str]:
        """
        Identify top highly variable genes based on variance

        Args:
            adata: AnnData object with expression data

        Returns:
            List of gene names for top highly variable genes
        """
        print(f"Identifying top {self.n_top_genes} highly variable genes...")

        # Memory-efficient variance calculation for sparse matrices
        if hasattr(adata.X, "toarray"):
            # For sparse matrices, calculate variance without converting entire matrix to dense
            print("Computing variance efficiently for sparse matrix...")
            n_genes = adata.X.shape[1]
            gene_variances = np.zeros(n_genes)

            # Convert to CSC format for efficient column access
            if not hasattr(adata.X, "getcol"):
                X_csc = adata.X.tocsc()
            else:
                X_csc = adata.X

            # Calculate variance for each gene individually to save memory
            for i in range(n_genes):
                if i % 1000 == 0:
                    print(f"Processing gene {i+1}/{n_genes}")

                # Get expression values for gene i
                gene_expr = X_csc[:, i].toarray().flatten()
                gene_variances[i] = np.var(gene_expr)

                # Clear memory
                del gene_expr
        else:
            # For dense matrices
            print("Computing variance for dense matrix...")
            gene_variances = np.var(adata.X, axis=0)

        # Get indices of top variable genes
        top_indices = np.argsort(gene_variances)[-self.n_top_genes :]

        # Get gene names
        hvg_genes = [adata.var_names[i] for i in top_indices]

        print(f"Selected {len(hvg_genes)} highly variable genes")
        print(
            f"Variance range: {gene_variances[top_indices].min():.4f} - {gene_variances[top_indices].max():.4f}"
        )

        # Clean up
        del gene_variances
        gc.collect()

        return hvg_genes

    def load_data(self, expression_file: Path, network_file: Path) -> None:
        """
        Load data into shared GPU memory, filtering to top highly variable genes
        Args:
            expression_file: Path to h5ad expression data file
            network_file: Path to network TSV file with regulator and regulated columns
        """
        print(f"Loading expression data into shared memory on {self.device}...")
        start_time = time.time()

        # Load expression data
        adata = sc.read_h5ad(expression_file)
        print(f"Original data shape: {adata.shape}")

        # Identify highly variable genes
        hvg_genes = self._identify_highly_variable_genes(adata)
        self.hvg_genes = set(hvg_genes)

        # Filter AnnData to only include HVGs
        hvg_mask = [gene in self.hvg_genes for gene in adata.var_names]
        adata_filtered = adata[:, hvg_mask].copy()
        print(f"Filtered data shape: {adata_filtered.shape}")

        # Convert sparse matrix to dense if needed - but only after filtering
        if hasattr(adata_filtered.X, "tocsc"):
            # Using CSC format for efficient column slicing later
            print("Converting filtered sparse matrix to dense...")
            expr_matrix = adata_filtered.X.tocsc().toarray()
            print(f"Converted sparse expression matrix to dense: {expr_matrix.shape}")
        else:
            expr_matrix = adata_filtered.X
            print(f"Using dense expression matrix: {expr_matrix.shape}")

        # Store gene and sample names (filtered)
        self.gene_names = np.array(adata_filtered.var_names.tolist())
        self.sample_names = np.array(adata_filtered.obs_names.tolist())
        self.memory_usage["num_genes"] = len(self.gene_names)
        self.memory_usage["num_samples"] = len(self.sample_names)

        # Create mapping dictionary for filtered genes
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}

        # Transfer entire matrix to GPU as a single tensor
        self.expr_tensor = torch.tensor(
            expr_matrix, dtype=torch.float32, device=self.device
        )

        print(f"Expression data loaded in {time.time() - start_time:.2f} seconds")
        print(
            f"Matrix shape: {self.expr_tensor.shape}, Features: {len(self.gene_names)}"
        )

        print("Loading network data...")
        network_start = time.time()

        # Load network data
        network_df = pd.read_csv(network_file, sep="\t", header=None)
        # Column 0: regulator, Column 1: regulated

        total_connections = len(network_df)
        print(f"Original network connections: {total_connections}")

        # Build network from genes that are in our filtered gene set
        filtered_network = defaultdict(list)
        valid_connections = 0

        for _, row in network_df.iterrows():
            regulator = row[0]
            regulated = row[1]

            # Only keep connections where both genes are in our filtered gene set
            if regulator in self.gene_to_idx and regulated in self.gene_to_idx:
                filtered_network[regulated].append(regulator)
                valid_connections += 1

        # Set the filtered network
        self.gene_network = filtered_network

        # Count unique source genes in the filtered network
        unique_source_genes = set()
        for target, sources in self.gene_network.items():
            unique_source_genes.update(sources)

        print(f"Network data loaded in {time.time() - network_start:.2f} seconds")
        print(f"Valid connections: {valid_connections}/{total_connections}")
        print(f"Target genes in network: {len(self.gene_network)}")
        print(f"Unique source genes in network: {len(unique_source_genes)}")

        self.is_initialized = True
        if self.device.type == "cuda":
            cuda_total_memory = torch.cuda.get_device_properties(
                self.device
            ).total_memory
            print(f"Total GPU memory: {cuda_total_memory / 1e9:.2f} GB")
            print(
                f"Current GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
            )

    def get_data_for_gene(
        self, target_gene: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[str], str]:
        """
        Get data for a specific target gene, creating GPU tensors on demand.

        Args:
            target_gene: Name of the target gene

        Returns:
            Tuple of (input tensor, target tensor, related genes list, target gene)
        """
        if not self.is_initialized:
            raise RuntimeError("Data manager not initialized. Call load_data() first.")

        if target_gene in self.gene_data_views:
            cached_data = self.gene_data_views[target_gene]
            # Return cached data with target gene name
            return cached_data[0], cached_data[1], cached_data[2], target_gene

        # Check if target gene is in our HVG set
        if target_gene not in self.hvg_genes:
            print(f"Warning: {target_gene} is not in the highly variable genes set")
            return None, None, [], target_gene

        # Get related genes for this target
        related_genes = self.gene_network.get(target_gene, [])
        if not related_genes:
            print(f"Warning: No related genes found for {target_gene}")
            return None, None, [], target_gene

        # Get indices for this target and its related genes
        target_idx = self.gene_to_idx[target_gene]
        related_indices = [self.gene_to_idx[gene] for gene in related_genes]

        try:
            # Create a view of the shared tensor
            X = self.expr_tensor[
                :, related_indices
            ]  # All samples, related genes features
            y = self.expr_tensor[
                :, target_idx
            ].squeeze()  # All samples, target gene expression

            # Store in cache to avoid recreating views (only store the 3 main values)
            self.gene_data_views[target_gene] = (X, y, related_genes)

            return X, y, related_genes, target_gene

        except Exception as e:
            print(f"Error creating data for gene {target_gene}: {e}")
            return None, None, [], target_gene

    def prefetch_gene_batch(self, gene_batch: List[str]) -> None:
        """
        Prefetch data for a batch of genes to GPU memory.

        Args:
            gene_batch: List of genes to prefetch
        """
        print(f"Prefetching data for {len(gene_batch)} genes...")
        for gene in gene_batch:
            if gene not in self.gene_data_views and gene in self.gene_network:
                # Just call get_data_for_gene which handles the loading
                self.get_data_for_gene(gene)

        if self.device.type == "cuda":
            print(
                f"Current GPU memory usage after prefetch: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
            )

    def get_model_config(self, target_gene: str) -> Dict:
        """Get model configuration for a gene."""
        X, y, _, _ = self.get_data_for_gene(target_gene)
        if X is None:
            return {}

        # Return configuration based on input dimensions
        return {
            "width": [X.shape[1], 2, 1],  # Input size -> hidden -> output
            "grid": 5,
            "k": 4,
            "seed": 63,
        }

    def evict_gene_data(self, genes_to_evict: List[str]) -> None:
        """
        Remove specific genes from GPU memory to free space.

        Args:
            genes_to_evict: List of genes to remove from cache
        """
        for gene in genes_to_evict:
            if gene in self.gene_data_views:
                del self.gene_data_views[gene]

        # Force garbage collection
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print(
                f"Current GPU memory usage after eviction: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB"
            )

    def get_all_target_genes(self) -> List[str]:
        """Returns list of all valid target genes in the network (filtered to HVGs)."""
        return list(self.gene_network.keys())

    def get_hvg_genes(self) -> List[str]:
        """Returns list of all highly variable genes."""
        return list(self.hvg_genes) if self.hvg_genes else []

    def cleanup(self):
        """Release GPU memory and clean up temporary files."""
        print("Cleaning up shared data manager resources...")
        self.expr_tensor = None
        self.gene_data_views = {}
        self.hvg_genes = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up scratch directory
        if self.scratch_dir and os.path.exists(self.scratch_dir):
            try:
                import shutil

                shutil.rmtree(self.scratch_dir)
                print(f"Removed temporary directory: {self.scratch_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory: {e}")

        self.is_initialized = False
        print("Shared data manager cleanup complete")
