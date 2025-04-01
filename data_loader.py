import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Iterator
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.sparse import csc_matrix
import time
import gc  # For garbage collection
import os


class OptimizedGeneDataProcessor:
    """Processes gene expression data for multiple target genes efficiently."""

    def __init__(self):
        self.gene_data = {}
        self.sample_names = None
        self.related_genes = {}
        self.expr_matrix = None
        self.gene_names = None
        self.gene_name_to_idx = {}
        self.gene_network = defaultdict(list)
        self.valid_targets = []
        self.start_time = None

    def load_data(self, expression_file: Path, network_file: Path):
        """
        Loads expression and network data once for all target genes.

        Args:
            expression_file: Path to h5ad expression data file
            network_file: Path to network TSV file
        """
        self.start_time = time.time()
        print("Loading expression data...")
        adata = sc.read_h5ad(expression_file)
        self.expr_matrix = (
            csc_matrix(adata.X) if not hasattr(adata.X, "tocsc") else adata.X.tocsc()
        )
        self.gene_names = np.array(adata.var_names.tolist())
        self.sample_names = np.array(adata.obs_names.tolist())
        self.gene_name_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}

        print("Loading network data...")
        network_df = pd.read_csv(network_file, sep="\t")
        source_col, target_col = network_df.columns[:2]

        for _, row in network_df.iterrows():
            if (
                row[target_col] in self.gene_name_to_idx
                and row[source_col] in self.gene_name_to_idx
            ):
                self.gene_network[row[target_col]].append(row[source_col])

        self.valid_targets = list(self.gene_network.keys())
        print(f"Found {len(self.valid_targets)} valid target genes in the network.")

    def get_all_target_genes(self) -> List[str]:
        """Returns list of all valid target genes in the network."""
        return self.valid_targets

    def prepare_training_data_for_gene(
        self, target_gene: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
        """
        Prepares training data for a specific target gene.

        Args:
            target_gene: Name of target gene

        Returns:
            Tuple of (input matrix, target values, model config, related genes list)
        """
        related_genes = self.gene_network.get(target_gene, [])
        if not related_genes:
            print(f"Warning: No related genes found for {target_gene}")
            return None, None, None, []

        target_idx = self.gene_name_to_idx[target_gene]
        related_indices = [self.gene_name_to_idx[gene] for gene in related_genes]

        try:
            X = self.expr_matrix[:, related_indices].toarray()
            y = self.expr_matrix[:, target_idx].toarray().flatten()

            config = {"width": [X.shape[1], 2, 1], "grid": 5, "k": 4, "seed": 42}

            return X.astype(np.float32), y.astype(np.float32), config, related_genes
        except MemoryError:
            print(
                f"Memory error processing gene {target_gene} with {len(related_genes)} related genes"
            )
            return None, None, None, []

    def prepare_all_training_data(
        self,
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray, Dict, List[str]]]:
        """
        Generator that yields prepared training data for all target genes.
        """
        for target_gene in tqdm(self.valid_targets, desc="Processing genes"):
            X, y, config, related_genes = self.prepare_training_data_for_gene(
                target_gene
            )
            if X is not None:
                yield target_gene, X, y, config, related_genes

    def process_in_batches(
        self, batch_size: int = 100, output_dir: str = None
    ) -> Dict[str, Dict]:
        """
        Process genes in memory-efficient batches and optionally save to disk.

        Args:
            batch_size: Number of genes to process in each batch
            output_dir: If provided, save data to this directory instead of returning

        Returns:
            Dictionary of processed gene data or info about saved files
        """
        if not self.valid_targets:
            raise ValueError("No data loaded. Call load_data() first.")

        expected_count = len(self.valid_targets)
        processed_count = 0
        error_genes = []
        processed_genes = {}

        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            data_dir = os.path.join(output_dir, "gene_data")
            os.makedirs(data_dir, exist_ok=True)

        self.start_time = time.time()
        print(f"Processing {expected_count} genes in batches of {batch_size}")

        # Process in batches
        for batch_start in range(0, expected_count, batch_size):
            batch_end = min(batch_start + batch_size, expected_count)
            print(
                f"\nBatch {batch_start//batch_size + 1}: Processing genes {batch_start+1}-{batch_end} of {expected_count}"
            )

            # Get the batch of genes
            batch_genes = self.valid_targets[batch_start:batch_end]

            # Process genes in this batch
            batch_data = {}
            for gene in tqdm(batch_genes, desc="Processing batch"):
                try:
                    X, y, config, related_genes = self.prepare_training_data_for_gene(
                        gene
                    )
                    if X is not None:
                        processed_count += 1

                        if output_dir:
                            # Save to disk instead of keeping in memory
                            gene_path = os.path.join(data_dir, f"{gene}")
                            os.makedirs(gene_path, exist_ok=True)

                            # Save arrays and metadata
                            np.save(os.path.join(gene_path, "X.npy"), X)
                            np.save(os.path.join(gene_path, "y.npy"), y)

                            # Save config and related genes as JSON
                            with open(
                                os.path.join(gene_path, "metadata.json"), "w"
                            ) as f:
                                import json

                                json.dump(
                                    {
                                        "config": config,
                                        "related_genes": related_genes,
                                        "X_shape": X.shape,
                                        "y_shape": y.shape,
                                    },
                                    f,
                                )

                            # Just store path info in memory
                            processed_genes[gene] = {
                                "path": gene_path,
                                "X_shape": X.shape,
                                "y_shape": y.shape,
                                "num_related_genes": len(related_genes),
                            }
                        else:
                            # Store in memory
                            batch_data[gene] = {
                                "X": X,
                                "y": y,
                                "config": config,
                                "related_genes": related_genes,
                            }
                except Exception as e:
                    print(f"Error processing gene {gene}: {e}")
                    error_genes.append((gene, f"Processing error: {str(e)}"))

            # Update processed genes dictionary if keeping in memory
            if not output_dir:
                processed_genes.update(batch_data)

            # Print progress
            elapsed = time.time() - self.start_time
            print(f"Progress: {processed_count}/{expected_count} genes processed")
            print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            print(
                f"Estimated remaining time: {(elapsed/processed_count)*(expected_count-processed_count)/60:.1f} minutes"
            )

            # Force garbage collection between batches
            gc.collect()

        # Print final summary
        elapsed_time = time.time() - self.start_time
        summary = {
            "processed_count": processed_count,
            "expected_count": expected_count,
            "error_count": len(error_genes),
            "total_time_seconds": elapsed_time,
            "errors": error_genes,
        }

        print(f"\nProcessing complete")
        print(f"Total genes processed: {processed_count}/{expected_count}")
        print(f"Genes with errors: {len(error_genes)}")
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

        if error_genes:
            print("\nFirst 10 genes with errors:")
            for gene, error in error_genes[:10]:
                print(f"  {gene}: {error}")
            if len(error_genes) > 10:
                print(f"  ... and {len(error_genes) - 10} more")

        if output_dir:
            # Save summary to the output directory
            with open(os.path.join(output_dir, "processing_summary.json"), "w") as f:
                import json

                json.dump(summary, f, indent=2)
            print(f"Data saved to {output_dir}")

            # Create a manifest file for easy loading
            with open(os.path.join(output_dir, "gene_manifest.json"), "w") as f:
                json.dump(processed_genes, f)

        return {"genes": processed_genes, "summary": summary}

    def load_gene_data(self, gene, data_dir):
        """
        Load a single gene's data from disk.

        Args:
            gene: Name of the gene
            data_dir: Directory where data is stored

        Returns:
            Tuple of (X, y, config, related_genes)
        """
        gene_path = os.path.join(data_dir, "gene_data", gene)

        # Load arrays
        X = np.load(os.path.join(gene_path, "X.npy"))
        y = np.load(os.path.join(gene_path, "y.npy"))

        # Load metadata
        with open(os.path.join(gene_path, "metadata.json"), "r") as f:
            import json

            metadata = json.load(f)

        return X, y, metadata["config"], metadata["related_genes"]
