from pathlib import Path
from data_loader import OptimizedGeneDataProcessor
import warnings
import os
import json
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", message="meta NOT subset.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="Importing read_umi_tools from `anndata` is deprecated",
    category=FutureWarning,
)

if __name__ == "__main__":
    # Initialize and load data
    processor = OptimizedGeneDataProcessor()
    processor.load_data(Path("Data/expression_data1.h5ad"), Path("Data/net_grn.tsv"))

    # Process data and save to disk
    data_dir = "processed_gene_data"
    output_dir = "kan_models"

    # Check if data is already processed, if not process it
    if not os.path.exists(os.path.join(data_dir, "gene_manifest.json")):
        print("Processing gene data and saving to disk...")
        results = processor.process_in_batches(batch_size=100, output_dir=data_dir)
    else:
        print(f"Found processed data in {data_dir}")

    # Load gene list for training
    with open(os.path.join(data_dir, "gene_manifest.json"), "r") as f:
        genes_list = list(json.load(f).keys())

    print(f"Found {len(genes_list)} genes to process")

    # Import training function only when needed to avoid circular imports
    from train import train_kan_models_parallel

    # Train models in parallel
    train_kan_models_parallel(
        gene_list=genes_list,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=32,
        max_models=4,
        epochs=5,
        patience=5,
        lr=0.001,
    )
