#!/usr/bin/env python3
"""
Advanced configuration example for KAN-GRN package
"""

from kan_grn import (
    KANGRNPipeline,
    PipelineConfig,
    ModelConfig,
    TrainingConfig,
    NetworkConfig,
)


def main():
    # Create detailed configuration
    config = PipelineConfig(
        expression_file="data/expression_data.h5ad",
        network_file="data/network_file.tsv",
        output_dir="advanced_results",
        n_top_genes=2000,
        device="cuda",
        model_config=ModelConfig(
            width=[None, 3, 1],  # Larger hidden layer
            grid=7,  # Higher resolution
            k=5,  # Higher polynomial degree
        ),
        training_config=TrainingConfig(
            batch_size=256,
            epochs=100,
            patience=10,
            learning_rate=0.001,
            max_models=8,  # More parallel models
            generate_symbolic=True,
        ),
        network_config=NetworkConfig(
            filter_method="zscore",
            zscore_threshold=1.5,  # More lenient threshold
            min_connections=2,  # Genes must have at least 2 connections
        ),
    )

    # Save configuration for reproducibility
    config.save_to_file("advanced_config.json")

    # Run pipeline
    pipeline = KANGRNPipeline(config)
    results = pipeline.run_complete_pipeline()

    print("Advanced pipeline completed!")


if __name__ == "__main__":
    main()
