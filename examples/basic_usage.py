#!/usr/bin/env python3
"""
Basic usage example for KAN-GRN package
"""

from kan_grn import KANGRNPipeline


def main():
    # Simple pipeline setup
    pipeline = KANGRNPipeline.from_files(
        expression_file="path/to/expression_data.h5ad",
        network_file="path/to/network_file.tsv",
        output_dir="results",
        n_top_genes=1000,  # Use top 1000 HVGs
        max_genes=50,  # Process only 50 genes for testing
    )

    # Run complete pipeline
    results = pipeline.run_complete_pipeline()

    print(f"Pipeline completed!")
    print(f"Network file: {results['network_results']['network_file']}")
    print(f"Total relationships: {results['network_results']['total_relationships']}")


if __name__ == "__main__":
    main()
