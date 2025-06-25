"""
Command line interface for KAN-GRN
"""

import argparse
import sys
import json
import os
from pathlib import Path

# Remove top-level imports - make them lazy instead
# from ..pipeline.main_pipeline import KANGRNPipeline
# from ..pipeline.config import PipelineConfig, ModelConfig, TrainingConfig, NetworkConfig


def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="KAN-GRN: Gene Regulatory Network inference using Kolmogorov-Arnold Networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Full pipeline command
    pipeline_parser = subparsers.add_parser("run", help="Run complete pipeline")
    add_pipeline_arguments(pipeline_parser)

    # Train only command
    train_parser = subparsers.add_parser("train", help="Train models only")
    add_training_arguments(train_parser)

    # Build network only command
    network_parser = subparsers.add_parser(
        "build-network", help="Build network from trained models"
    )
    add_network_arguments(network_parser)

    # Config command
    config_parser = subparsers.add_parser(
        "create-config", help="Create configuration file template"
    )
    config_parser.add_argument(
        "--output",
        "-o",
        default="kan_grn_config.json",
        help="Output configuration file",
    )

    return parser


def add_pipeline_arguments(parser):
    """Add arguments for full pipeline"""
    # Required arguments
    parser.add_argument(
        "expression_file", help="Path to gene expression data file (h5ad)"
    )
    parser.add_argument("network_file", help="Path to network file (TSV)")

    # Output settings
    parser.add_argument(
        "--output-dir",
        "-o",
        default="kan_grn_results",
        help="Output directory for results",
    )

    # Data settings
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Number of top highly variable genes to use",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Maximum number of genes to process (for testing)",
    )

    # Model settings
    parser.add_argument("--grid", type=int, default=5, help="KAN grid parameter")
    parser.add_argument("--k", type=int, default=4, help="KAN k parameter")

    # Training settings
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--max-models", type=int, default=6, help="Maximum parallel models"
    )

    # Hardware settings
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--scratch-dir", help="Scratch directory for temporary files")

    # Pipeline settings
    parser.add_argument(
        "--no-resume", action="store_true", help="Do not resume from checkpoint"
    )
    parser.add_argument(
        "--no-symbolic", action="store_true", help="Do not generate symbolic formulas"
    )

    # Network settings
    parser.add_argument(
        "--filter-method",
        choices=["zscore", "importance"],
        default="zscore",
        help="Network filtering method",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for network filtering",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.0,
        help="Importance threshold for network filtering",
    )

    # Configuration file option
    parser.add_argument("--config", help="Path to configuration file (JSON)")


def add_training_arguments(parser):
    """Add arguments for training only"""
    add_pipeline_arguments(parser)  # Reuse pipeline arguments


def add_network_arguments(parser):
    """Add arguments for network building only"""
    parser.add_argument("models_dir", help="Directory containing trained models")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="network_results",
        help="Output directory for network",
    )
    parser.add_argument(
        "--filter-method",
        choices=["zscore", "importance"],
        default="zscore",
        help="Network filtering method",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for network filtering",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.0,
        help="Importance threshold for network filtering",
    )


def run_pipeline(args):
    """Run the complete pipeline"""
    # Lazy import when actually needed
    from ..pipeline.main_pipeline import KANGRNPipeline
    from ..pipeline.config import (
        PipelineConfig,
        ModelConfig,
        TrainingConfig,
        NetworkConfig,
    )

    if args.config:
        # Load from config file
        config = PipelineConfig.load_from_file(args.config)
    else:
        # Create config from arguments
        config = PipelineConfig(
            expression_file=args.expression_file,
            network_file=args.network_file,
            output_dir=args.output_dir,
            n_top_genes=args.n_top_genes,
            max_genes=args.max_genes,
            device=args.device,
            scratch_dir=args.scratch_dir,
            resume_from_checkpoint=not args.no_resume,
            model_config=ModelConfig(grid=args.grid, k=args.k),
            training_config=TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                max_models=args.max_models,
                generate_symbolic=not args.no_symbolic,
            ),
            network_config=NetworkConfig(
                filter_method=args.filter_method,
                zscore_threshold=args.zscore_threshold,
                importance_threshold=args.importance_threshold,
            ),
        )

    # Run pipeline
    pipeline = KANGRNPipeline(config)
    results = pipeline.run_complete_pipeline()

    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {config.output_dir}")
    print(f"Total runtime: {results['total_runtime_hours']:.2f} hours")


def train_models(args):
    """Train models only"""
    # Lazy import when actually needed
    from ..pipeline.main_pipeline import KANGRNPipeline

    # Similar to run_pipeline but call train_models_only
    config = create_config_from_args(args)
    pipeline = KANGRNPipeline(config)
    results = pipeline.train_models_only()
    print("Model training completed successfully!")


def build_network(args):
    """Build network from trained models"""
    # Lazy import when actually needed
    from ..core.network_builder import GeneRegulatoryNetworkBuilder
    from ..pipeline.config import NetworkConfig

    network_config = NetworkConfig(
        filter_method=args.filter_method,
        zscore_threshold=args.zscore_threshold,
        importance_threshold=args.importance_threshold,
    )

    builder = GeneRegulatoryNetworkBuilder(
        models_dir=args.models_dir, network_config=network_config
    )

    results = builder.build_network()
    print(f"Network building completed successfully!")
    print(f"Network saved to: {results['network_file']}")


def create_config(args):
    """Create configuration file template"""
    # Lazy import when actually needed
    from ..pipeline.config import PipelineConfig

    config = PipelineConfig(
        expression_file="path/to/expression_data.h5ad",
        network_file="path/to/network_file.tsv",
        output_dir="kan_grn_results",
    )

    config.save_to_file(args.output)
    print(f"Configuration template created: {args.output}")


def create_config_from_args(args):
    """Helper function to create config from CLI arguments"""
    # Lazy import when actually needed
    from ..pipeline.config import (
        PipelineConfig,
        ModelConfig,
        TrainingConfig,
        NetworkConfig,
    )

    return PipelineConfig(
        expression_file=args.expression_file,
        network_file=args.network_file,
        output_dir=args.output_dir,
        n_top_genes=args.n_top_genes,
        max_genes=getattr(args, "max_genes", None),
        device=getattr(args, "device", "cuda"),
        scratch_dir=getattr(args, "scratch_dir", None),
        resume_from_checkpoint=not getattr(args, "no_resume", False),
        model_config=ModelConfig(
            grid=getattr(args, "grid", 5), k=getattr(args, "k", 4)
        ),
        training_config=TrainingConfig(
            batch_size=getattr(args, "batch_size", 512),
            epochs=getattr(args, "epochs", 50),
            patience=getattr(args, "patience", 5),
            learning_rate=getattr(args, "learning_rate", 0.0001),
            max_models=getattr(args, "max_models", 6),
            generate_symbolic=not getattr(args, "no_symbolic", False),
        ),
        network_config=NetworkConfig(
            filter_method=getattr(args, "filter_method", "zscore"),
            zscore_threshold=getattr(args, "zscore_threshold", 2.0),
            importance_threshold=getattr(args, "importance_threshold", 0.0),
        ),
    )


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "run":
            run_pipeline(args)
        elif args.command == "train":
            train_models(args)
        elif args.command == "build-network":
            build_network(args)
        elif args.command == "create-config":
            create_config(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
