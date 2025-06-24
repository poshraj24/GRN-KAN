"""
Main pipeline orchestrator for KAN-GRN inference
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import asdict

from ..core.data_manager import HPCSharedGeneDataManager
from ..core.trainer import KANTrainer
from ..core.network_builder import GeneRegulatoryNetworkBuilder
from .config import PipelineConfig


class KANGRNPipeline:
    """
    Main pipeline class for KAN-based Gene Regulatory Network inference.

    This class orchestrates the entire process from data loading to network generation.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the KAN-GRN pipeline.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.data_manager = None
        self.trainer = None
        self.network_builder = None

        # Setup logging
        self._setup_logging()

        # Validate configuration
        self._validate_config()

        self.logger.info("KAN-GRN Pipeline initialized")
        self.logger.info(f"Configuration: {self.config}")

    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.output_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.config.output_dir, "pipeline.log")
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("KANGRNPipeline")

    def _validate_config(self):
        """Validate pipeline configuration"""
        if not os.path.exists(self.config.expression_file):
            raise FileNotFoundError(
                f"Expression file not found: {self.config.expression_file}"
            )

        if not os.path.exists(self.config.network_file):
            raise FileNotFoundError(
                f"Network file not found: {self.config.network_file}"
            )

        if self.config.n_top_genes <= 0:
            raise ValueError("n_top_genes must be positive")

        self.logger.info("Configuration validation passed")

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete KAN-GRN inference pipeline.

        Returns:
            Dictionary containing pipeline results and statistics
        """
        start_time = time.time()

        try:
            # Step 1: Initialize data manager and load data
            self.logger.info("Step 1: Loading and preparing data...")
            self._initialize_data_manager()

            # Step 2: Train KAN models
            self.logger.info("Step 2: Training KAN models...")
            training_results = self._train_models()

            # Step 3: Build gene regulatory network
            self.logger.info("Step 3: Building gene regulatory network...")
            network_results = self._build_network()

            # Step 4: Generate final results
            self.logger.info("Step 4: Generating final results...")
            results = self._compile_results(training_results, network_results)

            total_time = time.time() - start_time
            results["total_runtime_hours"] = total_time / 3600

            self.logger.info(
                f"Pipeline completed successfully in {total_time/3600:.2f} hours"
            )

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

        finally:
            # Cleanup
            if self.data_manager:
                self.data_manager.cleanup()

    def _initialize_data_manager(self):
        """Initialize and load data into the data manager"""
        self.data_manager = HPCSharedGeneDataManager(
            device=self.config.device,
            scratch_dir=self.config.scratch_dir,
            n_top_genes=self.config.n_top_genes,
        )

        self.data_manager.load_data(
            Path(self.config.expression_file), Path(self.config.network_file)
        )

        self.logger.info(
            f"Loaded {len(self.data_manager.get_all_target_genes())} target genes"
        )
        self.logger.info(f"Using top {self.config.n_top_genes} highly variable genes")

    def _train_models(self) -> Dict[str, Any]:
        """Train KAN models for all target genes"""
        self.trainer = KANTrainer(
            data_manager=self.data_manager,
            config=self.config.training_config,
            output_dir=self.config.output_dir,
        )

        target_genes = self.data_manager.get_all_target_genes()

        if self.config.max_genes is not None and self.config.max_genes > 0:
            target_genes = target_genes[: self.config.max_genes]
            self.logger.info(f"Limited to {len(target_genes)} genes for training")

        training_results = self.trainer.train_all_models(
            gene_list=target_genes,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )

        return training_results

    def _build_network(self) -> Dict[str, Any]:
        """Build the gene regulatory network from trained models"""
        self.network_builder = GeneRegulatoryNetworkBuilder(
            models_dir=self.config.output_dir, network_config=self.config.network_config
        )

        network_results = self.network_builder.build_network()

        return network_results

    def _compile_results(
        self, training_results: Dict, network_results: Dict
    ) -> Dict[str, Any]:
        """Compile final pipeline results"""
        results = {
            "config": asdict(self.config),
            "training_results": training_results,
            "network_results": network_results,
            "pipeline_version": "1.0.0",
            "timestamp": time.time(),
        }

        # Save results to file
        results_file = os.path.join(self.config.output_dir, "pipeline_results.json")
        import json

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")

        return results

    @classmethod
    def from_files(
        cls,
        expression_file: str,
        network_file: str,
        output_dir: str = "kan_grn_results",
        **kwargs,
    ) -> "KANGRNPipeline":
        """
        Create pipeline from input files with default configuration.

        Args:
            expression_file: Path to gene expression data (h5ad format)
            network_file: Path to network file (TSV format)
            output_dir: Output directory for results
            **kwargs: Additional configuration parameters

        Returns:
            Configured KANGRNPipeline instance
        """
        config = PipelineConfig(
            expression_file=expression_file,
            network_file=network_file,
            output_dir=output_dir,
            **kwargs,
        )

        return cls(config)

    def train_models_only(self) -> Dict[str, Any]:
        """Run only the model training part of the pipeline"""
        self._initialize_data_manager()
        return self._train_models()

    def build_network_only(self) -> Dict[str, Any]:
        """Run only the network building part (requires pre-trained models)"""
        return self._build_network()
