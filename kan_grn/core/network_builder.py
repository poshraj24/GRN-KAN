"""
Gene Regulatory Network Builder from trained KAN models
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional

from ..pipeline.config import NetworkConfig


class GeneRegulatoryNetworkBuilder:
    """
    Build gene regulatory network from trained KAN model feature importance scores
    """

    def __init__(self, models_dir: str, network_config: NetworkConfig):
        """
        Initialize network builder

        Args:
            models_dir: Directory containing trained models
            network_config: Network building configuration
        """
        self.models_dir = models_dir
        self.config = network_config
        self.logger = logging.getLogger("NetworkBuilder")

    def build_network(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Build gene regulatory network from feature importance files

        Args:
            output_file: Optional output file path

        Returns:
            Dictionary containing network building results
        """
        if output_file is None:
            if self.config.filter_method == "zscore":
                output_file = os.path.join(
                    self.models_dir,
                    f"gene_regulatory_network_zscore_{self.config.zscore_threshold}.csv",
                )
            elif self.config.filter_method == "importance":
                output_file = os.path.join(
                    self.models_dir,
                    f"gene_regulatory_network_importance_{self.config.importance_threshold}.csv",
                )
            else:  # full network
                output_file = os.path.join(
                    self.models_dir,
                    "gene_regulatory_network_full.csv",
                )

        if self.config.filter_method == "zscore":
            self.logger.info(
                f"Building network using z-score filtering with threshold: {self.config.zscore_threshold}"
            )
        elif self.config.filter_method == "importance":
            self.logger.info(
                f"Building network using importance score filtering with threshold: {self.config.importance_threshold}"
            )
        else:
            self.logger.info("Building full network (importance score > 0.000)")

        # Find all gene folders
        gene_folders = [
            f
            for f in os.listdir(self.models_dir)
            if os.path.isdir(os.path.join(self.models_dir, f)) and not f.startswith(".")
        ]

        self.logger.info(f"Found {len(gene_folders)} gene folders to process")

        if not gene_folders:
            raise ValueError(f"No gene folders found in {self.models_dir}")

        # Process each gene folder
        network_relationships = []
        processed_genes = 0
        skipped_genes = 0

        for regulated_gene in tqdm(gene_folders, desc="Processing gene folders"):
            try:
                relationships = self._process_gene_folder(regulated_gene)
                network_relationships.extend(relationships)
                processed_genes += 1
            except Exception as e:
                self.logger.warning(f"Error processing {regulated_gene}: {str(e)}")
                skipped_genes += 1
                continue

        # Create network DataFrame
        if network_relationships:
            network_df = pd.DataFrame(network_relationships)

            # Apply additional filtering
            network_df = self._apply_post_filtering(network_df)

            # Save network
            network_df.to_csv(output_file, index=False)

            # Generate summary statistics
            summary = self._generate_network_summary(network_df)

            results = {
                "network_file": output_file,
                "total_relationships": len(network_df),
                "processed_genes": processed_genes,
                "skipped_genes": skipped_genes,
                "filtering_method": self.config.filter_method,
                "threshold": self._get_threshold(),
                "summary_statistics": summary,
            }

            self.logger.info(
                f"Network building completed: {len(network_df)} relationships"
            )
            self.logger.info(f"Network file created: {output_file}")

            # Log summary statistics
            self._log_summary_statistics(summary)

            # Save detailed results
            results_file = output_file.replace(".csv", "_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            return results

        else:
            raise ValueError("No valid relationships found meeting the criteria")

    def _process_gene_folder(self, regulated_gene: str) -> List[Dict[str, Any]]:
        """Process a single gene folder to extract relationships"""
        csv_path = os.path.join(
            self.models_dir, regulated_gene, "feature_importance.csv"
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"feature_importance.csv not found for {regulated_gene}"
            )

        # Read feature importance CSV
        df = pd.read_csv(csv_path)

        # Find column names (handle variations) - more flexible approach
        gene_col = self._find_column(df, ["gene"])
        importance_col = self._find_column(df, ["importance", "score"])

        if gene_col is None or importance_col is None:
            raise ValueError(f"Could not find required columns in {csv_path}")

        # Apply filtering based on method
        if self.config.filter_method == "zscore":
            filtered_df = self._apply_zscore_filtering(df, importance_col)
        elif self.config.filter_method == "importance":
            filtered_df = self._apply_importance_filtering(df, importance_col)
        else:  # full network
            filtered_df = self._apply_full_filtering(df, importance_col)

        # Create relationships
        relationships = []
        for _, row in filtered_df.iterrows():
            relationships.append(
                {
                    "regulator_gene": row[gene_col],
                    "regulated_gene": regulated_gene,
                    "importance_score": float(row[importance_col]),
                }
            )

        return relationships

    def _find_column(
        self, df: pd.DataFrame, possible_names: List[str]
    ) -> Optional[str]:
        """Find column by possible names (case insensitive)"""
        for name in possible_names:
            for col in df.columns:
                if name.lower() in col.lower():
                    return col
        return None

    def _apply_zscore_filtering(
        self, df: pd.DataFrame, importance_col: str
    ) -> pd.DataFrame:
        """Apply z-score based filtering"""
        # Calculate z-scores
        mean_score = df[importance_col].mean()
        std_score = df[importance_col].std()

        if std_score == 0:
            # If std is 0, return empty dataframe
            return df.iloc[0:0]

        df = df.copy()
        df["z_score"] = (df[importance_col] - mean_score) / std_score

        # Filter by z-score threshold
        return df[df["z_score"] > self.config.zscore_threshold]

    def _apply_importance_filtering(
        self, df: pd.DataFrame, importance_col: str
    ) -> pd.DataFrame:
        """Apply importance threshold filtering"""
        return df[df[importance_col] > self.config.importance_threshold]

    def _apply_full_filtering(
        self, df: pd.DataFrame, importance_col: str
    ) -> pd.DataFrame:
        """Apply minimal filtering for full network (importance > 0.000)"""
        return df[df[importance_col] > 0.000]

    def _apply_post_filtering(self, network_df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional post-processing filters"""
        # Remove self-loops
        network_df = network_df[
            network_df["regulator_gene"] != network_df["regulated_gene"]
        ]

        # Apply minimum connections filter if specified
        if hasattr(self.config, "min_connections") and self.config.min_connections > 0:
            gene_counts = network_df["regulated_gene"].value_counts()
            valid_genes = gene_counts[gene_counts >= self.config.min_connections].index
            network_df = network_df[network_df["regulated_gene"].isin(valid_genes)]

        return network_df

    def _generate_network_summary(self, network_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the network"""
        return {
            "total_relationships": len(network_df),
            "unique_regulators": network_df["regulator_gene"].nunique(),
            "unique_regulated": network_df["regulated_gene"].nunique(),
            "avg_importance_score": float(network_df["importance_score"].mean()),
            "importance_score_std": float(network_df["importance_score"].std()),
            "min_importance_score": float(network_df["importance_score"].min()),
            "max_importance_score": float(network_df["importance_score"].max()),
            "avg_connections_per_gene": len(network_df)
            / network_df["regulated_gene"].nunique(),
            "importance_score_range": f"{network_df['importance_score'].min():.6f} - {network_df['importance_score'].max():.6f}",
            "top_relationships": network_df.nlargest(10, "importance_score")[
                ["regulator_gene", "regulated_gene", "importance_score"]
            ].to_dict("records"),
            "top_regulators": network_df["regulator_gene"]
            .value_counts()
            .head(10)
            .to_dict(),
            "top_regulated": network_df["regulated_gene"]
            .value_counts()
            .head(10)
            .to_dict(),
        }

    def _log_summary_statistics(self, summary: Dict[str, Any]) -> None:
        """Log summary statistics"""
        self.logger.info("=" * 60)
        self.logger.info("NETWORK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total relationships: {summary['total_relationships']}")
        self.logger.info(f"Unique regulator genes: {summary['unique_regulators']}")
        self.logger.info(f"Unique regulated genes: {summary['unique_regulated']}")
        self.logger.info(
            f"Average importance score: {summary['avg_importance_score']:.6f}"
        )
        self.logger.info(f"Importance score range: {summary['importance_score_range']}")

        if summary["top_relationships"]:
            self.logger.info("\nTop 10 relationships by importance score:")
            for rel in summary["top_relationships"]:
                self.logger.info(
                    f"  {rel['regulator_gene']} -> {rel['regulated_gene']}: {rel['importance_score']:.6f}"
                )

    def _get_threshold(self) -> float:
        """Get the current threshold value"""
        if self.config.filter_method == "zscore":
            return self.config.zscore_threshold
        elif self.config.filter_method == "importance":
            return self.config.importance_threshold
        else:  # full
            return 0.000


def build_network_from_models(
    models_dir: str,
    filter_method: str = "zscore",
    zscore_threshold: float = 2.0,
    importance_threshold: float = 0.000,
    output_file: Optional[str] = None,
) -> str:
    """
    Convenience function to build network from models directory

    Args:
        models_dir: Directory containing trained models
        filter_method: Filtering method ('zscore', 'importance', or 'full')
        zscore_threshold: Z-score threshold for z-score filtering (default: 2.0)
        importance_threshold: Importance threshold for importance filtering (default: 0.000)
        output_file: Optional output file path

    Returns:
        Path to generated network file
    """
    config = NetworkConfig(
        filter_method=filter_method,
        zscore_threshold=zscore_threshold,
        importance_threshold=importance_threshold,
    )

    builder = GeneRegulatoryNetworkBuilder(models_dir, config)
    results = builder.build_network(output_file)

    return results["network_file"]


def create_full_network(models_dir: str, output_file: Optional[str] = None) -> str:
    """
    Convenience function to create a full network (importance > 0.000)

    Args:
        models_dir: Directory containing trained models
        output_file: Optional output file path

    Returns:
        Path to generated network file
    """
    return build_network_from_models(
        models_dir=models_dir, filter_method="full", output_file=output_file
    )


def create_zscore_network(
    models_dir: str, zscore_threshold: float = 2.0, output_file: Optional[str] = None
) -> str:
    """
    Convenience function to create a z-score filtered network

    Args:
        models_dir: Directory containing trained models
        zscore_threshold: Z-score cutoff threshold (default: 2.0)
        output_file: Optional output file path

    Returns:
        Path to generated network file
    """
    return build_network_from_models(
        models_dir=models_dir,
        filter_method="zscore",
        zscore_threshold=zscore_threshold,
        output_file=output_file,
    )


def create_importance_network(
    models_dir: str,
    importance_threshold: float = 0.000,
    output_file: Optional[str] = None,
) -> str:
    """
    Convenience function to create an importance threshold filtered network

    Args:
        models_dir: Directory containing trained models
        importance_threshold: Minimum importance score threshold (default: 0.000)
        output_file: Optional output file path

    Returns:
        Path to generated network file
    """
    return build_network_from_models(
        models_dir=models_dir,
        filter_method="importance",
        importance_threshold=importance_threshold,
        output_file=output_file,
    )
