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
            output_file = os.path.join(
                self.models_dir,
                f"gene_regulatory_network_{self.config.filter_method}_{self._get_threshold()}.csv",
            )

        self.logger.info(
            f"Building network using {self.config.filter_method} filtering"
        )
        self.logger.info(f"Threshold: {self._get_threshold()}")

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

        # Find column names (handle variations)
        gene_col = self._find_column(df, ["gene", "feature", "regulator"])
        importance_col = self._find_column(df, ["importance", "score", "weight"])

        if gene_col is None or importance_col is None:
            raise ValueError(f"Could not find required columns in {csv_path}")

        # Apply filtering
        if self.config.filter_method == "zscore":
            filtered_df = self._apply_zscore_filtering(df, importance_col)
        else:
            filtered_df = self._apply_importance_filtering(df, importance_col)

        # Create relationships
        relationships = []
        for _, row in filtered_df.iterrows():
            relationships.append(
                {
                    "regulator_gene": row[gene_col],
                    "regulated_gene": regulated_gene,
                    "importance_score": float(row[importance_col]),
                    "raw_score": float(row[importance_col]),
                }
            )

        return relationships

    def _find_column(
        self, df: pd.DataFrame, possible_names: List[str]
    ) -> Optional[str]:
        """Find column by possible names (case insensitive)"""
        df_columns_lower = [col.lower() for col in df.columns]

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
            "top_regulators": network_df["regulator_gene"]
            .value_counts()
            .head(10)
            .to_dict(),
            "top_regulated": network_df["regulated_gene"]
            .value_counts()
            .head(10)
            .to_dict(),
        }

    def _get_threshold(self) -> float:
        """Get the current threshold value"""
        if self.config.filter_method == "zscore":
            return self.config.zscore_threshold
        else:
            return self.config.importance_threshold


def build_network_from_models(
    models_dir: str,
    filter_method: str = "zscore",
    threshold: float = 1.0,
    output_file: Optional[str] = None,
) -> str:
    """
    Convenience function to build network from models directory

    Args:
        models_dir: Directory containing trained models
        filter_method: Filtering method ('zscore' or 'importance')
        threshold: Threshold value for filtering
        output_file: Optional output file path

    Returns:
        Path to generated network file
    """
    if filter_method == "zscore":
        config = NetworkConfig(filter_method=filter_method, zscore_threshold=threshold)
    else:
        config = NetworkConfig(
            filter_method=filter_method, importance_threshold=threshold
        )

    builder = GeneRegulatoryNetworkBuilder(models_dir, config)
    results = builder.build_network(output_file)

    return results["network_file"]
