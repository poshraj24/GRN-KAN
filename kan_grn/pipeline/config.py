"""
Configuration classes for KAN-GRN pipeline
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for KAN model architecture"""

    width: List[int] = field(
        default_factory=lambda: [None, 2, 1]
    )  # Input size determined automatically
    grid: int = 5
    k: int = 4
    seed: int = 63

    def __post_init__(self):
        """Validate model configuration"""
        if self.grid <= 0:
            raise ValueError("grid must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    batch_size: int = 512
    epochs: int = 50
    patience: int = 5
    learning_rate: float = 0.0001
    max_models: int = 6
    generate_symbolic: bool = True

    # Optimizer settings
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Scheduler settings
    scheduler: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-7

    def __post_init__(self):
        """Validate training configuration"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class NetworkConfig:
    """Configuration for network building"""

    filter_method: str = "zscore"  # "zscore", "importance", or "full"
    zscore_threshold: float = 2.0
    importance_threshold: float = 0.000
    min_connections: int = 1

    def __post_init__(self):
        """Validate network configuration"""
        if self.filter_method not in ["zscore", "importance", "full"]:
            raise ValueError("filter_method must be 'zscore', 'importance', or 'full'")
        if self.zscore_threshold < 0:
            raise ValueError("zscore_threshold must be positive")


@dataclass
class PipelineConfig:
    """Main configuration for KAN-GRN pipeline"""

    # Input files
    expression_file: str
    network_file: str

    # Output settings
    output_dir: str = "kan_grn_results"

    # Data settings
    n_top_genes: int = 2000
    max_genes: Optional[int] = None  # Limit number of genes for testing

    # Hardware settings
    device: str = "cuda"
    scratch_dir: Optional[str] = None

    # Pipeline settings
    resume_from_checkpoint: bool = True

    # Sub-configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    network_config: NetworkConfig = field(default_factory=NetworkConfig)

    def __post_init__(self):
        """Validate and process configuration"""
        if self.n_top_genes <= 0:
            raise ValueError("n_top_genes must be positive")

        # Ensure output directory ends with trailing slash
        self.output_dir = self.output_dir.rstrip("/").rstrip("\\")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary"""
        # Extract sub-configs
        model_config = ModelConfig(**config_dict.pop("model_config", {}))
        training_config = TrainingConfig(**config_dict.pop("training_config", {}))
        network_config = NetworkConfig(**config_dict.pop("network_config", {}))

        return cls(
            model_config=model_config,
            training_config=training_config,
            network_config=network_config,
            **config_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict

        return asdict(self)

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "PipelineConfig":
        """Load configuration from JSON file"""
        import json

        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
