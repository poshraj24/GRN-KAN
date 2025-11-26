# KAN-GRN

**Gene Regulatory Network Inference using Kolmogorov-Arnold Networks**

KAN-GRN infers Gene Regulatory Networks (GRNs) from single-cell RNA sequencing data using Kolmogorov-Arnold Networks (KANs).

## Installation

```bash
pip install kan-grn
```

**For GPU support:**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install kan-grn
```

## Quick Start

### Command Line

```bash
# Basic usage
kan-grn run expression_data.h5ad network_file.tsv

# With custom parameters
kan-grn run expression_data.h5ad network_file.tsv --output-dir results --n-top-genes 3000 --epochs 100 --device cuda
```

### Python API

```python
import kan_grn

pipeline_config = kan_grn.PipelineConfig(
    expression_file="expression_data.h5ad",
    network_file="network.tsv",
    output_dir="results",
    n_top_genes=3000,
    device="cuda",
    model_config=kan_grn.ModelConfig(grid=5, k=4),
    training_config=kan_grn.TrainingConfig(
        batch_size=128,
        epochs=100,
        learning_rate=0.001,
        generate_symbolic=True,
    ),
)

pipeline = kan_grn.KANGRNPipeline(pipeline_config)
results = pipeline.train_models_only()
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `results` | Output directory |
| `--n-top-genes` | `2000` | Top highly variable genes |
| `--grid` | `5` | KAN grid parameter |
| `--k` | `4` | KAN k parameter |
| `--batch-size` | `128` | Batch size |
| `--epochs` | `100` | Max epochs |
| `--learning-rate` | `0.001` | Learning rate |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--filter-method` | `zscore` | Network filtering method |

## Input Formats

- **Expression data**: `.h5ad` (AnnData) with cells × genes matrix
- **Network file**: `.tsv` with columns: `source_gene`, `target_gene`, `weight`

## Requirements

Python ≥ 3.8, PyTorch ≥ 1.9.0, Scanpy ≥ 1.9.0, PyKAN

## Links

- [GitHub Repository](https://github.com/poshraj24/GRN-KAN)
- [Documentation](https://github.com/poshraj24/GRN-KAN#readme)
- [Issues](https://github.com/poshraj24/GRN-KAN/issues)

## License

GPL-3.0

## Author

Posh Raj Dahal (dahal.poshraj24@gmail.com)