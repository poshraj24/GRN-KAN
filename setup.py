#!/usr/bin/env python3
"""
Setup script for KAN-GRN package
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure we're running Python 3.8+
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required.")

# Get the package directory
package_dir = Path(__file__).parent

# Read the README file
readme_file = package_dir / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = (
        "Gene Regulatory Network inference using Kolmogorov-Arnold Networks"
    )

# Read requirements
requirements_file = package_dir / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
else:
    # Fallback requirements if file doesn't exist
    requirements = [
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "scanpy>=1.7.0",
        "tqdm>=4.50.0",
        "psutil>=5.7.0",
        "h5py>=3.1.0",
        "anndata>=0.7.0",
    ]

# Read version from package
version_file = package_dir / "kan_grn" / "__init__.py"
version = "1.0.0"  # Default version
if version_file.exists():
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

# Development requirements
dev_requirements = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=21.0",
    "flake8>=3.8",
    "mypy>=0.812",
    "sphinx>=4.0",
    "sphinx-rtd-theme>=0.5",
    "jupyter>=1.0.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
]

# GPU requirements (CUDA-enabled PyTorch)
gpu_requirements = [
    "torch>=1.9.0+cu111",
    "torchvision>=0.10.0+cu111",
]

# HPC requirements
hpc_requirements = [
    "mpi4py>=3.0.0",
    "dask[distributed]>=2021.0.0",
    "joblib>=1.0.0",
]

setup(
    # Basic package information
    name="kan_grn",
    version=version,
    # Author information
    author="Posh Raj Dahal",
    author_email="dahal.poshraj24@gmail.com",
    maintainer="Posh Raj Dahal",
    maintainer_email="dahal.poshraj24@gmail.com",
    # Package description
    description="Gene Regulatory Network inference using Kolmogorov-Arnold Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # URLs
    url="https://github.com/poshraj24/GRN-KAN",
    download_url="https://github.com/poshraj24/GRN-KAN/archive/v{}.tar.gz".format(
        version
    ),
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    # Package data
    include_package_data=True,
    package_data={
        "kan_grn": [
            "data/*.json",
            "data/*.txt",
            "config/*.json",
            "examples/*.py",
        ],
    },
    # Requirements
    python_requires=">=3.8",
    install_requires=requirements,
    # Optional requirements
    extras_require={
        "dev": dev_requirements,
        "gpu": gpu_requirements,
        "hpc": hpc_requirements,
        "all": dev_requirements + hpc_requirements,
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
            "myst-parser>=0.15",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "networkx>=2.5",
        ],
    },
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "kan-grn=kan_grn.cli.commands:main",
            "kan-grn-train=kan_grn.cli.commands:train_models",
            "kan-grn-build-network=kan_grn.cli.commands:build_network",
            "kan-grn-config=kan_grn.cli.commands:create_config",
        ],
    },
    # Classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        # Topic
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # License
        "License :: OSI Approved :: MIT License",
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        # Environment
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        # Natural Language
        "Natural Language :: English",
    ],
    # Keywords for easier discovery
    keywords=[
        "gene regulatory networks",
        "bioinformatics",
        "machine learning",
        "kolmogorov arnold networks",
        "single cell RNA sequencing",
        "systems biology",
        "network inference",
        "symbolic regression",
        "computational biology",
        "genomics",
    ],
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/poshraj24/GRN-KAN/issues",
        "Source": "https://github.com/poshraj24/GRN-KAN",
        "Documentation": "https://kan-grn.readthedocs.io/",
        "Changelog": "https://github.com/poshraj24/GRN-KAN/blob/main/CHANGELOG.md",
        "Examples": "https://github.com/poshraj24/GRN-KAN/tree/main/examples",
        "CI/CD": "https://github.com/poshraj24/GRN-KAN/actions",
    },
    # Additional metadata
    license="MIT",
    platforms=["any"],
    # Zip safe
    zip_safe=False,
    # Test suite
    test_suite="tests",
    tests_require=[
        "pytest>=6.0",
        "pytest-cov>=2.0",
    ],
    # Command test
    cmdclass={},
)

# Post-installation message
print(
    """
🎉 KAN-GRN installation completed!

Quick Start:
1. Create a config file: kan-grn create-config
2. Run the pipeline: kan-grn run expression.h5ad network.tsv
3. Get help: kan-grn --help

For examples and documentation, visit:
https://github.com/poshraj24/GRN-KAN

If you encounter any issues, please report them at:
https://github.com/poshraj24/GRN-KAN/issues
"""
)
