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
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#") and ";" not in line
        ]
else:
    # Fallback requirements matching your current setup
    requirements = [
        "numpy>=1.21.0,<3.0.0",
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "scanpy>=1.9.0",
        "tqdm>=4.50.0",
        "psutil>=5.7.0",
        "h5py>=3.1.0",
        "anndata>=0.8.0",
    ]

# Read version from package
version = "1.0.1"  # Updated version

setup(
    # Basic package information
    name="kan-grn",
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
    download_url="https://github.com/poshraj24/GRN-KAN/archive/v1.0.1.tar.gz",
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    # Package data
    include_package_data=True,
    package_data={
        "kan_grn": [
            "data/*.json",
            "data/*.txt",
        ],
    },
    # Requirements
    python_requires=">=3.8",
    install_requires=requirements,
    # Optional requirements - Updated for PyTorch 2.5.0
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=2.5.0",
            "torchvision>=0.20.0",
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
        ],
    },
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
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
    ],
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/poshraj24/GRN-KAN/issues",
        "Source": "https://github.com/poshraj24/GRN-KAN",
        "Documentation": "https://github.com/poshraj24/GRN-KAN/blob/main/README.md",
    },
    # Additional metadata
    license="GPL-3.0",
    zip_safe=False,
)
