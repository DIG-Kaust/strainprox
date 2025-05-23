#!/bin/bash

# Exit on error
set -e

echo "Installing StrainProx environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "environment.yml not found. Make sure you're in the root directory of the project."
    exit 1
fi

# Create the conda environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment and install the package in development mode
echo "Installing package in development mode..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate strainprox
pip install -e .

echo "Done! StrainProx environment is now installed."
echo "Activate the environment with: conda activate strainprox" 